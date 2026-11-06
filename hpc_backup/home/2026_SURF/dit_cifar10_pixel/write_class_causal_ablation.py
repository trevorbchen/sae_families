#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import defaultdict

import torch
from torchvision import datasets, transforms

from train_dit_cifar import DiT, Diffusion, modulate
from write_onoff_ablation import TopKSAE, load_pt, parse_ints


T_VALUES = (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999)
CLASS_NAMES = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def load_model(path, device):
    model = DiT(learn_sigma=True).to(device)
    ckpt = load_pt(path, device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_write_sae(block_index, stream, block0_ckpt, remaining_ckpt, device, input_dim=384, dict_size=6144, k=32):
    sae = TopKSAE(input_dim=input_dim, dict_size=dict_size, k=k).to(device)
    if block_index == 0:
        ckpt = load_pt(block0_ckpt, device)
        key = "attn_sae" if stream == "attn" else "mlp_sae"
        sae.load_state_dict(ckpt[key])
    else:
        ckpt = load_pt(remaining_ckpt, device)
        sae.load_state_dict(ckpt["saes"][f"b{block_index:02d}_{stream}"])
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae


def load_all_saes(args, device):
    out = {}
    for b in range(12):
        for stream in ("attn", "mlp"):
            out[(b, stream)] = load_write_sae(b, stream, args.block0_sae_ckpt, args.remaining_sae_ckpt, device)
    return out


def load_cifar(data_dir):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    return datasets.CIFAR10(data_dir, train=True, transform=transform, download=True)


def make_class_batch(dataset, class_id, num_images, seed, device):
    gen = torch.Generator().manual_seed(seed + class_id * 100003)
    pool = [i for i, y in enumerate(dataset.targets) if int(y) == int(class_id)]
    indices = [pool[i] for i in torch.randperm(len(pool), generator=gen)[:num_images].tolist()]
    xs = []
    for idx in indices:
        x, _ = dataset[idx]
        xs.append(x)
    return torch.stack(xs).to(device), torch.full((len(xs),), class_id, device=device, dtype=torch.long), indices


def make_balanced_batch(dataset, per_class, seed, device):
    xs, ys, indices = [], [], []
    for class_id in range(10):
        x, y, idx = make_class_batch(dataset, class_id, per_class, seed + 1000000, device)
        xs.append(x)
        ys.append(y)
        indices.extend(idx)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0), indices


@torch.no_grad()
def encode_dense(sae, x_flat):
    pre = sae.encoder(x_flat - sae.dec_bias)
    acts = torch.relu(pre)
    vals, idx = torch.topk(acts, sae.k, dim=-1)
    dense = torch.zeros(x_flat.shape[0], sae.dict_size, device=x_flat.device, dtype=x_flat.dtype)
    dense.scatter_(1, idx, vals)
    return dense.reshape(-1, 256, sae.dict_size)


@torch.no_grad()
def model_eps(model, x_t, t, labels, precision):
    enabled = x_t.device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type=x_t.device.type, dtype=torch.bfloat16, enabled=enabled):
        out = model(x_t, t, labels)
    return torch.chunk(out.float(), 2, dim=1)[0]


@torch.no_grad()
def one_vs_rest_direction(model, x_t, t, class_id, precision):
    b = x_t.shape[0]
    eps_c = model_eps(model, x_t, t, torch.full((b,), class_id, device=x_t.device, dtype=torch.long), precision)
    others = []
    for other in range(10):
        if other == class_id:
            continue
        y = torch.full((b,), other, device=x_t.device, dtype=torch.long)
        others.append(model_eps(model, x_t, t, y, precision))
    d = eps_c - torch.stack(others, dim=0).mean(dim=0)
    d = d / d.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
    return d.detach(), eps_c.detach()


@torch.no_grad()
def baseline_means_for_t(model, diffusion, saes, x0, labels, t, precision, noise):
    x_t = diffusion.q_sample(x0, t, noise)
    enabled = x_t.device.type == "cuda" and precision == "bf16"
    means = {}
    with torch.autocast(device_type=x_t.device.type, dtype=torch.bfloat16, enabled=enabled):
        x = model.x_embedder(x_t).flatten(2).transpose(1, 2) + model.pos_embed
        c = model.t_embedder(t) + model.y_embedder(labels)
        for b, block in enumerate(model.blocks):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaLN(c).chunk(6, dim=1)
            attn_in = modulate(block.norm1(x), shift_msa, scale_msa)
            attn_out = block.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
            attn_write = gate_msa.unsqueeze(1) * attn_out
            means[(b, "attn")] = encode_dense(saes[(b, "attn")], attn_write.float().flatten(0, 1)).mean(dim=(0, 1)).detach()
            x = x + attn_write
            mlp_in = modulate(block.norm2(x), shift_mlp, scale_mlp)
            mlp_out = block.mlp(mlp_in)
            mlp_write = gate_mlp.unsqueeze(1) * mlp_out
            means[(b, "mlp")] = encode_dense(saes[(b, "mlp")], mlp_write.float().flatten(0, 1)).mean(dim=(0, 1)).detach()
            x = x + mlp_write
    return means


@torch.no_grad()
def ablate_write(write, sae, features, baseline_mean):
    if not features:
        return write
    z = encode_dense(sae, write.float().flatten(0, 1))
    clean_recon = torch.einsum("btd,dh->bth", z.float(), sae.decoder.float()) + sae.dec_bias.float()
    frozen_error = write.float() - clean_recon
    z_ablate = z.float().clone()
    f = torch.tensor(features, device=write.device, dtype=torch.long)
    z_ablate[:, :, f] = baseline_mean[f].view(1, 1, -1)
    recon = torch.einsum("btd,dh->bth", z_ablate, sae.decoder.float()) + sae.dec_bias.float()
    return recon + frozen_error


@torch.no_grad()
def forward_with_write_ablation(model, saes, x_t, t, labels, selected_by_site, baseline_means, precision):
    enabled = x_t.device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type=x_t.device.type, dtype=torch.bfloat16, enabled=enabled):
        x = model.x_embedder(x_t).flatten(2).transpose(1, 2) + model.pos_embed
        c = model.t_embedder(t) + model.y_embedder(labels)
        for b, block in enumerate(model.blocks):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaLN(c).chunk(6, dim=1)
            attn_in = modulate(block.norm1(x), shift_msa, scale_msa)
            attn_out = block.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
            attn_write = gate_msa.unsqueeze(1) * attn_out
            attn_features = selected_by_site.get((b, "attn"), [])
            if attn_features:
                attn_write = ablate_write(attn_write.float(), saes[(b, "attn")], attn_features, baseline_means[(b, "attn")])
            x = x + attn_write
            mlp_in = modulate(block.norm2(x), shift_mlp, scale_mlp)
            mlp_out = block.mlp(mlp_in)
            mlp_write = gate_mlp.unsqueeze(1) * mlp_out
            mlp_features = selected_by_site.get((b, "mlp"), [])
            if mlp_features:
                mlp_write = ablate_write(mlp_write.float(), saes[(b, "mlp")], mlp_features, baseline_means[(b, "mlp")])
            x = x + mlp_write
        out = model.unpatchify(model.final_layer(x, c))
    return torch.chunk(out.float(), 2, dim=1)[0]


def load_ranked_features(path, min_active, min_abs_attr, min_hits):
    by_class = defaultdict(list)
    seen = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if float(r.get("mean_active_frac", 0.0)) < min_active:
                continue
            if float(r.get("mean_abs_attribution", 0.0)) < min_abs_attr:
                continue
            if int(float(r.get("n_timestep_hits", 0))) < min_hits:
                continue
            cid = int(r["class_id"])
            block = int(r["block"].replace("block", ""))
            stream = r["stream"]
            feature = int(r["feature"])
            key = (block, stream, feature)
            if key in seen[cid]:
                continue
            seen[cid].add(key)
            by_class[cid].append({
                "class_id": cid,
                "class_name": r["class_name"],
                "block": block,
                "stream": stream,
                "feature": feature,
                "z": float(r.get("class_specific_z_filled", 0.0)),
                "mean_abs_attribution": float(r["mean_abs_attribution"]),
                "mean_active_frac": float(r["mean_active_frac"]),
            })
    for cid, rows in by_class.items():
        rows.sort(key=lambda r: (r["z"], r["mean_abs_attribution"], r["mean_active_frac"]), reverse=True)
    return by_class


def selected_sites(rows, k):
    out = defaultdict(list)
    for r in rows[:k]:
        out[(r["block"], r["stream"])].append(r["feature"])
    return dict(out)


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    dataset = load_cifar(args.data_dir)
    model = load_model(args.ckpt, device)
    diffusion = Diffusion(args.timesteps, device=device)
    saes = load_all_saes(args, device)
    ranked = load_ranked_features(args.feature_csv, args.min_active, args.min_abs_attribution, args.min_hits)
    classes = [int(x) for x in args.classes.split(",") if x.strip()]
    t_values = [int(x) for x in args.t_values]
    set_sizes = [int(x) for x in args.set_sizes.split(",") if x.strip()]

    x_base, y_base, _ = make_balanced_batch(dataset, args.baseline_per_class, args.seed + 77, device)
    rows = []
    fields = [
        "class_id", "class_name", "t", "set_size", "actual_set_size", "n_sites",
        "m_clean_mean", "m_ablate_mean", "m_drop_mean", "relative_drop",
        "m_clean_std", "m_ablate_std", "m_drop_std",
    ]

    baseline_cache = {}
    for t_int in t_values:
        t_base = torch.full((x_base.shape[0],), t_int, device=device, dtype=torch.long)
        gen_base = torch.Generator(device=device).manual_seed(args.seed + 50000 + t_int)
        noise_base = torch.randn(x_base.shape, device=device, generator=gen_base)
        baseline_cache[t_int] = baseline_means_for_t(model, diffusion, saes, x_base, y_base, t_base, args.precision, noise_base)
        print(f"computed same-t global means t={t_int}", flush=True)

    for class_id in classes:
        feats = ranked.get(class_id, [])
        if not feats:
            print(f"no selected features for class={class_id}", flush=True)
            continue
        x0, labels, _ = make_class_batch(dataset, class_id, args.num_images, args.seed, device)
        for t_int in t_values:
            t = torch.full((x0.shape[0],), t_int, device=device, dtype=torch.long)
            gen = torch.Generator(device=device).manual_seed(args.seed + class_id * 1000 + t_int)
            noise = torch.randn(x0.shape, device=device, generator=gen)
            x_t = diffusion.q_sample(x0, t, noise)
            direction, eps_clean = one_vs_rest_direction(model, x_t, t, class_id, args.precision)
            m_clean = (eps_clean * direction).flatten(1).sum(dim=1)
            for set_size in set_sizes:
                chosen = selected_sites(feats, min(set_size, len(feats)))
                eps_ablate = forward_with_write_ablation(
                    model, saes, x_t, t, labels, chosen, baseline_cache[t_int], args.precision
                )
                m_ablate = (eps_ablate * direction).flatten(1).sum(dim=1)
                drop = m_clean - m_ablate
                rec = {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "t": t_int,
                    "set_size": set_size,
                    "actual_set_size": min(set_size, len(feats)),
                    "n_sites": len(chosen),
                    "m_clean_mean": float(m_clean.mean().cpu()),
                    "m_ablate_mean": float(m_ablate.mean().cpu()),
                    "m_drop_mean": float(drop.mean().cpu()),
                    "relative_drop": float((drop.mean() / m_clean.abs().mean().clamp_min(1e-12)).cpu()),
                    "m_clean_std": float(m_clean.std(unbiased=False).cpu()),
                    "m_ablate_std": float(m_ablate.std(unbiased=False).cpu()),
                    "m_drop_std": float(drop.std(unbiased=False).cpu()),
                }
                rows.append(rec)
                print(
                    f"class={class_id} t={t_int} k={set_size} drop={rec['m_drop_mean']:.6g} rel={rec['relative_drop']:.4g}",
                    flush=True,
                )

    write_csv(os.path.join(args.out_root, "causal_ablation_by_class_timestep.csv"), rows, fields)

    summary = []
    by_class_k = defaultdict(list)
    for r in rows:
        by_class_k[(r["class_id"], r["class_name"], r["set_size"])].append(r)
    for (cid, cname, k), rs in by_class_k.items():
        summary.append({
            "class_id": cid,
            "class_name": cname,
            "set_size": k,
            "mean_drop_over_t": sum(float(r["m_drop_mean"]) for r in rs) / len(rs),
            "mean_relative_drop_over_t": sum(float(r["relative_drop"]) for r in rs) / len(rs),
            "max_relative_drop": max(float(r["relative_drop"]) for r in rs),
            "argmax_t": max(rs, key=lambda r: float(r["relative_drop"]))["t"],
            "n_timesteps": len(rs),
        })
    summary.sort(key=lambda r: (r["set_size"], r["class_id"]))
    write_csv(
        os.path.join(args.out_root, "causal_ablation_summary_by_class_setsize.csv"),
        summary,
        ["class_id", "class_name", "set_size", "mean_drop_over_t", "mean_relative_drop_over_t",
         "max_relative_drop", "argmax_t", "n_timesteps"],
    )
    with open(os.path.join(args.out_root, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Write Class Causal Necessity Ablation\n\n"
            "This is the exact causal follow-up to the write class-direction attribution filter.\n\n"
            "For each class, selected class-specific write SAE features are jointly ablated by setting their coefficients to "
            "the same-t mixed-class/global mean, not zero. The write is reconstructed as `decoder(z_ablated) + frozen_error`.\n\n"
            "Metric: `M_c = <eps_c, normalize(eps_c - mean eps_other)>`. Reported drop is `M_clean - M_ablate`; positive means "
            "the ablation reduced one-vs-rest class-c movement.\n"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/dit_s2_cifar10/last.pt")
    p.add_argument("--block0-sae-ckpt", default="runs/sae_block0_writes_16x_k32/last.pt")
    p.add_argument("--remaining-sae-ckpt", default="runs/sae_remaining_block_writes_16x_k32/last.pt")
    p.add_argument("--feature-csv", default="runs/write_class_direction_attribution_16x_k32/feature_class_specificity_zscore_filled_filtered.csv")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-root", default="runs/write_class_causal_ablation_16x_k32")
    p.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--t-values", type=parse_ints, default=T_VALUES)
    p.add_argument("--set-sizes", default="1,2,5,10,20,50,100")
    p.add_argument("--num-images", type=int, default=12)
    p.add_argument("--baseline-per-class", type=int, default=4)
    p.add_argument("--min-active", type=float, default=0.005)
    p.add_argument("--min-abs-attribution", type=float, default=1e-5)
    p.add_argument("--min-hits", type=int, default=2)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
