#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
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
    acts = F.relu(pre)
    vals, idx = torch.topk(acts, sae.k, dim=-1)
    dense = torch.zeros(x_flat.shape[0], sae.dict_size, device=x_flat.device, dtype=x_flat.dtype)
    dense.scatter_(1, idx, vals)
    return dense.reshape(-1, 256, sae.dict_size)


@torch.no_grad()
def prefix_to_write(model, diffusion, x0, labels, t, block_index, stream, precision, noise):
    x_t = diffusion.q_sample(x0, t, noise)
    enabled = x_t.device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type=x_t.device.type, dtype=torch.bfloat16, enabled=enabled):
        x = model.x_embedder(x_t).flatten(2).transpose(1, 2) + model.pos_embed
        c = model.t_embedder(t) + model.y_embedder(labels)
        for i in range(block_index):
            x = model.blocks[i](x, c)
        block = model.blocks[block_index]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaLN(c).chunk(6, dim=1)
        attn_in = modulate(block.norm1(x), shift_msa, scale_msa)
        attn_out = block.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        attn_write = gate_msa.unsqueeze(1) * attn_out
        x_after_attn = x + attn_write
        mlp_in = modulate(block.norm2(x_after_attn), shift_mlp, scale_mlp)
        mlp_out = block.mlp(mlp_in)
        mlp_write = gate_mlp.unsqueeze(1) * mlp_out
        x_after_block = x_after_attn + mlp_write
    return {
        "x_t": x_t.float(),
        "c": c.float(),
        "x_before_block": x.float(),
        "x_after_attn": x_after_attn.float(),
        "x_after_block": x_after_block.float(),
        "write": (attn_write if stream == "attn" else mlp_write).float(),
    }


def mlp_write_from_after_attn_grad(model, x_after_attn, c, block_index, precision):
    enabled = x_after_attn.device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type=x_after_attn.device.type, dtype=torch.bfloat16, enabled=enabled):
        block = model.blocks[block_index]
        _, _, _, shift_mlp, scale_mlp, gate_mlp = block.adaLN(c).chunk(6, dim=1)
        mlp_in = modulate(block.norm2(x_after_attn), shift_mlp, scale_mlp)
        mlp_out = block.mlp(mlp_in)
        return gate_mlp.unsqueeze(1) * mlp_out


def finish_after_block_grad(model, x, c, block_index, precision):
    enabled = x.device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=enabled):
        for i in range(block_index + 1, len(model.blocks)):
            x = model.blocks[i](x, c)
        return model.unpatchify(model.final_layer(x, c))


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
    return d.detach()


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(rows, out_dir):
    agg = defaultdict(lambda: {
        "score_sum": 0.0,
        "abs_score_sum": 0.0,
        "grad_abs_sum": 0.0,
        "disp_abs_sum": 0.0,
        "active_sum": 0.0,
        "n": 0,
    })
    for r in rows:
        key = (r["class_id"], r["class_name"], r["block"], r["stream"], r["feature"])
        a = agg[key]
        a["score_sum"] += float(r["attribution"])
        a["abs_score_sum"] += float(r["abs_attribution"])
        a["grad_abs_sum"] += float(r["grad_abs_mean"])
        a["disp_abs_sum"] += float(r["displacement_abs_mean"])
        a["active_sum"] += float(r["active_frac"])
        a["n"] += 1
    out_rows = []
    for (class_id, class_name, block, stream, feature), a in agg.items():
        n = max(a["n"], 1)
        out_rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "block": block,
            "stream": stream,
            "feature": feature,
            "mean_attribution": a["score_sum"] / n,
            "mean_abs_attribution": a["abs_score_sum"] / n,
            "mean_grad_abs": a["grad_abs_sum"] / n,
            "mean_displacement_abs": a["disp_abs_sum"] / n,
            "mean_active_frac": a["active_sum"] / n,
            "n_timesteps": n,
        })
    out_rows.sort(key=lambda r: r["mean_abs_attribution"], reverse=True)
    fields = ["class_id", "class_name", "block", "stream", "feature", "mean_attribution",
              "mean_abs_attribution", "mean_grad_abs", "mean_displacement_abs", "mean_active_frac", "n_timesteps"]
    write_csv(os.path.join(out_dir, "feature_class_attribution_overall.csv"), out_rows, fields)

    z_rows = []
    by_site_feature = defaultdict(list)
    for r in out_rows:
        by_site_feature[(r["block"], r["stream"], r["feature"])].append(r)
    for group_rows in by_site_feature.values():
        vals = torch.tensor([float(r["mean_abs_attribution"]) for r in group_rows])
        mu = vals.mean().item()
        sd = vals.std(unbiased=False).clamp_min(1e-12).item()
        for r in group_rows:
            rr = dict(r)
            rr["class_specific_z"] = (float(r["mean_abs_attribution"]) - mu) / sd
            z_rows.append(rr)
    z_rows.sort(key=lambda r: (r["class_specific_z"], r["mean_abs_attribution"]), reverse=True)
    fields_z = fields + ["class_specific_z"]
    write_csv(os.path.join(out_dir, "feature_class_specificity_zscore.csv"), z_rows, fields_z)


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
    classes = [int(x) for x in args.classes.split(",") if x.strip()]
    blocks = [int(x) for x in args.blocks.split(",") if x.strip()]
    streams = [x.strip() for x in args.streams.split(",") if x.strip()]
    t_values = [int(x) for x in args.t_values]

    x_base, y_base, _ = make_balanced_batch(dataset, args.baseline_per_class, args.seed + 77, device)
    rows = []
    fields = ["class_id", "class_name", "t", "block", "stream", "feature", "rank_abs", "attribution",
              "abs_attribution", "grad_abs_mean", "displacement_abs_mean", "active_frac", "coeff_abs_mean"]

    for block_index in blocks:
        for stream in streams:
            sae = load_write_sae(block_index, stream, args.block0_sae_ckpt, args.remaining_sae_ckpt, device)
            decoder = sae.decoder.float()
            site_rows = []
            print(f"site block={block_index:02d} stream={stream}", flush=True)
            for t_int in t_values:
                t_base = torch.full((x_base.shape[0],), t_int, device=device, dtype=torch.long)
                gen_base = torch.Generator(device=device).manual_seed(args.seed + 50000 + t_int)
                noise_base = torch.randn(x_base.shape, device=device, generator=gen_base)
                base_parts = prefix_to_write(model, diffusion, x_base, y_base, t_base, block_index, stream, args.precision, noise_base)
                baseline_mean = encode_dense(sae, base_parts["write"].flatten(0, 1)).mean(dim=(0, 1)).detach()
                del base_parts

                for class_id in classes:
                    x0, labels, _ = make_class_batch(dataset, class_id, args.num_images, args.seed, device)
                    t = torch.full((x0.shape[0],), t_int, device=device, dtype=torch.long)
                    gen = torch.Generator(device=device).manual_seed(args.seed + class_id * 1000 + t_int)
                    noise = torch.randn(x0.shape, device=device, generator=gen)
                    parts = prefix_to_write(model, diffusion, x0, labels, t, block_index, stream, args.precision, noise)
                    direction = one_vs_rest_direction(model, parts["x_t"], t, class_id, args.precision)

                    dense_clean = encode_dense(sae, parts["write"].flatten(0, 1)).detach()
                    clean_recon = torch.einsum("btd,dh->bth", dense_clean.float(), decoder) + sae.dec_bias.float()
                    frozen_error = (parts["write"] - clean_recon).detach()
                    z = dense_clean.clone().float().requires_grad_(True)
                    recon = torch.einsum("btd,dh->bth", z, decoder) + sae.dec_bias.float()
                    spliced_write = recon + frozen_error
                    if stream == "attn":
                        x_after_attn = parts["x_before_block"] + spliced_write
                        x_after_block = x_after_attn + mlp_write_from_after_attn_grad(model, x_after_attn, parts["c"], block_index, args.precision)
                    else:
                        x_after_block = parts["x_after_attn"] + spliced_write
                    out = finish_after_block_grad(model, x_after_block, parts["c"], block_index, args.precision)
                    eps = torch.chunk(out.float(), 2, dim=1)[0]
                    metric = (eps * direction).flatten(1).sum(dim=1).mean()
                    metric.backward()
                    grad = z.grad.detach()
                    disp = z.detach() - baseline_mean.view(1, 1, -1)
                    attr_tok = disp * grad
                    attr = attr_tok.sum(dim=(0, 1)) / z.shape[0]
                    grad_abs = grad.abs().mean(dim=(0, 1))
                    disp_abs = disp.abs().mean(dim=(0, 1))
                    active = (z.detach() > 0).float().mean(dim=(0, 1))
                    coeff_abs = z.detach().abs().mean(dim=(0, 1))
                    order = torch.argsort(attr.abs(), descending=True)[:args.top_features].tolist()
                    for rank, fid in enumerate(order, 1):
                        rec = {
                            "class_id": class_id,
                            "class_name": CLASS_NAMES[class_id],
                            "t": t_int,
                            "block": f"block{block_index:02d}",
                            "stream": stream,
                            "feature": int(fid),
                            "rank_abs": rank,
                            "attribution": float(attr[fid].cpu()),
                            "abs_attribution": float(attr[fid].abs().cpu()),
                            "grad_abs_mean": float(grad_abs[fid].cpu()),
                            "displacement_abs_mean": float(disp_abs[fid].cpu()),
                            "active_frac": float(active[fid].cpu()),
                            "coeff_abs_mean": float(coeff_abs[fid].cpu()),
                        }
                        rows.append(rec)
                        site_rows.append(rec)
                    model.zero_grad(set_to_none=True)
                    print(f"block={block_index:02d} {stream} t={t_int} class={class_id} top={order[:3]}", flush=True)
            site_dir = os.path.join(args.out_root, f"block{block_index:02d}_{stream}")
            write_csv(os.path.join(site_dir, "top_class_direction_attribution.csv"), site_rows, fields)

    write_csv(os.path.join(args.out_root, "top_class_direction_attribution_all_sites.csv"), rows, fields)
    summarize(rows, args.out_root)
    with open(os.path.join(args.out_root, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Write SAE Class-Direction Attribution\n\n"
            "This is a first-pass gradient attribution scan on frozen DDPM DiT write SAEs only.\n\n"
            "Metric per image/class/timestep: `M_c = <eps_theta(x_t,t,c), normalize(eps_c - mean(eps_other_classes))>`.\n\n"
            "Attribution uses `(clean_z - same_t_mixed_class_mean_z) * grad_z M_c`, summed over tokens and averaged over images. "
            "The SAE is spliced as `decoder(z) + frozen_reconstruction_error`, so gradients target SAE feature coefficients while preserving the original reconstruction gap.\n\n"
            "These are filter rankings, not final causal claims; exact joint ablation should confirm top feature sets.\n"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/dit_s2_cifar10/last.pt")
    p.add_argument("--block0-sae-ckpt", default="runs/sae_block0_writes_16x_k32/last.pt")
    p.add_argument("--remaining-sae-ckpt", default="runs/sae_remaining_block_writes_16x_k32/last.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-root", default="runs/write_class_direction_attribution_16x_k32")
    p.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--blocks", default="0,1,2,3,4,5,6,7,8,9,10,11")
    p.add_argument("--streams", default="attn,mlp")
    p.add_argument("--t-values", type=parse_ints, default=T_VALUES)
    p.add_argument("--num-images", type=int, default=8)
    p.add_argument("--baseline-per-class", type=int, default=2)
    p.add_argument("--top-features", type=int, default=200)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
