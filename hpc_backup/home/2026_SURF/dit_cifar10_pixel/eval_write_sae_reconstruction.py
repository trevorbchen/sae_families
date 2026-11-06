#!/usr/bin/env python3
import argparse
import csv
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from train_dit_cifar import DiT, Diffusion, modulate
from train_remaining_block_write_saes import TopKSAE


def load_pt(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(path, device):
    model = DiT(learn_sigma=True).to(device)
    ckpt = load_pt(path, device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_saes(block0_path, remaining_path, device, input_dim=384, dict_size=6144, k=32):
    saes = nn.ModuleDict()
    block0 = load_pt(block0_path, device)
    for stream, key in (("attn", "attn_sae"), ("mlp", "mlp_sae")):
        sae = TopKSAE(input_dim, dict_size, k).to(device)
        sae.load_state_dict(block0[key])
        sae.eval()
        saes[f"b00_{stream}"] = sae

    remaining = load_pt(remaining_path, device)
    for block in range(1, 12):
        for stream in ("attn", "mlp"):
            name = f"b{block:02d}_{stream}"
            sae = TopKSAE(input_dim, dict_size, k).to(device)
            sae.load_state_dict(remaining["saes"][name])
            sae.eval()
            saes[name] = sae

    for sae in saes.values():
        for p in sae.parameters():
            p.requires_grad_(False)
    return saes


@torch.no_grad()
def get_writes(model, diffusion, x0, labels, t, precision):
    noise = torch.randn_like(x0)
    x_t = diffusion.q_sample(x0, t, noise)
    enabled = x_t.device.type == "cuda" and precision == "bf16"
    writes = {}
    with torch.autocast(device_type=x_t.device.type, dtype=torch.bfloat16, enabled=enabled):
        x = model.x_embedder(x_t).flatten(2).transpose(1, 2) + model.pos_embed
        c = model.t_embedder(t) + model.y_embedder(labels)
        for i, block in enumerate(model.blocks):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaLN(c).chunk(6, dim=1)
            attn_in = modulate(block.norm1(x), shift_msa, scale_msa)
            attn_out = block.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
            attn_write = gate_msa.unsqueeze(1) * attn_out
            x = x + attn_write
            mlp_in = modulate(block.norm2(x), shift_mlp, scale_mlp)
            mlp_out = block.mlp(mlp_in)
            mlp_write = gate_mlp.unsqueeze(1) * mlp_out
            x = x + mlp_write
            writes[f"b{i:02d}_attn"] = attn_write.float().flatten(0, 1)
            writes[f"b{i:02d}_mlp"] = mlp_write.float().flatten(0, 1)
    return writes


@torch.no_grad()
def encode_decode(sae, x):
    pre = sae.encoder(x - sae.dec_bias)
    acts = F.relu(pre)
    vals, idx = torch.topk(acts, sae.k, dim=-1)
    recon = (vals.unsqueeze(-1) * sae.decoder[idx]).sum(dim=1) + sae.dec_bias
    return recon, vals, idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/dit_s2_cifar10/last.pt")
    p.add_argument("--block0-sae-ckpt", default="runs/sae_block0_writes_16x_k32/last.pt")
    p.add_argument("--remaining-sae-ckpt", default="runs/sae_remaining_block_writes_16x_k32/last.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs/write_sae_reconstruction_eval_1k")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = load_model(args.ckpt, device)
    diffusion = Diffusion(args.timesteps, device=device)
    saes = load_saes(args.block0_sae_ckpt, args.remaining_sae_ckpt, device)
    names = [f"b{i:02d}_{stream}" for i in range(12) for stream in ("attn", "mlp")]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    dataset = datasets.CIFAR10(args.data_dir, train=True, transform=transform, download=True)
    gen = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=gen)[: args.num_samples].tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    sums = {name: {"mse": 0.0, "target_sq": 0.0, "l0": 0.0, "tokens": 0} for name in names}
    active_counts = {name: torch.zeros(saes[name].dict_size, dtype=torch.long, device=device) for name in names}
    t_hist = torch.zeros(args.timesteps, dtype=torch.long)

    sample_offset = 0
    for batch_i, (x0, labels) in enumerate(loader):
        x0 = x0.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_n = x0.shape[0]
        t_gen = torch.Generator(device=device).manual_seed(args.seed + 100000 + sample_offset)
        t = torch.randint(0, args.timesteps, (batch_n,), generator=t_gen, device=device)
        t_hist.scatter_add_(0, t.cpu(), torch.ones(batch_n, dtype=torch.long))
        writes = get_writes(model, diffusion, x0, labels, t, args.precision)
        for name in names:
            x = writes[name]
            recon, vals, idx = encode_decode(saes[name], x)
            sums[name]["mse"] += F.mse_loss(recon, x, reduction="sum").item()
            sums[name]["target_sq"] += x.pow(2).sum().item()
            sums[name]["l0"] += (vals > 0).sum().item()
            sums[name]["tokens"] += x.shape[0]
            active_counts[name].scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel(), dtype=torch.long, device=device))
        sample_offset += batch_n
        if (batch_i + 1) % 5 == 0:
            print(f"processed_samples={sample_offset}", flush=True)

    rows = []
    for name in names:
        block = int(name[1:3])
        stream = name.split("_")[1]
        tokens = sums[name]["tokens"]
        mse = sums[name]["mse"] / max(tokens * 384, 1)
        target_mean_sq = sums[name]["target_sq"] / max(tokens * 384, 1)
        ev = 1.0 - mse / max(target_mean_sq, 1e-12)
        counts = active_counts[name].cpu()
        active_features = int((counts > 0).sum().item())
        sorted_counts = torch.sort(counts.float(), descending=True).values
        total_slots = sorted_counts.sum().clamp_min(1)
        cumsum = sorted_counts.cumsum(0)
        rows.append({
            "block": block,
            "stream": stream,
            "name": name,
            "num_image_timestep_samples": args.num_samples,
            "num_tokens": int(tokens),
            "mse": mse,
            "target_mean_sq": target_mean_sq,
            "var_explained": ev,
            "mean_l0": sums[name]["l0"] / max(tokens, 1),
            "unique_active_features": active_features,
            "features_for_50pct_slots": int(torch.searchsorted(cumsum, 0.50 * total_slots).item() + 1),
            "features_for_90pct_slots": int(torch.searchsorted(cumsum, 0.90 * total_slots).item() + 1),
            "features_for_99pct_slots": int(torch.searchsorted(cumsum, 0.99 * total_slots).item() + 1),
            "top1_feature_slot_frac": float(sorted_counts[0] / total_slots),
            "top10_feature_slot_frac": float(sorted_counts[:10].sum() / total_slots),
            "top100_feature_slot_frac": float(sorted_counts[:100].sum() / total_slots),
        })

        top_path = os.path.join(args.out_dir, f"{name}_top_active_features.csv")
        top_ids = torch.argsort(counts, descending=True)[:1000].tolist()
        with open(top_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["rank", "feature", "active_slots", "slot_frac"])
            w.writeheader()
            total = int(counts.sum().item())
            for rank, fid in enumerate(top_ids, 1):
                w.writerow({
                    "rank": rank,
                    "feature": int(fid),
                    "active_slots": int(counts[fid].item()),
                    "slot_frac": float(counts[fid].item() / max(total, 1)),
                })

    summary_path = os.path.join(args.out_dir, "write_sae_reconstruction_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(args.out_dir, "config_and_t_hist.json"), "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "cifar_indices": indices, "t_hist_nonzero": int((t_hist > 0).sum().item())}, f, indent=2)

    print(json.dumps({"out_dir": args.out_dir, "summary": summary_path, "rows": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
