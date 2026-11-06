#!/usr/bin/env python3
"""Generate Diff-AE perturbation grids for the polynomial TopK SAE."""

import argparse
import json
import os
import sys

import torch
from PIL import Image, ImageDraw


def save_grid(images, path, labels, cell=128):
    rows, cols = len(images), len(images[0])
    out = Image.new("RGB", (cols * cell, rows * (cell + 18)), "white")
    draw = ImageDraw.Draw(out)
    for r, row in enumerate(images):
        for c, image in enumerate(row):
            arr = (image.detach().float().cpu().clamp(0, 1)
                   .permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            out.paste(Image.fromarray(arr), (c * cell, r * (cell + 18)))
            draw.text((c * cell + 2, r * (cell + 18) + cell), labels[c], fill="black")
    out.save(path)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--codes", required=True)
    p.add_argument("--sae", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--examples", type=int, default=12)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--steps", type=int, default=20)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, os.path.abspath(args.repo))

    from experiment import LitModel
    from templates import ffhq128_autoenc_130M
    from train_zsem_polysae import PolyTopKSAE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf = ffhq128_autoenc_130M(); conf.pretrain = None; conf.latent_infer_path = None
    lit = LitModel(conf)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    lit.load_state_dict(state["state_dict"], strict=False); lit.eval().to(device)

    codes_state = torch.load(args.codes, map_location="cpu", weights_only=False)
    codes = codes_state["conds"].float()
    gen = torch.Generator().manual_seed(args.seed)
    ids = torch.randperm(len(codes), generator=gen)[:args.examples]
    raw = codes[ids].to(device)
    noise = torch.randn(len(ids), 3, 128, 128, generator=gen).to(device)

    s = torch.load(args.sae, map_location="cpu", weights_only=False)
    cfg = s["config"]
    sae = PolyTopKSAE(512, int(cfg["dict_mult"]) * 512, int(cfg["k"]),
                     int(cfg.get("aux_k", cfg["k"]))).to(device)
    sae.load_state_dict(s["model"], strict=True); sae.eval()
    mean = s["input_mean"].float().to(device)
    std = s["input_std"].float().to(device)
    x = (raw - mean) / std
    _, vals, idx, _ = sae.encode(x)

    # Estimate population activation means/stds, including implicit zeros, so
    # the sigma scale matches the earlier linear SAE grids.
    total = torch.zeros(sae.h, device=device)
    second = torch.zeros(sae.h, device=device)
    all_x = (codes - mean.cpu()) / std.cpu()
    for start in range(0, len(all_x), 2048):
        _, vv, ii, _ = sae.encode(all_x[start:start + 2048].to(device))
        total.scatter_add_(0, ii.reshape(-1), vv.reshape(-1))
        second.scatter_add_(0, ii.reshape(-1), (vv * vv).reshape(-1))
    mu = total / len(all_x)
    sigma = (second / len(all_x) - mu.square()).clamp_min(0).sqrt()
    rank = torch.argsort(mu, descending=True)

    base = lit.render(noise, cond=raw, T=args.steps).clamp(0, 1)
    labels = ["base"] + [f"{m:+d}s" for m in (-10, -4, -2, 0, 2, 4, 10)]
    for set_size in (1, 2, 4, 8):
        rows = [base]
        for mult in (-10, -4, -2, 0, 2, 4, 10):
            modified = raw.clone()
            for row in range(len(ids)):
                active_ids = idx[row].tolist()
                active_ids.sort(key=lambda f: int((rank == f).nonzero()[0]))
                for f in active_ids[:set_size]:
                    pos = (idx[row] == f).nonzero()[0, 0]
                    clean_value = vals[row, pos]
                    target = torch.clamp(clean_value + mult * sigma[f], min=0)
                    # Reconstruct the normalized SAE vector with the changed
                    # polynomial coefficient, preserving all other support.
                    new_vals = vals[row].clone()
                    new_vals[pos] = target
                    recon = sae.decode_sparse(new_vals.unsqueeze(0), idx[row].unsqueeze(0))[0]
                    clean_recon = sae.decode_sparse(vals[row].unsqueeze(0), idx[row].unsqueeze(0))[0]
                    modified[row] += (recon - clean_recon) * std
            rows.append(lit.render(noise, cond=modified, T=args.steps).clamp(0, 1))
        images = [[rows[c][r] for c in range(len(rows))] for r in range(len(ids))]
        save_grid(images, os.path.join(args.out, f"polynomial_k{cfg['k']}_set{set_size}.png"), labels)

    with open(os.path.join(args.out, "polynomial_metadata.json"), "w") as f:
        json.dump({"ids": ids.tolist(), "k": int(cfg["k"]),
                   "dict_size": sae.h, "selected_by": "population mean activation among clean active features",
                   "sigma_definition": "all-sample std including implicit zeros",
                   "polynomial_decoder": "z + z^2 + z^3", "frozen_support": True}, f, indent=2)
    print(f"saved polynomial grids to {args.out}", flush=True)


if __name__ == "__main__":
    main()
