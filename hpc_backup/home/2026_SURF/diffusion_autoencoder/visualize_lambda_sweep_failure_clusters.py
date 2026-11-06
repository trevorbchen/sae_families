#!/usr/bin/env python3
"""Decode the CelebA (OOD) faces that are uniquely hard for one objective
cluster but not another, so the blind-spot structure found in
evaluate_lambda_sweep_reconstruction_failures.py can be inspected visually
instead of just as Jaccard-overlap numbers.

One row per requested objective, columns = that objective's hardest
uniquely-hard examples (top-N by topk_mse, restricted to the "not in any
other objective's worst-2N" set computed the same way as the overlap
analysis).
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torchvision.utils import save_image

from spatial_conv_autoencoder import load_spatial_autoencoder
from vector_conv_autoencoder import load_vector_autoencoder

OBJECTIVES = ("latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high")
CACHE_BY_SOURCE = {
    "celeba": {
        "vector": ("linear_probe_vector16x_lambda_sweep/celeba_vector_32768.pt", "clean_latents"),
        "spatial": ("linear_probe_spatial16x_lambda_sweep/celeba_spatial_32768.pt", "clean_latents"),
    },
    "fairface": {
        "vector": ("latents/fairface_vector_8192.pt", "latents"),
        "spatial": ("latents/fairface_spatial_8192.pt", "latents"),
    },
}
AE_PATH = {"vector": "autoencoders/vector/last.pt", "spatial": "autoencoders/spatial/last.pt"}


def uniquely_hard(sub, objective, n, factor=2):
    worst = {}
    worst2 = {}
    for obj in OBJECTIVES:
        s = sub[sub["objective"] == obj].sort_values("topk_mse", ascending=False)
        worst[obj] = list(s["latent_index"].head(n))
        worst2[obj] = set(s["latent_index"].head(factor * n))
    others_union = set().union(*[worst2[o] for o in OBJECTIVES if o != objective])
    return [idx for idx in worst[objective] if idx not in others_union]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--ae-type", choices=("vector", "spatial"), required=True)
    parser.add_argument("--source", choices=("celeba", "fairface"), default="celeba")
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--rank-n", type=int, default=200, help="must match the N used to build --csv's overlap analysis")
    parser.add_argument("--per-objective", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = Path(args.base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv)
    sub = df[(df["arm"] == args.arm) & (df["seed"] == args.seed)]

    cache_path, latent_key = CACHE_BY_SOURCE[args.source][args.ae_type]
    cache = torch.load(base / cache_path, map_location="cpu", weights_only=False)
    latents = cache[latent_key].float()

    if args.ae_type == "vector":
        autoencoder = load_vector_autoencoder(str(base / AE_PATH[args.ae_type]), device)
    else:
        autoencoder = load_spatial_autoencoder(str(base / AE_PATH[args.ae_type]), device)
    autoencoder.eval().requires_grad_(False)

    rows = []
    labels = []
    for objective in OBJECTIVES:
        picked = uniquely_hard(sub, objective, args.rank_n)[:args.per_objective]
        print(f"{objective}: {len(picked)} uniquely-hard examples picked (of requested {args.per_objective})", flush=True)
        labels.append((objective, len(picked)))
        images = []
        with torch.no_grad():
            for idx in picked:
                raw = latents[idx:idx + 1].to(device)
                decoded = autoencoder.decode(raw).add(1).div(2).clamp(0, 1)
                images.append(decoded.cpu())
            while len(images) < args.per_objective:
                images.append(torch.zeros_like(images[0]) if images else torch.zeros(1, 3, 64, 64))
        rows.append(torch.cat(images, dim=0))

    grid = torch.cat(rows, dim=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, args.out, nrow=args.per_objective)
    print("rows (top to bottom):", labels, flush=True)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
