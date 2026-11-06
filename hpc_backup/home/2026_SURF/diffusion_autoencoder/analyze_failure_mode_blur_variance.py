#!/usr/bin/env python3
"""Quantify the "motion blur" theme found on FairFace (replacing the
CelebA-genre-specific "stage lighting" claim) with the classical
variance-of-Laplacian blur metric, computed on decoded pixels -- works on
any dataset since it needs no ground-truth labels, unlike the CelebA
attribute check.

Low variance of the image Laplacian = blurry (little high-frequency
content); high variance = sharp. For each objective's uniquely-hard set,
compare mean blur score against a random baseline sample from the same
pool, with a two-sample t-test.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from spatial_conv_autoencoder import load_spatial_autoencoder

OBJECTIVES = ("latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high")
LAPLACIAN_KERNEL = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def hard_set(sub, objective, n):
    s = sub[sub["objective"] == objective].sort_values("topk_mse", ascending=False)
    return list(s["latent_index"].head(n))


def uniquely_hard(sub, objective, n, factor=2):
    worst = {obj: set(hard_set(sub, obj, n)) for obj in OBJECTIVES}
    worst2 = {obj: set(hard_set(sub, obj, factor * n)) for obj in OBJECTIVES}
    others_union = set().union(*[worst2[o] for o in OBJECTIVES if o != objective])
    return [idx for idx in worst[objective] if idx not in others_union]


@torch.no_grad()
def blur_scores(indices, latents, autoencoder, device, batch_size=64):
    scores = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        raw = latents[batch_idx].to(device)
        decoded = autoencoder.decode(raw).add(1).div(2).clamp(0, 1)
        gray = (0.299 * decoded[:, 0] + 0.587 * decoded[:, 1] + 0.114 * decoded[:, 2])[:, None]
        kernel = LAPLACIAN_KERNEL.to(device)[None, None]
        lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kernel)
        variance = lap.flatten(1).var(dim=1)
        scores.extend(variance.cpu().tolist())
    return scores


def welch_t(a, b):
    a, b = np.asarray(a), np.asarray(b)
    mean_a, mean_b = a.mean(), b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (mean_a - mean_b) / se if se > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--latent-cache", default="latents/fairface_spatial_8192.pt")
    parser.add_argument("--autoencoder", default="autoencoders/spatial/last.pt")
    parser.add_argument("--arms", nargs="+", default=["spatial_4x_k8", "spatial_16x_k32"])
    parser.add_argument("--rank-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--baseline-sample", type=int, default=500)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    base = Path(args.base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = load_spatial_autoencoder(str(base / args.autoencoder), device)
    autoencoder.eval().requires_grad_(False)

    cache = torch.load(base / args.latent_cache, map_location="cpu", weights_only=False)
    latents = cache["latents"].float()
    n_total = len(latents)

    generator = torch.Generator().manual_seed(0)
    baseline_idx = torch.randperm(n_total, generator=generator)[:args.baseline_sample].tolist()
    baseline_scores = blur_scores(baseline_idx, latents, autoencoder, device)
    baseline_mean = float(np.mean(baseline_scores))
    print(f"baseline: n={len(baseline_scores)} mean_blur_variance={baseline_mean:.6f}", flush=True)

    df = pd.read_csv(args.csv)
    rows = []
    for arm in args.arms:
        sub = df[(df["arm"] == arm) & (df["seed"] == args.seed)]
        if sub.empty:
            continue
        for obj in OBJECTIVES:
            unique_idx = uniquely_hard(sub, obj, args.rank_n)
            if not unique_idx:
                continue
            scores = blur_scores(unique_idx, latents, autoencoder, device)
            t = welch_t(scores, baseline_scores)
            rows.append({
                "arm": arm, "objective": obj, "n_uniquely_hard": len(unique_idx),
                "mean_blur_variance": float(np.mean(scores)), "baseline_mean_blur_variance": baseline_mean,
                "ratio_vs_baseline": float(np.mean(scores)) / baseline_mean,
                "t_stat": t, "significant_p05": abs(t) >= 1.96,
            })
            print(f"{arm}/{obj}: n={len(unique_idx)} mean_blur={np.mean(scores):.6f} "
                  f"(baseline={baseline_mean:.6f}, ratio={np.mean(scores)/baseline_mean:.2f}, t={t:+.2f})", flush=True)

    out = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    print("\nratio < 1 means BLURRIER than baseline (lower Laplacian variance); significant cells:")
    for _, r in out[out["significant_p05"]].iterrows():
        direction = "blurrier" if r["ratio_vs_baseline"] < 1 else "sharper"
        print(f"  {r['arm']:<18} {r['objective']:<12} {direction} than baseline "
              f"(x{r['ratio_vs_baseline']:.2f}, t={r['t_stat']:+.2f}, n={r['n_uniquely_hard']})")


if __name__ == "__main__":
    main()
