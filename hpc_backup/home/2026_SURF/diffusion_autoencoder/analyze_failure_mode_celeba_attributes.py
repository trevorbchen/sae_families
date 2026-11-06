#!/usr/bin/env python3
"""Quantify the eyeballed "occlusion" theme instead of trusting a handful of
example images: for each objective's uniquely-hard CelebA image set, check
whether occlusion-coded ground-truth attributes (Eyeglasses, Wearing_Hat,
Blurry, ...) are enriched relative to their base rate across all 8192
images, with a two-proportion z-test.
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OBJECTIVES = ("latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high")
ATTRS_OF_INTEREST = ("Eyeglasses", "Wearing_Hat", "Blurry", "Bangs", "Wearing_Necklace", "Smiling", "Mouth_Slightly_Open")


def hard_set(sub, objective, n):
    s = sub[sub["objective"] == objective].sort_values("topk_mse", ascending=False)
    return list(s["latent_index"].head(n))


def uniquely_hard(sub, objective, n, factor=2):
    worst = {obj: set(hard_set(sub, obj, n)) for obj in OBJECTIVES}
    worst2 = {obj: set(hard_set(sub, obj, factor * n)) for obj in OBJECTIVES}
    others_union = set().union(*[worst2[o] for o in OBJECTIVES if o != objective])
    return [idx for idx in worst[objective] if idx not in others_union]


def two_proportion_z(count_a, n_a, count_b, n_b):
    p_a, p_b = count_a / n_a, count_b / n_b
    p_pool = (count_a + count_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0
    return (p_a - p_b) / se


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--attributes-cache", default="linear_probe_spatial16x_lambda_sweep/celeba_spatial_32768.pt")
    parser.add_argument("--arms", nargs="+", default=["spatial_4x_k8", "spatial_16x_k32"])
    parser.add_argument("--rank-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    base = Path(args.base)
    cache = torch.load(base / args.attributes_cache, map_location="cpu", weights_only=False)
    attribute_names = list(cache["attribute_names"])
    attributes = cache["attributes"].numpy()  # [32768, 40], uint8
    name_to_col = {name: i for i, name in enumerate(attribute_names)}
    n_total = attributes.shape[0]
    base_rates = {name: attributes[:, name_to_col[name]].mean() for name in ATTRS_OF_INTEREST if name in name_to_col}

    df = pd.read_csv(args.csv)
    rows = []
    for arm in args.arms:
        sub = df[(df["arm"] == arm) & (df["seed"] == args.seed)]
        if sub.empty:
            continue
        for obj in OBJECTIVES:
            unique_idx = uniquely_hard(sub, obj, args.rank_n)
            n_hard = len(unique_idx)
            if n_hard == 0:
                continue
            hard_attrs = attributes[unique_idx]
            for name, base_rate in base_rates.items():
                col = name_to_col[name]
                count_hard = int(hard_attrs[:, col].sum())
                rate_hard = count_hard / n_hard
                count_base = int(attributes[:, col].sum())
                z = two_proportion_z(count_hard, n_hard, count_base, n_total)
                rows.append({
                    "arm": arm, "objective": obj, "n_uniquely_hard": n_hard,
                    "attribute": name, "rate_in_hard_set": rate_hard, "base_rate": base_rate,
                    "enrichment_ratio": rate_hard / base_rate if base_rate > 0 else float("nan"),
                    "z_score": z, "significant_p05": abs(z) >= 1.96,
                })

    out = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")

    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 200)
    for arm in args.arms:
        print(f"\n=== {arm} ===")
        piv = out[out["arm"] == arm].pivot(index="objective", columns="attribute", values="enrichment_ratio")
        piv = piv.reindex(OBJECTIVES)
        print("enrichment ratio (hard-set rate / base rate), >1 = enriched:")
        print(piv.round(2))
        sig = out[(out["arm"] == arm) & out["significant_p05"]]
        print("significant (|z|>=1.96) cells:")
        for _, r in sig.iterrows():
            print(f"  {r['objective']:<12} {r['attribute']:<20} {r['rate_in_hard_set']:.2f} vs base {r['base_rate']:.2f}  "
                  f"(x{r['enrichment_ratio']:.2f}, z={r['z_score']:+.2f}, n={r['n_uniquely_hard']})")


if __name__ == "__main__":
    main()
