#!/usr/bin/env python3
"""Per-objective SAE vocabulary concentration on the hard-image set: does an
objective's SAE reuse a small set of "workhorse" features across most hard
images (narrow vocabulary), or spread activation across many different
features (broad vocabulary)? Each objective has its own dictionary, so this
only compares concentration shape, not feature identity across objectives.
"""
import argparse
from pathlib import Path

import numpy as np
import torch


def get(records, source, objective, seed):
    for r in records:
        if r["source"] == source and r["objective"] == objective and r["seed"] == seed:
            return r
    return None


def concentration_stats(feature_indices, feature_count):
    # feature_indices: (n_images, k) int tensor of which dictionary atoms fired
    flat = feature_indices.flatten().numpy()
    counts = np.bincount(flat, minlength=feature_count).astype(np.float64)
    total = counts.sum()
    p = counts / total
    distinct_used = int((counts > 0).sum())
    # effective number of features (inverse Simpson index): 1 if one feature
    # does everything, feature_count if usage were perfectly uniform
    effective_n = 1.0 / np.sum(p ** 2)
    # Gini coefficient of the usage distribution (0 = uniform, 1 = maximally concentrated)
    sorted_counts = np.sort(counts)
    n = feature_count
    cum = np.cumsum(sorted_counts)
    gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    # fraction of total usage covered by the top-K most used features
    top_sorted = np.sort(counts)[::-1]
    cov20 = top_sorted[:20].sum() / total
    cov50 = top_sorted[:50].sum() / total
    cov100 = top_sorted[:100].sum() / total
    return {
        "distinct_used": distinct_used, "effective_n": effective_n, "gini": gini,
        "cov_top20": cov20, "cov_top50": cov50, "cov_top100": cov100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usage", required=True)
    parser.add_argument("--source", default="ffhq_heldout")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2030, 2031, 2032])
    args = parser.parse_args()

    data = torch.load(args.usage, map_location="cpu", weights_only=False)
    records, feature_count, k = data["records"], data["feature_count"], data["k"]
    print(f"arm={data['arm']} feature_count={feature_count} k={k} source={args.source}\n")

    objectives = ["latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high"]
    header = f"{'objective':14s} {'seed':>6s} {'distinct':>9s} {'eff_n':>8s} {'gini':>6s} {'cov@20':>7s} {'cov@50':>7s} {'cov@100':>8s}"
    print(header)
    print("-" * len(header))
    per_obj_eff_n = {obj: [] for obj in objectives}
    for obj in objectives:
        for seed in args.seeds:
            rec = get(records, args.source, obj, seed)
            if rec is None:
                continue
            stats = concentration_stats(rec["feature_indices"], feature_count)
            per_obj_eff_n[obj].append(stats["effective_n"])
            print(f"{obj:14s} {seed:6d} {stats['distinct_used']:9d} {stats['effective_n']:8.1f} "
                  f"{stats['gini']:6.3f} {stats['cov_top20']:7.3f} {stats['cov_top50']:7.3f} {stats['cov_top100']:8.3f}")
        print()

    print("mean effective_n across seeds (lower = narrower/more concentrated vocabulary):")
    for obj in objectives:
        vals = per_obj_eff_n[obj]
        if vals:
            print(f"  {obj:14s} {np.mean(vals):8.1f}  (seeds: {[f'{v:.0f}' for v in vals]})")


if __name__ == "__main__":
    main()
