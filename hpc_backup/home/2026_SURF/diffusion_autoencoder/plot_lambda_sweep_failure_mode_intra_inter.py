#!/usr/bin/env python3
"""Plot same-objective/different-seed ('intra') vs. different-objective/
same-seed ('inter') overlap of each SAE's worst-N hard-image set, across
arms and probe datasets -- the same same-seed-vs-cross-seed logic already
used elsewhere in the sweep (decoder cosine, CKA), applied to hard-image
identity instead of decoder geometry.

Higher intra than inter means which objective trained the SAE predicts
which images it fails on more than which random seed did -- i.e. the
objective-dependent blind spot is a real signal, not seed noise.
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OBJECTIVES = ("latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high")
SEEDS = (2030, 2031, 2032)
ARMS = ("spatial_4x_k8", "spatial_16x_k32", "vector_4x_k8", "vector_16x_k32")
ARM_LABELS = {
    "spatial_4x_k8": "spatial\n4x k8", "spatial_16x_k32": "spatial\n16x k32",
    "vector_4x_k8": "vector\n4x k8", "vector_16x_k32": "vector\n16x k32",
}
INTRA_COLOR = "#2a78d6"
INTER_COLOR = "#eb6834"
HATCH_COLOR = "#9a9a95"


def hard_set(sub, objective, seed, n):
    s = sub[(sub["objective"] == objective) & (sub["seed"] == seed)].sort_values("topk_mse", ascending=False)
    return set(s["latent_index"].head(n))


def jaccard(a, b):
    return len(a & b) / len(a | b)


def intra_vs_inter(df, arm, n):
    sub = df[df["arm"] == arm]
    sets = {}
    for obj in OBJECTIVES:
        for seed in SEEDS:
            s = sub[(sub["objective"] == obj) & (sub["seed"] == seed)]
            if not s.empty:
                sets[(obj, seed)] = hard_set(sub, obj, seed, n)

    intra_by_obj = {}
    for obj in OBJECTIVES:
        available = [seed for seed in SEEDS if (obj, seed) in sets]
        values = [jaccard(sets[(obj, s1)], sets[(obj, s2)]) for s1, s2 in combinations(available, 2)]
        if values:
            intra_by_obj[obj] = values

    inter = []
    for seed in SEEDS:
        objs_at_seed = [obj for obj in OBJECTIVES if (obj, seed) in sets]
        for o1, o2 in combinations(objs_at_seed, 2):
            inter.append(jaccard(sets[(o1, seed)], sets[(o2, seed)]))

    intra_all = [v for vs in intra_by_obj.values() for v in vs]
    # True if every objective contributed at least one seed-pair -- i.e. this
    # arm has full 3-seed coverage, not just the MSE control standing in for
    # the whole arm (the known lambda=100 spatial-arm coverage gap).
    full_coverage = len(intra_by_obj) == len(OBJECTIVES)
    return intra_all, inter, full_coverage


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, 1.96 * sem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction-failures-dir", required=True,
                         help="results/family_sae/lambda_sweep/reconstruction_failures")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rf_dir = Path(args.reconstruction_failures_dir)
    datasets = [
        ("FFHQ held-out", rf_dir / "lambda_100p0_per_example_mse.csv", 100),
        ("CelebA", rf_dir / "lambda_100p0_celeba_per_example_mse.csv", 200),
        ("CelebA (resample B)", rf_dir / "lambda_100p0_celeba_resampleB_per_example_mse.csv", 200),
        ("FairFace", rf_dir / "lambda_100p0_fairface_per_example_mse.csv", 200),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3), sharey=True)
    bar_width = 0.32

    for ax, (label, csv_path, n) in zip(axes.flat, datasets):
        df = pd.read_csv(csv_path)
        x = np.arange(len(ARMS))
        for offset, (key, color, name) in enumerate([
            ("intra", INTRA_COLOR, "intra: same objective,\ndifferent seed"),
            ("inter", INTER_COLOR, "inter: different objective,\nmatched seed"),
        ]):
            means, cis, hatch_flags = [], [], []
            for arm in ARMS:
                intra, inter, full_coverage = intra_vs_inter(df, arm, n)
                values = intra if key == "intra" else inter
                mean, ci = mean_ci95(values)
                means.append(mean)
                cis.append(ci)
                hatch_flags.append(key == "intra" and not full_coverage)
            xpos = x + (offset - 0.5) * bar_width
            bars = ax.bar(xpos, means, bar_width, yerr=cis, capsize=3,
                          color=color, edgecolor="white", linewidth=0.6,
                          error_kw={"linewidth": 1.1, "ecolor": "#333333"},
                          label=name if ax is axes.flat[0] else None, zorder=3)
            for bar, hatched in zip(bars, hatch_flags):
                if hatched:
                    bar.set_hatch("///")
                    bar.set_edgecolor(HATCH_COLOR)

        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABELS[a] for a in ARMS], fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color="#e6e6e2", linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.015, 0.5, "Jaccard overlap of worst-N hard-image sets", fontsize=10,
             rotation="vertical", va="center", ha="center")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    hatch_patch = plt.Rectangle((0, 0), 1, 1, facecolor=INTRA_COLOR, edgecolor=HATCH_COLOR,
                                 hatch="///", label="intra, hatched: only latent_mse has\n3 seeds "
                                 "at this lambda -- not a full 5-objective average")
    fig.legend(handles + [hatch_patch], labels + [hatch_patch.get_label()],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=9)

    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.09, right=0.98, hspace=0.32, wspace=0.10)
    fig.suptitle(
        "Does objective identity predict which images an SAE fails on, more than seed does?\n"
        "Mean $\\pm$ 95% CI Jaccard overlap of each SAE's worst-100/200 held-out reconstruction-error images, "
        "$\\lambda$=100, seed 2030-2032",
        fontsize=11.5, y=0.975)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "failure_mode_intra_vs_inter_objective.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
