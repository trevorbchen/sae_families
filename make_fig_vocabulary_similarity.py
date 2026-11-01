"""Re-render the CVPR2026 vocabulary-similarity heatmap (Figure 4,
fig:vocabulary-similarity-heatmap) as a single row of four panels with much
larger fonts -- addresses reviewer feedback that the original 2x2 composite
(hand-assembled from plot_cross_seed_similarity_heatmaps.py's four separate
per-metric PNGs) was illegible at print size and "a little sad looking".

Reads the same pairwise_category_summary.csv that backed the original
figure; does not recompute anything, only re-renders. Verified to reproduce
the exact same numbers as the deployed figA18_vocabulary_similarity_heatmap.png
(e.g. MSE/MSE diagonal: decoder-cosine 0.529, activation-CKA 0.377,
activation-correlation 0.279, atom-recurrence 0.047).
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

OBJECTIVES = ("latent_mse", "weighted_laplacian", "weighted_fourier_low",
              "weighted_fourier_mid", "weighted_fourier_high")
OBJECTIVE_LABELS = {
    "latent_mse": "MSE",
    "weighted_laplacian": "MSE + Lap",
    "weighted_fourier_low": "MSE + Low",
    "weighted_fourier_mid": "MSE + Mid",
    "weighted_fourier_high": "MSE + High",
}
METRIC_ORDER = ("decoder_cosine", "activation_cka", "activation_correlation", "shared_atoms_0p5")
METRIC_TITLES = {
    "decoder_cosine": "Decoder-vector\nalignment",
    "activation_cka": "Activation CKA",
    "activation_correlation": "Activation\ncorrelation",
    "shared_atoms_0p5": "Strict atom\nrecurrence",
}
CMAP = "coolwarm"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary-csv", type=Path, required=True,
                    help="pairwise_category_summary.csv from plot_cross_seed_similarity_heatmaps.py")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def load(summary_csv):
    rows = list(csv.DictReader(open(summary_csv)))
    by_metric = {}
    for r in rows:
        by_metric.setdefault(r["metric"], {})[(r["objective_a"], r["objective_b"])] = r
        by_metric[r["metric"]][(r["objective_b"], r["objective_a"])] = r
    return by_metric


def panel(ax, by_metric, metric, show_cbar_label):
    records = by_metric[metric]
    size = len(OBJECTIVES)
    full = np.empty((size, size))
    for i, a in enumerate(OBJECTIVES):
        for j, b in enumerate(OBJECTIVES):
            full[i, j] = float(records[(a, b)]["mean"])
    baseline = float(full.mean())
    relative = 100.0 * (full / max(abs(baseline), 1e-12) - 1.0)
    matrix = np.where(np.tri(size, dtype=bool), relative, np.nan)
    visible = matrix[np.isfinite(matrix)]
    limit = max(1.0, float(np.abs(visible).max()))

    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=CMAP, vmin=-limit, vmax=limit)
    labels = [OBJECTIVE_LABELS[o] for o in OBJECTIVES]
    ax.set_xticks(range(size), labels, fontsize=14.5)
    ax.set_yticks(range(size), labels, fontsize=14.5)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    ax.tick_params(axis="y", length=0)
    plt.setp(ax.get_xticklabels(), rotation=32, ha="left", rotation_mode="anchor")
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, a in enumerate(OBJECTIVES):
        for j, b in enumerate(OBJECTIVES):
            if i < j:
                continue
            record = records[(a, b)]
            red, green, blue, _ = image.cmap(image.norm(relative[i, j]))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = "white" if luminance < 0.5 else "#1a1f26"
            ax.text(j, i - 0.14, f'{float(record["mean"]):.2f}', ha="center", va="center",
                     fontsize=15.5, fontweight="bold", color=color)
            ax.text(j, i + 0.23,
                     f'[{float(record["bootstrap_ci_low"]):.2f}, {float(record["bootstrap_ci_high"]):.2f}]',
                     ha="center", va="center", fontsize=9.5, color=color)
            if i == j:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                        edgecolor="#1a1f26", linewidth=2.8))

    ax.set_title(METRIC_TITLES[metric], fontsize=17.5, pad=15)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03, shrink=0.88)
    cbar.ax.tick_params(labelsize=10.5)
    if show_cbar_label:
        cbar.set_label("vs. five-vocabulary mean (%)", fontsize=10.5)


def main():
    args = parse_args()
    by_metric = load(args.summary_csv)
    fig, axes = plt.subplots(1, 4, figsize=(25, 7.2), constrained_layout=True)
    for k, (ax, metric) in enumerate(zip(axes, METRIC_ORDER)):
        panel(ax, by_metric, metric, show_cbar_label=(k == len(axes) - 1))
    fig.suptitle(
        "Dictionary similarity across five objectives (vector_4x_k8, λ=10): "
        "same-objective pairs (diagonal, boxed) are more similar than cross-objective pairs",
        fontsize=16.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
