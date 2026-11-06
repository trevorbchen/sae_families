#!/usr/bin/env python3
"""Per-objective intra/inter breakdown of hard-image-set overlap, one 5x5
grid per arm (2x2 grid of arms). Off-diagonal (i,j) = mean Jaccard overlap
of objective i's and objective j's worst-N hard-image sets at matched seed
(inter -- different objective, same seed). Diagonal (i,i) = mean Jaccard of
objective i's own hard-image set across different seeds (intra -- same
objective, different seed) -- replacing the usual trivial 1.0 self-overlap
with the actually informative number, so each objective's self-consistency
sits right next to its overlap with every other specific objective.
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OBJECTIVES = ("latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high")
SHORT = {"latent_mse": "MSE", "laplacian": "Lap", "fourier_low": "F-lo", "fourier_mid": "F-mid", "fourier_high": "F-hi"}
SEEDS = (2030, 2031, 2032)
ARMS = ("spatial_4x_k8", "spatial_16x_k32", "vector_4x_k8", "vector_16x_k32")
ARM_LABELS = {
    "spatial_4x_k8": "spatial 4x k8", "spatial_16x_k32": "spatial 16x k32",
    "vector_4x_k8": "vector 4x k8", "vector_16x_k32": "vector 16x k32",
}


def hard_set(sub, objective, seed, n):
    s = sub[(sub["objective"] == objective) & (sub["seed"] == seed)].sort_values("topk_mse", ascending=False)
    return set(s["latent_index"].head(n))


def jaccard(a, b):
    return len(a & b) / len(a | b)


def build_grid(df, arm, n):
    sub = df[df["arm"] == arm]
    sets = {}
    for obj in OBJECTIVES:
        for seed in SEEDS:
            s = sub[(sub["objective"] == obj) & (sub["seed"] == seed)]
            if not s.empty:
                sets[(obj, seed)] = hard_set(sub, obj, seed, n)

    grid = np.full((5, 5), np.nan)
    diag_underpowered = [False] * 5
    for i, oi in enumerate(OBJECTIVES):
        available_i = [seed for seed in SEEDS if (oi, seed) in sets]
        if len(available_i) >= 2:
            values = [jaccard(sets[(oi, s1)], sets[(oi, s2)]) for s1, s2 in combinations(available_i, 2)]
            grid[i, i] = np.mean(values)
        else:
            diag_underpowered[i] = True
        for j, oj in enumerate(OBJECTIVES):
            if i == j:
                continue
            matched_seeds = [seed for seed in SEEDS if (oi, seed) in sets and (oj, seed) in sets]
            if matched_seeds:
                values = [jaccard(sets[(oi, seed)], sets[(oj, seed)]) for seed in matched_seeds]
                grid[i, j] = np.mean(values)
    return grid, diag_underpowered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--rank-n", type=int, default=200)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    arms = args.arms
    n_rows = 1 if len(arms) <= 2 else 2
    n_cols = 2 if len(arms) > 1 else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 4.5 * n_rows), squeeze=False)
    axes_flat = list(axes.flat)
    cmap = plt.get_cmap("coolwarm")

    # Same color convention as the vocabulary-similarity heatmap (Figure 2):
    # coolwarm, diverging around the pooled mean across all shown arms, so
    # the two figures read the same way at a glance.
    grids = {}
    diag_flags = {}
    for arm in arms:
        grid, diag_underpowered = build_grid(df, arm, args.rank_n)
        grids[arm] = grid
        diag_flags[arm] = diag_underpowered
    pooled = np.concatenate([g[np.isfinite(g)] for g in grids.values()])
    baseline = float(pooled.mean())
    relative_grids = {arm: 100.0 * (g / baseline - 1.0) for arm, g in grids.items()}
    limit = max(1.0, float(np.nanmax(np.abs(np.concatenate([
        r[np.isfinite(r)] for r in relative_grids.values()
    ])))))

    # Matrix is symmetric (grid[i,j] == grid[j,i]), so -- same convention as
    # the vocabulary-similarity heatmap (Figure 2) -- only the lower triangle
    # (including the diagonal) is drawn; the redundant upper triangle is left
    # blank instead of duplicating every off-diagonal value.
    triu = ~np.tri(5, dtype=bool)

    im = None
    for ax, arm in zip(axes_flat, arms):
        grid = np.where(triu, np.nan, grids[arm])
        relative = np.where(triu, np.nan, relative_grids[arm])
        diag_underpowered = diag_flags[arm]
        masked = np.ma.masked_invalid(relative)
        im = ax.imshow(masked, cmap=cmap, vmin=-limit, vmax=limit, aspect="equal")

        ax.set_xticks(range(5), [SHORT[o] for o in OBJECTIVES], fontsize=10)
        ax.set_yticks(range(5), [SHORT[o] for o in OBJECTIVES], fontsize=10)
        ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
        ax.tick_params(length=0)
        plt.setp(ax.get_xticklabels(), rotation=28, ha="left", rotation_mode="anchor")
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in range(5):
            for j in range(5):
                if j > i:
                    continue
                value = grid[i, j]
                if np.isnan(value):
                    ax.text(j, i, "single\nseed", ha="center", va="center", fontsize=7.5, color="#999")
                    continue
                red, green, blue, _ = cmap((relative[i, j] + limit) / (2 * limit))[:4]
                lum = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                color = "white" if lum < 0.55 else "#1f2430"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9.5, color=color,
                        fontweight="bold" if i == j else "normal")

        # Highlight the diagonal -- each objective's own cross-seed
        # consistency -- so it reads as the reference value each row/column
        # of "inter" comparisons should be compared against.
        for i in range(5):
            edge_color = "#c9a227" if not diag_underpowered[i] else "#b6373a"
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=edge_color, linewidth=2.5, zorder=5))

        ax.set_title(ARM_LABELS[arm], fontsize=12.5, fontweight="bold", pad=28)

    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.06, right=0.84, hspace=0.4, wspace=0.3)

    cax = fig.add_axes([0.87, 0.32, 0.02, 0.42])
    colorbar = fig.colorbar(im, cax=cax)
    colorbar.set_label("Jaccard overlap, relative to the pooled mean (%)", fontsize=9.5)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#c9a227", linewidth=2.5,
                      label="diagonal = intra (same objective, different seed)"),
        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#b6373a", linewidth=2.5,
                      label="diagonal, red = only 1 seed trained -- no intra value yet"),
    ]
    off_diag_patch = plt.Rectangle((0, 0), 1, 1, facecolor=cmap(0.75), edgecolor="none",
                                    label="off-diagonal = inter (different objective, matched seed)")
    fig.legend(handles=[off_diag_patch] + handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
              ncol=1, frameon=False, fontsize=10)

    fig.suptitle(f"Per-objective hard-image overlap -- {args.dataset_label}", fontsize=13, y=0.98)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
