#!/usr/bin/env python3
"""Re-plot the FFHQ held-out per-image reconstruction t-SNE panels from the
cached embedding (results/tsne_hard_images/ffhq_heldout_reconstruction_embedding.npz),
using the current tsne_viz_utils.kde_contours_multi (line-only contours),
instead of re-running explore_reconstruction_tsne.py's full TSNE fit. Gives
the main text's now-FFHQ-primary Figure 2 (LDA) a dataset-matched appendix
counterpart at the per-image level, alongside the existing CelebA one.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from tsne_viz_utils import kde_contours_multi

OBJECTIVES = ("latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high")
OBJECTIVE_COLORS = {
    "latent_mse": "#984ea3", "laplacian": "#e41a1c", "fourier_low": "#4daf4a",
    "fourier_mid": "#ff7f00", "fourier_high": "#377eb8",
}
OBJECTIVE_LABELS = {
    "latent_mse": "MSE", "laplacian": "Laplacian", "fourier_low": "Fourier low",
    "fourier_mid": "Fourier mid", "fourier_high": "Fourier high",
}
SEED_COLORS = {2030: "#1b9e77", 2031: "#d95f02", 2032: "#7570b3"}
SEED_LABELS = {s: f"seed {s}" for s in SEED_COLORS}


def plot_categorical_contours(emb2d, keys_arr, color_map, label_map, draw_order, title, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    present = {k for k in keys_arr}
    keys = [k for k in draw_order if k in present]
    xy_by_class = {k: emb2d[keys_arr == k] for k in keys}
    kde_contours_multi(ax, xy_by_class, color_map)
    handles = [plt.Line2D([0], [0], color=color_map.get(k, "#999999"), lw=2, label=label_map.get(k, str(k)))
               for k in keys]
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(handles=handles, fontsize=9, frameon=False, loc="best")
    ax.set_title(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def plot_fidelity(emb2d, fidelity, title, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    norm = Normalize(vmin=np.percentile(fidelity, 2), vmax=np.percentile(fidelity, 98))
    ax.scatter(emb2d[:, 0], emb2d[:, 1], s=5, alpha=0.6, c=fidelity, cmap="coolwarm",
               norm=norm, linewidths=0, zorder=2)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="coolwarm"), ax=ax, shrink=0.8)
    cbar.set_label("reconstruction fidelity (hot=good, cool=bad)", fontsize=8)
    ax.set_title(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def main():
    npz_path = Path("results/tsne_hard_images/ffhq_heldout_reconstruction_embedding.npz")
    out_dir = Path("results/tsne_hard_images")
    prefix = "ffhq_heldout_reconstruction_replot"

    d = np.load(npz_path, allow_pickle=True)
    emb2d = d["embedding"][:, :2]
    objective_arr = d["objective"]
    seed_arr = d["seed"]
    fidelity_arr = d["fidelity"]
    n = emb2d.shape[0]
    print(f"loaded {npz_path}: {n} points", flush=True)

    plot_categorical_contours(emb2d, objective_arr, OBJECTIVE_COLORS, OBJECTIVE_LABELS, OBJECTIVES,
                               f"ffhq held-out, reconstruction t-SNE, by objective ({n} pts)",
                               out_dir / f"{prefix}_by_objective.png")
    plot_categorical_contours(emb2d, seed_arr, SEED_COLORS, SEED_LABELS, list(SEED_COLORS),
                               f"ffhq held-out, reconstruction t-SNE, by seed ({n} pts)",
                               out_dir / f"{prefix}_by_seed.png")
    plot_fidelity(emb2d, fidelity_arr, f"ffhq held-out, reconstruction t-SNE, by fidelity ({n} pts)",
                  out_dir / f"{prefix}_by_fidelity.png")


if __name__ == "__main__":
    main()
