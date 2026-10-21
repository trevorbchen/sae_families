#!/usr/bin/env python3
"""Re-plot the FairFace LDA by_objective/by_seed/by_fidelity figures from the
cached embedding (ffhq_heldout_lda_embedding.npz), using the current (fixed)
tsne_viz_utils.kde_contours_multi -- fill_alpha=0 by default, lines only --
instead of re-running the full explore_feature_lda.py pipeline, which would
need the original SAE checkpoints and latent caches. Matches the CelebA
panels, which were already regenerated with this fix earlier in the session.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from tsne_viz_utils import kde_contours_multi

OBJECTIVES = ("latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high")
SEEDS = (2030, 2031, 2032)
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
    ax.set_xlabel("LD1", fontsize=9)
    ax.set_ylabel("LD2", fontsize=9)
    ax.legend(handles=handles, fontsize=9, frameon=False, loc="best")
    ax.set_title(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def plot_fidelity(emb2d, fidelity, title, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    norm = Normalize(vmin=np.percentile(fidelity, 2), vmax=np.percentile(fidelity, 98))
    ax.scatter(emb2d[:, 0], emb2d[:, 1], s=4, alpha=0.5, c=fidelity, cmap="coolwarm",
               norm=norm, linewidths=0, zorder=2)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="coolwarm"), ax=ax, shrink=0.8)
    cbar.set_label("reconstruction fidelity (hot=good, cool=bad)", fontsize=8)
    ax.set_xlabel("LD1", fontsize=9)
    ax.set_ylabel("LD2", fontsize=9)
    ax.set_title(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def main():
    npz_path = Path("results/tsne_hard_images/lda/ffhq_heldout_lda_embedding.npz")
    out_dir = Path("results/tsne_hard_images/lda")
    prefix = "ffhq_heldout"

    d = np.load(npz_path, allow_pickle=True)
    emb2d = d["embedding"][:, :2]
    objective_arr = d["objective"]
    seed_arr = d["seed"]
    fidelity_arr = d["fidelity"]
    acc = float(d["held_out_accuracy"])
    print(f"loaded {npz_path}: {emb2d.shape[0]} points, held-out accuracy {acc:.1%}", flush=True)

    plot_categorical_contours(emb2d, objective_arr, OBJECTIVE_COLORS, OBJECTIVE_LABELS, OBJECTIVES,
                               "ffhq held-out, LDA by objective",
                               out_dir / f"{prefix}_lda_by_objective.png")
    plot_categorical_contours(emb2d, seed_arr, SEED_COLORS, SEED_LABELS, list(SEED_COLORS),
                               "ffhq held-out, LDA by seed",
                               out_dir / f"{prefix}_lda_by_seed.png")
    plot_fidelity(emb2d, fidelity_arr, "ffhq held-out, LDA by fidelity",
                  out_dir / f"{prefix}_lda_by_fidelity.png")


if __name__ == "__main__":
    main()
