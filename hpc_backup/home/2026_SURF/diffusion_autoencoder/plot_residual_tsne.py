#!/usr/bin/env python3
"""t-SNE/UMAP of SAE reconstruction residuals, to visualize whether different
training objectives occupy different regions of "what the SAE failed to
capture" -- the spatial complement to the Jaccard-overlap hard-image
analysis (plot_lambda_sweep_failure_mode_intra_inter.py /
plot_lambda_sweep_failure_mode_objective_grid.py).

Two embeddings, matching the intra/inter logic used elsewhere in the sweep:
  - same seed, different objective: do residuals cluster by objective?
  - same objective, different seed: do residuals NOT cluster as cleanly
    (if the paper's main finding holds, this should look mixed/overlapping)?

Run on both raw and unit-normalized residuals: a Fourier-high objective's
residual differs systematically in magnitude from a Laplacian objective's by
construction (different term is the one being suppressed), not just in
shape, so raw-residual structure could just be reflecting that scale
difference rather than anything about which directions are represented.
Normalizing removes it and leaves only direction/shape.

Input is the .pt file written by extract_residual_vectors_for_tsne.py.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

OBJECTIVE_COLORS = {
    "latent_mse": "#999999", "laplacian": "#d6604d", "fourier_low": "#4393c3",
    "fourier_mid": "#92c5de", "fourier_high": "#2166ac",
}
OBJECTIVE_LABELS = {
    "latent_mse": "MSE (baseline density)", "laplacian": "Laplacian", "fourier_low": "Fourier low",
    "fourier_mid": "Fourier mid", "fourier_high": "Fourier high",
}
OBJECTIVE_MARKERS = {
    "latent_mse": "o", "laplacian": "s", "fourier_low": "^",
    "fourier_mid": "D", "fourier_high": "v",
}
# Draw order (for the scattered, non-background categories), bottom to top.
OBJECTIVE_DRAW_ORDER = ["fourier_mid", "fourier_low", "fourier_high", "laplacian"]
# latent_mse is the "everything basically lands here" baseline -- not informative
# as discrete points, so it's rendered as a soft background density blob instead
# of a layer of gray dots competing with the categories we actually care about.
OBJECTIVE_BACKGROUND_KEY = "latent_mse"

SEED_COLORS = {2030: "#1b9e77", 2031: "#d95f02", 2032: "#7570b3"}
SEED_MARKERS = {2030: "o", 2031: "s", 2032: "^"}
SEED_DRAW_ORDER = [2030, 2031, 2032]


def load_records(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["arm"], data["lambda_value"], data["records"]


def subsample(residual, n, generator):
    if residual.shape[0] <= n:
        return residual
    perm = torch.randperm(residual.shape[0], generator=generator)[:n]
    return residual[perm]


def build_matrix(records, source, filter_fn, per_cell_cap, normalize, generator):
    vectors, meta = [], []
    for rec in records:
        if rec["source"] != source or not filter_fn(rec):
            continue
        residual = subsample(rec["residual"], per_cell_cap, generator).float()
        if normalize:
            residual = residual / residual.norm(dim=1, keepdim=True).clamp_min(1e-8)
        vectors.append(residual)
        meta.extend([{"objective": rec["objective"], "seed": rec["seed"]}] * residual.shape[0])
    if not vectors:
        return None, []
    return torch.cat(vectors, dim=0).numpy(), meta


def run_embedding(matrix, method, perplexity):
    if method == "tsne":
        from sklearn.manifold import TSNE
        effective = min(perplexity, (matrix.shape[0] - 1) // 3)
        return TSNE(n_components=2, init="pca", perplexity=effective, random_state=0).fit_transform(matrix)
    import umap
    return umap.UMAP(n_components=2, random_state=0).fit_transform(matrix)


def plot_background_blob(ax, points, color):
    """Render a point cloud as a soft filled-density silhouette instead of
    discrete dots -- used for a baseline/reference category whose job is just
    to show "most things land around here", not to be read point-by-point."""
    if points.shape[0] < 5:
        ax.scatter(points[:, 0], points[:, 1], s=40, alpha=0.15, color=color, linewidths=0, zorder=0)
        return
    kde = gaussian_kde(points.T)
    pad_x = max((points[:, 0].max() - points[:, 0].min()) * 0.15, 1e-3)
    pad_y = max((points[:, 1].max() - points[:, 1].min()) * 0.15, 1e-3)
    xmin, xmax = points[:, 0].min() - pad_x, points[:, 0].max() + pad_x
    ymin, ymax = points[:, 1].min() - pad_y, points[:, 1].max() + pad_y
    xx, yy = np.mgrid[xmin:xmax:150j, ymin:ymax:150j]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    threshold = zz.max() * 0.08
    ax.contourf(xx, yy, zz, levels=[threshold, zz.max()], colors=[color], alpha=0.3, zorder=0)


def plot(embedding, meta, color_key, color_map, label_map, title, out_path,
         draw_order=None, marker_map=None, background_key=None):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    present = {m[color_key] for m in meta}
    marker_map = marker_map or {}
    handles = []

    if background_key is not None and background_key in present:
        mask = np.array([m[color_key] == background_key for m in meta])
        plot_background_blob(ax, embedding[mask], color_map.get(background_key, "#999999"))
        handles.append(Patch(facecolor=color_map.get(background_key, "#999999"), alpha=0.35,
                              label=label_map.get(background_key, str(background_key))))

    ordered = draw_order or sorted(present, key=str)
    keys = [k for k in ordered if k in present and k != background_key]
    keys += sorted(present - set(ordered) - {background_key}, key=str)

    for key in keys:
        mask = np.array([m[color_key] == key for m in meta])
        sc = ax.scatter(embedding[mask, 0], embedding[mask, 1], s=6, alpha=0.6,
                         color=color_map.get(key, "#999999"), marker=marker_map.get(key, "o"),
                         linewidths=0, label=label_map.get(key, str(key)), zorder=2)
        handles.append(sc)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(handles=handles, markerscale=3, fontsize=9, frameon=False, loc="best")
    ax.set_title(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residuals", required=True)
    parser.add_argument("--source", choices=("ffhq_heldout", "celeba", "fairface"), default="ffhq_heldout")
    parser.add_argument("--method", choices=("tsne", "umap"), default="tsne")
    parser.add_argument("--per-cell-cap", type=int, default=2000,
                         help="5 objectives * this <= budget for the diff-objective plot; "
                              "3 seeds * this <= budget for the diff-seed plot")
    parser.add_argument("--fixed-seed", type=int, default=2030, help="for the diff-objective plot")
    parser.add_argument("--fixed-objective", default="latent_mse", help="for the diff-seed plot")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity (clamped to the data size)")
    parser.add_argument("--skip-diff-objective", action="store_true")
    parser.add_argument("--skip-diff-seed", action="store_true")
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    arm, lambda_value, records = load_records(args.residuals)
    generator = torch.Generator().manual_seed(0)

    for normalize, tag in ((False, "raw"), (True, "normalized")):
        if not args.skip_diff_objective:
            matrix, meta = build_matrix(
                records, args.source, lambda rec: rec["seed"] == args.fixed_seed,
                args.per_cell_cap, normalize, generator,
            )
            if matrix is not None:
                embedding = run_embedding(matrix, args.method, args.perplexity)
                plot(embedding, meta, "objective", OBJECTIVE_COLORS, OBJECTIVE_LABELS,
                     f"{arm}, {args.source}, seed={args.fixed_seed}, {tag} residuals, perp={args.perplexity} ({len(meta)} pts)",
                     Path(f"{args.out_prefix}_diff_objective_{tag}_perp{args.perplexity}.png"),
                     draw_order=OBJECTIVE_DRAW_ORDER, marker_map=OBJECTIVE_MARKERS,
                     background_key=OBJECTIVE_BACKGROUND_KEY)

        if args.skip_diff_seed:
            continue
        matrix, meta = build_matrix(
            records, args.source, lambda rec: rec["objective"] == args.fixed_objective,
            args.per_cell_cap, normalize, generator,
        )
        if matrix is not None:
            embedding = run_embedding(matrix, args.method, args.perplexity)
            seed_label_map = {s: f"seed {s}" for s in SEED_COLORS}
            plot(embedding, meta, "seed", SEED_COLORS, seed_label_map,
                 f"{arm}, {args.source}, objective={args.fixed_objective}, {tag} residuals, perp={args.perplexity} ({len(meta)} pts)",
                 Path(f"{args.out_prefix}_diff_seed_{args.fixed_objective}_{tag}_perp{args.perplexity}.png"),
                 draw_order=SEED_DRAW_ORDER, marker_map=SEED_MARKERS)


if __name__ == "__main__":
    main()
