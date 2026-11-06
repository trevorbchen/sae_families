#!/usr/bin/env python3
"""One combined t-SNE over ALL 5 objectives x 3 seeds (15 cells) pooled
together, rendered twice -- colored by objective, colored by seed -- using
the exact same 2D coordinates. This is the apples-to-apples version of the
by_objective/by_seed comparison: plot_feature_usage_tsne.py instead runs two
*separate* embeddings on two *different* point subsets (5 objectives at a
single fixed seed; 1 objective at 3 seeds), so those two plots are not
directly comparable to each other. Only the aligned ("difference")
representation is used here -- pooling all 15 independently-initialized
dictionaries unaligned would trivially separate by cell identity and tell
us nothing new.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

OBJECTIVE_COLORS = {
    "latent_mse": "#984ea3", "laplacian": "#e41a1c", "fourier_low": "#4daf4a",
    "fourier_mid": "#ff7f00", "fourier_high": "#377eb8",
}
OBJECTIVE_LABELS = {
    "latent_mse": "MSE (baseline)", "laplacian": "Laplacian", "fourier_low": "Fourier low",
    "fourier_mid": "Fourier mid", "fourier_high": "Fourier high",
}
OBJECTIVE_MARKERS = {
    "latent_mse": "o", "laplacian": "s", "fourier_low": "^", "fourier_mid": "D", "fourier_high": "v",
}
OBJECTIVE_DRAW_ORDER = ["latent_mse", "fourier_mid", "fourier_low", "fourier_high", "laplacian"]

SEED_COLORS = {2030: "#1b9e77", 2031: "#d95f02", 2032: "#7570b3"}
SEED_MARKERS = {2030: "o", 2031: "s", 2032: "^"}
SEED_DRAW_ORDER = [2030, 2031, 2032]


def subsample(n_total, n, generator):
    if n_total <= n:
        return torch.arange(n_total)
    return torch.randperm(n_total, generator=generator)[:n]


def densify(feature_values, feature_indices, feature_count, normalize):
    n = feature_values.shape[0]
    dense = torch.zeros(n, feature_count, dtype=torch.float32)
    dense.scatter_add_(1, feature_indices, feature_values.float())
    if normalize:
        dense = dense / dense.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return dense


def build_matrix(records, source, per_cell_cap, feature_count, generator, normalize):
    vectors, meta = [], []
    for rec in records:
        if rec["source"] != source:
            continue
        n_total = rec["feature_indices"].shape[0]
        idx = subsample(n_total, per_cell_cap, generator)
        dense = densify(rec["feature_values"][idx], rec["aligned_feature_indices"][idx], feature_count, normalize)
        vectors.append(dense)
        meta.extend([{"objective": rec["objective"], "seed": rec["seed"]}] * dense.shape[0])
    return torch.cat(vectors, dim=0).numpy(), meta


def run_embedding(matrix, perplexity, early_exaggeration, n_iter, pca_dims):
    if pca_dims and matrix.shape[1] > pca_dims:
        matrix = PCA(n_components=pca_dims, random_state=0).fit_transform(matrix)
    effective = min(perplexity, (matrix.shape[0] - 1) // 3)
    return TSNE(n_components=2, init="pca", perplexity=effective, early_exaggeration=early_exaggeration,
                max_iter=n_iter, n_jobs=-1, random_state=0).fit_transform(matrix)


def plot(embedding, meta, color_key, color_map, label_map, title, out_path, draw_order, marker_map):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    present = {m[color_key] for m in meta}
    handles = []
    keys = [k for k in draw_order if k in present]
    for key in keys:
        mask = np.array([m[color_key] == key for m in meta])
        sc = ax.scatter(embedding[mask, 0], embedding[mask, 1], s=5, alpha=0.5,
                         color=color_map.get(key, "#999999"), marker=marker_map.get(key, "o"),
                         linewidths=0, label=label_map.get(key, str(key)), zorder=2)
        handles.append(sc)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(handles=handles, markerscale=3, fontsize=9, frameon=False, loc="best")
    ax.set_title(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def plot_highlight(embedding, meta, color_key, highlight, color_map, label_map, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    mask_other = np.array([m[color_key] != highlight for m in meta])
    mask_high = ~mask_other
    ax.scatter(embedding[mask_other, 0], embedding[mask_other, 1], s=5, alpha=0.25,
               color="#bbbbbb", marker="o", linewidths=0, label="other", zorder=1)
    ax.scatter(embedding[mask_high, 0], embedding[mask_high, 1], s=6, alpha=0.7,
               color=color_map.get(highlight, "#e41a1c"), marker="o", linewidths=0,
               label=label_map.get(highlight, str(highlight)), zorder=2)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(markerscale=3, fontsize=9, frameon=False, loc="best")
    ax.set_title(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usage", required=True)
    parser.add_argument("--source", choices=("ffhq_heldout", "celeba", "fairface"), default="ffhq_heldout")
    parser.add_argument("--per-cell-cap", type=int, default=2000)
    parser.add_argument("--perplexity", type=int, default=15)
    parser.add_argument("--early-exaggeration", type=float, default=24.0)
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--pca-dims", type=int, default=100)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--only-normalized", action="store_true")
    parser.add_argument("--with-highlights", action="store_true",
                         help="also emit one by_objective plot per objective with all others grayed out")
    args = parser.parse_args()

    data = torch.load(args.usage, map_location="cpu", weights_only=False)
    records, feature_count = data["records"], data["feature_count"]

    tags = ((True, "normalized"),) if args.only_normalized else ((False, "unnormalized"), (True, "normalized"))
    for normalize, norm_tag in tags:
        tag = f"joint_difference_{norm_tag}"
        generator = torch.Generator().manual_seed(0)
        t0 = time.time()
        matrix, meta = build_matrix(records, args.source, args.per_cell_cap, feature_count, generator, normalize)
        print(f"{args.source} {tag}: embedding {matrix.shape[0]} points (all 15 objective x seed cells pooled)", flush=True)
        embedding = run_embedding(matrix, args.perplexity, args.early_exaggeration, args.n_iter, args.pca_dims)
        print(f"  t-SNE took {time.time()-t0:.1f}s", flush=True)

        plot(embedding, meta, "objective", OBJECTIVE_COLORS, OBJECTIVE_LABELS,
             f"{args.source}, joint SAE code-vector t-SNE, by objective, {norm_tag} ({len(meta)} pts)",
             Path(f"{args.out_prefix}_{tag}_by_objective.png"),
             OBJECTIVE_DRAW_ORDER, OBJECTIVE_MARKERS)
        seed_label_map = {s: f"seed {s}" for s in SEED_COLORS}
        plot(embedding, meta, "seed", SEED_COLORS, seed_label_map,
             f"{args.source}, joint SAE code-vector t-SNE, by seed, {norm_tag} ({len(meta)} pts)",
             Path(f"{args.out_prefix}_{tag}_by_seed.png"),
             SEED_DRAW_ORDER, SEED_MARKERS)

        if args.with_highlights:
            for obj in OBJECTIVE_DRAW_ORDER:
                plot_highlight(embedding, meta, "objective", obj, OBJECTIVE_COLORS, OBJECTIVE_LABELS,
                               f"{args.source}, joint SAE code-vector t-SNE, {OBJECTIVE_LABELS[obj]} highlighted, "
                               f"{norm_tag} ({len(meta)} pts)",
                               Path(f"{args.out_prefix}_{tag}_highlight_{obj}.png"))


if __name__ == "__main__":
    main()
