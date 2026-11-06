#!/usr/bin/env python3
"""t-SNE of the SAE's own sparse code vectors (the vocabulary itself, not the
reconstruction residual) on the hard-image set, colored by objective/seed.

Two independent axes, mirroring plot_residual_tsne_hprojected.py:
  - absolute vs difference: "absolute" uses each cell's own raw local
    dictionary indices (feature_indices). Every (objective, seed) cell is
    its own independently-initialized 8192-wide dictionary (see
    FlatTopKSAE.__init__ in train_spatial_map_sae.py -- decoder ~
    N(0, d^-0.5) fresh per training run), so two cells' raw code vectors are
    near-certain to have almost-disjoint nonzero support (expected overlap
    in a k=32-of-8192 draw is ~0.13 indices) regardless of any real
    objective effect -- confirmed by the diff_seed control. "difference"
    instead uses aligned_feature_indices, produced by
    extract_feature_usage_for_hard_images.py by remapping every cell's
    dictionary atoms into one fixed reference cell's atom space via nearest
    decoder-direction cosine similarity -- meaningful because every cell's
    decoder reconstructs directions in the same underlying 512-dim ConvAE
    latent space, so this removes the "which arbitrary dictionary" nuisance
    the same way h-projection removes the "which original image" nuisance
    in the residual pipeline.
  - normalized vs unnormalized: whether the dense code vector is L2-unit
    normalized before PCA/t-SNE (direction only) or left at its raw
    magnitude.
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


def get(records, source, objective, seed):
    for r in records:
        if r["source"] == source and r["objective"] == objective and r["seed"] == seed:
            return r
    return None


def subsample(n_total, n, generator):
    if n_total <= n:
        return torch.arange(n_total)
    return torch.randperm(n_total, generator=generator)[:n]


def densify(feature_values, feature_indices, feature_count, normalize):
    n = feature_values.shape[0]
    dense = torch.zeros(n, feature_count, dtype=torch.float32)
    # scatter_add_ (not scatter_) because aligned indices can collide within a row
    dense.scatter_add_(1, feature_indices, feature_values.float())
    if normalize:
        dense = dense / dense.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return dense


def build_matrix(records, source, filter_fn, per_cell_cap, feature_count, generator, align, normalize):
    vectors, meta = [], []
    index_key = "aligned_feature_indices" if align else "feature_indices"
    for rec in records:
        if rec["source"] != source or not filter_fn(rec):
            continue
        n_total = rec["feature_indices"].shape[0]
        idx = subsample(n_total, per_cell_cap, generator)
        dense = densify(rec["feature_values"][idx], rec[index_key][idx], feature_count, normalize)
        vectors.append(dense)
        meta.extend([{"objective": rec["objective"], "seed": rec["seed"]}] * dense.shape[0])
    if not vectors:
        return None, []
    return torch.cat(vectors, dim=0).numpy(), meta


def run_embedding(matrix, perplexity, early_exaggeration, n_iter, pca_dims):
    if pca_dims and matrix.shape[1] > pca_dims:
        matrix = PCA(n_components=pca_dims, random_state=0).fit_transform(matrix)
    effective = min(perplexity, (matrix.shape[0] - 1) // 3)
    return TSNE(n_components=2, init="pca", perplexity=effective, early_exaggeration=early_exaggeration,
                max_iter=n_iter, n_jobs=-1, random_state=0).fit_transform(matrix)


def plot(embedding, meta, color_key, color_map, label_map, title, out_path, draw_order=None, marker_map=None):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    present = {m[color_key] for m in meta}
    marker_map = marker_map or {}
    handles = []
    ordered = draw_order or sorted(present, key=str)
    keys = [k for k in ordered if k in present]
    keys += sorted(present - set(ordered), key=str)
    for key in keys:
        mask = np.array([m[color_key] == key for m in meta])
        sc = ax.scatter(embedding[mask, 0], embedding[mask, 1], s=6, alpha=0.6,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usage", required=True)
    parser.add_argument("--source", choices=("ffhq_heldout", "celeba", "fairface"), default="ffhq_heldout")
    parser.add_argument("--per-cell-cap", type=int, default=2000)
    parser.add_argument("--fixed-seed", type=int, default=2030)
    parser.add_argument("--fixed-objective", default="latent_mse")
    parser.add_argument("--perplexity", type=int, default=15)
    parser.add_argument("--early-exaggeration", type=float, default=24.0)
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--pca-dims", type=int, default=100)
    parser.add_argument("--skip-by-objective", action="store_true")
    parser.add_argument("--skip-by-seed", action="store_true")
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    data = torch.load(args.usage, map_location="cpu", weights_only=False)
    records, feature_count = data["records"], data["feature_count"]
    if "align_quality" in data:
        print(f"ref_cell={data['ref_cell']}")
        for cell, cos in data["align_quality"].items():
            print(f"  align_quality {cell}: mean_top1_cosine={cos:.3f}")

    for align, align_tag in ((True, "difference"), (False, "absolute")):
        for normalize, norm_tag in ((False, "unnormalized"), (True, "normalized")):
            tag = f"{align_tag}_{norm_tag}"
            generator = torch.Generator().manual_seed(0)

            if not args.skip_by_objective:
                t0 = time.time()
                matrix, meta = build_matrix(
                    records, args.source, lambda rec: rec["seed"] == args.fixed_seed,
                    args.per_cell_cap, feature_count, generator, align, normalize,
                )
                if matrix is not None:
                    embedding = run_embedding(matrix, args.perplexity, args.early_exaggeration, args.n_iter, args.pca_dims)
                    plot(embedding, meta, "objective", OBJECTIVE_COLORS, OBJECTIVE_LABELS,
                         f"{args.source}, SAE code vectors, by objective, {tag} ({len(meta)} pts)",
                         Path(f"{args.out_prefix}_by_objective_{tag}.png"),
                         draw_order=OBJECTIVE_DRAW_ORDER, marker_map=OBJECTIVE_MARKERS)
                    print(f"  by_objective {tag} took {time.time()-t0:.1f}s", flush=True)

            if not args.skip_by_seed:
                t0 = time.time()
                matrix, meta = build_matrix(
                    records, args.source, lambda rec: rec["objective"] == args.fixed_objective,
                    args.per_cell_cap, feature_count, generator, align, normalize,
                )
                if matrix is not None:
                    embedding = run_embedding(matrix, args.perplexity, args.early_exaggeration, args.n_iter, args.pca_dims)
                    seed_label_map = {s: f"seed {s}" for s in SEED_COLORS}
                    plot(embedding, meta, "seed", SEED_COLORS, seed_label_map,
                         f"{args.source}, SAE code vectors, by seed ({args.fixed_objective}), {tag} ({len(meta)} pts)",
                         Path(f"{args.out_prefix}_by_seed_{tag}.png"),
                         draw_order=SEED_DRAW_ORDER, marker_map=SEED_MARKERS)
                    print(f"  by_seed {tag} took {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
