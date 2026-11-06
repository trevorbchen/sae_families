#!/usr/bin/env python3
"""t-SNE of SAE code vectors (same embedding as plot_feature_usage_tsne.py's
by_objective panel), with the actual decoded face images for a handful of
points sampled from each spatial region -- answers "what do the images in
this part of the embedding actually look like" rather than just "what color
is this region."

Procedure: embed with the aligned ("difference"), L2-normalized code
vectors (the one tag where cross-cell alignment quality is good, since
by_objective fixes seed=2030=the reference seed); k-means the 2D embedding
into spatial regions; for each region, decode a few representative points'
*actual* latents (not the SAE reconstruction) through the base ConvAE
decoder to get real face thumbnails; lay the thumbnails out under the
scatter plot, keyed by region number, alongside each region's objective
color composition.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from vector_conv_autoencoder import load_vector_autoencoder
from spatial_conv_autoencoder import load_spatial_autoencoder

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

CACHE_BY_SOURCE = {
    "ffhq_heldout": {"vector": "latents/vector_ffhq_32768.pt", "spatial": "latents/spatial_ffhq_32768.pt"},
    "celeba": {
        "vector": "linear_probe_vector16x_lambda_sweep/celeba_vector_32768.pt",
        "spatial": "linear_probe_spatial16x_lambda_sweep/celeba_spatial_32768.pt",
    },
    "fairface": {"vector": "latents/fairface_vector_8192.pt", "spatial": "latents/fairface_spatial_8192.pt"},
}
AE_PATH = {"vector": "autoencoders/vector/last.pt", "spatial": "autoencoders/spatial/last.pt"}


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


def build_matrix(records, source, fixed_seed, per_cell_cap, feature_count, generator, align, normalize):
    vectors, meta = [], []
    index_key = "aligned_feature_indices" if align else "feature_indices"
    for rec in records:
        if rec["source"] != source or rec["seed"] != fixed_seed:
            continue
        n_total = rec["feature_indices"].shape[0]
        idx = subsample(n_total, per_cell_cap, generator)
        dense = densify(rec["feature_values"][idx], rec[index_key][idx], feature_count, normalize)
        vectors.append(dense)
        latent_idx = rec["latent_index"][idx]
        for i in range(dense.shape[0]):
            meta.append({"objective": rec["objective"], "seed": rec["seed"], "latent_index": int(latent_idx[i])})
    return torch.cat(vectors, dim=0).numpy(), meta


def run_embedding(matrix, perplexity, early_exaggeration, n_iter, pca_dims):
    if pca_dims and matrix.shape[1] > pca_dims:
        matrix = PCA(n_components=pca_dims, random_state=0).fit_transform(matrix)
    effective = min(perplexity, (matrix.shape[0] - 1) // 3)
    return TSNE(n_components=2, init="pca", perplexity=effective, early_exaggeration=early_exaggeration,
                max_iter=n_iter, n_jobs=-1, random_state=0).fit_transform(matrix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usage", required=True)
    parser.add_argument("--base", default="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1")
    parser.add_argument("--arm", default="vector_16x_k32")
    parser.add_argument("--source", choices=("ffhq_heldout", "celeba", "fairface"), default="ffhq_heldout")
    parser.add_argument("--per-cell-cap", type=int, default=2000)
    parser.add_argument("--fixed-seed", type=int, default=2030)
    parser.add_argument("--n-regions", type=int, default=8)
    parser.add_argument("--thumbs-per-region", type=int, default=5)
    parser.add_argument("--perplexity", type=int, default=15)
    parser.add_argument("--early-exaggeration", type=float, default=24.0)
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--pca-dims", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(args.base)
    ae_type = "vector" if "vector" in args.arm else "spatial"

    data = torch.load(args.usage, map_location="cpu", weights_only=False)
    records, feature_count = data["records"], data["feature_count"]

    generator = torch.Generator().manual_seed(0)
    matrix, meta = build_matrix(records, args.source, args.fixed_seed, args.per_cell_cap,
                                 feature_count, generator, align=True, normalize=True)
    print(f"embedding {matrix.shape[0]} points, {matrix.shape[1]} dims (aligned, normalized)", flush=True)
    embedding = run_embedding(matrix, args.perplexity, args.early_exaggeration, args.n_iter, args.pca_dims)

    km = KMeans(n_clusters=args.n_regions, random_state=0, n_init=10).fit(embedding)
    region_id = km.labels_
    centers = km.cluster_centers_

    cache = torch.load(base / CACHE_BY_SOURCE[args.source][ae_type], map_location="cpu", weights_only=False)
    latent_key = "latents" if "latents" in cache else "clean_latents"
    latents = cache[latent_key].float()
    if ae_type == "vector":
        autoencoder = load_vector_autoencoder(str(base / AE_PATH[ae_type]), device)
    else:
        autoencoder = load_spatial_autoencoder(str(base / AE_PATH[ae_type]), device)
    autoencoder.eval().requires_grad_(False)

    fig = plt.figure(figsize=(9, 4 + 1.4 * args.n_regions))
    gs = GridSpec(args.n_regions + 1, args.thumbs_per_region, figure=fig,
                  height_ratios=[4 * args.thumbs_per_region / 5] + [1] * args.n_regions)
    ax = fig.add_subplot(gs[0, :])
    marker_map = OBJECTIVE_MARKERS
    for obj in OBJECTIVE_DRAW_ORDER:
        mask = np.array([m["objective"] == obj for m in meta])
        if mask.any():
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=5, alpha=0.55,
                       color=OBJECTIVE_COLORS[obj], marker=marker_map[obj], linewidths=0,
                       label=OBJECTIVE_LABELS[obj], zorder=2)
    for r in range(args.n_regions):
        ax.annotate(str(r), centers[r], fontsize=13, fontweight="bold", ha="center", va="center",
                    color="black", zorder=4,
                    bbox=dict(boxstyle="circle", facecolor="white", edgecolor="black", alpha=0.85))
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="best", markerscale=2)
    ax.set_title(f"{args.source}: SAE code-vector t-SNE (aligned, normalized), {len(meta)} pts, "
                 f"{args.n_regions} k-means regions", fontsize=10)

    rng = np.random.default_rng(0)
    with torch.no_grad():
        for r in range(args.n_regions):
            idxs_in_region = np.where(region_id == r)[0]
            objs_in_region = [meta[i]["objective"] for i in idxs_in_region]
            comp = {o: objs_in_region.count(o) / len(objs_in_region) for o in OBJECTIVE_DRAW_ORDER if objs_in_region.count(o) > 0}
            comp_str = ", ".join(f"{OBJECTIVE_LABELS[o]} {p:.0%}" for o, p in sorted(comp.items(), key=lambda kv: -kv[1]))
            pick = rng.choice(idxs_in_region, size=min(args.thumbs_per_region, len(idxs_in_region)), replace=False)
            for c, i in enumerate(pick):
                latent_idx = meta[i]["latent_index"]
                raw = latents[latent_idx:latent_idx + 1].to(device)
                decoded = autoencoder.decode(raw).add(1).div(2).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy()
                axt = fig.add_subplot(gs[r + 1, c])
                axt.imshow(decoded)
                axt.set_xticks([]); axt.set_yticks([])
                for spine in axt.spines.values():
                    spine.set_edgecolor(OBJECTIVE_COLORS[meta[i]["objective"]])
                    spine.set_linewidth(2.5)
                if c == 0:
                    axt.set_ylabel(f"region {r}", fontsize=9, rotation=0, labelpad=35, va="center")
            for c in range(len(pick), args.thumbs_per_region):
                axt = fig.add_subplot(gs[r + 1, c])
                axt.axis("off")
            row_bbox = gs[r + 1, :].get_position(fig)
            fig.text(0.5, row_bbox.y1 + 0.004, comp_str, fontsize=7.5, ha="center",
                      transform=fig.transFigure)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
