#!/usr/bin/env python3
"""Index-agnostic version of the feature-usage t-SNE: instead of densifying
each image's active features into an 8192-dim vector indexed by the SAE's
own (arbitrary, per-model-random) dictionary slots -- which only becomes
comparable across cells after a lossy nearest-neighbor alignment step -- this
represents every active feature directly by its causal effect in the shared
512-dim ConvAE latent space, where no alignment is needed at all because
every cell's SAE reconstructs into the *same* space.

The representation is the exact ablation effect of turning that one active
feature off: since FlatTopKSAE.forward() computes
    reconstruction = sum_i(values_i * decoder[indices_i]) + dec_bias
with a fixed bias independent of which features are active, removing feature
i's contribution from the sum is *exactly* `values_i * decoder[indices_i]`
(no need to actually re-run the forward pass -- this is the closed form of
the ablation, not an approximation). Un-standardizing by the cell's own
channel_std turns this into a real direction in raw ConvAE latent space,
directly comparable point-for-point across every objective and seed with no
remapping step, because the coordinate system (the 512-dim ConvAE latent) was
shared from the start.

One point is produced per (image, active feature) pair -- not per image --
so the point budget is much larger than the feature-usage pipeline's; a
--features-per-image cap subsamples each image's ~32 active features down to
a manageable number before pooling and embedding.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from tsne_viz_utils import kde_contours_multi, kde_contour_single

OBJECTIVES = ("latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high")
OBJECTIVE_COLORS = {
    "latent_mse": "#984ea3", "laplacian": "#e41a1c", "fourier_low": "#4daf4a",
    "fourier_mid": "#ff7f00", "fourier_high": "#377eb8",
}
OBJECTIVE_LABELS = {
    "latent_mse": "MSE", "laplacian": "Laplacian", "fourier_low": "Fourier low",
    "fourier_mid": "Fourier mid", "fourier_high": "Fourier high",
}
OBJECTIVE_MARKERS = {
    "latent_mse": "o", "laplacian": "s", "fourier_low": "^", "fourier_mid": "D", "fourier_high": "v",
}
OBJECTIVE_DRAW_ORDER = ["latent_mse", "fourier_mid", "fourier_low", "fourier_high", "laplacian"]

SEED_COLORS = {2030: "#1b9e77", 2031: "#d95f02", 2032: "#7570b3"}
SEED_MARKERS = {2030: "o", 2031: "s", 2032: "^"}
SEED_DRAW_ORDER = [2030, 2031, 2032]

CACHE_BY_SOURCE = {
    "ffhq_heldout": "latents/vector_ffhq_32768.pt",
    "celeba": "linear_probe_vector16x_lambda_sweep/celeba_vector_32768.pt",
    "fairface": "latents/fairface_vector_8192.pt",
}


def load_sae_parts(checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    decoder = state["model"]["decoder"].float()      # [feature_count, d]
    dec_bias = state["model"]["dec_bias"].float()     # [d]
    mean = state["channel_mean"].float().view(-1)     # [d]
    std = state["channel_std"].float().view(-1)       # [d]
    return decoder, dec_bias, mean, std


def subsample(n, k, generator):
    if n <= k:
        return torch.arange(n)
    return torch.randperm(n, generator=generator)[:k]


def build_points(records, source, base, arm, lambda_tag, per_cell_images, features_per_image, generator,
                  latents=None):
    """One row per (image, active feature): delta = value * decoder[index] * std,
    in raw ConvAE latent space. meta tracks objective/seed/latent_index/value.
    If `latents` is given, also computes each image's own reconstruction
    fidelity (-||residual||, using ALL its active features, not just the
    subsampled ones) and stores it in meta as "fidelity", broadcast to every
    point belonging to that image."""
    vectors, meta = [], []
    sae_cache = {}
    for rec in records:
        if rec["source"] != source:
            continue
        objective, seed = rec["objective"], rec["seed"]
        key = (objective, seed)
        if key not in sae_cache:
            subdir = "mse_control" if objective == "latent_mse" else f"lambda_{lambda_tag}"
            ckpt = Path(base) / "experiments" / arm / "checkpoints" / subdir / objective / f"seed_{seed}" / "last.pt"
            sae_cache[key] = load_sae_parts(ckpt)
        decoder, dec_bias, mean, std = sae_cache[key]

        n_images = rec["feature_indices"].shape[0]
        img_idx = subsample(n_images, per_cell_images, generator)
        values_full = rec["feature_values"][img_idx]          # [n, k]
        indices_full = rec["feature_indices"][img_idx]         # [n, k]
        latent_idx = rec["latent_index"][img_idx]

        fidelity_full = None
        if latents is not None:
            raw = latents[latent_idx].float()
            h_std = (raw - mean) / std
            recon_full = (values_full.unsqueeze(-1) * decoder[indices_full]).sum(1) + dec_bias
            residual_full = h_std - recon_full
            fidelity_full = -residual_full.norm(dim=1)

        k = values_full.shape[1]
        feat_idx = subsample(k, features_per_image, generator)
        values = values_full[:, feat_idx]
        indices = indices_full[:, feat_idx]

        atoms = decoder[indices]                           # [n, f, d]
        deltas = values.unsqueeze(-1) * atoms * std.view(1, 1, -1)  # [n, f, d], raw AE-space

        n, f, d = deltas.shape
        vectors.append(deltas.reshape(n * f, d))
        for i in range(n):
            for j in range(f):
                point = {
                    "objective": objective, "seed": seed,
                    "latent_index": int(latent_idx[i]), "value": float(values[i, j]),
                }
                if fidelity_full is not None:
                    point["fidelity"] = float(fidelity_full[i])
                meta.append(point)
    return torch.cat(vectors, dim=0).numpy(), meta


def run_embedding(matrix, perplexity, early_exaggeration, n_iter, pca_dims):
    if pca_dims and matrix.shape[1] > pca_dims:
        matrix = PCA(n_components=pca_dims, random_state=0).fit_transform(matrix)
    effective = min(perplexity, (matrix.shape[0] - 1) // 3)
    return TSNE(n_components=2, init="pca", perplexity=effective, early_exaggeration=early_exaggeration,
                max_iter=n_iter, n_jobs=-1, random_state=0).fit_transform(matrix)


def plot(embedding, meta, color_key, color_map, label_map, title, out_path, draw_order, marker_map):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    present = {m[color_key] for m in meta}
    keys = [k for k in draw_order if k in present]
    xy_by_class = {key: embedding[np.array([m[color_key] == key for m in meta])] for key in keys}
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


def plot_highlight(embedding, meta, highlight, title, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    mask_other = np.array([m["objective"] != highlight for m in meta])
    mask_high = ~mask_other
    ax.scatter(embedding[mask_other, 0], embedding[mask_other, 1], s=4, alpha=0.15,
               color="#bbbbbb", marker="o", linewidths=0, label="other", zorder=1)
    kde_contour_single(ax, embedding[mask_high], OBJECTIVE_COLORS[highlight])
    handles = [plt.Line2D([0], [0], marker="o", color="#bbbbbb", lw=0, markersize=5, label="other"),
               plt.Line2D([0], [0], color=OBJECTIVE_COLORS[highlight], lw=2, label=OBJECTIVE_LABELS[highlight])]
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(handles=handles, fontsize=9, frameon=False, loc="best")
    ax.set_title(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def plot_fidelity(embedding, meta, title, out_path):
    fidelity = np.array([m["fidelity"] for m in meta])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    norm = Normalize(vmin=np.percentile(fidelity, 2), vmax=np.percentile(fidelity, 98))
    ax.scatter(embedding[:, 0], embedding[:, 1], s=4, alpha=0.5, c=fidelity,
               cmap="coolwarm", norm=norm, linewidths=0, zorder=2)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--usage", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--arm", default="vector_16x_k32")
    parser.add_argument("--lambda-tag", default="10p0")
    parser.add_argument("--source", choices=("ffhq_heldout", "celeba", "fairface"), default="ffhq_heldout")
    parser.add_argument("--per-cell-images", type=int, default=400,
                         help="images sampled per (objective, seed) cell before exploding into per-feature points")
    parser.add_argument("--features-per-image", type=int, default=8,
                         help="active features sampled per image (out of k=32)")
    parser.add_argument("--perplexity", type=int, default=15)
    parser.add_argument("--early-exaggeration", type=float, default=24.0)
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--pca-dims", type=int, default=0, help="0 disables PCA (already only 512-dim)")
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    data = torch.load(args.usage, map_location="cpu", weights_only=False)
    records = data["records"]
    cache = torch.load(Path(args.base) / CACHE_BY_SOURCE[args.source], map_location="cpu", weights_only=False)
    latent_key = "latents" if "latents" in cache else "clean_latents"
    latents = cache[latent_key].float()

    generator = torch.Generator().manual_seed(0)
    matrix, meta = build_points(records, args.source, args.base, args.arm, args.lambda_tag,
                                 args.per_cell_images, args.features_per_image, generator, latents=latents)
    print(f"{args.source}: {matrix.shape[0]} (image, active-feature) points, {matrix.shape[1]}-dim raw AE-space "
          f"ablation deltas, no alignment", flush=True)

    embedding = run_embedding(matrix, args.perplexity, args.early_exaggeration, args.n_iter, args.pca_dims)

    plot(embedding, meta, "objective", OBJECTIVE_COLORS, OBJECTIVE_LABELS,
         f"{args.source}, feature t-SNE, by objective ({len(meta)} pts)",
         Path(f"{args.out_prefix}_by_objective.png"), OBJECTIVE_DRAW_ORDER, OBJECTIVE_MARKERS)
    seed_label_map = {s: f"seed {s}" for s in SEED_COLORS}
    plot(embedding, meta, "seed", SEED_COLORS, seed_label_map,
         f"{args.source}, feature t-SNE, by seed ({len(meta)} pts)",
         Path(f"{args.out_prefix}_by_seed.png"), SEED_DRAW_ORDER, SEED_MARKERS)
    plot_fidelity(embedding, meta,
                  f"{args.source}, feature t-SNE, by fidelity ({len(meta)} pts)",
                  Path(f"{args.out_prefix}_by_fidelity.png"))
    for obj in OBJECTIVES:
        plot_highlight(embedding, meta, obj,
                        f"{args.source}, feature t-SNE, "
                        f"{OBJECTIVE_LABELS[obj]} highlighted ({len(meta)} pts)",
                        Path(f"{args.out_prefix}_highlight_{obj}.png"))

    out_npz = Path(f"{args.out_prefix}_embedding.npz")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, embedding=embedding,
             objective=np.array([m["objective"] for m in meta]),
             seed=np.array([m["seed"] for m in meta]),
             latent_index=np.array([m["latent_index"] for m in meta]),
             fidelity=np.array([m["fidelity"] for m in meta]))
    print(f"wrote {out_npz}", flush=True)


if __name__ == "__main__":
    main()
