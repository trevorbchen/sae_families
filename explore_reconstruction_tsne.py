#!/usr/bin/env python3
"""Joint (15-cell-pooled) version of the residual/"by SAE" t-SNE -- same
h-projection fix as plot_residual_tsne_hprojected.py (raw residual is mostly
just "which image is this"; projecting out each image's own standardized
activation h leaves the objective-dependent part), but pooling all 5
objectives x 3 seeds into one shared embedding (consistent with the
feature-usage joint fix) instead of two separate by_objective/by_seed
subsets. Rendered three ways on the same coordinates: by objective, by seed,
and continuously by reconstruction fidelity (cool=low fidelity/high error,
hot=high fidelity/low error) to see whether the regions that separate by
objective also happen to correspond to systematically better or worse
reconstruction.

Residuals are computed directly here (not loaded from a separate
extraction), since every SAE checkpoint and the raw latents needed are
already available locally.
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


def load_sae(checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    decoder = state["model"]["decoder"].float()
    encoder = state["model"]["encoder"].float()
    encoder_bias = state["model"]["encoder_bias"].float()
    dec_bias = state["model"]["dec_bias"].float()
    mean = state["channel_mean"].float().view(1, -1)
    std = state["channel_std"].float().view(1, -1)
    k = state["config"]["k"]
    return decoder, encoder, encoder_bias, dec_bias, mean, std, k


@torch.no_grad()
def sae_forward(x, decoder, encoder, encoder_bias, dec_bias, k):
    activations = torch.relu(torch.nn.functional.linear(x - dec_bias, encoder, encoder_bias))
    values, indices = torch.topk(activations, k, dim=1)
    reconstruction = (values.unsqueeze(-1) * decoder[indices]).sum(1) + dec_bias
    return reconstruction


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def project_out_h(residual, h_std):
    h_u = unit(h_std)
    coeff = (residual * h_u).sum(dim=1, keepdim=True)
    return residual - coeff * h_u


def subsample(n, k, generator):
    if n <= k:
        return torch.arange(n)
    return torch.randperm(n, generator=generator)[:k]


def build_joint(records, source, latents, base, arm, lambda_tag, per_cell_cap, generator):
    vectors, meta, fidelity = [], [], []
    sae_cache = {}
    for rec in records:
        if rec["source"] != source:
            continue
        objective, seed = rec["objective"], rec["seed"]
        key = (objective, seed)
        if key not in sae_cache:
            subdir = "mse_control" if objective == "latent_mse" else f"lambda_{lambda_tag}"
            ckpt = Path(base) / "experiments" / arm / "checkpoints" / subdir / objective / f"seed_{seed}" / "last.pt"
            sae_cache[key] = load_sae(ckpt)
        decoder, encoder, encoder_bias, dec_bias, mean, std, k = sae_cache[key]

        n_total = rec["feature_indices"].shape[0]
        img_idx = subsample(n_total, per_cell_cap, generator)
        latent_idx = rec["latent_index"][img_idx]

        raw = latents[latent_idx].float()
        h_std = (raw - mean) / std
        reconstruction = sae_forward(h_std, decoder, encoder, encoder_bias, dec_bias, k)
        residual = h_std - reconstruction
        fid = -residual.norm(dim=1)  # higher = better reconstruction = "hotter"
        vec = unit(project_out_h(residual, h_std))

        vectors.append(vec)
        fidelity.append(fid)
        for i in range(vec.shape[0]):
            meta.append({"objective": objective, "seed": seed, "latent_index": int(latent_idx[i])})
    return torch.cat(vectors, dim=0).numpy(), meta, torch.cat(fidelity, dim=0).numpy()


def run_embedding(matrix, perplexity, early_exaggeration, n_iter, pca_dims):
    if pca_dims and matrix.shape[1] > pca_dims:
        matrix = PCA(n_components=pca_dims, random_state=0).fit_transform(matrix)
    effective = min(perplexity, (matrix.shape[0] - 1) // 3)
    return TSNE(n_components=2, init="pca", perplexity=effective, early_exaggeration=early_exaggeration,
                max_iter=n_iter, n_jobs=-1, random_state=0).fit_transform(matrix)


def plot_categorical(embedding, meta, color_key, color_map, label_map, title, out_path, draw_order, marker_map):
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
    ax.scatter(embedding[mask_other, 0], embedding[mask_other, 1], s=5, alpha=0.15,
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


def plot_fidelity(embedding, fidelity, title, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    norm = Normalize(vmin=np.percentile(fidelity, 2), vmax=np.percentile(fidelity, 98))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.6, c=fidelity,
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
    parser.add_argument("--usage", required=True, help="feature_usage .pt (reused just for its cell list / image sampling)")
    parser.add_argument("--base", required=True)
    parser.add_argument("--arm", default="vector_16x_k32")
    parser.add_argument("--lambda-tag", default="10p0")
    parser.add_argument("--source", choices=("ffhq_heldout", "celeba", "fairface"), default="ffhq_heldout")
    parser.add_argument("--per-cell-cap", type=int, default=2000)
    parser.add_argument("--perplexity", type=int, default=15)
    parser.add_argument("--early-exaggeration", type=float, default=24.0)
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--pca-dims", type=int, default=100)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    base = Path(args.base)
    data = torch.load(args.usage, map_location="cpu", weights_only=False)
    records = data["records"]

    cache = torch.load(base / CACHE_BY_SOURCE[args.source], map_location="cpu", weights_only=False)
    latent_key = "latents" if "latents" in cache else "clean_latents"
    latents = cache[latent_key].float()

    generator = torch.Generator().manual_seed(0)
    matrix, meta, fidelity = build_joint(records, args.source, latents, base, args.arm, args.lambda_tag,
                                          args.per_cell_cap, generator)
    print(f"{args.source}: {matrix.shape[0]} pts, joint residual (h-projected, normalized), "
          f"fidelity range [{fidelity.min():.3f}, {fidelity.max():.3f}]", flush=True)

    embedding = run_embedding(matrix, args.perplexity, args.early_exaggeration, args.n_iter, args.pca_dims)

    plot_categorical(embedding, meta, "objective", OBJECTIVE_COLORS, OBJECTIVE_LABELS,
                      f"{args.source}, reconstruction t-SNE, by objective ({len(meta)} pts)",
                      Path(f"{args.out_prefix}_by_objective.png"), OBJECTIVE_DRAW_ORDER, OBJECTIVE_MARKERS)
    seed_label_map = {s: f"seed {s}" for s in SEED_COLORS}
    plot_categorical(embedding, meta, "seed", SEED_COLORS, seed_label_map,
                      f"{args.source}, reconstruction t-SNE, by seed ({len(meta)} pts)",
                      Path(f"{args.out_prefix}_by_seed.png"), SEED_DRAW_ORDER, SEED_MARKERS)
    plot_fidelity(embedding, fidelity,
                  f"{args.source}, reconstruction t-SNE, by fidelity ({len(meta)} pts)",
                  Path(f"{args.out_prefix}_by_fidelity.png"))
    for obj in OBJECTIVES:
        plot_highlight(embedding, meta, obj,
                        f"{args.source}, reconstruction t-SNE, {OBJECTIVE_LABELS[obj]} highlighted ({len(meta)} pts)",
                        Path(f"{args.out_prefix}_highlight_{obj}.png"))

    out_npz = Path(f"{args.out_prefix}_embedding.npz")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, embedding=embedding,
             objective=np.array([m["objective"] for m in meta]),
             seed=np.array([m["seed"] for m in meta]),
             latent_index=np.array([m["latent_index"] for m in meta]),
             fidelity=fidelity)
    print(f"wrote {out_npz}", flush=True)


if __name__ == "__main__":
    main()
