#!/usr/bin/env python3
"""t-SNE of SAE reconstruction residuals, with each image's own standardized
activation h projected out first.

Why: the raw residual (standardized_input - reconstruction) turns out to be
heavily parallel to the standardized input itself (cos(residual, h) ~ 0.72-0.81
measured on vector_16x_k32/lambda10/ffhq_heldout) -- i.e. most of a residual
vector's direction is just "which image is this", not anything about the
training objective. Pooled t-SNE on raw residuals is therefore dominated by
~2000 near-orthogonal per-image directions and renders as a uniform blob
regardless of objective. Projecting out each image's own h direction removes
that confound and leaves the part of the residual that is actually about
*how the SAE failed*, which does carry a real, objective-dependent,
cross-image-consistent signature for at least Laplacian and Fourier-high
(checked via cosine similarity of different-image/same-objective pairs:
~0.14-0.18 after projection vs ~0.01-0.08 before, against a ~0.03-0.07 random
floor).

This replaces plot_residual_tsne.py's figures entirely (same arm/lambda,
same objective/color/marker scheme), but needs each source's raw latents and
a reference checkpoint's standardization stats to compute h, which the saved
residuals.pt file does not include on its own.
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

CELEBA_CACHE = {
    "vector": "linear_probe_vector16x_lambda_sweep/celeba_vector_32768.pt",
    "spatial": "linear_probe_spatial16x_lambda_sweep/celeba_spatial_32768.pt",
}
FAIRFACE_CACHE = {
    "vector": "latents/fairface_vector_8192.pt",
    "spatial": "latents/fairface_spatial_8192.pt",
}


def load_raw_latents(base, source, arm):
    ae_type = "vector" if "vector" in arm else "spatial"
    if source == "ffhq_heldout":
        cache = torch.load(base / "latents" / f"{ae_type}_ffhq_32768.pt", map_location="cpu", weights_only=False)
        return cache["latents"].float()
    if source == "celeba":
        cache = torch.load(base / CELEBA_CACHE[ae_type], map_location="cpu", weights_only=False)
        return cache["clean_latents"].float()
    if source == "fairface":
        cache = torch.load(base / FAIRFACE_CACHE[ae_type], map_location="cpu", weights_only=False)
        return cache["latents"].float()
    raise ValueError(source)


def load_standardization(base, arm, lambda_tag, reference_objective="latent_mse", reference_seed=2030):
    subdir = "mse_control" if reference_objective == "latent_mse" else f"lambda_{lambda_tag}"
    ckpt_path = base / "experiments" / arm / "checkpoints" / subdir / reference_objective / f"seed_{reference_seed}" / "last.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mean = state["channel_mean"].float().view(1, -1)
    std = state["channel_std"].float().view(1, -1)
    return mean, std


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def load_records(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["arm"], data["lambda_value"], data["records"]


def get(records, source, objective, seed):
    for r in records:
        if r["source"] == source and r["objective"] == objective and r["seed"] == seed:
            return r
    return None


def subsample(residual, latent_index, n, generator):
    if residual.shape[0] <= n:
        return residual, latent_index
    perm = torch.randperm(residual.shape[0], generator=generator)[:n]
    return residual[perm], latent_index[perm]


def project_out_h(residual, h_std):
    h_u = unit(h_std)
    coeff = (residual * h_u).sum(dim=1, keepdim=True)
    return residual - coeff * h_u


def build_matrix(records, source, filter_fn, per_cell_cap, subtract_h, normalize, generator, h_by_index):
    vectors, meta = [], []
    for rec in records:
        if rec["source"] != source or not filter_fn(rec):
            continue
        residual, idx = subsample(rec["residual"].float(), rec["latent_index"], per_cell_cap, generator)
        vec = residual
        if subtract_h:
            h_std = torch.stack([h_by_index[int(i)] for i in idx.tolist()], dim=0)
            vec = project_out_h(residual, h_std)
        if normalize:
            vec = unit(vec)
        vectors.append(vec)
        meta.extend([{"objective": rec["objective"], "seed": rec["seed"]}] * vec.shape[0])
    if not vectors:
        return None, []
    return torch.cat(vectors, dim=0).numpy(), meta


def run_embedding(matrix, perplexity, early_exaggeration, n_iter, pca_dims):
    if pca_dims and matrix.shape[1] > pca_dims:
        matrix = PCA(n_components=pca_dims, random_state=0).fit_transform(matrix)
    effective = min(perplexity, (matrix.shape[0] - 1) // 3)
    return TSNE(n_components=2, init="pca", perplexity=effective, early_exaggeration=early_exaggeration,
                max_iter=n_iter, n_jobs=-1, random_state=0).fit_transform(matrix)


def plot(embedding, meta, color_key, color_map, label_map, title, out_path,
         draw_order=None, marker_map=None):
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
    print(f"wrote {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residuals", required=True)
    parser.add_argument("--base", default="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1")
    parser.add_argument("--source", choices=("ffhq_heldout", "celeba", "fairface"), default="ffhq_heldout")
    parser.add_argument("--per-cell-cap", type=int, default=2000)
    parser.add_argument("--fixed-seed", type=int, default=2030)
    parser.add_argument("--fixed-objective", default="latent_mse")
    parser.add_argument("--perplexity", type=int, default=15)
    parser.add_argument("--early-exaggeration", type=float, default=24.0)
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--pca-dims", type=int, default=100, help="0 to disable PCA pre-reduction")
    parser.add_argument("--skip-diff-objective", action="store_true")
    parser.add_argument("--skip-diff-seed", action="store_true")
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    base = Path(args.base)
    arm, lambda_value, records = load_records(args.residuals)
    lambda_tag = f"{lambda_value:g}".replace(".", "p")
    generator = torch.Generator().manual_seed(0)

    raw_latents = load_raw_latents(base, args.source, arm)
    mean, std = load_standardization(base, arm, lambda_tag)
    h_full = (raw_latents - mean) / std
    all_idx = set()
    for rec in records:
        if rec["source"] == args.source:
            all_idx.update(int(i) for i in rec["latent_index"].tolist())
    h_by_index = {i: h_full[i] for i in all_idx}
    print(f"h computed for {len(h_by_index)} images in {args.source}", flush=True)

    # Two independent axes:
    #  - subtract_h: "absolute" = raw residual r, h NOT subtracted (this is the
    #    old pre-fix pipeline -- expect no real separation). "difference" = h
    #    projected out of r first (see project_out_h above) -- this is the fix.
    #  - normalize: "unnormalized" keeps whatever magnitude survives;
    #    "normalized" additionally divides it out, keeping only direction.
    for subtract_h, h_tag in ((True, "difference"), (False, "absolute")):
        for normalize, norm_tag in ((False, "unnormalized"), (True, "normalized")):
            tag = f"{h_tag}_{norm_tag}"
            if not args.skip_diff_objective:
                t0 = time.time()
                matrix, meta = build_matrix(
                    records, args.source, lambda rec: rec["seed"] == args.fixed_seed,
                    args.per_cell_cap, subtract_h, normalize, generator, h_by_index,
                )
                if matrix is not None:
                    embedding = run_embedding(matrix, args.perplexity, args.early_exaggeration, args.n_iter, args.pca_dims)
                    plot(embedding, meta, "objective", OBJECTIVE_COLORS, OBJECTIVE_LABELS,
                         f"{args.source}, by objective, {tag} ({len(meta)} pts)",
                         Path(f"{args.out_prefix}_by_objective_{tag}.png"),
                         draw_order=OBJECTIVE_DRAW_ORDER, marker_map=OBJECTIVE_MARKERS)
                    print(f"  by_objective/{tag} took {time.time()-t0:.1f}s", flush=True)

            if args.skip_diff_seed:
                continue
            t0 = time.time()
            matrix, meta = build_matrix(
                records, args.source, lambda rec: rec["objective"] == args.fixed_objective,
                args.per_cell_cap, subtract_h, normalize, generator, h_by_index,
            )
            if matrix is not None:
                embedding = run_embedding(matrix, args.perplexity, args.early_exaggeration, args.n_iter, args.pca_dims)
                seed_label_map = {s: f"seed {s}" for s in SEED_COLORS}
                plot(embedding, meta, "seed", SEED_COLORS, seed_label_map,
                     f"{args.source}, by seed ({args.fixed_objective}), {tag} ({len(meta)} pts)",
                     Path(f"{args.out_prefix}_by_seed_{tag}.png"),
                     draw_order=SEED_DRAW_ORDER, marker_map=SEED_MARKERS)
                print(f"  by_seed/{tag} took {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
