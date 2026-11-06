#!/usr/bin/env python3
"""Find "interesting" clusters in the joint (15-cell) feature-usage t-SNE --
ones that are mostly one objective AND hold up across multiple seeds (so they
aren't just the seed-2030-reference-alignment artifact) -- then show what a
few representative images from each such cluster actually look like when
reconstructed through *every* objective's SAE, at a fixed seed, so the visual
effect of the objective (not just its dictionary-usage footprint) is visible
directly.

Cluster selection is anchored on *local neighborhood purity*, not k-means
space-partitioning: k-means carves the embedding into roughly-equal-volume
Voronoi cells irrespective of where the actual single-objective-dense pockets
sit, so a cell's centroid can land near the edge of a color blob rather than
its visual core (seen directly: a k-means-based pass circled a point well
outside the dense red Laplacian blob). Instead, every point's own k=60
nearest neighbors are checked; a point's score is
(fraction of its neighbors sharing its own objective) x (distinct seeds among
its neighbors / 3) -- the second term penalizes the known alignment-to-
reference-cell artifact (same-seed-as-reference cells align far more tightly,
~0.45-0.49 cosine, than cross-seed, ~0.21-0.26, which can otherwise produce
misleading objective-looking pockets that are really just "close to the
reference seed"). The highest-scoring points are then greedily selected as
cluster anchors with a minimum pairwise separation (non-max suppression), so
each anchor sits, by construction, in the interior of a real dense single-
objective pocket, and the chosen anchors don't all pile into the same blob.

Run across both the "normalized" and "unnormalized" L2 variants (same two
tags plot_feature_usage_tsne_joint.py produces), always on the aligned
("difference") representation -- unaligned pooling would trivially separate
by cell identity and isn't meaningful here.

For each chosen anchor, a handful of distinct images (by latent_index) from
its own k-NN neighborhood are picked, and for each one every objective's SAE
(at one fixed seed, for apples-to-apples comparison) is run on that image's
actual latent to get its *own* reconstruction -- not just which dictionary
atoms fired -- decoded back through the shared ConvAE decoder.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from train_spatial_map_sae import FlatTopKSAE
from vector_conv_autoencoder import load_vector_autoencoder
from spatial_conv_autoencoder import load_spatial_autoencoder

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


def build_matrix(records, source, per_cell_cap, feature_count, generator, normalize):
    vectors, meta = [], []
    for rec in records:
        if rec["source"] != source:
            continue
        n_total = rec["feature_indices"].shape[0]
        idx = subsample(n_total, per_cell_cap, generator)
        dense = densify(rec["feature_values"][idx], rec["aligned_feature_indices"][idx], feature_count, normalize)
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


def load_sae(checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_shape = tuple(state["input_shape"])
    config = state["config"]
    model = FlatTopKSAE(input_shape, config["feature_count"], config["k"]).to(device)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    spatial_axes = max(0, len(input_shape) - 1)
    stats_shape = (1, -1) + (1,) * spatial_axes
    mean = state["channel_mean"].float().to(device).view(stats_shape)
    std = state["channel_std"].float().to(device).view(stats_shape)
    return model, mean, std


def find_anchor_clusters(embedding, meta, k_neighbors, n_clusters, min_sep_frac):
    """Score every point by (local objective purity) x (local seed diversity)
    over its own k nearest neighbors, then greedily pick the top-scoring
    points as cluster anchors subject to a minimum pairwise separation. Each
    anchor sits, by construction, inside the dense core of a real single-
    objective pocket (unlike a k-means cell centroid, which can land near a
    cell boundary irrespective of where the color is actually concentrated).
    """
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(embedding)
    _, indices = nn.kneighbors(embedding)
    neighbor_idxs_all = indices[:, 1:]

    obj_to_code = {o: i for i, o in enumerate(OBJECTIVES)}
    obj_codes = np.array([obj_to_code[m["objective"]] for m in meta])
    seed_list = sorted({m["seed"] for m in meta})
    seed_to_code = {s: i for i, s in enumerate(seed_list)}
    seed_codes = np.array([seed_to_code[m["seed"]] for m in meta])

    n = embedding.shape[0]
    scores = np.zeros(n)
    top_codes = np.zeros(n, dtype=int)
    purities = np.zeros(n)
    n_seeds_arr = np.zeros(n, dtype=int)
    for i in range(n):
        neigh = neighbor_idxs_all[i]
        counts = np.bincount(obj_codes[neigh], minlength=len(OBJECTIVES))
        top_code = int(counts.argmax())
        purity = counts[top_code] / k_neighbors
        n_seeds = len(np.unique(seed_codes[neigh]))
        top_codes[i] = top_code
        purities[i] = purity
        n_seeds_arr[i] = n_seeds
        scores[i] = purity * (n_seeds / len(seed_list))

    order = np.argsort(-scores)
    coord_range = embedding.max(axis=0) - embedding.min(axis=0)
    min_sep = min_sep_frac * float(np.linalg.norm(coord_range))

    anchors = []
    for i in order:
        c = embedding[i]
        if any(np.linalg.norm(c - embedding[a["anchor_idx"]]) < min_sep for a in anchors):
            continue
        anchors.append({
            "anchor_idx": int(i), "coord": c, "top_objective": OBJECTIVES[top_codes[i]],
            "purity": float(purities[i]), "n_seeds": int(n_seeds_arr[i]), "score": float(scores[i]),
            "neighbor_idxs": neighbor_idxs_all[i],
        })
        if len(anchors) >= n_clusters:
            break
    return anchors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usage", required=True)
    parser.add_argument("--base", default="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1")
    parser.add_argument("--arm", default="vector_16x_k32")
    parser.add_argument("--lambda-tag", default="10p0")
    parser.add_argument("--source", choices=("ffhq_heldout", "celeba", "fairface"), default="ffhq_heldout")
    parser.add_argument("--per-cell-cap", type=int, default=2000)
    parser.add_argument("--k-neighbors", type=int, default=60, help="neighborhood size used for local purity scoring")
    parser.add_argument("--min-sep-frac", type=float, default=0.08,
                         help="minimum pairwise anchor separation, as a fraction of the embedding's coordinate span")
    parser.add_argument("--n-top-clusters", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=4, help="distinct images shown per chosen cluster")
    parser.add_argument("--recon-seed", type=int, default=2030, help="fixed seed used for the objective-comparison row")
    parser.add_argument("--perplexity", type=int, default=15)
    parser.add_argument("--early-exaggeration", type=float, default=24.0)
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--pca-dims", type=int, default=100)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(args.base)
    ae_type = "vector" if "vector" in args.arm else "spatial"

    data = torch.load(args.usage, map_location="cpu", weights_only=False)
    records, feature_count = data["records"], data["feature_count"]

    # load raw latents + base autoencoder + all 5 SAEs (at the fixed recon-seed) once, reused across tags
    cache = torch.load(base / CACHE_BY_SOURCE[args.source][ae_type], map_location="cpu", weights_only=False)
    latent_key = "latents" if "latents" in cache else "clean_latents"
    latents = cache[latent_key].float()
    if ae_type == "vector":
        autoencoder = load_vector_autoencoder(str(base / AE_PATH[ae_type]), device)
    else:
        autoencoder = load_spatial_autoencoder(str(base / AE_PATH[ae_type]), device)
    autoencoder.eval().requires_grad_(False)

    sae_by_objective = {}
    for objective in OBJECTIVES:
        subdir = "mse_control" if objective == "latent_mse" else f"lambda_{args.lambda_tag}"
        ckpt = base / "experiments" / args.arm / "checkpoints" / subdir / objective / f"seed_{args.recon_seed}" / "last.pt"
        if not ckpt.is_file():
            print(f"missing checkpoint for {objective}/seed_{args.recon_seed}, skipping", flush=True)
            continue
        sae_by_objective[objective] = load_sae(ckpt, device)
    print(f"loaded {len(sae_by_objective)} SAEs at seed {args.recon_seed} for reconstruction comparison", flush=True)

    @torch.no_grad()
    def decode_raw(latent_index):
        raw = latents[latent_index:latent_index + 1].to(device).float()
        return autoencoder.decode(raw).add(1).div(2).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy()

    @torch.no_grad()
    def decode_sae_recon(latent_index, objective):
        model, mean, std = sae_by_objective[objective]
        raw = latents[latent_index:latent_index + 1].to(device).float()
        standardized = (raw - mean) / std
        reconstruction, _, _ = model(standardized)
        recon_raw = reconstruction * std + mean
        return autoencoder.decode(recon_raw).add(1).div(2).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy()

    columns = ["original"] + list(OBJECTIVES)
    col_labels = ["Original"] + [OBJECTIVE_LABELS[o] for o in OBJECTIVES]

    for normalize, norm_tag in ((False, "unnormalized"), (True, "normalized")):
        generator = torch.Generator().manual_seed(0)
        matrix, meta = build_matrix(records, args.source, args.per_cell_cap, feature_count, generator, normalize)
        print(f"{norm_tag}: embedding {matrix.shape[0]} points, {matrix.shape[1]} dims (joint, aligned)", flush=True)
        embedding = run_embedding(matrix, args.perplexity, args.early_exaggeration, args.n_iter, args.pca_dims)

        chosen = find_anchor_clusters(embedding, meta, args.k_neighbors, args.n_top_clusters, args.min_sep_frac)

        print(f"{norm_tag} top anchor clusters (anchor_idx, top_objective, purity, n_seeds, score):", flush=True)
        for d in chosen:
            print(f"  {d['anchor_idx']:6d}  {d['top_objective']:12s}  "
                  f"purity={d['purity']:.2f}  seeds={d['n_seeds']}/3  score={d['score']:.3f}", flush=True)

        # overview scatter, anchors circled at their own (in-blob) coordinate
        fig, ax = plt.subplots(figsize=(7, 6.5))
        for obj in OBJECTIVE_DRAW_ORDER:
            mask = np.array([m["objective"] == obj for m in meta])
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=4, alpha=0.35,
                       color=OBJECTIVE_COLORS[obj], marker=OBJECTIVE_MARKERS[obj], linewidths=0,
                       label=OBJECTIVE_LABELS[obj], zorder=1)
        for rank, d in enumerate(chosen):
            c = d["coord"]
            ax.scatter([c[0]], [c[1]], s=260, facecolors="none", edgecolors="black", linewidths=2, zorder=3)
            ax.annotate(f"#{rank+1}", c, fontsize=12, fontweight="bold", ha="center", va="center", zorder=4)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="best", markerscale=3)
        ax.set_title(f"{args.source}: joint feature-usage t-SNE ({norm_tag}), "
                     f"top {len(chosen)} single-objective clusters circled", fontsize=10)
        overview_path = Path(f"{args.out_prefix}_{norm_tag}_overview.png")
        overview_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(overview_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {overview_path}", flush=True)

        for rank, d in enumerate(chosen):
            neigh = d["neighbor_idxs"]
            obj_counts = {o: int(np.sum([meta[j]["objective"] == o for j in neigh])) for o in OBJECTIVES}
            obj_counts = {o: c for o, c in obj_counts.items() if c > 0}
            seen_images, picked = set(), []
            for j in neigh:
                li = meta[int(j)]["latent_index"]
                if li in seen_images:
                    continue
                seen_images.add(li)
                picked.append(li)
                if len(picked) >= args.n_samples:
                    break

            comp_str = ", ".join(f"{OBJECTIVE_LABELS[o]} {c / len(neigh):.0%}"
                                  for o, c in sorted(obj_counts.items(), key=lambda kv: -kv[1]))
            fig = plt.figure(figsize=(2.0 * len(columns), 2.1 * len(picked) + 0.8))
            gs = GridSpec(len(picked), len(columns), figure=fig)
            for row, li in enumerate(picked):
                for col, tag in enumerate(columns):
                    axt = fig.add_subplot(gs[row, col])
                    if tag == "original":
                        img = decode_raw(li)
                        edgecolor = "black"
                    elif tag in sae_by_objective:
                        img = decode_sae_recon(li, tag)
                        edgecolor = OBJECTIVE_COLORS[tag]
                    else:
                        axt.axis("off")
                        continue
                    axt.imshow(img)
                    axt.set_xticks([]); axt.set_yticks([])
                    lw = 3.5 if tag == d["top_objective"] else 1.2
                    for spine in axt.spines.values():
                        spine.set_edgecolor(edgecolor)
                        spine.set_linewidth(lw)
                    if row == 0:
                        axt.set_title(col_labels[col], fontsize=9)
                    if col == 0:
                        axt.set_ylabel(f"img {li}", fontsize=8, rotation=0, labelpad=28, va="center")
            fig.suptitle(f"{args.source}, cluster #{rank+1} ({norm_tag}, anchor {d['anchor_idx']}, "
                         f"dominant={OBJECTIVE_LABELS[d['top_objective']]}, purity={d['purity']:.0%}, "
                         f"{d['n_seeds']}/3 seeds)\ncomposition: {comp_str}  |  reconstructions at seed {args.recon_seed}",
                         fontsize=9)
            out_path = Path(f"{args.out_prefix}_{norm_tag}_cluster{rank+1}_anchor{d['anchor_idx']}.png")
            fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
