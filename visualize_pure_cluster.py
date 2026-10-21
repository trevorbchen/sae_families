#!/usr/bin/env python3
"""Decode real images from a specific, visually-chosen area of interest in a
saved feature/reconstruction t-SNE embedding -- not a k-means/PCA-selected
region. The anchor point is found by local density of one objective's own
points (how many of its k nearest neighbors share its objective), which
directly operationalizes "where is the reddest/densest patch" without
space-partitioning the whole embedding the way k-means does. A --center-x/
--center-y override lets you instead point at an exact spot by eye (e.g.
read off the saved embedding's coordinate range and match it to where you
see the densest color in the rendered PNG).

For each of a handful of real images nearest that anchor, shows the original
decoded image alongside its actual SAE reconstruction under every objective
(fixed seed, for apples-to-apples comparison) -- same idea as the earlier
cluster-explore galleries, just anchored properly this time.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.neighbors import NearestNeighbors

from vector_conv_autoencoder import load_vector_autoencoder
from spatial_conv_autoencoder import load_spatial_autoencoder
from train_spatial_map_sae import FlatTopKSAE

OBJECTIVES = ("latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high")
OBJECTIVE_COLORS = {
    "latent_mse": "#984ea3", "laplacian": "#e41a1c", "fourier_low": "#4daf4a",
    "fourier_mid": "#ff7f00", "fourier_high": "#377eb8",
}
OBJECTIVE_LABELS = {
    "latent_mse": "MSE", "laplacian": "Laplacian", "fourier_low": "Fourier low",
    "fourier_mid": "Fourier mid", "fourier_high": "Fourier high",
}

CACHE_BY_SOURCE = {
    "ffhq_heldout": "latents/vector_ffhq_32768.pt",
    "celeba": "linear_probe_vector16x_lambda_sweep/celeba_vector_32768.pt",
    "fairface": "latents/fairface_vector_8192.pt",
}
AE_PATH = "autoencoders/vector/last.pt"


def load_sae(checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_shape = tuple(state["input_shape"])
    config = state["config"]
    model = FlatTopKSAE(input_shape, config["feature_count"], config["k"])
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    mean = state["channel_mean"].float().view(1, -1)
    std = state["channel_std"].float().view(1, -1)
    return model, mean, std


def find_density_anchor(embedding, objective_arr, highlight, k_neighbors):
    mask = objective_arr == highlight
    idxs = np.where(mask)[0]
    pts = embedding[idxs]
    nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(pts))).fit(pts)
    dist, _ = nn.kneighbors(pts)
    density = -dist.mean(axis=1)  # smaller mean distance to own-objective neighbors = denser
    best = idxs[np.argmax(density)]
    return embedding[best]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", required=True, help="{dataset}_{method}_embedding.npz")
    parser.add_argument("--base", required=True)
    parser.add_argument("--arm", default="vector_16x_k32")
    parser.add_argument("--lambda-tag", default="10p0")
    parser.add_argument("--source", choices=("ffhq_heldout", "celeba", "fairface"), default="ffhq_heldout")
    parser.add_argument("--highlight", default="laplacian", choices=OBJECTIVES,
                         help="which objective's density peak to anchor on (ignored if --center-x/y given)")
    parser.add_argument("--density-k", type=int, default=40)
    parser.add_argument("--center-x", type=float, default=None)
    parser.add_argument("--center-y", type=float, default=None)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--neighborhood-k", type=int, default=60,
                         help="how many nearby points to use for composition stats")
    parser.add_argument("--recon-seed", type=int, default=2030)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = Path(args.base)
    device = torch.device("cpu")

    npz = np.load(args.embedding)
    embedding = npz["embedding"]
    objective_arr = npz["objective"]
    seed_arr = npz["seed"]
    latent_idx_arr = npz["latent_index"]

    if args.center_x is not None and args.center_y is not None:
        center = np.array([args.center_x, args.center_y])
        print(f"using manual center ({args.center_x:.2f}, {args.center_y:.2f})", flush=True)
    else:
        center = find_density_anchor(embedding, objective_arr, args.highlight, args.density_k)
        print(f"density-peak anchor for {args.highlight}: ({center[0]:.2f}, {center[1]:.2f})", flush=True)

    dist = np.linalg.norm(embedding - center, axis=1)
    order = np.argsort(dist)
    neigh = order[:args.neighborhood_k]

    obj_counts = {o: int((objective_arr[neigh] == o).sum()) for o in OBJECTIVES}
    obj_counts = {o: c for o, c in obj_counts.items() if c > 0}
    comp_str = ", ".join(f"{OBJECTIVE_LABELS[o]} {c / len(neigh):.0%}"
                          for o, c in sorted(obj_counts.items(), key=lambda kv: -kv[1]))
    print(f"neighborhood composition (k={len(neigh)}): {comp_str}", flush=True)

    seen, picked = set(), []
    for i in order:
        li = int(latent_idx_arr[i])
        if li in seen:
            continue
        seen.add(li)
        picked.append(li)
        if len(picked) >= args.n_samples:
            break

    cache = torch.load(base / CACHE_BY_SOURCE[args.source], map_location="cpu", weights_only=False)
    latent_key = "latents" if "latents" in cache else "clean_latents"
    latents = cache[latent_key].float()
    autoencoder = load_vector_autoencoder(str(base / AE_PATH), device)
    autoencoder.eval().requires_grad_(False)

    sae_by_objective = {}
    for objective in OBJECTIVES:
        subdir = "mse_control" if objective == "latent_mse" else f"lambda_{args.lambda_tag}"
        ckpt = base / "experiments" / args.arm / "checkpoints" / subdir / objective / f"seed_{args.recon_seed}" / "last.pt"
        if ckpt.is_file():
            sae_by_objective[objective] = load_sae(ckpt)

    @torch.no_grad()
    def decode_raw(latent_index):
        raw = latents[latent_index:latent_index + 1].float()
        return autoencoder.decode(raw).add(1).div(2).clamp(0, 1)[0].permute(1, 2, 0).numpy()

    @torch.no_grad()
    def decode_sae_recon(latent_index, objective):
        model, mean, std = sae_by_objective[objective]
        raw = latents[latent_index:latent_index + 1].float()
        standardized = (raw - mean) / std
        reconstruction, _, _ = model(standardized)
        recon_raw = reconstruction * std + mean
        return autoencoder.decode(recon_raw).add(1).div(2).clamp(0, 1)[0].permute(1, 2, 0).numpy()

    columns = ["original"] + list(OBJECTIVES)
    col_labels = ["Original"] + [OBJECTIVE_LABELS[o] for o in OBJECTIVES]

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
            for spine in axt.spines.values():
                spine.set_edgecolor(edgecolor)
                spine.set_linewidth(1.2)
            if row == 0:
                axt.set_title(col_labels[col], fontsize=9)
            if col == 0:
                axt.set_ylabel(f"img {li}", fontsize=8, rotation=0, labelpad=28, va="center")
    fig.suptitle(f"{args.source}, pure-cluster gallery, anchor=({center[0]:.1f},{center[1]:.1f})\n"
                 f"composition: {comp_str}  |  reconstructions at seed {args.recon_seed}", fontsize=9)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
