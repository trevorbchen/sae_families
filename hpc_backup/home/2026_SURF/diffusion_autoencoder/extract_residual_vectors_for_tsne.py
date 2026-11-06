#!/usr/bin/env python3
"""Extract per-image reconstruction residuals (standardized_input - reconstruction)
for the hard-image t-SNE/UMAP analysis: one residual vector per (objective,
seed, image), for a fixed arm and lambda, across an in-distribution (FFHQ
held-out) and two out-of-distribution (CelebA, FairFace) sources.

Mirrors evaluate_lambda_sweep_reconstruction_failures.py's checkpoint/cache
loading exactly (same classes, same paths), but keeps the residual tensor
instead of collapsing it to a scalar MSE -- reconstructions themselves mostly
just look like the input image and don't separate by objective, so it's the
residual (what the SAE failed to capture) that should show structure.

The same subsample of image indices is used across every (objective, seed)
cell per source, so any structure in a later embedding reflects what the SAE
did to those images, not which images happened to be sampled.
"""

import argparse
from pathlib import Path

import torch

from train_spatial_map_sae import FlatTopKSAE

OBJECTIVES = ("latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high")
SEEDS = (2030, 2031, 2032)

CELEBA_CACHE = {
    "vector": "linear_probe_vector16x_lambda_sweep/celeba_vector_32768.pt",
    "spatial": "linear_probe_spatial16x_lambda_sweep/celeba_spatial_32768.pt",
}
FAIRFACE_CACHE = {
    "vector": "latents/fairface_vector_8192.pt",
    "spatial": "latents/fairface_spatial_8192.pt",
}


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


@torch.no_grad()
def per_example_residual(model, mean, std, latents, indices, batch_size, device):
    out = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        raw = latents[batch_idx].to(device).float()
        standardized = (raw - mean) / std
        reconstruction, _, _ = model(standardized)
        residual = (standardized - reconstruction).flatten(1)
        out.append(residual.double().cpu())
    return torch.cat(out, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1")
    parser.add_argument("--arm", default="vector_16x_k32")
    parser.add_argument("--lambda-tag", default="10p0")
    parser.add_argument("--lambda-value", type=float, default=10.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--objectives", nargs="+", default=list(OBJECTIVES))
    parser.add_argument("--sources", nargs="+", default=["ffhq_heldout", "celeba", "fairface"])
    parser.add_argument("--per-cell", type=int, default=2000,
                         help="images sampled per (objective, seed) cell, per source -- "
                              "generous on purpose; the plotting script subsamples further "
                              "per comparison, this just avoids re-running inference")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="output .pt path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(args.base)
    ae_type = "vector" if "vector" in args.arm else "spatial"

    sources = {}
    if "ffhq_heldout" in args.sources:
        cache = torch.load(base / "latents" / f"{ae_type}_ffhq_32768.pt", map_location="cpu", weights_only=False)
        sources["ffhq_heldout"] = (cache["latents"].float(), cache["valid_indices"].long())
    if "celeba" in args.sources:
        cache = torch.load(base / CELEBA_CACHE[ae_type], map_location="cpu", weights_only=False)
        latents = cache["clean_latents"].float()
        sources["celeba"] = (latents, torch.arange(len(latents)))
    if "fairface" in args.sources:
        cache = torch.load(base / FAIRFACE_CACHE[ae_type], map_location="cpu", weights_only=False)
        latents = cache["latents"].float()
        sources["fairface"] = (latents, torch.arange(len(latents)))

    generator = torch.Generator().manual_seed(args.sample_seed)
    sampled_indices = {}
    for source_name, (_, valid_indices) in sources.items():
        n = min(args.per_cell, len(valid_indices))
        perm = torch.randperm(len(valid_indices), generator=generator)[:n]
        sampled_indices[source_name] = valid_indices[perm]
        print(f"{source_name}: sampling {n}/{len(valid_indices)} images", flush=True)

    records = []
    for objective in args.objectives:
        for seed in args.seeds:
            subdir = "mse_control" if objective == "latent_mse" else f"lambda_{args.lambda_tag}"
            checkpoint_path = base / "experiments" / args.arm / "checkpoints" / subdir / objective / f"seed_{seed}" / "last.pt"
            if not checkpoint_path.is_file():
                print(f"skipping {objective}/seed_{seed}: no checkpoint at {checkpoint_path}", flush=True)
                continue
            model, mean, std = load_sae(checkpoint_path, device)
            for source_name, (latents, _) in sources.items():
                idx = sampled_indices[source_name]
                residual = per_example_residual(model, mean, std, latents, idx, args.batch_size, device)
                records.append({
                    "objective": objective, "seed": seed, "source": source_name,
                    "latent_index": idx.clone(), "residual": residual,
                })
            print(f"{objective}/seed_{seed}: done", flush=True)
            del model

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arm": args.arm, "lambda_value": args.lambda_value, "records": records}, out_path)
    total_points = sum(r["residual"].shape[0] for r in records)
    print(f"wrote {len(records)} cells, {total_points} total residual vectors, to {out_path}", flush=True)


if __name__ == "__main__":
    main()
