#!/usr/bin/env python3
"""Per-example held-out SAE reconstruction error for the matched512 lambda
sweep -- NOT on planted perturbations, on ordinary held-out latents. The
question this answers: for real images, which ones does each objective's
SAE struggle to represent sparsely, and do the hard-example sets differ by
objective (an inspectable blind spot) or are they the same universally-hard
images regardless of vocabulary?

Writes one row per (arm, objective, seed, held-out latent index) with the
SAE's own top-k reconstruction MSE on that example -- cheap, no autoencoder
decode needed here, that only happens later for whichever examples turn out
interesting.
"""

import argparse
import csv
from pathlib import Path

import torch

from train_spatial_map_sae import FlatTopKSAE

OBJECTIVES = ("latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high")
ARM_AE_TYPE = {
    "vector_4x_k8": "vector", "vector_16x_k32": "vector",
    "spatial_4x_k8": "spatial", "spatial_16x_k32": "spatial",
}
# 4x and 16x arms of the same latent type share one base autoencoder (only
# the SAE's k/feature_count differ), confirmed earlier by bit-identical
# channel_mean/channel_std across capacity variants -- so one CelebA cache
# per latent type, already built for the linear-probe pipeline against the
# same autoencoder checkpoints, covers all 4 arms as an out-of-distribution
# probe (CelebA vs. the FFHQ training/held-out distribution).
CELEBA_CACHE = {
    "vector": "linear_probe_vector16x_lambda_sweep/celeba_vector_32768.pt",
    "spatial": "linear_probe_spatial16x_lambda_sweep/celeba_spatial_32768.pt",
}
# FairFace (YFCC-100M Flickr, no celebrity/press bias) -- a second OOD probe
# set with a genuinely different content genre than CelebA, to check whether
# a failure-mode finding is about the objective or about CelebA's specific
# red-carpet/press-photography skew.
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
def per_example_mse(model, mean, std, latents, indices, batch_size, device):
    out = torch.empty(len(indices), dtype=torch.float64)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        raw = latents[batch_idx].to(device).float()
        standardized = (raw - mean) / std
        reconstruction, _, _ = model(standardized)
        mse = (standardized.flatten(1) - reconstruction.flatten(1)).square().mean(1)
        out[start:start + len(batch_idx)] = mse.double().cpu()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1")
    parser.add_argument("--arms", nargs="+", default=list(ARM_AE_TYPE))
    parser.add_argument("--lambda-tag", default="100p0")
    parser.add_argument("--lambda-value", type=float, default=100.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=(2030, 2031, 2032))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--source", choices=("ffhq_heldout", "celeba", "fairface"), default="ffhq_heldout")
    parser.add_argument("--celeba-sample", type=int, default=8192,
                         help="random subsample size from the 32768-image CelebA cache")
    parser.add_argument("--celeba-seed", type=int, default=0)
    parser.add_argument("--celeba-offset", type=int, default=0,
                         help="slice permutation[offset:offset+sample] instead of [:sample] -- "
                              "e.g. offset=8192 with the same seed as an earlier run gives a "
                              "disjoint resample from the same underlying 32768-image cache")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(args.base)

    rows = []
    latent_cache_cache = {}
    for arm in args.arms:
        ae_type = ARM_AE_TYPE[arm]
        if ae_type not in latent_cache_cache:
            if args.source == "ffhq_heldout":
                cache = torch.load(base / "latents" / f"{ae_type}_ffhq_32768.pt", map_location="cpu", weights_only=False)
                latent_cache_cache[ae_type] = (cache["latents"].float(), cache["valid_indices"].long())
            elif args.source == "celeba":
                cache = torch.load(base / CELEBA_CACHE[ae_type], map_location="cpu", weights_only=False)
                latents = cache["clean_latents"].float()
                generator = torch.Generator().manual_seed(args.celeba_seed)
                permutation = torch.randperm(len(latents), generator=generator)
                start, end = args.celeba_offset, min(args.celeba_offset + args.celeba_sample, len(latents))
                indices = permutation[start:end]
                latent_cache_cache[ae_type] = (latents, indices)
            else:  # fairface
                cache = torch.load(base / FAIRFACE_CACHE[ae_type], map_location="cpu", weights_only=False)
                latents = cache["latents"].float()
                latent_cache_cache[ae_type] = (latents, torch.arange(len(latents)))
        latents, valid_indices = latent_cache_cache[ae_type]

        for objective in OBJECTIVES:
            for seed in args.seeds:
                # latent_mse (the MSE control/anchor) has no lambda-weighted
                # transform term, so it's trained once and stored under
                # mse_control/ rather than duplicated per lambda folder --
                # confirmed byte-identical to the few per-lambda copies that
                # do exist (vector arms) via md5sum.
                subdir = "mse_control" if objective == "latent_mse" else f"lambda_{args.lambda_tag}"
                checkpoint_path = base / "experiments" / arm / "checkpoints" / subdir / objective / f"seed_{seed}" / "last.pt"
                if not checkpoint_path.is_file():
                    print(f"skipping {arm}/{objective}/seed_{seed}: no checkpoint", flush=True)
                    continue
                model, mean, std = load_sae(checkpoint_path, device)
                mse = per_example_mse(model, mean, std, latents, valid_indices, args.batch_size, device)
                for local_pos, latent_index in enumerate(valid_indices.tolist()):
                    rows.append({
                        "arm": arm, "lambda": args.lambda_value, "objective": objective, "seed": seed,
                        "source": args.source, "latent_index": latent_index, "topk_mse": float(mse[local_pos]),
                    })
                print(f"{arm}/{objective}/seed_{seed}: mean_topk_mse={float(mse.mean()):.6f}", flush=True)
                del model

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arm", "lambda", "objective", "seed", "source", "latent_index", "topk_mse"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
