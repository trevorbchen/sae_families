#!/usr/bin/env python3
"""Evaluate an MSE anchor combined with independently trained specialists."""

import argparse
import json
from pathlib import Path

import torch

from spatial_conv_autoencoder import load_spatial_autoencoder
from train_spatial_residual_specialist import load_base
from train_grouped_objective_sae import GroupSpec, PartitionedTopKSAE
from train_grouped_spatial_ae_sae import spatial_geometry_losses


OBJECTIVES = ("laplacian", "fourier_low", "fourier_mid", "fourier_high")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--base-sae", required=True)
    parser.add_argument("--specialist-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def load_specialist(path, input_dim, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    specs = tuple(GroupSpec(**record) for record in checkpoint["groups"])
    model = PartitionedTopKSAE(input_dim, specs)
    model.load_state_dict(checkpoint["model"])
    return model.eval().requires_grad_(False).to(device)


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.data, map_location="cpu", weights_only=False)
    raw_latents = cache["latents"]
    valid_indices = cache["valid_indices"].long()
    base, base_checkpoint = load_base(args.base_sae, device)
    mean = base_checkpoint["input_mean"].to(device)
    std = base_checkpoint["input_std"].to(device)
    input_shape = tuple(base_checkpoint["input_shape"])
    specialists = {
        name: load_specialist(
            Path(args.specialist_root) / name / "best_feasible.pt",
            base.input_dim,
            device,
        )
        for name in OBJECTIVES
    }
    autoencoder = load_spatial_autoencoder(args.autoencoder, device)
    autoencoder.eval().requires_grad_(False)

    mode_names = (
        "base",
        *(f"base_plus_{name}" for name in OBJECTIVES),
        "base_plus_raw_sum",
        "base_plus_mean",
    )
    totals = {
        mode: {
            "latent_squared_error": 0.0,
            "pixel_mse": 0.0,
            **{name: 0.0 for name in OBJECTIVES},
        }
        for mode in mode_names
    }
    target_sum = target_squared_sum = 0.0
    target_elements = count = 0

    for start in range(0, len(valid_indices), args.batch_size):
        selected = valid_indices[start:start + args.batch_size]
        raw = raw_latents[selected].to(device).float()
        standardized = ((raw - mean) / std).flatten(1)
        base_reconstruction = base(standardized)[0]
        residual = standardized - base_reconstruction
        deltas = {
            name: model(residual)[0]
            for name, model in specialists.items()
        }
        summed = torch.stack(tuple(deltas.values())).sum(0)
        reconstructions = {
            "base": base_reconstruction,
            **{
                f"base_plus_{name}": base_reconstruction + delta
                for name, delta in deltas.items()
            },
            "base_plus_raw_sum": base_reconstruction + summed,
            "base_plus_mean": base_reconstruction + summed / len(deltas),
        }
        target_sum += float(standardized.sum())
        target_squared_sum += float(standardized.square().sum())
        target_elements += standardized.numel()
        count += len(selected)
        for mode, reconstruction in reconstructions.items():
            totals[mode]["latent_squared_error"] += float(
                (standardized - reconstruction).square().sum()
            )
            reconstructed_raw = reconstruction.view(-1, *input_shape) * std + mean
            transformed, pixel_mse = spatial_geometry_losses(
                autoencoder, raw, reconstructed_raw, powers=None
            )
            totals[mode]["pixel_mse"] += float(pixel_mse.sum())
            for name in OBJECTIVES:
                totals[mode][name] += float(transformed[name].sum())

    target_mean = target_sum / target_elements
    target_variance = target_squared_sum / target_elements - target_mean ** 2
    results = {}
    for mode, values in totals.items():
        latent_mse = values.pop("latent_squared_error") / target_elements
        results[mode] = {
            "latent_mse": latent_mse,
            "latent_explained_variance": 1.0 - latent_mse / target_variance,
            **{name: value / count for name, value in values.items()},
        }
    payload = {
        "validation_examples": count,
        "target_variance": target_variance,
        "capacity": {
            "dictionary_size": 8192 + 4 * 2048,
            "active_features": 32 + 4 * 8,
        },
        "aggregation_note": (
            "Raw sum is the literal five-branch decoder. Mean compensates for all "
            "four independently trained specialists targeting the same residual."
        ),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
