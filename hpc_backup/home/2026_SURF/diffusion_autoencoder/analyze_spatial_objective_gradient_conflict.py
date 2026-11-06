#!/usr/bin/env python3
"""Measure transformed-objective gradient conflict with latent MSE."""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from spatial_conv_autoencoder import load_spatial_autoencoder
from train_grouped_objective_sae import GroupSpec, PartitionedTopKSAE, REGULAR_SPECS
from train_grouped_spatial_ae_sae import spatial_geometry_losses


OBJECTIVES = ("laplacian", "fourier_low", "fourier_mid", "fourier_high")


def load_trained_sae(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    specs = tuple(GroupSpec(**record) for record in checkpoint["groups"])
    input_dim = checkpoint["model"]["decoder"].shape[1]
    sae = PartitionedTopKSAE(input_dim, specs)
    sae.load_state_dict(checkpoint["model"])
    sae.eval().to(device)
    return sae, checkpoint["input_mean"], checkpoint["input_std"]


def summarize(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "p05": float(torch.quantile(tensor, 0.05)),
        "p25": float(torch.quantile(tensor, 0.25)),
        "p75": float(torch.quantile(tensor, 0.75)),
        "p95": float(torch.quantile(tensor, 0.95)),
    }


def evaluate_context(
    name,
    sae,
    autoencoder,
    raw_latents,
    indices,
    mean,
    std,
    batch_size,
    device,
):
    rows = []
    device_mean = mean.to(device)
    device_std = std.to(device)
    input_shape = tuple(raw_latents.shape[1:])
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        raw = raw_latents[selected].to(device).float()
        standardized = ((raw - device_mean) / device_std).flatten(1)
        with torch.no_grad():
            initial = sae(standardized)[0]
        reconstruction = initial.detach().requires_grad_(True)
        latent_loss = (standardized - reconstruction).square().mean(1)
        reconstructed_raw = (
            reconstruction.view(-1, *input_shape) * device_std + device_mean
        )
        transformed, _ = spatial_geometry_losses(
            autoencoder, raw, reconstructed_raw, powers=None
        )
        mse_gradient = torch.autograd.grad(
            latent_loss.sum(), reconstruction, retain_graph=True
        )[0].flatten(1)
        mse_norm = mse_gradient.norm(dim=1).clamp_min(1e-12)
        for objective in OBJECTIVES:
            gradient = torch.autograd.grad(
                transformed[objective].sum(), reconstruction, retain_graph=True
            )[0].flatten(1)
            cosine = F.cosine_similarity(mse_gradient, gradient, dim=1)
            norm_ratio = gradient.norm(dim=1) / mse_norm
            for offset, example_index in enumerate(selected.tolist()):
                rows.append({
                    "context": name,
                    "example_index": example_index,
                    "objective": objective,
                    "cosine": float(cosine[offset]),
                    "norm_ratio_to_mse": float(norm_ratio[offset]),
                    "mse_gradient_norm": float(mse_norm[offset]),
                    "transformed_gradient_norm": float(gradient[offset].norm()),
                })
    return rows


def build_summary(rows):
    summary = {}
    contexts = sorted({row["context"] for row in rows})
    for context in contexts:
        summary[context] = {}
        for objective in OBJECTIVES:
            selected = [
                row for row in rows
                if row["context"] == context and row["objective"] == objective
            ]
            cosines = [row["cosine"] for row in selected]
            ratios = [row["norm_ratio_to_mse"] for row in selected]
            summary[context][objective] = {
                "examples": len(selected),
                "cosine": summarize(cosines),
                "fraction_negative": sum(value < 0 for value in cosines) / len(cosines),
                "fraction_below_minus_0_1": (
                    sum(value < -0.1 for value in cosines) / len(cosines)
                ),
                "norm_ratio_to_mse": summarize(ratios),
            }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--regular-sae", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.data, map_location="cpu", weights_only=False)
    raw_latents = cache["latents"]
    indices = cache["valid_indices"].long()[:args.examples]
    autoencoder = load_spatial_autoencoder(args.autoencoder, device)

    trained, mean, std = load_trained_sae(args.regular_sae, device)
    input_dim = trained.input_dim
    del trained
    torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    initialized = PartitionedTopKSAE(input_dim, REGULAR_SPECS).to(device).eval()
    rows = evaluate_context(
        "initialized",
        initialized,
        autoencoder,
        raw_latents,
        indices,
        mean,
        std,
        args.batch_size,
        device,
    )
    del initialized
    torch.cuda.empty_cache()

    trained, trained_mean, trained_std = load_trained_sae(args.regular_sae, device)
    rows.extend(evaluate_context(
        "trained_mse",
        trained,
        autoencoder,
        raw_latents,
        indices,
        trained_mean,
        trained_std,
        args.batch_size,
        device,
    ))

    summary = build_summary(rows)
    with (args.output / "gradient_conflict_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (args.output / "gradient_conflict_per_example.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
