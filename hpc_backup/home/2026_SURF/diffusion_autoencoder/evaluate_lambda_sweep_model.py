#!/usr/bin/env python3
"""Evaluate one lambda-sweep SAE's reconstruction and feature-output geometry."""

import argparse
import json
from pathlib import Path

import torch

from evaluate_feature_output_geometry import (
    METRICS,
    aggregate_feature_rows,
    collect_examples,
    image_geometry,
    load_sae,
    rank_features,
    write_csv,
)
from spatial_conv_autoencoder import load_spatial_autoencoder
from vector_conv_autoencoder import load_vector_autoencoder


LABELS = {
    "latent_mse": "MSE",
    "weighted_laplacian": "Laplacian",
    "weighted_fourier_low": "Fourier low",
    "weighted_fourier_mid": "Fourier mid",
    "weighted_fourier_high": "Fourier high",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--objective", choices=tuple(LABELS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lambda-value", type=float, required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument(
        "--autoencoder-type", choices=("spatial", "vector"), default="spatial"
    )
    parser.add_argument("--latent-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-features", type=int, default=100)
    parser.add_argument("--examples-per-feature", type=int, default=4)
    parser.add_argument("--min-examples", type=int, default=4)
    parser.add_argument("--rank-examples", type=int, default=13107)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--multipliers", type=float, nargs="+", default=(-2.0, 2.0))
    return parser.parse_args()


@torch.no_grad()
def reconstruction_metrics(model, mean, std, latents, indices, batch_size, device):
    target_sum = target_square_sum = residual_sum = residual_square_sum = 0.0
    count = 0
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        target = (latents[selected].to(device).float() - mean) / std
        reconstruction, _, _ = model(target)
        residual = target - reconstruction
        target_sum += float(target.double().sum())
        target_square_sum += float(target.double().square().sum())
        residual_sum += float(residual.double().sum())
        residual_square_sum += float(residual.double().square().sum())
        count += target.numel()
    target_variance = target_square_sum / count - (target_sum / count) ** 2
    residual_variance = residual_square_sum / count - (residual_sum / count) ** 2
    return {
        "validation_elements": count,
        "latent_mse": residual_square_sum / count,
        "target_variance": target_variance,
        "residual_variance": residual_variance,
        "latent_explained_variance": 1.0 - residual_variance / target_variance,
    }


@torch.no_grad()
def feature_geometry(args, model, mean, std, autoencoder, latents, train_indices,
                     valid_indices, device):
    rank_indices = train_indices[:min(args.rank_examples, len(train_indices))]
    stats = rank_features(model, mean, std, latents, rank_indices, args.batch_size, device)
    alive = [feature for feature in stats["order"] if stats["counts"][feature] > 0]
    examples = collect_examples(
        model, mean, std, latents, valid_indices, alive, args.batch_size, device
    )
    chosen = [feature for feature in alive if len(examples[feature]) >= args.min_examples]
    chosen = chosen[:args.top_features]
    rows = []
    for rank, feature in enumerate(chosen):
        selected = examples[feature][:args.examples_per_feature]
        coefficients = torch.tensor([value for value, _ in selected], device=device)
        example_indices = torch.tensor([index for _, index in selected], dtype=torch.long)
        raw = latents[example_indices].to(device).float()
        standardized = (raw - mean) / std
        clean = autoencoder.decode(raw).add(1).div(2).clamp(0, 1)
        scale = stats["active_rms"][feature].to(device).clamp_min(1e-6)
        for multiplier in args.multipliers:
            target = torch.clamp(coefficients + multiplier * scale, min=0)
            coefficient_delta = target - coefficients
            modified = standardized.flatten(1) + coefficient_delta[:, None] * model.decoder[feature]
            modified_raw = modified.view_as(raw) * std + mean
            edited = autoencoder.decode(modified_raw).add(1).div(2).clamp(0, 1)
            for measurement, values in {
                "edited_output": image_geometry(edited),
                "causal_delta": image_geometry(edited - clean),
            }.items():
                for row_index, example_index in enumerate(example_indices.tolist()):
                    row = {
                        "objective": args.objective,
                        "objective_label": LABELS[args.objective],
                        "seed": args.seed,
                        "lambda": args.lambda_value,
                        "feature_rank": rank,
                        "feature": feature,
                        "example_index": example_index,
                        "multiplier": multiplier,
                        "coefficient": float(coefficients[row_index]),
                        "active_rms": float(scale),
                        "coefficient_delta": float(coefficient_delta[row_index]),
                        "measurement": measurement,
                    }
                    row.update({name: float(tensor[row_index]) for name, tensor in values.items()})
                    rows.append(row)
        if (rank + 1) % 25 == 0:
            print(f"geometry {rank + 1}/{len(chosen)}", flush=True)
    return rows, chosen


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, state = load_sae(Path(args.checkpoint), device)
    if args.autoencoder_type == "vector":
        autoencoder = load_vector_autoencoder(args.autoencoder, device)
    else:
        autoencoder = load_spatial_autoencoder(args.autoencoder, device)
    autoencoder.eval().requires_grad_(False)
    cache = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
    latents = cache["latents"].float()
    train_indices = cache["train_indices"].long()
    valid_indices = cache["valid_indices"].long()

    metrics = reconstruction_metrics(
        model, mean, std, latents, valid_indices, args.batch_size, device
    )
    rows, chosen = feature_geometry(
        args, model, mean, std, autoencoder, latents, train_indices, valid_indices, device
    )
    feature_rows = aggregate_feature_rows(rows)
    for row in feature_rows:
        row["lambda"] = args.lambda_value
    write_csv(out_dir / "raw_measurements.csv", rows)
    write_csv(out_dir / "feature_means.csv", feature_rows)
    metrics.update({
        "objective": args.objective,
        "objective_label": LABELS[args.objective],
        "seed": args.seed,
        "lambda": args.lambda_value,
        "checkpoint": args.checkpoint,
        "autoencoder_type": args.autoencoder_type,
        "feature_count": state["config"]["feature_count"],
        "k": state["config"]["k"],
        "evaluated_features": len(chosen),
        "geometry_metrics": list(METRICS),
    })
    (out_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
