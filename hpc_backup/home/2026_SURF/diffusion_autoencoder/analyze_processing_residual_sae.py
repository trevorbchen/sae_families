#!/usr/bin/env python3
"""Rank residual-SAE features by exact leave-one-feature-out reconstruction effect."""

import argparse
import csv
import json
import os

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from spatial_conv_autoencoder import load_spatial_autoencoder
from vector_conv_autoencoder import load_vector_autoencoder
from train_spatial_map_sae import FlatTopKSAE


def load_autoencoder(path, autoencoder_type, device):
    if autoencoder_type == "vector":
        return load_vector_autoencoder(path, device)
    return load_spatial_autoencoder(path, device)


def stats_broadcast_shape(input_shape, with_batch_dim):
    # Vector latents are [d] (or [batch, d]); spatial latents are
    # [channels, ...] (or [batch, channels, ...]). Only the leading axis
    # after any batch dim is the channel axis; broadcast over the rest.
    spatial_axes = max(0, len(input_shape) - 1)
    trailing = (1,) * spatial_axes
    return ((1, -1) + trailing) if with_batch_dim else ((-1,) + trailing)


def load_sae(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["architecture"] != "flat":
        raise ValueError("residual analysis expects a flat SAE")
    sae = FlatTopKSAE(
        checkpoint["input_shape"], config["feature_count"], config["k"]
    )
    sae.load_state_dict(checkpoint["model"])
    sae.eval().to(device)
    return checkpoint, sae


@torch.no_grad()
def encode_batch(sae, standardized):
    reconstruction, values, indices = sae(standardized)
    return reconstruction, values, indices


@torch.no_grad()
def rank_by_exact_ablation(
    sae, residuals, positions, mean, std, batch_size, device,
):
    feature_count = sae.h
    importance_sum = torch.zeros(feature_count, dtype=torch.float64, device=device)
    activation_sum = torch.zeros(feature_count, dtype=torch.float64, device=device)
    fire_count = torch.zeros(feature_count, dtype=torch.long, device=device)
    dimension = sae.d
    for start in range(0, len(positions), batch_size):
        selected_positions = positions[start:start + batch_size]
        raw = residuals[selected_positions].to(device).float()
        standardized = (raw - mean) / std
        reconstruction, values, indices = encode_batch(sae, standardized)
        error = (standardized - reconstruction).flatten(1)
        atoms = sae.decoder[indices]
        dot = (atoms * error[:, None]).sum(-1)
        norm_squared = atoms.square().sum(-1)
        effects = (2 * values * dot + values.square() * norm_squared) / dimension
        active = values > 0
        importance_sum.scatter_add_(0, indices[active], effects[active].double())
        activation_sum.scatter_add_(0, indices[active], values[active].double())
        fire_count.scatter_add_(
            0, indices[active], torch.ones_like(indices[active], dtype=torch.long)
        )
    count = len(positions)
    return {
        "mean_importance": (importance_sum / count).cpu(),
        "mean_activation": (activation_sum / count).cpu(),
        "fire_count": fire_count.cpu(),
        "activation_sum": activation_sum.cpu(),
    }


@torch.no_grad()
def evaluate_top_sets(
    sae, residuals, positions, feature_order, mean, std, batch_size, device,
):
    counts = (1, 5, 10)
    top_masks = {}
    for count in counts:
        mask = torch.zeros(sae.h, dtype=torch.bool, device=device)
        mask[feature_order[:count].to(device)] = True
        top_masks[count] = mask
    sums = {"bias": 0.0, "full": 0.0, **{f"top_{n}": 0.0 for n in counts}}
    all_values = []
    all_indices = []
    for start in range(0, len(positions), batch_size):
        selected_positions = positions[start:start + batch_size]
        raw = residuals[selected_positions].to(device).float()
        standardized = (raw - mean) / std
        reconstruction, values, indices = encode_batch(sae, standardized)
        target = standardized.flatten(1)
        bias = sae.dec_bias[None]
        sums["bias"] += float((target - bias).square().mean(1).sum())
        sums["full"] += float((target - reconstruction.flatten(1)).square().mean(1).sum())
        atoms = sae.decoder[indices]
        for count in counts:
            keep = top_masks[count][indices]
            partial = (
                (values * keep).unsqueeze(-1) * atoms
            ).sum(1) + sae.dec_bias
            sums[f"top_{count}"] += float((target - partial).square().mean(1).sum())
        all_values.append(values.cpu())
        all_indices.append(indices.cpu())
    losses = {key: value / len(positions) for key, value in sums.items()}
    # Fraction of the *target signal itself* (not the SAE's own bias-to-full
    # gap) explained by the top-k features -- i.e. (bias_mse - topk_mse) /
    # bias_mse, the standard R2/variance-explained framing. bias_mse is the
    # raw perturbation's own magnitude (MSE of predicting the constant decoder
    # bias), which is never ~0 by construction, unlike (bias_mse - full_mse)
    # -- the SAE's own ceiling over doing nothing -- which routinely goes
    # near-zero when a given seed's SAE happens to show little total
    # improvement for a condition, causing 0/0 blowups and huge seed-to-seed
    # variance in the fraction even when the underlying detection is stable.
    fractions = {
        count: (
            (losses["bias"] - losses[f"top_{count}"]) / losses["bias"]
            if losses["bias"] > 0 else None
        )
        for count in counts
    }
    return losses, fractions, torch.cat(all_values), torch.cat(all_indices)


@torch.no_grad()
def save_feature_visuals(
    output_dir, run_name, feature_order, importance, values, indices,
    evaluation_positions, valid_indices, residuals, base_latents,
    checkpoint, sae, autoencoder, device,
):
    std = checkpoint["channel_std"].float().view(
        stats_broadcast_shape(checkpoint["input_shape"], with_batch_dim=True)
    )
    # One scalar activation per chosen example, broadcast against the full
    # atom shape (every axis of input_shape, not just a per-channel stat).
    activation_shape = (-1,) + (1,) * len(checkpoint["input_shape"])
    rows = []
    exact = {}
    for rank, feature_tensor in enumerate(feature_order[:10], start=1):
        feature = int(feature_tensor)
        activations = torch.where(
            indices == feature, values, torch.zeros_like(values)
        ).max(1).values
        chosen = torch.argsort(activations, descending=True)[:8]
        local_positions = evaluation_positions[chosen]
        global_indices = valid_indices[local_positions]
        base = base_latents[global_indices].to(device).float()
        target_residual = residuals[local_positions].to(device).float()
        atom = sae.decoder[feature].detach().cpu().view(
            1, *checkpoint["input_shape"]
        )
        raw_write = activations[chosen].view(activation_shape) * atom * std
        base_image = autoencoder.decode(base).add(1).div(2)
        target_image = autoencoder.decode(base + target_residual).add(1).div(2)
        feature_image = autoencoder.decode(base + raw_write.to(device)).add(1).div(2)
        true_delta = (target_image - base_image).cpu()
        feature_delta = (feature_image - base_image).cpu()
        grid = torch.cat((true_delta, feature_delta)).mul(16).add(0.5).clamp(0, 1)
        filename = f"{run_name}_rank_{rank:02d}_feature_{feature}_signed_delta_x16.png"
        save_image(grid, os.path.join(output_dir, filename), nrow=8)
        rows.append({
            "rank": rank,
            "feature": feature,
            "mean_exact_ablation_loss_increase": float(importance[feature]),
            "mean_active_activation_in_examples": float(
                activations[activations > 0].mean() if (activations > 0).any() else 0
            ),
            "evaluation_activation_rate": float((activations > 0).float().mean()),
            "visualization": filename,
        })
        exact[f"rank_{rank:02d}_feature_{feature}"] = {
            "global_indices": global_indices,
            "activations": activations[chosen],
            "true_transformed_minus_clean": true_delta,
            "feature_write_minus_clean": feature_delta,
        }
    torch.save(exact, os.path.join(output_dir, run_name + "_top10_exact_deltas.pt"))
    return rows


def analyze_one(
    path, residual_cache, base_latents, autoencoder, output_dir, batch_size, device,
):
    checkpoint, sae = load_sae(path, device)
    residuals_all = residual_cache["latents"]
    valid_indices = residual_cache["valid_indices"].long()
    residuals = residuals_all[valid_indices]
    order = torch.randperm(
        len(valid_indices), generator=torch.Generator().manual_seed(260810)
    )
    split = len(order) // 2
    discovery_positions = order[:split]
    evaluation_positions = order[split:]
    batch_stats_shape = stats_broadcast_shape(checkpoint["input_shape"], with_batch_dim=True)
    mean = checkpoint["channel_mean"].to(device).float().view(batch_stats_shape)
    std = checkpoint["channel_std"].to(device).float().view(batch_stats_shape)

    stats = rank_by_exact_ablation(
        sae, residuals, discovery_positions, mean, std, batch_size, device
    )
    importance = stats["mean_importance"]
    feature_order = torch.argsort(importance, descending=True)
    positive_indices = residual_cache.get(
        "train_positive_indices", residual_cache["train_indices"]
    ).long()
    mean_positive_residual = residuals_all[positive_indices].float().mean(0)
    standardized_direction = (
        mean_positive_residual
        / checkpoint["channel_std"].float().view(
            stats_broadcast_shape(checkpoint["input_shape"], with_batch_dim=False)
        )
    ).flatten()
    direction_cosines = F.cosine_similarity(
        sae.decoder.detach().cpu(), standardized_direction[None], dim=1
    )
    direction_feature = int(direction_cosines.abs().argmax())
    direction_feature_rank = int(
        torch.where(feature_order == direction_feature)[0][0]
    ) + 1
    losses, fractions, values, indices = evaluate_top_sets(
        sae, residuals, evaluation_positions, feature_order,
        mean, std, batch_size, device,
    )
    objective = checkpoint["config"]["objective"]
    seed = checkpoint["config"]["seed"]
    run_name = f"{objective}_seed_{seed}"
    rows = save_feature_visuals(
        output_dir, run_name, feature_order, importance, values, indices,
        evaluation_positions, valid_indices, residuals, base_latents,
        checkpoint, sae, autoencoder, device,
    )
    for row in rows:
        feature = row["feature"]
        count = int(stats["fire_count"][feature])
        row["discovery_activation_rate"] = count / len(discovery_positions)
        row["discovery_mean_activation_when_active"] = (
            float(stats["activation_sum"][feature] / count) if count else 0.0
        )
    with open(os.path.join(output_dir, run_name + "_top10.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return {
        "processing_feature": residual_cache["processing_feature"]["feature"],
        "objective": objective,
        "seed": seed,
        "evaluation_bias_only_mse": losses["bias"],
        "evaluation_full_sae_mse": losses["full"],
        "evaluation_top1_mse": losses["top_1"],
        "evaluation_top5_mse": losses["top_5"],
        "evaluation_top10_mse": losses["top_10"],
        "top1_fraction_of_target_explained": fractions[1],
        "top5_fraction_of_target_explained": fractions[5],
        "top10_fraction_of_target_explained": fractions[10],
        "mean_positive_direction_best_feature": direction_feature,
        "mean_positive_direction_best_abs_decoder_cosine": float(
            direction_cosines[direction_feature].abs()
        ),
        "mean_positive_direction_feature_ablation_rank": direction_feature_rank,
        "top_features": rows,
        "checkpoint": path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-cache", required=True)
    parser.add_argument("--base-latent-cache", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--autoencoder-type", choices=("spatial", "vector"), default="spatial")
    parser.add_argument("--sae-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objectives", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(2030,))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    residual_cache = torch.load(
        args.residual_cache, map_location="cpu", weights_only=False
    )
    base_cache = torch.load(
        args.base_latent_cache, map_location="cpu", weights_only=False
    )
    base_latents = base_cache["latents"]
    autoencoder = load_autoencoder(args.autoencoder, args.autoencoder_type, device)
    summaries = []
    for objective in args.objectives:
        for seed in args.seeds:
            path = os.path.join(args.sae_root, objective, f"seed_{seed}", "last.pt")
            summaries.append(analyze_one(
                path, residual_cache, base_latents, autoencoder,
                args.output_dir, args.batch_size, device,
            ))
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump({
            "processing_feature": residual_cache["processing_feature"],
            "runs": summaries,
        }, handle, indent=2)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
