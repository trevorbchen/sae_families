#!/usr/bin/env python3
"""Measure image-space geometry induced by matched standalone SAE vocabularies."""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from spatial_conv_autoencoder import load_spatial_autoencoder
from train_spatial_map_sae import FlatTopKSAE


OBJECTIVES = (
    "latent_mse",
    "weighted_laplacian",
    "weighted_fourier_low",
    "weighted_fourier_mid",
    "weighted_fourier_high",
)
LABELS = {
    "latent_mse": "MSE",
    "weighted_laplacian": "Laplacian",
    "weighted_fourier_low": "Fourier low",
    "weighted_fourier_mid": "Fourier mid",
    "weighted_fourier_high": "Fourier high",
}
METRICS = (
    "laplacian_norm",
    "fourier_low_fraction",
    "fourier_mid_fraction",
    "fourier_high_fraction",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--latent-cache", required=True)
    parser.add_argument("--sae-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-features", type=int, default=500)
    parser.add_argument("--examples-per-feature", type=int, default=8)
    parser.add_argument("--min-examples", type=int, default=4)
    parser.add_argument("--rank-examples", type=int, default=13107)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--multipliers", type=float, nargs="+", default=(-2.0, 2.0))
    return parser.parse_args()


def checkpoint_path(root, objective, seed):
    root = Path(root)
    if objective == "latent_mse" and seed == 2030:
        return root / "spatial_map_saes/flat/latent_mse/last.pt"
    if objective == "latent_mse":
        return root / f"spatial_map_sae_seed_replicates/flat/latent_mse/seed_{seed}/last.pt"
    return root / f"spatial_map_sae_weighted_seed_replicates/flat/{objective}/seed_{seed}/last.pt"


def load_sae(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state["config"]
    model = FlatTopKSAE(state["input_shape"], config["feature_count"], config["k"])
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False).to(device)
    # Vector latents are [batch, d], while spatial latents are
    # [batch, channels, ...]. Broadcast channel statistics without inventing
    # spatial axes for the vector case.
    spatial_axes = max(0, len(state["input_shape"]) - 1)
    stats_shape = (1, -1) + (1,) * spatial_axes
    mean = state["channel_mean"].float().to(device).view(stats_shape)
    std = state["channel_std"].float().to(device).view(stats_shape)
    return model, mean, std, state


def selected_codes(model, standardized):
    flat = standardized.flatten(1)
    activations = F.relu(F.linear(flat - model.dec_bias, model.encoder, model.encoder_bias))
    return torch.topk(activations, model.k, dim=1)


@torch.no_grad()
def rank_features(model, mean, std, latents, indices, batch_size, device):
    sums = torch.zeros(model.h, device=device)
    squares = torch.zeros(model.h, device=device)
    counts = torch.zeros(model.h, device=device)
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        standardized = (latents[selected].to(device).float() - mean) / std
        values, feature_ids = selected_codes(model, standardized)
        positive = values > 0
        ids = feature_ids[positive]
        vals = values[positive]
        sums.scatter_add_(0, ids, vals)
        squares.scatter_add_(0, ids, vals.square())
        counts.scatter_add_(0, ids, torch.ones_like(vals))
    return {
        "order": torch.argsort(sums, descending=True).cpu().tolist(),
        "sums": sums.cpu(),
        "counts": counts.cpu(),
        "active_rms": (squares / counts.clamp_min(1)).sqrt().cpu(),
    }


@torch.no_grad()
def collect_examples(model, mean, std, latents, indices, candidate_features, batch_size, device):
    feature_to_slot = torch.full((model.h,), -1, dtype=torch.long, device=device)
    feature_to_slot[torch.tensor(candidate_features, device=device)] = torch.arange(
        len(candidate_features), device=device
    )
    candidates = [[] for _ in candidate_features]
    for start in range(0, len(indices), batch_size):
        selected_indices = indices[start:start + batch_size]
        standardized = (latents[selected_indices].to(device).float() - mean) / std
        values, feature_ids = selected_codes(model, standardized)
        slots = feature_to_slot[feature_ids]
        rows = torch.arange(len(selected_indices), device=device)[:, None].expand_as(feature_ids)
        selected = (slots >= 0) & (values > 0)
        triples = torch.stack((slots[selected], values[selected], rows[selected]), dim=1).cpu().tolist()
        for slot, value, row in triples:
            candidates[int(slot)].append((float(value), int(selected_indices[int(row)])))
    return {
        feature: sorted(feature_candidates, reverse=True)
        for feature, feature_candidates in zip(candidate_features, candidates)
    }


def image_geometry(images):
    """Return normalized Laplacian energy and non-DC Fourier-band fractions."""
    gray = (
        0.299 * images[:, 0]
        + 0.587 * images[:, 1]
        + 0.114 * images[:, 2]
    )
    kernel = images.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    padded = F.pad(gray[:, None], (1, 1, 1, 1), mode="reflect")
    laplacian = F.conv2d(padded, kernel[None, None])[:, 0]
    image_energy = gray.square().mean(dim=(1, 2))
    laplacian_energy = laplacian.square().mean(dim=(1, 2))

    height, width = gray.shape[-2:]
    spectrum = torch.fft.rfft2(gray - gray.mean(dim=(1, 2), keepdim=True), norm="ortho")
    power = spectrum.abs().square()
    # rfft2 stores only nonnegative horizontal frequencies; restore their omitted mirrors.
    horizontal_weight = torch.full(
        (spectrum.shape[-1],), 2.0, device=images.device, dtype=power.dtype
    )
    horizontal_weight[0] = 1.0
    if width % 2 == 0:
        horizontal_weight[-1] = 1.0
    power = power * horizontal_weight[None, None, :]
    fy = torch.fft.fftfreq(height, device=images.device) * height
    fx = torch.fft.rfftfreq(width, device=images.device) * width
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    masks = {
        "low": (radius > 0) & (radius < 4),
        "mid": (radius >= 4) & (radius < 16),
        "high": radius >= 16,
    }
    band_energy = {name: power[:, mask].sum(1) for name, mask in masks.items()}
    non_dc_energy = sum(band_energy.values())
    eps = torch.finfo(images.dtype).eps
    result = {
        "image_energy": image_energy,
        "laplacian_energy": laplacian_energy,
        "laplacian_norm": laplacian_energy / image_energy.clamp_min(eps),
        "fourier_non_dc_energy": non_dc_energy,
    }
    for name, energy in band_energy.items():
        result[f"fourier_{name}_energy"] = energy
        result[f"fourier_{name}_fraction"] = energy / non_dc_energy.clamp_min(eps)
    return result


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_feature_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["objective"], row["feature_rank"], row["feature"], row["measurement"])
        grouped.setdefault(key, []).append(row)
    output = []
    for (objective, rank, feature, measurement), records in grouped.items():
        result = {
            "objective": objective,
            "objective_label": LABELS[objective],
            "seed": records[0]["seed"],
            "feature_rank": rank,
            "feature": feature,
            "measurement": measurement,
            "examples": len(records),
        }
        for metric in METRICS:
            result[metric] = sum(float(record[metric]) for record in records) / len(records)
        output.append(result)
    return output


@torch.no_grad()
def evaluate_objective(args, objective, autoencoder, latents, train_indices, valid_indices, device):
    path = checkpoint_path(args.sae_root, objective, args.seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    model, mean, std, state = load_sae(path, device)
    rank_indices = train_indices[:min(args.rank_examples, len(train_indices))]
    stats = rank_features(model, mean, std, latents, rank_indices, args.batch_size, device)
    # Scan all non-dead features, then retain the most active features with enough held-out examples.
    alive_order = [feature for feature in stats["order"] if stats["counts"][feature] > 0]
    examples = collect_examples(
        model, mean, std, latents, valid_indices, alive_order,
        args.batch_size, device,
    )
    chosen = [
        feature for feature in alive_order
        if len(examples[feature]) >= args.min_examples
    ][:args.top_features]
    print(
        f"{objective}: evaluating {len(chosen)} features; "
        f"{sum(len(examples[f]) >= args.min_examples for f in alive_order)} eligible",
        flush=True,
    )
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
            measurements = {
                "edited_output": image_geometry(edited),
                "causal_delta": image_geometry(edited - clean),
            }
            for measurement, values in measurements.items():
                for row_index, example_index in enumerate(example_indices.tolist()):
                    row = {
                        "objective": objective,
                        "objective_label": LABELS[objective],
                        "seed": args.seed,
                        "checkpoint": str(path),
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
            print(f"{objective}: {rank + 1}/{len(chosen)} features", flush=True)
    metadata = {
        "objective": objective,
        "seed": args.seed,
        "checkpoint": str(path),
        "feature_count": state["config"]["feature_count"],
        "k": state["config"]["k"],
        "eligible_features": len(chosen),
    }
    return rows, metadata


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = load_spatial_autoencoder(args.autoencoder, device)
    autoencoder.eval().requires_grad_(False)
    cache = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
    latents = cache["latents"].float()
    train_indices = cache["train_indices"].long()
    valid_indices = cache["valid_indices"].long()
    all_rows = []
    metadata = []
    for objective in OBJECTIVES:
        rows, record = evaluate_objective(
            args, objective, autoencoder, latents, train_indices, valid_indices, device
        )
        all_rows.extend(rows)
        metadata.append(record)
    feature_rows = aggregate_feature_rows(all_rows)
    write_csv(out_dir / "raw_measurements.csv", all_rows)
    write_csv(out_dir / "feature_means.csv", feature_rows)
    manifest = {
        "method": {
            "feature_selection": "training-split activation-mass rank",
            "evaluation": "highest-activation held-out examples per feature",
            "intervention": "original latent plus selected decoder-atom coefficient delta",
            "multipliers": args.multipliers,
            "top_features": args.top_features,
            "examples_per_feature": args.examples_per_feature,
            "minimum_held_out_examples": args.min_examples,
            "laplacian": "squared 4-neighbor discrete Laplacian / image squared energy",
            "fourier": "low [1,4), mid [4,16), high [16,inf) fractions of non-DC power",
        },
        "models": metadata,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(all_rows)} raw rows to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
