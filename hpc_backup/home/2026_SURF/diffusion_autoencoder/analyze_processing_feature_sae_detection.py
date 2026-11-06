#!/usr/bin/env python3
"""Evaluate SAE detection of paired image-processing features."""

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


def paired_auc(negative, positive):
    comparisons = positive[:, None] - negative[None, :]
    return float(
        ((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean()
    )


def safe_cosine(left, right):
    denominator = left.norm() * right.norm()
    if denominator <= 1e-12:
        return 0.0
    return float(torch.dot(left.flatten(), right.flatten()) / denominator)


def sparse_activations(sae, standardized, batch_size, device):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(standardized), batch_size):
            batch = standardized[start:start + batch_size].to(device)
            flat = batch.flatten(1)
            dense = F.relu(F.linear(
                flat - sae.dec_bias, sae.encoder, sae.encoder_bias
            ))
            values, indices = torch.topk(dense, sae.k, dim=1)
            sparse = torch.zeros_like(dense)
            sparse.scatter_(1, indices, values)
            outputs.append(sparse.cpu())
    return torch.cat(outputs)


def ridge_probe(clean, transformed, discovery, evaluation, selected, ridge=1.0):
    train_x = torch.cat((clean[discovery][:, selected], transformed[discovery][:, selected]))
    train_y = torch.cat((
        torch.zeros(len(discovery)), torch.ones(len(discovery))
    ))
    mean = train_x.mean(0, keepdim=True)
    std = train_x.std(0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    centered_y = train_y - train_y.mean()
    identity = torch.eye(train_x.shape[1], dtype=train_x.dtype)
    weights = torch.linalg.solve(
        train_x.T @ train_x + ridge * identity,
        train_x.T @ centered_y,
    )
    clean_score = ((clean[evaluation][:, selected] - mean) / std) @ weights
    transformed_score = ((transformed[evaluation][:, selected] - mean) / std) @ weights
    return paired_auc(clean_score, transformed_score)


def load_sae(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["architecture"] != "flat":
        raise ValueError("processing-feature analysis expects flat spatial SAEs")
    sae = FlatTopKSAE(
        checkpoint["input_shape"], config["feature_count"], config["k"]
    )
    sae.load_state_dict(checkpoint["model"])
    sae.eval().to(device)
    return checkpoint, sae


def write_probe_svg(path, rows):
    width, height = 700, 420
    left, right, top, bottom = 76, 24, 48, 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    def y_position(value):
        return top + (1 - value) / 0.5 * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.grid{stroke:#d1d5db}.axis{stroke:#374151}</style>',
        f'<text x="{width / 2}" y="27" text-anchor="middle" font-size="18">Processing-feature detection</text>',
    ]
    for tick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        y = y_position(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="12">{tick:.1f}</text>')
    slot = plot_width / len(rows)
    for index, (label, value) in enumerate(rows):
        x = left + slot * (index + 0.5)
        y = y_position(value)
        base = y_position(0.5)
        parts.append(f'<rect x="{x-slot*0.28:.1f}" y="{y:.1f}" width="{slot*0.56:.1f}" height="{base-y:.1f}" fill="#2563eb"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-38}" text-anchor="middle" font-size="12">{label}</text>')
        parts.append(f'<text x="{x:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="12">{value:.3f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{y_position(0.5):.1f}"/>')
    parts.append('</svg>')
    with open(path, "w") as handle:
        handle.write("\n".join(parts))


@torch.no_grad()
def decoded_write(autoencoder, clean_raw, raw_write, batch_size, device):
    differences = []
    limit = min(128, len(clean_raw))
    for start in range(0, limit, batch_size):
        clean = clean_raw[start:start + batch_size].to(device).float()
        differences.append(
            (autoencoder.decode(clean + raw_write.to(device)) - autoencoder.decode(clean)).cpu()
        )
    return torch.cat(differences).mean(0)


def analyze_one(path, cache, autoencoder, output_dir, batch_size, device):
    checkpoint, sae = load_sae(path, device)
    clean_raw = cache["paired_clean_latents"].float()
    transformed_raw = cache["paired_transformed_latents"].float()
    # Vector latents are [batch, d]; spatial latents are [batch, channels, ...].
    # Broadcast channel stats without assuming a fixed number of spatial axes.
    spatial_axes = max(0, len(checkpoint["input_shape"]) - 1)
    stats_shape = (1, -1) + (1,) * spatial_axes
    mean = checkpoint["channel_mean"].float().view(stats_shape)
    std = checkpoint["channel_std"].float().view(stats_shape)
    clean = (clean_raw - mean) / std
    transformed = (transformed_raw - mean) / std
    clean_activation = sparse_activations(sae, clean, batch_size, device)
    transformed_activation = sparse_activations(sae, transformed, batch_size, device)

    order = torch.randperm(
        len(clean), generator=torch.Generator().manual_seed(260810)
    )
    split = len(order) // 2
    discovery, evaluation = order[:split], order[split:]
    discovery_clean = clean_activation[discovery]
    discovery_transformed = transformed_activation[discovery]
    combined = torch.cat((discovery_clean, discovery_transformed))
    mean_shift = discovery_transformed.mean(0) - discovery_clean.mean(0)
    pooled_std = combined.std(0).clamp_min(1e-6)
    standardized_shift = mean_shift / pooled_std
    feature_order = torch.argsort(standardized_shift.abs(), descending=True)

    top_counts = (1, 5, 10)
    probe_aucs = {
        count: ridge_probe(
            clean_activation, transformed_activation, discovery, evaluation,
            feature_order[:count],
        )
        for count in top_counts
    }
    selected = int(feature_order[0])
    single_auc = paired_auc(
        clean_activation[evaluation, selected],
        transformed_activation[evaluation, selected],
    )
    single_separation = max(single_auc, 1 - single_auc)
    latent_shift = (transformed[evaluation] - clean[evaluation]).mean(0).flatten()
    atom = sae.decoder[selected].detach().cpu()
    signed_atom = atom * torch.sign(mean_shift[selected])
    atom_alignment = safe_cosine(signed_atom, latent_shift)
    pair_shift_norm = float(
        (transformed - clean).flatten(1).norm(dim=1).mean()
    )

    objective = checkpoint["config"]["objective"]
    seed = checkpoint["config"]["seed"]
    run_name = f"{objective}_seed_{seed}"
    ranking_rows = []
    for rank, feature in enumerate(feature_order[:10].tolist(), start=1):
        auc = paired_auc(
            clean_activation[evaluation, feature],
            transformed_activation[evaluation, feature],
        )
        ranking_rows.append({
            "rank": rank,
            "feature": feature,
            "discovery_mean_activation_shift": float(mean_shift[feature]),
            "discovery_standardized_shift": float(standardized_shift[feature]),
            "heldout_auc": auc,
            "heldout_separation_auc": max(auc, 1 - auc),
            "clean_activation_rate": float((clean_activation[:, feature] > 0).float().mean()),
            "transformed_activation_rate": float((transformed_activation[:, feature] > 0).float().mean()),
        })
    with open(os.path.join(output_dir, run_name + "_ranking.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ranking_rows[0].keys())
        writer.writeheader()
        writer.writerows(ranking_rows)

    preview_indices = evaluation[:8]
    clean_preview = clean_raw[preview_indices].to(device)
    transformed_preview = transformed_raw[preview_indices].to(device)
    with torch.no_grad():
        clean_images = autoencoder.decode(clean_preview).add(1).div(2).cpu()
        transformed_images = autoencoder.decode(transformed_preview).add(1).div(2).cpu()
    true_delta = transformed_images - clean_images
    raw_writes = {}
    decoded_deltas = {}
    for count in top_counts:
        features = feature_order[:count]
        standardized_write = (
            mean_shift[features, None] * sae.decoder[features].detach().cpu()
        ).sum(0).view(1, *checkpoint["input_shape"])
        raw_write = standardized_write * std
        raw_writes[count] = raw_write
        with torch.no_grad():
            altered = autoencoder.decode(clean_preview + raw_write.to(device)).add(1).div(2).cpu()
        decoded_deltas[count] = altered - clean_images

    # Store the exact signed differences; PNGs use a fixed 16x gain around neutral gray.
    torch.save(
        {
            "preview_indices": preview_indices,
            "true_transformed_minus_clean": true_delta,
            **{f"top_{count}_feature_minus_clean": decoded_deltas[count] for count in top_counts},
        },
        os.path.join(output_dir, run_name + "_feature_only_deltas.pt"),
    )
    for label, delta in [
        ("true_transform", true_delta),
        *[(f"top_{count}", decoded_deltas[count]) for count in top_counts],
    ]:
        save_image(
            delta.mul(16).add(0.5).clamp(0, 1),
            os.path.join(output_dir, run_name + f"_{label}_signed_delta_x16.png"),
            nrow=8,
        )
    mean_image_writes = {
        count: decoded_write(
            autoencoder, clean_raw[evaluation], raw_writes[count], batch_size, device
        )
        for count in top_counts
    }

    summary = {
        "processing_feature": cache["processing_feature"]["feature"],
        "objective": objective,
        "seed": seed,
        "selected_feature": selected,
        "heldout_single_feature_auc": single_auc,
        "heldout_single_feature_separation_auc": single_separation,
        "heldout_top1_probe_auc": probe_aucs[1],
        "heldout_top5_probe_auc": probe_aucs[5],
        "heldout_top10_probe_auc": probe_aucs[10],
        "clean_activation_rate": float((clean_activation[:, selected] > 0).float().mean()),
        "transformed_activation_rate": float((transformed_activation[:, selected] > 0).float().mean()),
        "decoder_atom_alignment_with_mean_latent_shift": atom_alignment,
        "mean_standardized_pair_shift_norm": pair_shift_norm,
        "mean_decoded_top1_write_norm": float(mean_image_writes[1].norm()),
        "mean_decoded_top5_write_norm": float(mean_image_writes[5].norm()),
        "mean_decoded_top10_write_norm": float(mean_image_writes[10].norm()),
        "checkpoint": path,
    }
    write_probe_svg(
        os.path.join(output_dir, run_name + "_probe_aucs.svg"),
        [(f"top {count}", probe_aucs[count]) for count in top_counts],
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
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
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    autoencoder = load_autoencoder(args.autoencoder, args.autoencoder_type, device)
    summaries = []
    for objective in args.objectives:
        for seed in args.seeds:
            path = os.path.join(args.sae_root, objective, f"seed_{seed}", "last.pt")
            summaries.append(analyze_one(
                path, cache, autoencoder, args.output_dir, args.batch_size, device
            ))
    with open(os.path.join(args.output_dir, "summary.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    payload = {"processing_feature": cache["processing_feature"], "runs": summaries}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
