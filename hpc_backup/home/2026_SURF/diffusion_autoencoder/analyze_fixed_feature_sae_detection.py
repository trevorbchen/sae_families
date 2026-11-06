#!/usr/bin/env python3
"""Measure whether SAE vocabularies isolate a known injected Fourier feature."""

import argparse
import csv
import json
import os

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from spatial_conv_autoencoder import load_spatial_autoencoder
from train_spatial_map_sae import FlatTopKSAE


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


def write_comparison_svg(path, grouped):
    width, height = 720, 440
    left, right, top, bottom = 82, 24, 52, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    labels = list(grouped)
    colors = ("#6b7280", "#dc2626", "#2563eb", "#059669")

    def y_position(value):
        return top + (1.0 - value) / 0.55 * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.axis{stroke:#374151;stroke-width:1}.grid{stroke:#d1d5db;stroke-width:1}</style>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">Recovery of fixed high-frequency feature</text>',
    ]
    for tick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        y = y_position(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12">{tick:.1f}</text>')
    slot = plot_width / max(1, len(labels))
    bar_width = min(120, slot * 0.5)
    for index, label in enumerate(labels):
        values = torch.tensor(grouped[label], dtype=torch.float32)
        mean = float(values.mean())
        error = float(values.std(unbiased=True)) if len(values) > 1 else 0.0
        x = left + slot * (index + 0.5)
        y = y_position(mean)
        baseline = y_position(0.45)
        parts.append(
            f'<rect x="{x - bar_width / 2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{baseline - y:.2f}" fill="{colors[index % len(colors)]}"/>'
        )
        error_top, error_bottom = y_position(min(1.0, mean + error)), y_position(max(0.45, mean - error))
        parts.append(f'<line class="axis" x1="{x:.2f}" y1="{error_top:.2f}" x2="{x:.2f}" y2="{error_bottom:.2f}"/>')
        parts.append(f'<line class="axis" x1="{x - 8:.2f}" y1="{error_top:.2f}" x2="{x + 8:.2f}" y2="{error_top:.2f}"/>')
        parts.append(f'<line class="axis" x1="{x - 8:.2f}" y1="{error_bottom:.2f}" x2="{x + 8:.2f}" y2="{error_bottom:.2f}"/>')
        parts.append(f'<text x="{x:.2f}" y="{height - 38}" text-anchor="middle" font-size="13">{label}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{y_position(0.45):.2f}"/>')
    parts.append(f'<text transform="translate(20 {height / 2}) rotate(-90)" text-anchor="middle" font-size="13">Held-out single-feature separation AUROC</text>')
    parts.append('</svg>')
    with open(path, "w") as handle:
        handle.write("\n".join(parts))


def load_sae(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["architecture"] != "flat":
        raise ValueError("fixed-feature analysis currently expects flat spatial SAEs")
    sae = FlatTopKSAE(
        checkpoint["input_shape"], config["feature_count"], config["k"]
    )
    sae.load_state_dict(checkpoint["model"])
    sae.eval().to(device)
    return checkpoint, sae


@torch.no_grad()
def decoded_write(autoencoder, clean_raw, raw_write, batch_size, device):
    differences = []
    limit = min(128, len(clean_raw))
    for start in range(0, limit, batch_size):
        clean = clean_raw[start:start + batch_size].to(device).float()
        altered = clean + raw_write.to(device)
        differences.append(
            (autoencoder.decode(altered) - autoencoder.decode(clean)).cpu()
        )
    return torch.cat(differences).mean(0)


def analyze_one(path, cache, autoencoder, output_dir, batch_size, device):
    checkpoint, sae = load_sae(path, device)
    clean_raw = cache["paired_clean_latents"].float()
    injected_raw = cache["paired_injected_latents"].float()
    mean = checkpoint["channel_mean"][None, :, None, None].float()
    std = checkpoint["channel_std"][None, :, None, None].float()
    clean = (clean_raw - mean) / std
    injected = (injected_raw - mean) / std

    generator = torch.Generator().manual_seed(260810)
    order = torch.randperm(len(clean), generator=generator)
    split = len(order) // 2
    discovery = order[:split]
    evaluation = order[split:]
    clean_activation = sparse_activations(sae, clean, batch_size, device)
    injected_activation = sparse_activations(sae, injected, batch_size, device)
    delta = injected_activation - clean_activation
    discovery_delta = delta[discovery]
    delta_mean = discovery_delta.mean(0)
    delta_std = discovery_delta.std(0).clamp_min(1e-8)
    standardized_effect = delta_mean / delta_std
    selected = int(delta_mean.abs().argmax())

    auc = paired_auc(
        clean_activation[evaluation, selected],
        injected_activation[evaluation, selected],
    )
    separation_auc = max(auc, 1 - auc)
    eval_delta = delta[evaluation, selected]
    atom = sae.decoder[selected].detach().cpu()
    mean_latent_shift = (injected[evaluation] - clean[evaluation]).mean(0).flatten()
    signed_atom = atom * torch.sign(delta_mean[selected]).clamp(min=-1, max=1)
    atom_alignment = safe_cosine(signed_atom, mean_latent_shift)

    discovery_shift = (injected[discovery] - clean[discovery]).mean(0).flatten()
    direction = F.normalize(discovery_shift, dim=0)
    decoder_alignments = F.normalize(sae.decoder.detach().cpu(), dim=1) @ direction
    clean_projection = clean[evaluation].flatten(1) @ direction
    injected_projection = injected[evaluation].flatten(1) @ direction
    raw_latent_auc = paired_auc(clean_projection, injected_projection)

    raw_write = (
        delta_mean[selected]
        * atom.view(1, *checkpoint["input_shape"])
        * std
    )
    image_write = decoded_write(
        autoencoder, clean_raw[evaluation], raw_write, batch_size, device
    )
    pattern = cache["pattern"].float()
    grayscale_write = image_write.mean(0)
    pattern_alignment = safe_cosine(grayscale_write, pattern)

    objective = checkpoint["config"]["objective"]
    seed = checkpoint["config"]["seed"]
    run_name = f"{objective}_seed_{seed}"
    rows = []
    order_features = torch.argsort(delta_mean.abs(), descending=True)
    for rank, feature in enumerate(order_features.tolist(), start=1):
        rows.append({
            "rank": rank,
            "feature": feature,
            "discovery_mean_delta": float(delta_mean[feature]),
            "discovery_mean_write_norm": float(delta_mean[feature].abs()),
            "discovery_standardized_effect": float(standardized_effect[feature]),
            "decoder_alignment_with_latent_shift": float(decoder_alignments[feature]),
            "clean_activation_rate": float((clean_activation[:, feature] > 0).float().mean()),
            "injected_activation_rate": float((injected_activation[:, feature] > 0).float().mean()),
        })
    with open(os.path.join(output_dir, run_name + "_feature_ranking.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    preview_indices = evaluation[:8]
    clean_preview = clean_raw[preview_indices].to(device)
    injected_preview = injected_raw[preview_indices].to(device)
    altered_preview = clean_preview + raw_write.to(device)
    with torch.no_grad():
        clean_images = autoencoder.decode(clean_preview).add(1).div(2).cpu()
        injected_images = autoencoder.decode(injected_preview).add(1).div(2).cpu()
        altered_images = autoencoder.decode(altered_preview).add(1).div(2).cpu()
    amplified = (altered_images - clean_images).mul(16).add(0.5).clamp(0, 1)
    save_image(
        torch.cat((clean_images, injected_images, altered_images, amplified)),
        os.path.join(output_dir, run_name + "_selected_feature.png"),
        nrow=8,
    )

    return {
        "objective": objective,
        "seed": seed,
        "selected_feature": selected,
        "selection_direction": "activates" if auc >= 0.5 else "suppresses",
        "heldout_auc": auc,
        "heldout_separation_auc": separation_auc,
        "heldout_paired_direction_accuracy": float(
            ((eval_delta > 0) if auc >= 0.5 else (eval_delta < 0)).float().mean()
        ),
        "discovery_mean_delta": float(delta_mean[selected]),
        "discovery_standardized_effect": float(standardized_effect[selected]),
        "clean_activation_rate": float((clean_activation[:, selected] > 0).float().mean()),
        "injected_activation_rate": float((injected_activation[:, selected] > 0).float().mean()),
        "decoder_atom_alignment_with_latent_shift": atom_alignment,
        "decoded_write_alignment_with_pattern": pattern_alignment,
        "raw_latent_direction_auc": raw_latent_auc,
        "checkpoint": path,
    }, signed_atom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--sae-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objectives", nargs="+", default=("latent_mse", "fourier_high"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(2030, 2031, 2032))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    autoencoder = load_spatial_autoencoder(args.autoencoder, device)
    summaries = []
    selected_atoms = {}
    for objective in args.objectives:
        selected_atoms[objective] = []
        for seed in args.seeds:
            path = os.path.join(args.sae_root, objective, f"seed_{seed}", "last.pt")
            summary, atom = analyze_one(
                path, cache, autoencoder, args.output_dir, args.batch_size, device
            )
            summaries.append(summary)
            selected_atoms[objective].append(atom)

    for objective, atoms in selected_atoms.items():
        cosines = []
        for left in range(len(atoms)):
            for right in range(left + 1, len(atoms)):
                cosines.append(safe_cosine(atoms[left], atoms[right]))
        value = sum(cosines) / len(cosines) if cosines else None
        for summary in summaries:
            if summary["objective"] == objective:
                summary["cross_seed_selected_atom_cosine"] = value

    fields = list(summaries[0].keys())
    with open(os.path.join(args.output_dir, "summary.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump({"fixed_feature": cache["fixed_feature"], "runs": summaries}, handle, indent=2)

    grouped = {}
    for objective in args.objectives:
        values = [
            row["heldout_separation_auc"] for row in summaries
            if row["objective"] == objective
        ]
        grouped[objective] = values
    write_comparison_svg(
        os.path.join(args.output_dir, "objective_detection_comparison.svg"), grouped
    )
    print(json.dumps({"fixed_feature": cache["fixed_feature"], "runs": summaries}, indent=2))


if __name__ == "__main__":
    main()
