#!/usr/bin/env python3
"""Cache spatial-AE latents for controlled image-processing features."""

import argparse
import json
import math
import os

import torch
from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg
from torchvision.utils import save_image

from spatial_conv_autoencoder import load_spatial_autoencoder


def choose_subset(indices, fraction, seed):
    count = max(1, round(len(indices) * fraction))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(indices), generator=generator)
    return indices[order[:count]]


def transform_images(images, indices, args, phases=None):
    selected = images[indices].contiguous()
    if args.feature == "jpeg":
        encoded = encode_jpeg(list(selected), quality=args.jpeg_quality)
        decoded = decode_jpeg(encoded, mode=ImageReadMode.RGB)
        return torch.stack(decoded).float()
    height, width = selected.shape[-2:]
    y = torch.arange(height, dtype=torch.float32)[:, None]
    x = torch.arange(width, dtype=torch.float32)[None, :]
    angle = 2 * math.pi * (
        args.frequency_x * x / width + args.frequency_y * y / height
    )
    phase = phases[indices].float()[:, None, None]
    pattern = torch.sin(angle[None] + phase)
    return (selected.float() + args.amplitude * pattern[:, None]).clamp(0, 255)


@torch.no_grad()
def encode_transformed(model, images, indices, args, phases, batch_size, device):
    latents = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        transformed = transform_images(images, selected, args, phases)
        normalized = transformed.to(device, non_blocking=True).div(127.5).sub(1)
        latents.append(model.encode(normalized).half().cpu())
    return torch.cat(latents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", choices=("jpeg", "random_phase"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--clean-latent-cache", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--prevalence", type=float, default=0.10)
    parser.add_argument("--jpeg-quality", type=int, default=60)
    parser.add_argument("--frequency-x", type=int, default=17)
    parser.add_argument("--frequency-y", type=int, default=17)
    parser.add_argument("--amplitude", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=260810)
    args = parser.parse_args()

    if not 0 < args.prevalence < 1:
        raise ValueError("prevalence must lie strictly between zero and one")
    if args.feature == "random_phase":
        radius = math.hypot(args.frequency_x, args.frequency_y)
        if not 16 <= radius < 32:
            raise ValueError(f"frequency radius {radius:.3f} is outside [16, 32)")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.preview), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    images = data["images_uint8"]
    clean = torch.load(args.clean_latent_cache, map_location="cpu", weights_only=False)
    clean_latents = clean["latents"]
    train_indices = clean["train_indices"].long()
    valid_indices = clean["valid_indices"].long()
    if len(images) != len(clean_latents):
        raise ValueError("image data and clean latent cache have different lengths")

    phases = None
    if args.feature == "random_phase":
        generator = torch.Generator().manual_seed(args.seed + 2)
        phases = torch.rand(len(images), generator=generator).mul(2 * math.pi)
    model = load_spatial_autoencoder(args.autoencoder, device)
    train_positive = choose_subset(train_indices, args.prevalence, args.seed)
    valid_positive = choose_subset(valid_indices, args.prevalence, args.seed + 1)
    all_positive = torch.cat((train_positive, valid_positive))
    positive_latents = encode_transformed(
        model, images, all_positive, args, phases, args.batch_size, device
    )
    mixed_latents = clean_latents.clone()
    mixed_latents[all_positive] = positive_latents
    paired_transformed = encode_transformed(
        model, images, valid_indices, args, phases, args.batch_size, device
    )

    transformed_mask = torch.zeros(len(images), dtype=torch.bool)
    transformed_mask[all_positive] = True
    metadata = {
        "feature": args.feature,
        "prevalence": args.prevalence,
        "train_positive_count": len(train_positive),
        "valid_positive_count": len(valid_positive),
        "paired_valid_count": len(valid_indices),
        "jpeg_quality": args.jpeg_quality if args.feature == "jpeg" else None,
        "frequency_x": args.frequency_x if args.feature == "random_phase" else None,
        "frequency_y": args.frequency_y if args.feature == "random_phase" else None,
        "frequency_radius": (
            math.hypot(args.frequency_x, args.frequency_y)
            if args.feature == "random_phase" else None
        ),
        "amplitude_8bit": args.amplitude if args.feature == "random_phase" else None,
        "phase": "deterministic_uniform_per_image" if args.feature == "random_phase" else None,
        "seed": args.seed,
    }
    torch.save(
        {
            "latents": mixed_latents,
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "transformed_mask": transformed_mask,
            "train_positive_indices": train_positive,
            "valid_positive_indices": valid_positive,
            "paired_indices": valid_indices,
            "paired_clean_latents": clean_latents[valid_indices],
            "paired_transformed_latents": paired_transformed,
            "source_data": args.data,
            "clean_latent_cache": args.clean_latent_cache,
            "autoencoder_checkpoint": args.autoencoder,
            "latent_shape": tuple(clean_latents.shape[1:]),
            "processing_feature": metadata,
        },
        args.output,
    )

    preview_indices = valid_indices[:8]
    clean_images = images[preview_indices].float().div(255)
    transformed_images = transform_images(images, preview_indices, args, phases).div(255)
    difference = (transformed_images - clean_images).mul(8).add(0.5).clamp(0, 1)
    save_image(
        torch.cat((clean_images, transformed_images, difference)), args.preview, nrow=8
    )
    with open(os.path.splitext(args.output)[0] + ".json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
