#!/usr/bin/env python3
"""Inject one fixed Fourier feature and cache paired spatial-AE latents."""

import argparse
import json
import math
import os

import torch
from torchvision.utils import save_image

from spatial_conv_autoencoder import load_spatial_autoencoder


def fixed_pattern(height, width, frequency_x, frequency_y, phase, device):
    y = torch.arange(height, device=device, dtype=torch.float32)[:, None]
    x = torch.arange(width, device=device, dtype=torch.float32)[None, :]
    angle = 2 * math.pi * (
        frequency_x * x / width + frequency_y * y / height
    ) + phase
    return torch.sin(angle)


def inject(images_uint8, pattern, amplitude):
    images = images_uint8.float().to(pattern.device)
    images = (images + amplitude * pattern[None, None]).clamp(0, 255)
    return images.div(127.5).sub(1)


@torch.no_grad()
def encode_injected(model, images, indices, pattern, amplitude, batch_size, device):
    latents = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        batch = inject(images[selected].to(device, non_blocking=True), pattern, amplitude)
        latents.append(model.encode(batch).half().cpu())
    return torch.cat(latents)


def choose_subset(indices, fraction, seed):
    count = max(1, round(len(indices) * fraction))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(indices), generator=generator)
    return indices[order[:count]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--clean-latent-cache", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--frequency-x", type=int, default=17)
    parser.add_argument("--frequency-y", type=int, default=17)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--amplitude", type=float, default=3.0,
                        help="Sinusoid peak amplitude in 8-bit pixel levels.")
    parser.add_argument("--prevalence", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=260810)
    args = parser.parse_args()

    if not 0 < args.prevalence < 1:
        raise ValueError("prevalence must lie strictly between zero and one")
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

    height, width = images.shape[-2:]
    pattern = fixed_pattern(
        height, width, args.frequency_x, args.frequency_y, args.phase, device
    )
    radius = math.hypot(args.frequency_x, args.frequency_y)
    if not 16 <= radius < 32:
        raise ValueError(
            f"frequency radius {radius:.3f} is outside the first high band [16, 32)"
        )
    model = load_spatial_autoencoder(args.autoencoder, device)

    train_positive = choose_subset(train_indices, args.prevalence, args.seed)
    valid_positive = choose_subset(valid_indices, args.prevalence, args.seed + 1)
    all_positive = torch.cat((train_positive, valid_positive))
    positive_latents = encode_injected(
        model, images, all_positive, pattern, args.amplitude, args.batch_size, device
    )
    mixed_latents = clean_latents.clone()
    mixed_latents[all_positive] = positive_latents

    paired_injected = encode_injected(
        model, images, valid_indices, pattern, args.amplitude, args.batch_size, device
    )
    injected_mask = torch.zeros(len(images), dtype=torch.bool)
    injected_mask[all_positive] = True
    metadata = {
        "feature": "fixed_diagonal_sinusoid",
        "frequency_x": args.frequency_x,
        "frequency_y": args.frequency_y,
        "frequency_radius": radius,
        "phase": args.phase,
        "amplitude_8bit": args.amplitude,
        "prevalence": args.prevalence,
        "train_positive_count": len(train_positive),
        "valid_positive_count": len(valid_positive),
        "paired_valid_count": len(valid_indices),
        "seed": args.seed,
    }
    torch.save(
        {
            "latents": mixed_latents,
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "injected_mask": injected_mask,
            "train_positive_indices": train_positive,
            "valid_positive_indices": valid_positive,
            "paired_indices": valid_indices,
            "paired_clean_latents": clean_latents[valid_indices],
            "paired_injected_latents": paired_injected,
            "pattern": pattern.half().cpu(),
            "source_data": args.data,
            "clean_latent_cache": args.clean_latent_cache,
            "autoencoder_checkpoint": args.autoencoder,
            "latent_shape": tuple(clean_latents.shape[1:]),
            "fixed_feature": metadata,
        },
        args.output,
    )

    preview_indices = valid_indices[:8]
    clean_images = images[preview_indices].float().div(255)
    injected_images = inject(
        images[preview_indices].to(device), pattern, args.amplitude
    ).add(1).div(2).cpu()
    difference = (injected_images - clean_images).mul(32).add(0.5).clamp(0, 1)
    save_image(
        torch.cat((clean_images, injected_images, difference)),
        args.preview,
        nrow=8,
    )
    with open(os.path.splitext(args.output)[0] + ".json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
