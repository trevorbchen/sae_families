#!/usr/bin/env python3
"""Cache paired spatial-AE residuals for controlled image transformations."""

import argparse
import json
import math
import os

import torch
from torchvision.utils import save_image

from prepare_processing_feature_spatial_latents import transform_images
from spatial_conv_autoencoder import load_spatial_autoencoder


@torch.no_grad()
def encode_transformed_float(model, images, indices, args, phases, batch_size, device):
    latents = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        transformed = transform_images(images, selected, args, phases)
        normalized = transformed.to(device, non_blocking=True).div(127.5).sub(1)
        latents.append(model.encode(normalized).float().cpu())
    return torch.cat(latents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", choices=("jpeg", "random_phase"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--clean-latent-cache", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--jpeg-quality", type=int, default=60)
    parser.add_argument("--frequency-x", type=int, default=17)
    parser.add_argument("--frequency-y", type=int, default=17)
    parser.add_argument("--amplitude", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=260810)
    args = parser.parse_args()

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
    all_indices = torch.arange(len(images))
    transformed_latents = encode_transformed_float(
        model, images, all_indices, args, phases, args.batch_size, device
    )
    residual_latents = transformed_latents - clean_latents.float()

    metadata = {
        "feature": args.feature,
        "examples": len(images),
        "target": "transformed_latent_minus_clean_latent",
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
            "latents": residual_latents,
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "clean_latent_cache": args.clean_latent_cache,
            "autoencoder_checkpoint": args.autoencoder,
            "latent_shape": tuple(residual_latents.shape[1:]),
            "processing_feature": metadata,
        },
        args.output,
    )

    preview_indices = valid_indices[:8]
    clean_images = images[preview_indices].float().div(255)
    transformed_images = transform_images(images, preview_indices, args, phases).div(255)
    signed_delta = (transformed_images - clean_images).mul(16).add(0.5).clamp(0, 1)
    save_image(signed_delta, args.preview, nrow=8)
    with open(os.path.splitext(args.output)[0] + ".json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
