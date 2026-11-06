#!/usr/bin/env python3
"""Create sparse controlled residual-latent datasets for SAE discovery tests."""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg
from torchvision.utils import save_image

from prepare_processing_feature_spatial_latents import choose_subset
from spatial_conv_autoencoder import load_spatial_autoencoder
from vector_conv_autoencoder import load_vector_autoencoder


def load_autoencoder(path, autoencoder_type, device):
    if autoencoder_type == "vector":
        return load_vector_autoencoder(path, device)
    return load_spatial_autoencoder(path, device)


PIXEL_FEATURES = (
    "sine_fixed",
    "sine_random_phase",
    "checkerboard",
    "color_red",
    "jpeg",
    "brightness",
    "contrast",
    "hue_rotate",
    "gaussian_blur",
    "unsharp_mask",
    "edge_enhance",
    "local_gabor",
    "local_blob",
    "sparse_impulse",
    "pixelate",
    "local_occlusion",
)


def transform_images(images, indices, phases, args):
    selected = images[indices].contiguous()
    if args.feature == "jpeg":
        encoded = encode_jpeg(list(selected), quality=args.jpeg_quality)
        return torch.stack(decode_jpeg(encoded, mode=ImageReadMode.RGB)).float()

    output = selected.float()
    height, width = output.shape[-2:]
    y = torch.arange(height, dtype=torch.float32)[:, None]
    x = torch.arange(width, dtype=torch.float32)[None, :]
    if args.feature in ("sine_fixed", "sine_random_phase"):
        angle = 2 * math.pi * (
            args.frequency_x * x / width + args.frequency_y * y / height
        )
        if args.feature == "sine_random_phase":
            phase = phases[indices].float()[:, None, None]
        else:
            phase = torch.full(
                (len(indices), 1, 1), args.phase, dtype=torch.float32
            )
        pattern = torch.sin(angle[None] + phase)
        output = output + args.amplitude * pattern[:, None]
    elif args.feature == "checkerboard":
        pattern = ((x.long() + y.long()).remainder(2).mul(2).sub(1)).float()
        output = output + args.amplitude * pattern[None, None]
    elif args.feature == "color_red":
        output[:, 0] += args.amplitude
    elif args.feature == "brightness":
        output += args.amplitude
    elif args.feature == "contrast":
        image_mean = output.mean(dim=(-2, -1), keepdim=True)
        output = image_mean + (1 + args.amplitude) * (output - image_mean)
    elif args.feature == "hue_rotate":
        output = TF.adjust_hue(output.div(255), args.amplitude).mul(255)
    elif args.feature == "gaussian_blur":
        output = TF.gaussian_blur(
            output, [args.blur_kernel, args.blur_kernel], [args.blur_sigma] * 2
        )
    elif args.feature == "unsharp_mask":
        blurred = TF.gaussian_blur(
            output, [args.blur_kernel, args.blur_kernel], [args.blur_sigma] * 2
        )
        output = output + args.amplitude * (output - blurred)
    elif args.feature == "edge_enhance":
        gray = output.mean(1, keepdim=True)
        sobel_x = output.new_tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
        ).unsqueeze(1)
        sobel_y = sobel_x.transpose(-1, -2)
        dx = F.conv2d(gray, sobel_x, padding=1)
        dy = F.conv2d(gray, sobel_y, padding=1)
        edges = (dx.square() + dy.square()).sqrt()
        edges = edges / edges.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        output += args.amplitude * edges
    elif args.feature in ("local_gabor", "local_blob"):
        center_x = 0.68 * (width - 1)
        center_y = 0.32 * (height - 1)
        sigma = args.local_sigma * min(height, width)
        envelope = torch.exp(
            -((x - center_x).square() + (y - center_y).square()) / (2 * sigma**2)
        )
        if args.feature == "local_gabor":
            carrier = torch.cos(
                2
                * math.pi
                * (args.frequency_x * x / width + args.frequency_y * y / height)
            )
            pattern = envelope * carrier
        else:
            pattern = envelope
        output += args.amplitude * pattern[None, None]
    elif args.feature == "sparse_impulse":
        mask = ((x.long() * 37 + y.long() * 17).remainder(97) == 0).float()
        signs = ((x.long() + 3 * y.long()).remainder(2).mul(2).sub(1)).float()
        output += args.amplitude * (mask * signs)[None, None]
    elif args.feature == "pixelate":
        pooled = F.avg_pool2d(
            output, kernel_size=args.block_size, stride=args.block_size
        )
        output = F.interpolate(pooled, size=(height, width), mode="nearest")
    elif args.feature == "local_occlusion":
        side = args.occlusion_size
        top = max(0, height // 3 - side // 2)
        left = max(0, 2 * width // 3 - side // 2)
        output[:, :, top:top + side, left:left + side] *= 1 - args.amplitude
    else:
        raise ValueError(f"unknown pixel feature {args.feature}")
    return output.clamp(0, 255)


@torch.no_grad()
def encode_transformed_float(model, images, indices, phases, args, device):
    latents = []
    for start in range(0, len(indices), args.batch_size):
        chosen = indices[start:start + args.batch_size]
        transformed = transform_images(images, chosen, phases, args)
        normalized = transformed.to(device, non_blocking=True).div(127.5).sub(1)
        latents.append(model.encode(normalized).float().cpu())
    return torch.cat(latents)


def latent_random_residual(clean_latents, positive_indices, amplitude, seed):
    # Vector latents are [N, d]; spatial latents are [N, channels, ...]. Only
    # dim 1 is the channel axis in either case; reduce over every other dim.
    generator = torch.Generator().manual_seed(seed + 17)
    direction = torch.randn(clean_latents.shape[1:], generator=generator)
    direction = direction / direction.square().mean().sqrt().clamp_min(1e-12)
    reduce_dims = tuple(d for d in range(clean_latents.dim()) if d != 1)
    channel_std = clean_latents.float().std(dim=reduce_dims, keepdim=False)
    broadcast_shape = (-1,) + (1,) * (clean_latents.dim() - 2)
    direction = direction * channel_std.view(broadcast_shape) * amplitude
    residuals = torch.zeros_like(clean_latents, dtype=torch.float32)
    residuals[positive_indices] = direction
    return residuals, direction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature", choices=(*PIXEL_FEATURES, "latent_random"), required=True
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--clean-latent-cache", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--autoencoder-type", choices=("spatial", "vector"), default="spatial")
    parser.add_argument("--output", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--prevalence", type=float, default=0.10)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--frequency-x", type=int, default=17)
    parser.add_argument("--frequency-y", type=int, default=17)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--jpeg-quality", type=int, default=60)
    parser.add_argument("--blur-kernel", type=int, default=9)
    parser.add_argument("--blur-sigma", type=float, default=2.0)
    parser.add_argument("--local-sigma", type=float, default=0.12)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--occlusion-size", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=260810)
    args = parser.parse_args()

    if not 0 < args.prevalence < 1:
        raise ValueError("prevalence must lie strictly between zero and one")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.preview), exist_ok=True)
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    images = data["images_uint8"]
    clean = torch.load(args.clean_latent_cache, map_location="cpu", weights_only=False)
    clean_latents = clean["latents"]
    train_indices = clean["train_indices"].long()
    valid_indices = clean["valid_indices"].long()
    train_positive = choose_subset(train_indices, args.prevalence, args.seed)
    valid_positive = choose_subset(valid_indices, args.prevalence, args.seed + 1)
    positive_indices = torch.cat((train_positive, valid_positive))

    phases = None
    if args.feature == "sine_random_phase":
        generator = torch.Generator().manual_seed(args.seed + 2)
        phases = torch.rand(len(images), generator=generator).mul(2 * math.pi)

    planted_direction = None
    if args.feature == "latent_random":
        residuals, planted_direction = latent_random_residual(
            clean_latents, positive_indices, args.amplitude, args.seed
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_autoencoder(args.autoencoder, args.autoencoder_type, device)
        transformed = encode_transformed_float(
            model, images, positive_indices, phases, args, device
        )
        residuals = torch.zeros_like(clean_latents, dtype=torch.float32)
        residuals[positive_indices] = (
            transformed - clean_latents[positive_indices].float()
        )

    metadata = {
        "feature": args.feature,
        "target": "sparse_transformed_latent_minus_clean_latent",
        "prevalence": args.prevalence,
        "train_positive_count": len(train_positive),
        "valid_positive_count": len(valid_positive),
        "amplitude": args.amplitude,
        "frequency_x": args.frequency_x if "sine" in args.feature else None,
        "frequency_y": args.frequency_y if "sine" in args.feature else None,
        "phase": (
            "uniform_per_positive_image"
            if args.feature == "sine_random_phase"
            else args.phase if args.feature == "sine_fixed" else None
        ),
        "jpeg_quality": args.jpeg_quality if args.feature == "jpeg" else None,
        "blur_kernel": (
            args.blur_kernel
            if args.feature in ("gaussian_blur", "unsharp_mask")
            else None
        ),
        "blur_sigma": (
            args.blur_sigma
            if args.feature in ("gaussian_blur", "unsharp_mask")
            else None
        ),
        "block_size": args.block_size if args.feature == "pixelate" else None,
        "occlusion_size": (
            args.occlusion_size if args.feature == "local_occlusion" else None
        ),
        "seed": args.seed,
    }
    torch.save(
        {
            "latents": residuals,
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "positive_mask": torch.zeros(len(images), dtype=torch.bool).scatter_(
                0, positive_indices, True
            ),
            "train_positive_indices": train_positive,
            "valid_positive_indices": valid_positive,
            "planted_direction": planted_direction,
            "clean_latent_cache": args.clean_latent_cache,
            "processing_feature": metadata,
        },
        args.output,
    )

    preview_indices = valid_positive[:8]
    if args.feature == "latent_random":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_autoencoder(args.autoencoder, args.autoencoder_type, device)
        base = clean_latents[preview_indices].to(device).float()
        changed = base + planted_direction.to(device)[None]
        clean_images = model.decode(base).add(1).div(2).cpu()
        transformed_images = model.decode(changed).add(1).div(2).cpu()
    else:
        clean_images = images[preview_indices].float().div(255)
        transformed_images = transform_images(
            images, preview_indices, phases, args
        ).div(255)
    signed_delta = (transformed_images - clean_images).mul(16).add(0.5).clamp(0, 1)
    save_image(signed_delta, args.preview, nrow=8)
    with open(os.path.splitext(args.output)[0] + ".json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
