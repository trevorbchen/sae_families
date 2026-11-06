#!/usr/bin/env python3
"""Encode CelebA and controlled geometry variants with a frozen ConvAE."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torchvision.transforms import functional as TF

from spatial_conv_autoencoder import load_spatial_autoencoder
from vector_conv_autoencoder import load_vector_autoencoder


ATTRIBUTE_NAMES = (
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young",
)

GEOMETRY_NAMES = (
    "luminance_mean", "contrast_rms", "gradient_log_power",
    "laplacian_log_power", "fourier_low_log_power",
    "fourier_mid_log_power", "fourier_high_log_power", "spectral_centroid",
)

CONTROL_NAMES = (
    "clean", "blur", "unsharp", "sine_low", "sine_mid", "sine_high",
    "white_noise",
)


def depthwise_filter(image, kernel):
    channels = image.shape[1]
    weight = kernel.to(image)[None, None].expand(channels, 1, -1, -1)
    return F.conv2d(F.pad(image, (1, 1, 1, 1), mode="reflect"), weight,
                    groups=channels)


def gaussian_blur(image):
    weights = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=image.device)
    weights = weights.to(image).div(16)
    channels = image.shape[1]
    horizontal = weights[None, None, None].expand(channels, 1, 1, -1)
    vertical = weights[None, None, :, None].expand(channels, 1, -1, 1)
    image = F.conv2d(F.pad(image, (2, 2, 0, 0), mode="reflect"), horizontal,
                     groups=channels)
    return F.conv2d(F.pad(image, (0, 0, 2, 2), mode="reflect"), vertical,
                    groups=channels)


def frequency_grid(height, width, device):
    fy = torch.fft.fftfreq(height, device=device) * height
    fx = torch.fft.rfftfreq(width, device=device) * width
    return (fy[:, None].square() + fx[None, :].square()).sqrt()


@torch.no_grad()
def geometry_targets(image):
    gray = (
        0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
    )[:, None]
    dx = gray[..., :, 1:] - gray[..., :, :-1]
    dy = gray[..., 1:, :] - gray[..., :-1, :]
    gradient_power = 0.5 * (
        dx.square().flatten(1).mean(1) + dy.square().flatten(1).mean(1)
    )
    laplacian = depthwise_filter(
        gray,
        torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]),
    )
    laplacian_power = laplacian.square().flatten(1).mean(1)
    spectrum = torch.fft.rfft2(gray, norm="ortho").abs().square()[:, 0]
    radius = frequency_grid(gray.shape[-2], gray.shape[-1], gray.device)
    bands = ((0, 4), (4, 16), (16, float("inf")))
    band_powers = []
    for low, high in bands:
        selected = spectrum[:, (radius >= low) & (radius < high)]
        band_powers.append(selected.mean(1))
    non_dc = radius > 0
    centroid = (
        spectrum[:, non_dc].mul(radius[non_dc]).sum(1)
        / spectrum[:, non_dc].sum(1).clamp_min(1e-12)
        / math.hypot(gray.shape[-2] / 2, gray.shape[-1] / 2)
    )
    eps = 1e-10
    return torch.stack((
        gray.flatten(1).mean(1),
        gray.flatten(1).std(1),
        gradient_power.add(eps).log10(),
        laplacian_power.add(eps).log10(),
        band_powers[0].add(eps).log10(),
        band_powers[1].add(eps).log10(),
        band_powers[2].add(eps).log10(),
        centroid,
    ), dim=1)


def sinusoid(image, sample_ids, radius, amplitude):
    batch, _, height, width = image.shape
    y = torch.arange(height, device=image.device, dtype=image.dtype)[:, None]
    x = torch.arange(width, device=image.device, dtype=image.dtype)[None, :]
    angles = (sample_ids % 8).to(image.dtype) * (math.pi / 8)
    fx = radius * angles.cos()
    fy = radius * angles.sin()
    phase = ((sample_ids * 2654435761) % 1009).to(image.dtype) / 1009 * 2 * math.pi
    argument = (
        2 * math.pi * (
            fx[:, None, None] * x[None] / width
            + fy[:, None, None] * y[None] / height
        ) + phase[:, None, None]
    )
    return (image + amplitude * argument.sin()[:, None]).clamp(0, 1)


@torch.no_grad()
def controlled_images(image, labels, sample_ids):
    output = image.clone()
    blurred = gaussian_blur(image)
    for label in range(len(CONTROL_NAMES)):
        mask = labels == label
        if not mask.any() or label == 0:
            continue
        if label == 1:
            output[mask] = blurred[mask]
        elif label == 2:
            output[mask] = (image[mask] + 1.5 * (image[mask] - blurred[mask])).clamp(0, 1)
        elif label == 3:
            output[mask] = sinusoid(image[mask], sample_ids[mask], 2.0, 8 / 255)
        elif label == 4:
            output[mask] = sinusoid(image[mask], sample_ids[mask], 10.0, 6 / 255)
        elif label == 5:
            output[mask] = sinusoid(image[mask], sample_ids[mask], 24.0, 4 / 255)
        elif label == 6:
            generators = [
                torch.Generator(device=image.device).manual_seed(int(index) + 9173)
                for index in sample_ids[mask]
            ]
            noise = torch.stack([
                torch.randn(image.shape[1:], generator=generator,
                            device=image.device, dtype=image.dtype)
                for generator in generators
            ])
            output[mask] = (image[mask] + 5 / 255 * noise).clamp(0, 1)
    return output


def split_from_identity(identity, seed):
    bucket = (identity.long() * 1103515245 + seed * 12345).remainder(100)
    split = torch.full_like(bucket, 2, dtype=torch.uint8)
    split[bucket < 85] = 1
    split[bucket < 70] = 0
    return split


def image_tensor(pil_image):
    image = TF.center_crop(TF.resize(pil_image.convert("RGB"), 128), [128, 128])
    return TF.pil_to_tensor(image).float().div(255)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument(
        "--autoencoder-type", choices=("spatial", "vector"), default="spatial"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="flwrlabs/celeba")
    parser.add_argument(
        "--local-parquet-dir",
        help="Load train-*.parquet directly from this local directory instead of "
             "downloading flwrlabs/celeba through the Hub (workaround for a cluster "
             "network path where huggingface_hub's own download code hangs even "
             "though the files are reachable via plain curl).",
    )
    parser.add_argument("--max-images", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=260818)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = (
        load_spatial_autoencoder
        if args.autoencoder_type == "spatial"
        else load_vector_autoencoder
    )
    model = loader(args.autoencoder, device)
    if args.local_parquet_dir:
        import glob
        files = sorted(glob.glob(os.path.join(args.local_parquet_dir, "train-*.parquet")))
        dataset = load_dataset("parquet", data_files=files, split="train")
    else:
        dataset = load_dataset(args.dataset, split="train", streaming=False)
    if args.max_images < len(dataset):
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_images))
    clean_latents, controlled_latents, attributes = [], [], []
    identities, geometry, source_geometry, control_labels = [], [], [], []
    for start in range(0, len(dataset), args.batch_size):
        rows = dataset[start:start + args.batch_size]
        images = torch.stack([image_tensor(image) for image in rows["image"]]).to(device)
        sample_ids = torch.arange(start, start + len(images), device=device)
        labels = (sample_ids + args.seed).remainder(len(CONTROL_NAMES)).long()
        transformed = controlled_images(images, labels, sample_ids)
        normalized = images.mul(2).sub(1)
        transformed_normalized = transformed.mul(2).sub(1)
        with torch.no_grad():
            clean_latent = model.encode(normalized)
            clean_latents.append(clean_latent.half().cpu())
            controlled_latents.append(model.encode(transformed_normalized).half().cpu())
            reconstructed = model.decode(clean_latent).add(1).div(2).clamp(0, 1)
            geometry.append(geometry_targets(reconstructed).float().cpu())
            source_geometry.append(geometry_targets(images).float().cpu())
        attributes.append(torch.tensor([
            [bool(rows[name][index]) for name in ATTRIBUTE_NAMES]
            for index in range(len(images))
        ], dtype=torch.uint8))
        identities.append(torch.tensor(rows["celeb_id"], dtype=torch.int32))
        control_labels.append(labels.to(torch.uint8).cpu())
        if start == 0 or (start + len(images)) % 2048 == 0:
            print(f"encoded {start + len(images)} / {len(dataset)}", flush=True)

    identities = torch.cat(identities)
    payload = {
        "clean_latents": torch.cat(clean_latents),
        "controlled_latents": torch.cat(controlled_latents),
        "attributes": torch.cat(attributes),
        "attribute_names": ATTRIBUTE_NAMES,
        "identities": identities,
        "split": split_from_identity(identities, args.seed),
        "split_names": ("train", "validation", "test"),
        "geometry_targets": torch.cat(geometry),
        "source_geometry_targets": torch.cat(source_geometry),
        "geometry_names": GEOMETRY_NAMES,
        "control_labels": torch.cat(control_labels),
        "control_names": CONTROL_NAMES,
        "dataset": args.dataset,
        "seed": args.seed,
        "autoencoder_checkpoint": args.autoencoder,
        "autoencoder_type": args.autoencoder_type,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    summary = {
        "examples": len(identities),
        "split_counts": {
            name: int((payload["split"] == index).sum())
            for index, name in enumerate(payload["split_names"])
        },
        "control_counts": {
            name: int((payload["control_labels"] == index).sum())
            for index, name in enumerate(CONTROL_NAMES)
        },
        "attribute_prevalence": {
            name: float(payload["attributes"][:, index].float().mean())
            for index, name in enumerate(ATTRIBUTE_NAMES)
        },
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
