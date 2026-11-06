#!/usr/bin/env python3
"""Encode a FairFace sample with the matched512 base autoencoders, as a
second out-of-distribution probe set -- unlike CelebA (celebrity/press
photography), FairFace is sourced from YFCC-100M Flickr with no celebrity
bias, so it tests whether a failure-mode finding is about content genre
(red-carpet/stage photography) rather than the objective itself.

Same preprocessing convention as prepare_celeba_spatial_probe_cache.py
(resize shorter side to 128, center-crop 128x128) so latents are directly
comparable; no face detection/alignment needed since FairFace images (config
"1.25") are already single-face, loosely-cropped portraits like CelebA/FFHQ.
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from torchvision.transforms import functional as TF

from spatial_conv_autoencoder import load_spatial_autoencoder
from vector_conv_autoencoder import load_vector_autoencoder


def to_tensor(pil_image):
    image = TF.center_crop(TF.resize(pil_image.convert("RGB"), 128), [128, 128])
    return TF.to_tensor(image)


@torch.no_grad()
def encode(images, autoencoder, batch_size, device):
    latents = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device).mul(2).sub(1)
        latents.append(autoencoder.encode(batch).half().cpu())
    return torch.cat(latents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-autoencoder", required=True)
    parser.add_argument("--spatial-autoencoder", required=True)
    parser.add_argument("--dataset", default="HuggingFaceM4/FairFace")
    parser.add_argument("--config", default="1.25")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-images", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out-vector", required=True)
    parser.add_argument("--out-spatial", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(args.dataset, args.config, split=args.split, streaming=False)
    n = min(args.max_images, len(dataset))
    print(f"encoding {n} of {len(dataset)} images from {args.dataset} ({args.config}, {args.split})", flush=True)

    images = torch.empty(n, 3, 128, 128)
    for i in range(n):
        images[i] = to_tensor(dataset[i]["image"])
        if (i + 1) % 1024 == 0:
            print(f"loaded {i + 1}/{n}", flush=True)

    vector_ae = load_vector_autoencoder(args.vector_autoencoder, device)
    vector_ae.eval().requires_grad_(False)
    spatial_ae = load_spatial_autoencoder(args.spatial_autoencoder, device)
    spatial_ae.eval().requires_grad_(False)

    vector_latents = encode(images, vector_ae, args.batch_size, device)
    print(f"vector latents: {vector_latents.shape}", flush=True)
    spatial_latents = encode(images, spatial_ae, args.batch_size, device)
    print(f"spatial latents: {spatial_latents.shape}", flush=True)

    Path(args.out_vector).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_spatial).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": vector_latents.float(), "dataset": args.dataset, "config": args.config,
                "split": args.split, "autoencoder_checkpoint": args.vector_autoencoder}, args.out_vector)
    torch.save({"latents": spatial_latents.float(), "dataset": args.dataset, "config": args.config,
                "split": args.split, "autoencoder_checkpoint": args.spatial_autoencoder}, args.out_spatial)
    print(f"wrote {args.out_vector} and {args.out_spatial}", flush=True)


if __name__ == "__main__":
    main()
