#!/usr/bin/env python3
"""Encode an image cache with a frozen spatial autoencoder."""

import argparse
import os

import torch

from spatial_conv_autoencoder import load_spatial_autoencoder


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=2026)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = torch.load(
        args.data, map_location="cpu", weights_only=False
    )["images_uint8"]
    model = load_spatial_autoencoder(args.autoencoder, device)
    latents = []
    for start in range(0, len(images), args.batch_size):
        batch = (
            images[start:start + args.batch_size]
            .to(device)
            .float()
            .div(127.5)
            .sub(1)
        )
        latents.append(model.encode(batch).half().cpu())
        print(
            f"encoded {min(start + len(batch), len(images))}/{len(images)}",
            flush=True,
        )

    generator = torch.Generator().manual_seed(args.split_seed)
    permutation = torch.randperm(len(images), generator=generator)
    split = int(0.9 * len(images))
    state = {
        "latents": torch.cat(latents),
        "train_indices": permutation[:split],
        "valid_indices": permutation[split:],
        "source_data": args.data,
        "autoencoder_checkpoint": args.autoencoder,
        "latent_shape": tuple(latents[0].shape[1:]),
        "split_seed": args.split_seed,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(state, args.out)
    print({
        "out": args.out,
        "examples": len(images),
        "latent_shape": state["latent_shape"],
    })


if __name__ == "__main__":
    main()
