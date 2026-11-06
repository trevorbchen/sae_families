#!/usr/bin/env python3
"""Train and cache a deterministic FFHQ convolutional autoencoder."""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

from spatial_conv_autoencoder import SpatialConvAutoencoder


def image_batch(images_uint8, indices, device, augment=False):
    images = images_uint8[indices].to(device, non_blocking=True).float().div(127.5).sub(1)
    if augment:
        flip = torch.rand(len(images), device=device) < 0.5
        images[flip] = images[flip].flip(-1)
    return images


@torch.no_grad()
def evaluate(model, images, indices, batch_size, device):
    l1_sum = mse_sum = 0.0
    count = 0
    for start in range(0, len(indices), batch_size):
        batch = image_batch(images, indices[start:start + batch_size], device)
        reconstruction = model(batch)
        l1_sum += float((batch - reconstruction).abs().flatten(1).mean(1).sum())
        mse_sum += float((batch - reconstruction).square().flatten(1).mean(1).sum())
        count += len(batch)
    mse = mse_sum / count
    return {"l1": l1_sum / count, "mse": mse, "psnr": 10 * math.log10(4.0 / mse)}


@torch.no_grad()
def cache_latents(model, images, batch_size, device):
    latents = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device).float().div(127.5).sub(1)
        latents.append(model.encode(batch).half().cpu())
    return torch.cat(latents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--latent-cache", required=True)
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2030)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.latent_cache), exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    images = data["images_uint8"]
    split_generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(images), generator=split_generator)
    split = int(0.9 * len(images))
    train_indices = permutation[:split]
    valid_indices = permutation[split:]
    loader_generator = torch.Generator().manual_seed(args.seed + 1)
    loader = DataLoader(
        TensorDataset(train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
    )
    iterator = iter(loader)

    model = SpatialConvAutoencoder(args.latent_channels, args.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_l1 = float("inf")
    eval_indices = valid_indices[:min(512, len(valid_indices))]

    for step in range(1, args.steps + 1):
        try:
            indices = next(iterator)[0]
        except StopIteration:
            iterator = iter(loader)
            indices = next(iterator)[0]
        batch = image_batch(images, indices, device, augment=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            reconstruction = model(batch)
            loss = F.l1_loss(reconstruction, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if args.warmup and step <= args.warmup:
            for group in optimizer.param_groups:
                group["lr"] = args.lr * step / args.warmup
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 100 == 0:
            record = {
                "step": step,
                "train_l1": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
            }
            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate(model, images, eval_indices, args.batch_size, device)
                record.update({f"valid_{key}": value for key, value in metrics.items()})
                if metrics["l1"] < best_l1:
                    best_l1 = metrics["l1"]
                    torch.save(
                        {"model": model.state_dict(), "config": vars(args), "step": step},
                        os.path.join(args.out_dir, "best.pt"),
                    )
            history.append(record)
            print(record, flush=True)

    best = torch.load(os.path.join(args.out_dir, "best.pt"), map_location="cpu",
                      weights_only=False)
    model.load_state_dict(best["model"])
    model.to(device).eval()
    final_metrics = evaluate(model, images, valid_indices, args.batch_size, device)
    checkpoint = {
        "model": model.state_dict(),
        "config": vars(args),
        "best_step": best["step"],
        "metrics": final_metrics,
        "history": history,
    }
    checkpoint_path = os.path.join(args.out_dir, "last.pt")
    torch.save(checkpoint, checkpoint_path)

    preview_indices = valid_indices[:16]
    with torch.no_grad():
        preview = image_batch(images, preview_indices, device)
        preview_reconstruction = model(preview)
    save_image(
        torch.cat((preview, preview_reconstruction)).add(1).div(2),
        os.path.join(args.out_dir, "validation_reconstructions.png"),
        nrow=8,
    )

    latents = cache_latents(model, images, args.batch_size, device)
    torch.save(
        {
            "latents": latents,
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "source_data": args.data,
            "autoencoder_checkpoint": checkpoint_path,
            "latent_shape": tuple(latents.shape[1:]),
            "seed": args.seed,
        },
        args.latent_cache,
    )
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump({"metrics": final_metrics, "config": vars(args)}, handle, indent=2)
    print(json.dumps(final_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
