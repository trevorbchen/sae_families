#!/usr/bin/env python3
import argparse
import os

import torch
from torchvision import utils

from train_dit_cifar import DiT, Diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/dit_s2_cifar10/last.pt")
    parser.add_argument("--out-dir", default="runs/sanity_samples")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learn-sigma", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = DiT(learn_sigma=args.learn_sigma).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model.eval()

    labels = torch.arange(args.num_samples, device=device) % 10
    diffusion = Diffusion(args.timesteps, device=device)
    samples = diffusion.p_sample_loop(
        model,
        (args.num_samples, 3, 32, 32),
        labels,
        args.learn_sigma,
        device,
    )
    samples = (samples.clamp(-1, 1) + 1) / 2

    grid_path = os.path.join(args.out_dir, f"grid_seed{args.seed}_n{args.num_samples}.png")
    utils.save_image(samples, grid_path, nrow=10)
    for i, image in enumerate(samples):
        utils.save_image(image, os.path.join(args.out_dir, f"sample_{i:03d}_class{int(labels[i])}.png"))

    print(f"Saved {args.num_samples} samples to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
