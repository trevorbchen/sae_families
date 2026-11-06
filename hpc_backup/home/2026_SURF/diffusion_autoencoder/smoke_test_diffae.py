#!/usr/bin/env python3
"""Load the official FFHQ-128 Diff-AE checkpoint and test z_sem reconstruction."""

import argparse
import os
import sys

import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.utils import save_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo))
    from experiment import LitModel
    from templates import ffhq128_autoenc_130M

    os.makedirs(args.out_dir, exist_ok=True)
    conf = ffhq128_autoenc_130M()
    conf.pretrain = None
    conf.latent_infer_path = None
    lit = LitModel(conf)
    # The official checkpoint contains Lightning metadata; it is trusted here.
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    lit.load_state_dict(state["state_dict"], strict=False)
    lit.eval().cuda()

    transform = Compose([
        Resize((128, 128)),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    image = transform(Image.open(args.image).convert("RGB"))[None].cuda()
    with torch.no_grad():
        z_sem = lit.encode(image)
        x_T = torch.randn_like(image)
        recon = lit.render(x_T, cond=z_sem, T=args.steps)

    torch.save({"z_sem": z_sem.cpu(), "x_T": x_T.cpu()},
               os.path.join(args.out_dir, "smoke_latents.pt"))
    save_image((image.cpu() + 1) / 2,
               os.path.join(args.out_dir, "input.png"))
    save_image(recon.cpu(), os.path.join(args.out_dir, "decode_random_xT.png"))
    print("z_sem shape:", tuple(z_sem.shape))
    print("input range:", float(image.min()), float(image.max()))
    print("reconstruction range:", float(recon.min()), float(recon.max()))


if __name__ == "__main__":
    main()
