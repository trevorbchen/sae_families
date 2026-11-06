#!/usr/bin/env python3
"""Measure whether repeated Diff-AE z_sem extraction is stochastic."""

import argparse
import json
import os
import sys

import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


def summarize(z):
    ref = z[:1]
    delta = z - ref
    return {
        "shape": list(z.shape),
        "mean_abs_delta": float(delta.abs().mean()),
        "max_abs_delta": float(delta.abs().max()),
        "mean_sample_std": float(z.std(dim=0).mean()),
        "max_sample_std": float(z.std(dim=0).max()),
        "mean_sample_var": float(z.var(dim=0).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repeats", type=int, default=32)
    args = p.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo))
    from experiment import LitModel
    from templates import ffhq128_autoenc_130M

    conf = ffhq128_autoenc_130M()
    conf.pretrain = None
    conf.latent_infer_path = None
    lit = LitModel(conf)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    lit.load_state_dict(state["state_dict"], strict=False)
    lit.eval().cuda()

    transform = Compose([
        Resize((128, 128)), ToTensor(),
        Normalize([0.5] * 3, [0.5] * 3),
    ])
    image = transform(Image.open(args.image).convert("RGB"))[None].cuda()
    batch = image.repeat(args.repeats, 1, 1, 1)

    with torch.no_grad():
        z_eval = lit.ema_model.encoder(batch).float()
    result = {"eval_mode": summarize(z_eval)}

    # Stress test only: this is not how the released latent.pkl was created.
    lit.ema_model.encoder.train()
    with torch.no_grad():
        z_train = lit.ema_model.encoder(batch).float()
    result["train_mode_stress_test"] = summarize(z_train)
    result["sae_validation_mse_reference"] = 0.5237665772438049
    result["note"] = "latent.pkl uses eval mode and deterministic flip_prob=0 preprocessing"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
