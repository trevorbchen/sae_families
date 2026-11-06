#!/usr/bin/env python3
"""Compute the normalized target-latent variance for one arm's held-out
validation split, using any one checkpoint's channel_mean/channel_std
(identical across lambda/objective/seed within an arm -- verified before
writing this). This is the denominator `evaluate_lambda_sweep_model.py`
uses for `latent_explained_variance` (1 - residual_variance/target_variance),
computed once so training-time latent_mse_monitor curves can be converted
into an approximate explained-variance-over-steps curve without re-running
any model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="any one last.pt from this arm")
    parser.add_argument("--latent-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    spatial_axes = max(0, len(state["input_shape"]) - 1)
    stats_shape = (1, -1) + (1,) * spatial_axes
    mean = state["channel_mean"].float().view(stats_shape)
    std = state["channel_std"].float().view(stats_shape)

    cache = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
    latents = cache["latents"].float()
    valid_indices = cache["valid_indices"].long()
    target = (latents[valid_indices] - mean) / std

    target_sum = float(target.double().sum())
    target_square_sum = float(target.double().square().sum())
    count = target.numel()
    target_variance = target_square_sum / count - (target_sum / count) ** 2

    Path(args.output).write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "latent_cache": args.latent_cache,
        "validation_elements": count,
        "target_variance": target_variance,
    }, indent=2))
    print(f"target_variance={target_variance!r} -> {args.output}")


if __name__ == "__main__":
    main()
