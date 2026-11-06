#!/usr/bin/env python3
"""Compare routed specialists with architecture-matched MSE-only branches."""

import argparse
import csv
import json
from pathlib import Path

import torch

from spatial_conv_autoencoder import load_spatial_autoencoder
from train_independent_routed_union_sae import load_independent_routed_union
from train_spatial_map_sae import transformed_loss


OBJECTIVES = ("laplacian", "fourier_low", "fourier_mid", "fourier_high")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routed", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    routed, routed_state = load_independent_routed_union(args.routed, device)
    control, control_state = load_independent_routed_union(args.control, device)
    if routed_state["input_shape"] != control_state["input_shape"]:
        raise ValueError("checkpoint input shapes differ")
    for key in ("channel_mean", "channel_std"):
        torch.testing.assert_close(routed_state[key], control_state[key])

    cache = torch.load(args.data, map_location="cpu", weights_only=False)
    raw_latents = cache["latents"]
    valid_indices = cache["valid_indices"].long()
    mean = routed_state["channel_mean"].float().to(device)[None, :, None, None]
    std = routed_state["channel_std"].float().to(device)[None, :, None, None]
    powers = {
        name: torch.tensor(
            routed_state["metrics"]["component_powers"][name], device=device
        )
        for name in OBJECTIVES
    }
    autoencoder = load_spatial_autoencoder(args.autoencoder, device)
    autoencoder.eval().requires_grad_(False)
    routed_sums = {name: 0.0 for name in OBJECTIVES}
    control_sums = {name: 0.0 for name in OBJECTIVES}
    count = 0

    with torch.no_grad():
        for start in range(0, len(valid_indices), args.batch_size):
            selected = valid_indices[start:start + args.batch_size]
            raw = raw_latents[selected].float().to(device)
            standardized = (raw - mean) / std
            clean = autoencoder.decode(raw).add(1).div(2)
            _, routed_outputs = routed(standardized)
            _, control_outputs = control(standardized)
            for index, objective in enumerate(OBJECTIVES, start=1):
                routed_raw = (
                    routed_outputs[objective]["reconstruction"] * std + mean
                )
                control_raw = (
                    control_outputs[f"control_{index}"]["reconstruction"] * std + mean
                )
                routed_image = autoencoder.decode(routed_raw).add(1).div(2)
                control_image = autoencoder.decode(control_raw).add(1).div(2)
                routed_sums[objective] += float(transformed_loss(
                    objective, clean - routed_image, powers[objective]
                ).sum())
                control_sums[objective] += float(transformed_loss(
                    objective, clean - control_image, powers[objective]
                ).sum())
            count += len(raw)

    rows = []
    for objective in OBJECTIVES:
        routed_loss = routed_sums[objective] / count
        control_loss = control_sums[objective] / count
        rows.append({
            "objective": objective,
            "routed_transformed_loss": routed_loss,
            "matched_mse_branch_transformed_loss": control_loss,
            "relative_reduction": 1 - routed_loss / control_loss,
        })
    result = {
        "valid_examples": count,
        "routed_checkpoint": args.routed,
        "control_checkpoint": args.control,
        "routed_union_ev": routed_state["metrics"]["final_latent"]["union_ev"],
        "control_union_ev": control_state["metrics"]["final_latent"]["union_ev"],
        "objective_comparisons": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "specialist_vs_mse_control.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "specialist_vs_mse_control.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
