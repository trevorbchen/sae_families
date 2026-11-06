#!/usr/bin/env python3
"""Vector-latent (512-d ConvAE z_sem) variant of train_independent_routed_union_sae.py.

Same five-branch routed-tandem SAE (one MSE branch + four specialist branches, branch-local
TopK, independent weights, uniform-mean output), but trained on the matched-512 *vector*
autoencoder latents (the same space and standardization as the vector_16x_k32 single-objective
SAEs) and with the same lambda weighting (loss = latent_mse + lambda * transformed). The original
trainer is hard-wired to 64x8x8 spatial latents, lambda = 1 and the spatial decoder, so its
checkpoints live in a different space from Figure 11. Everything else is imported unchanged.
"""

import argparse
from dataclasses import asdict
import json
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from vector_conv_autoencoder import load_vector_autoencoder
from train_spatial_map_sae import calibrate, rescue_dead, transformed_loss
from train_independent_routed_union_sae import (
    IndependentRoutedUnionSAE, archival_payload, build_specs,
    evaluate_latent, evaluate_targets,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objectives", required=True, help="Comma-separated subset")
    parser.add_argument("--all-mse-control", action="store_true")
    parser.add_argument("--data", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mse-dict-size", type=int, default=8192)
    parser.add_argument("--mse-k", type=int, default=32)
    parser.add_argument("--specialist-dict-size", type=int, default=8192)
    parser.add_argument("--specialist-k", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--calibration-examples", type=int, default=2048)
    parser.add_argument("--reinit-every", type=int, default=1000)
    parser.add_argument("--dead-after", type=int, default=3000)
    parser.add_argument("--max-rescue", type=int, default=64)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--transform-weight", type=float, default=10.0,
                        help="lambda on the transformed loss; 10.0 matches the vector_16x_k32 "
                             "single-objective arm (loss = latent_mse + lambda * transformed)")
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-valid-examples", type=int)
    args = parser.parse_args()

    requested_objectives, specs = build_specs(args)
    if sum(spec.feature_count for spec in specs) != (
        args.mse_dict_size + args.specialist_dict_size
    ):
        raise AssertionError("dictionary allocation changed total capacity")
    if sum(spec.k for spec in specs) != args.mse_k + args.specialist_k:
        raise AssertionError("TopK allocation changed total active budget")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.data, map_location="cpu", weights_only=False)
    raw_latents = cache["latents"]
    train_indices = cache["train_indices"].long()
    valid_indices = cache["valid_indices"].long()
    if args.max_train_examples:
        train_indices = train_indices[:args.max_train_examples]
    if args.max_valid_examples:
        valid_indices = valid_indices[:args.max_valid_examples]
    input_shape = tuple(raw_latents.shape[1:])
    train_latents = raw_latents[train_indices].float()
    if train_latents.ndim == 2:  # vector latents (N, 512): per-dimension stats, as in train_spatial_map_sae
        channel_mean = train_latents.mean(dim=0)
        channel_std = train_latents.std(dim=0).clamp_min(1e-6)
        mean = channel_mean.to(device)[None, :]
        std = channel_std.to(device)[None, :]
        host_mean, host_std = channel_mean[None, :], channel_std[None, :]
    else:
        channel_mean = train_latents.mean(dim=(0, 2, 3))
        channel_std = train_latents.std(dim=(0, 2, 3)).clamp_min(1e-6)
        mean = channel_mean.to(device)[None, :, None, None]
        std = channel_std.to(device)[None, :, None, None]
        host_mean, host_std = channel_mean[None, :, None, None], channel_std[None, :, None, None]
    del train_latents
    valid_standardized = (raw_latents[valid_indices].float() - host_mean) / host_std
    valid_variance = float(valid_standardized.var().clamp_min(1e-8))
    del valid_standardized

    autoencoder = load_vector_autoencoder(args.autoencoder, device)
    model = IndependentRoutedUnionSAE(input_shape, specs).to(device)
    branches = model.branches
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(train_indices), batch_size=args.batch_size, shuffle=True,
        drop_last=True, generator=torch.Generator().manual_seed(args.seed),
    )
    iterator = iter(loader)

    transformed_objectives = sorted({
        spec.objective for spec in specs if spec.objective != "latent_mse"
    })
    powers = {}
    calibration_indices = train_indices[:min(args.calibration_examples, len(train_indices))]
    for objective in transformed_objectives:
        powers[objective] = calibrate(
            autoencoder, objective, raw_latents, calibration_indices,
            args.batch_size, device,
        )

    feature_counts = {
        spec.name: torch.zeros(spec.feature_count, dtype=torch.long, device=device)
        for spec in specs
    }
    last_fire = {
        spec.name: torch.zeros(spec.feature_count, dtype=torch.long, device=device)
        for spec in specs
    }
    rescued_total = {spec.name: 0 for spec in specs}
    history = []
    validation_history = []

    for step in range(1, args.steps + 1):
        try:
            indices = next(iterator)[0]
        except StopIteration:
            iterator = iter(loader)
            indices = next(iterator)[0]
        raw = raw_latents[indices].to(device).float()
        standardized = (raw - mean) / std
        clean_image = None
        optimizer.zero_grad(set_to_none=True)
        step_records = {}
        branch_outputs = {}

        for spec in specs:
            branch = branches[spec.name]
            reconstruction, values, selected = branch(standardized)
            latent_mse = (standardized - reconstruction).square().flatten(1).mean(1)
            transformed = None
            if spec.objective == "latent_mse":
                loss = latent_mse.mean()
            else:
                if clean_image is None:
                    with torch.no_grad():
                        clean_image = autoencoder.decode(raw).add(1).div(2)
                reconstructed_raw = reconstruction * std + mean
                reconstructed_image = autoencoder.decode(reconstructed_raw).add(1).div(2)
                transformed = transformed_loss(
                    spec.objective,
                    clean_image - reconstructed_image,
                    powers[spec.objective],
                )
                loss = (latent_mse + args.transform_weight * transformed).mean()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                branch.parameters(), args.grad_clip
            )
            branch_outputs[spec.name] = (
                reconstruction.detach(), values.detach(), selected.detach()
            )
            step_records[spec.name] = {
                "loss": float(loss.detach()),
                "latent_mse": float(latent_mse.mean().detach()),
                "transformed_target": (
                    None if transformed is None else float(transformed.mean().detach())
                ),
                "gradient_norm_before_clip": float(gradient_norm),
            }

        if args.warmup and step <= args.warmup:
            for group in optimizer.param_groups:
                group["lr"] = args.lr * step / args.warmup
        optimizer.step()
        for branch in branches.values():
            branch.normalize_decoder()

        for spec in specs:
            reconstruction, values, selected = branch_outputs[spec.name]
            active = values > 0
            feature_counts[spec.name].scatter_add_(
                0, selected[active],
                torch.ones_like(selected[active], dtype=torch.long),
            )
            last_fire[spec.name][selected[active].unique()] = step
            rescued = 0
            if args.reinit_every and step % args.reinit_every == 0:
                rescued = rescue_dead(
                    branches[spec.name], standardized, reconstruction,
                    last_fire[spec.name], step, args.dead_after, args.max_rescue,
                )
                rescued_total[spec.name] += rescued
            step_records[spec.name]["rescued"] = rescued
            step_records[spec.name]["used_features"] = int(
                (feature_counts[spec.name] > 0).sum()
            )

        if step == 1 or step % 100 == 0:
            record = {"step": step, "branches": step_records}
            history.append(record)
            print(record, flush=True)
        if args.validate_every and (
            step == 1 or step % args.validate_every == 0 or step == args.steps
        ):
            validation = evaluate_latent(
                branches, specs, raw_latents, valid_indices, mean, std,
                args.batch_size, device, valid_variance,
            )
            validation["step"] = step
            validation_history.append(validation)
            print({"validation": validation}, flush=True)

    final_latent = evaluate_latent(
        branches, specs, raw_latents, valid_indices, mean, std,
        args.batch_size, device, valid_variance,
    )
    final_targets = evaluate_targets(
        branches, specs, powers, autoencoder, raw_latents, valid_indices,
        mean, std, args.batch_size, device,
    )
    metrics = {
        "requested_objectives": requested_objectives,
        "all_mse_control": args.all_mse_control,
        "specs": [asdict(spec) for spec in specs],
        "total_dictionary": sum(spec.feature_count for spec in specs),
        "total_k": sum(spec.k for spec in specs),
        "valid_variance": valid_variance,
        "final_latent": final_latent,
        "final_transformed_targets": final_targets,
        "used_features": {
            name: int((counts > 0).sum()) for name, counts in feature_counts.items()
        },
        "rescued_total": rescued_total,
        "component_powers": {
            name: value.cpu().tolist() for name, value in powers.items()
        },
        "history": history,
        "validation_history": validation_history,
        "config": vars(args),
    }
    torch.save(
        archival_payload(
            branches, specs, channel_mean, channel_std, input_shape,
            args, args.steps, metrics,
        ),
        os.path.join(args.out_dir, "last.pt"),
    )
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    with open(os.path.join(args.out_dir, "COMPLETED.json"), "w") as handle:
        json.dump({"step": args.steps, "union_ev": final_latent["union_ev"]}, handle, indent=2)
    print(json.dumps({
        "final_latent": final_latent,
        "final_transformed_targets": final_targets,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
