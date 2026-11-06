#!/usr/bin/env python3
"""Train matched grouped TopK SAEs on full spatial Conv-AE latent maps."""

import argparse
import json
import os
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from spatial_conv_autoencoder import load_spatial_autoencoder
from train_grouped_objective_sae import (
    GROUPED_PURE_SPECS,
    GROUPED_SPECS,
    GroupSpec,
    PartitionedTopKSAE,
    append_jsonl,
    build_run_manifest,
    group_parameter_gradient_norms,
    model_payload,
    rescue_group,
    routed_gradient_proxy,
    write_diagnostics,
)
from train_spatial_map_sae import (
    calibrate,
    transform_objective_name,
    transformed_loss,
)
from train_zsem_transformed_sae import component_power, transform_components


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("regular", "grouped", "grouped_pure"), required=True
    )
    parser.add_argument("--data", required=True, help="Full Conv-AE latent cache")
    parser.add_argument("--checkpoint", required=True, help="Spatial Conv-AE checkpoint")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--transform-weight", type=float, default=1.0)
    parser.add_argument(
        "--loss-normalization", choices=("none", "power"), default="none"
    )
    parser.add_argument(
        "--lambda-mode",
        choices=("dynamic_gradient", "initial_gradient", "fixed"),
        default="dynamic_gradient",
    )
    parser.add_argument("--gradient-target-ratio", type=float, default=1.0)
    parser.add_argument("--regular-dict-size", type=int, default=16_384)
    parser.add_argument("--regular-k", type=int, default=64)
    parser.add_argument("--grouped-mse-dict-size", type=int, default=8_192)
    parser.add_argument("--grouped-mse-k", type=int, default=32)
    parser.add_argument("--grouped-specialist-dict-size", type=int, default=2_048)
    parser.add_argument("--grouped-specialist-k", type=int, default=8)
    parser.add_argument(
        "--parameter-gradient-cap",
        choices=("none", "initial_mse"),
        default="initial_mse",
    )
    parser.add_argument("--calibration-examples", type=int, default=2_048)
    parser.add_argument("--aux-coeff", type=float, default=0.1)
    parser.add_argument("--reinit-every", type=int, default=1_000)
    parser.add_argument("--dead-after", type=int, default=3_000)
    parser.add_argument("--max-rescue-per-group", type=int, default=64)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--state-every", type=int, default=500)
    parser.add_argument("--trace-flush-every", type=int, default=100)
    parser.add_argument("--skip-file-hashes", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=2026)
    return parser.parse_args()


def raw_transformed_loss(objective, error):
    losses = torch.stack([
        component_power(component)
        for component in transform_components(objective, error)
    ], dim=1)
    return losses.mean(1)


def grouped_specs(args, pure=False):
    specialists = (
        ("laplacian", "laplacian"),
        ("fourier_low", "fourier_low"),
        ("fourier_mid", "fourier_mid"),
        ("fourier_high", "fourier_high"),
    )
    if not pure:
        specialists = tuple(
            (name, f"weighted_{objective}")
            for name, objective in specialists
        )
    return (
        GroupSpec(
            "mse", "latent_mse",
            args.grouped_mse_dict_size, args.grouped_mse_k,
        ),
        *(
            GroupSpec(
                name, objective,
                args.grouped_specialist_dict_size,
                args.grouped_specialist_k,
            )
            for name, objective in specialists
        ),
    )


def spatial_geometry_losses(autoencoder, raw, reconstructed_raw, powers=None):
    with torch.no_grad():
        clean_image = autoencoder.decode(raw).add(1).div(2)
    reconstructed_image = autoencoder.decode(reconstructed_raw).add(1).div(2)
    error = clean_image - reconstructed_image
    objectives = ("laplacian", "fourier_low", "fourier_mid", "fourier_high")
    if powers:
        losses = {
            name: transformed_loss(name, error, powers[name])
            for name in objectives
        }
    else:
        losses = {name: raw_transformed_loss(name, error) for name in objectives}
    pixel_mse = error.square().flatten(1).mean(1)
    return losses, pixel_mse


def spatial_group_objective_losses(mode, latent_loss, transformed, lambdas):
    losses = {"mse": latent_loss}
    for name, geometry_loss in transformed.items():
        weighted = lambdas[name] * geometry_loss
        losses[name] = latent_loss + weighted if mode == "grouped" else weighted
    return losses


def calibrate_gradient_lambdas(
    sae,
    autoencoder,
    raw_latents,
    indices,
    mean,
    std,
    input_shape,
    batch_size,
    device,
    target_ratio,
    powers=None,
):
    """Match each raw transformed objective's output-gradient norm to MSE."""
    norms = {
        name: []
        for name in ("mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high")
    }
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        raw = raw_latents[selected].to(device).float()
        standardized = ((raw - mean) / std).flatten(1)
        with torch.no_grad():
            initial_reconstruction = sae(standardized)[0]
        reconstruction = initial_reconstruction.detach().requires_grad_(True)
        latent_loss = (standardized - reconstruction).square().mean(1)
        reconstructed_raw = reconstruction.view(-1, *input_shape) * std + mean
        transformed, _ = spatial_geometry_losses(
            autoencoder, raw, reconstructed_raw, powers
        )
        mse_gradient = torch.autograd.grad(
            latent_loss.mean(), reconstruction, retain_graph=True
        )[0]
        norms["mse"].append(float(mse_gradient.norm()))
        for name, loss in transformed.items():
            gradient = torch.autograd.grad(
                loss.mean(), reconstruction, retain_graph=True
            )[0]
            norms[name].append(float(gradient.norm()))

    medians = {
        name: float(torch.tensor(values, dtype=torch.float64).median())
        for name, values in norms.items()
    }
    mse_median = medians["mse"]
    lambdas = {
        name: target_ratio * mse_median / max(medians[name], 1e-12)
        for name in medians
        if name != "mse"
    }
    return lambdas, {
        "method": "median_output_gradient_norm",
        "target_ratio": target_ratio,
        "batches": len(norms["mse"]),
        "examples": len(indices),
        "raw_gradient_norm_median": medians,
        "lambda_by_group": lambdas,
    }


@torch.no_grad()
def evaluate_latent(
    sae, raw_latents, indices, mean, std, batch_size, device, collect_used=False
):
    was_training = sae.training
    sae.eval()
    squared_error = 0.0
    count = 0
    used = {spec.name: set() for spec in sae.specs}
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        raw = raw_latents[selected].to(device).float()
        standardized = ((raw - mean) / std).flatten(1)
        reconstruction, groups = sae(standardized)
        squared_error += float((standardized - reconstruction).square().sum())
        count += standardized.numel()
        if collect_used:
            for spec in sae.specs:
                used[spec.name].update(
                    groups[spec.name]["indices"].unique().cpu().tolist()
                )
    sae.train(was_training)
    return squared_error / count, used


def half_archival_payload(payload):
    archived = dict(payload)
    archived["model"] = {
        name: (
            tensor.detach().cpu().half()
            if tensor.is_floating_point() else tensor.detach().cpu()
        )
        for name, tensor in payload["model"].items()
    }
    return archived


@torch.no_grad()
def cap_group_parameter_gradients(sae, target):
    """Cap each partition's actual parameter-gradient norm independently."""
    before = group_parameter_gradient_norms(sae)
    scales = {}
    for spec in sae.specs:
        norm = float(before[spec.name])
        scale = min(1.0, target / max(norm, 1e-12))
        start, end = sae.ranges[spec.name]
        for gradient in (
            sae.decoder.grad,
            sae.encoder.weight.grad,
            sae.encoder.bias.grad,
        ):
            if gradient is not None:
                gradient[start:end].mul_(scale)
        scales[spec.name] = scale
    return before, group_parameter_gradient_norms(sae), scales


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    trace_path = os.path.join(args.out_dir, "step_trace.jsonl")
    validation_path = os.path.join(args.out_dir, "validation_trace.jsonl")
    for path in (trace_path, validation_path):
        with open(path, "w", encoding="utf-8"):
            pass

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = build_run_manifest(args, extra_source_files=(__file__,))
    with open(os.path.join(args.out_dir, "run_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)

    cache = torch.load(args.data, map_location="cpu", weights_only=False)
    raw_latents = cache["latents"]
    train_indices = cache["train_indices"].long()
    valid_indices = cache["valid_indices"].long()
    input_shape = tuple(raw_latents.shape[1:])
    input_dim = int(torch.tensor(input_shape).prod())
    train_latents = raw_latents[train_indices].float()
    mean = train_latents.mean(0)
    std = train_latents.std(0).clamp_min(1e-6)
    del train_latents
    device_mean = mean.to(device)
    device_std = std.to(device)
    valid_standardized = (
        (raw_latents[valid_indices].float() - mean) / std
    ).flatten(1)
    valid_variance = float(valid_standardized.var().clamp_min(1e-8))
    del valid_standardized

    specs_by_mode = {
        "regular": (
            GroupSpec(
                "mse", "latent_mse", args.regular_dict_size, args.regular_k
            ),
        ),
        "grouped": grouped_specs(args),
        "grouped_pure": grouped_specs(args, pure=True),
    }
    specs = specs_by_mode[args.mode]
    autoencoder = None
    powers = {}
    if args.mode != "regular":
        autoencoder = load_spatial_autoencoder(args.checkpoint, device)
        calibration_indices = train_indices[
            :min(len(train_indices), args.calibration_examples)
        ]
        if args.loss_normalization == "power":
            for spec in specs[1:]:
                objective = transform_objective_name(spec.objective)
                powers[objective] = calibrate(
                    autoencoder,
                    objective,
                    raw_latents,
                    calibration_indices,
                    args.batch_size,
                    device,
                )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    sae = PartitionedTopKSAE(input_dim, specs).to(device)
    objective_lambdas = {}
    gradient_calibration = None
    if args.mode != "regular":
        if args.lambda_mode == "initial_gradient":
            objective_lambdas, gradient_calibration = calibrate_gradient_lambdas(
                sae,
                autoencoder,
                raw_latents,
                calibration_indices,
                device_mean,
                device_std,
                input_shape,
                args.batch_size,
                device,
                args.gradient_target_ratio,
                powers or None,
            )
        elif args.lambda_mode == "fixed":
            objective_lambdas = {
                spec.name: args.transform_weight for spec in specs[1:]
            }
            gradient_calibration = {
                "method": "fixed",
                "lambda_by_group": objective_lambdas,
            }
        else:
            objective_lambdas = {spec.name: 1.0 for spec in specs[1:]}
            gradient_calibration = {
                "method": "per_batch_output_gradient_norm",
                "target_ratio": args.gradient_target_ratio,
                "loss_normalization": args.loss_normalization,
            }
        with open(
            os.path.join(args.out_dir, "gradient_calibration.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(gradient_calibration, handle, indent=2)
        print({"gradient_calibration": gradient_calibration}, flush=True)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
    )
    iterator = iter(loader)
    feature_counts = torch.zeros(sae.dict_size, dtype=torch.long, device=device)
    last_fire = torch.zeros_like(feature_counts)
    rescued_by_group = {spec.name: 0 for spec in specs}
    history = []
    validation_history = []
    trace_buffer = []
    gradient_norm_history = []
    clip_count = 0
    clip_streak = 0
    longest_clip_streak = 0
    dominant_clip_counts = {spec.name: 0 for spec in specs}
    lambda_history = {spec.name: [] for spec in specs[1:]}
    parameter_gradient_target = None
    best_valid_ev = float("-inf")
    started_at = time.time()

    for step in range(1, args.steps + 1):
        try:
            indices = next(iterator)[0]
        except StopIteration:
            iterator = iter(loader)
            indices = next(iterator)[0]
        raw = raw_latents[indices].to(device).float()
        standardized = ((raw - device_mean) / device_std).flatten(1)
        reconstruction, groups = sae(standardized)
        latent_per_example = (standardized - reconstruction).square().mean(1)
        per_group_losses = {"mse": latent_per_example}
        transformed = {}
        pixel_mse = torch.full_like(latent_per_example, float("nan"))
        if args.mode != "regular":
            reconstructed_raw = (
                reconstruction.view(-1, *input_shape) * device_std + device_mean
            )
            transformed, pixel_mse = spatial_geometry_losses(
                autoencoder, raw, reconstructed_raw, powers or None
            )

        residual = (standardized - reconstruction).detach()
        aux_loss = torch.stack([
            F.mse_loss(groups[spec.name]["aux_contribution"], residual)
            for spec in specs
        ]).mean()
        write_norms, cancellation_ratio, pairwise_cosines = write_diagnostics(
            specs, groups
        )
        optimizer.zero_grad(set_to_none=True)
        if args.mode == "regular":
            routed_gradients = {
                "mse": 2 * (reconstruction - standardized) / reconstruction.numel()
            }
            training_loss = latent_per_example.mean() + args.aux_coeff * aux_loss
            training_loss.backward()
        else:
            mse_gradient = torch.autograd.grad(
                latent_per_example.mean(), reconstruction, retain_graph=True
            )[0]
            geometry_gradients = {
                name: torch.autograd.grad(
                    loss.mean(), reconstruction, retain_graph=True
                )[0]
                for name, loss in transformed.items()
            }
            if args.lambda_mode == "dynamic_gradient":
                mse_norm = mse_gradient.detach().norm()
                objective_lambdas = {
                    name: float(
                        args.gradient_target_ratio
                        * mse_norm
                        / gradient.detach().norm().clamp_min(1e-12)
                    )
                    for name, gradient in geometry_gradients.items()
                }
            per_group_losses = spatial_group_objective_losses(
                args.mode,
                latent_per_example,
                transformed,
                objective_lambdas,
            )
            routed_gradients = {"mse": mse_gradient}
            for name, gradient in geometry_gradients.items():
                scaled = objective_lambdas[name] * gradient
                routed_gradients[name] = (
                    mse_gradient + scaled if args.mode == "grouped" else scaled
                )
                lambda_history[name].append(objective_lambdas[name])
            proxy = routed_gradient_proxy(sae, groups, routed_gradients)
            training_loss = (
                torch.stack([
                    loss.mean() for loss in per_group_losses.values()
                ]).mean()
                + args.aux_coeff * aux_loss.detach()
            )
            (proxy + args.aux_coeff * aux_loss).backward()

        raw_parameter_gradient_norms = group_parameter_gradient_norms(sae)
        parameter_gradient_scales = {spec.name: 1.0 for spec in specs}
        parameter_gradient_norms = raw_parameter_gradient_norms
        if args.parameter_gradient_cap == "initial_mse":
            if parameter_gradient_target is None:
                parameter_gradient_target = float(raw_parameter_gradient_norms["mse"])
            (
                raw_parameter_gradient_norms,
                parameter_gradient_norms,
                parameter_gradient_scales,
            ) = cap_group_parameter_gradients(sae, parameter_gradient_target)
        gradient_norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip)
        gradient_norm_value = float(gradient_norm)
        clipped = gradient_norm_value > args.grad_clip
        if clipped:
            clip_count += 1
            clip_streak += 1
            longest_clip_streak = max(longest_clip_streak, clip_streak)
        else:
            clip_streak = 0
        dominant_gradient_group = max(
            raw_parameter_gradient_norms,
            key=lambda name: float(raw_parameter_gradient_norms[name]),
        )
        if clipped:
            dominant_clip_counts[dominant_gradient_group] += 1
        gradient_norm_history.append(gradient_norm_value)

        if args.warmup and step <= args.warmup:
            optimizer.param_groups[0]["lr"] = args.lr * step / args.warmup
        optimizer.step()
        sae.normalize_decoder()

        for spec in specs:
            values = groups[spec.name]["values"]
            selected = groups[spec.name]["indices"]
            active = values > 0
            feature_counts.scatter_add_(
                0,
                selected[active],
                torch.ones_like(selected[active], dtype=torch.long),
            )
            last_fire[selected[active].unique()] = step

        rescued_step = {}
        if args.reinit_every and step % args.reinit_every == 0 and step < args.steps:
            for spec in specs:
                rescued = rescue_group(
                    sae,
                    spec,
                    standardized,
                    reconstruction.detach(),
                    per_group_losses[spec.name].detach(),
                    last_fire,
                    step,
                    args.dead_after,
                    args.max_rescue_per_group,
                )
                rescued_step[spec.name] = rescued
                rescued_by_group[spec.name] += rescued

        trace_record = {
            "step": step,
            "elapsed_seconds": time.time() - started_at,
            "loss_by_group_mean": {
                name: float(loss.mean().detach())
                for name, loss in per_group_losses.items()
            },
            "loss_by_group_per_example": {
                name: loss.detach().cpu().tolist()
                for name, loss in per_group_losses.items()
            },
            "transformed_loss_per_example": {
                name: loss.detach().cpu().tolist()
                for name, loss in transformed.items()
            },
            "latent_mse_per_example": latent_per_example.detach().cpu().tolist(),
            "pixel_mse_per_example": (
                None if args.mode == "regular" else pixel_mse.detach().cpu().tolist()
            ),
            "aux_mse": float(aux_loss.detach()),
            "objective_lambda_by_group": objective_lambdas,
            "routed_gradient_norm_by_group": {
                name: float(gradient.detach().norm())
                for name, gradient in routed_gradients.items()
            },
            "parameter_gradient_norm_by_group": {
                name: float(value) for name, value in parameter_gradient_norms.items()
            },
            "raw_parameter_gradient_norm_by_group": {
                name: float(value)
                for name, value in raw_parameter_gradient_norms.items()
            },
            "parameter_gradient_scale_by_group": parameter_gradient_scales,
            "parameter_gradient_target": parameter_gradient_target,
            "dominant_gradient_group": dominant_gradient_group,
            "gradient_norm_before_clip": gradient_norm_value,
            "clipped": clipped,
            "clip_count": clip_count,
            "clip_fraction": clip_count / step,
            "current_clip_streak": clip_streak,
            "longest_clip_streak": longest_clip_streak,
            "write_norm_mean_by_group": {
                spec.name: float(write_norms[:, position].mean())
                for position, spec in enumerate(specs)
            },
            "cancellation_ratio_mean": float(cancellation_ratio.mean()),
            "cancellation_ratio_per_example": cancellation_ratio.cpu().tolist(),
            "pairwise_write_cosine_mean": {
                name: float(values.mean())
                for name, values in pairwise_cosines.items()
            },
            "active_count_mean_by_group": {
                spec.name: float(
                    (groups[spec.name]["values"] > 0).sum(1).float().mean()
                )
                for spec in specs
            },
            "active_magnitude_mean_by_group": {
                spec.name: float(
                    (
                        groups[spec.name]["values"]
                        * (groups[spec.name]["values"] > 0)
                    ).sum()
                    / (groups[spec.name]["values"] > 0).sum().clamp_min(1)
                )
                for spec in specs
            },
            "rescued": rescued_step,
        }
        if step == 1 or step % 100 == 0:
            trace_record["parameter_statistics"] = {
                spec.name: {
                    "encoder_weight_norm_mean": float(
                        sae.encoder.weight[slice(*sae.ranges[spec.name])]
                        .norm(dim=1).mean()
                    ),
                    "encoder_bias_abs_mean": float(
                        sae.encoder.bias[slice(*sae.ranges[spec.name])].abs().mean()
                    ),
                    "encoder_bias_abs_max": float(
                        sae.encoder.bias[slice(*sae.ranges[spec.name])].abs().max()
                    ),
                }
                for spec in specs
            }
        trace_buffer.append(trace_record)
        if step % args.trace_flush_every == 0 or step == args.steps:
            append_jsonl(trace_path, trace_buffer)
            trace_buffer.clear()

        if step == 1 or step % args.validate_every == 0 or step == args.steps:
            valid_mse, _ = evaluate_latent(
                sae,
                raw_latents,
                valid_indices,
                device_mean,
                device_std,
                args.batch_size,
                device,
            )
            valid_ev = 1.0 - valid_mse / valid_variance
            validation_record = {
                "step": step,
                "valid_mse": valid_mse,
                "valid_ev": valid_ev,
                "elapsed_seconds": time.time() - started_at,
                "clip_count": clip_count,
                "clip_fraction": clip_count / step,
                "longest_clip_streak": longest_clip_streak,
            }
            validation_history.append(validation_record)
            append_jsonl(validation_path, [validation_record])
            if valid_ev > best_valid_ev:
                best_valid_ev = valid_ev
                best = model_payload(
                    sae, args, mean, std, step, feature_counts, last_fire
                )
                best["input_shape"] = input_shape
                best["objective_lambda_by_group"] = objective_lambdas
                best["gradient_calibration"] = gradient_calibration
                best["parameter_gradient_target"] = parameter_gradient_target
                best["validation"] = validation_record
                torch.save(best, os.path.join(args.out_dir, "best_valid.pt"))
            print({"validation": validation_record}, flush=True)

        if step % args.checkpoint_every == 0 or step == args.steps:
            archived = model_payload(
                sae, args, mean, std, step, feature_counts, last_fire
            )
            archived["input_shape"] = input_shape
            archived["objective_lambda_by_group"] = objective_lambdas
            archived["gradient_calibration"] = gradient_calibration
            archived["parameter_gradient_target"] = parameter_gradient_target
            torch.save(
                half_archival_payload(archived),
                os.path.join(checkpoint_dir, f"step_{step:06d}.pt"),
            )
        if step % args.state_every == 0 or step == args.steps:
            state = model_payload(
                sae, args, mean, std, step, feature_counts, last_fire
            )
            state.update({
                "input_shape": input_shape,
                "objective_lambda_by_group": objective_lambdas,
                "gradient_calibration": gradient_calibration,
                "parameter_gradient_target": parameter_gradient_target,
                "optimizer": optimizer.state_dict(),
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
                "loader_generator_state": loader_generator.get_state(),
                "clip_count": clip_count,
                "clip_streak": clip_streak,
                "longest_clip_streak": longest_clip_streak,
                "best_valid_ev": best_valid_ev,
            })
            torch.save(state, os.path.join(args.out_dir, "latest_training_state.pt"))

        if step == 1 or step % 100 == 0:
            record = {
                "step": step,
                "training_objective_mean": float(training_loss.detach()),
                "latent_mse": float(latent_per_example.mean().detach()),
                "pixel_mse": (
                    None if args.mode == "regular" else float(pixel_mse.mean().detach())
                ),
                "aux_mse": float(aux_loss.detach()),
                "gradient_norm_before_clip": gradient_norm_value,
                "clipped": clipped,
                "clip_count": clip_count,
                "clip_fraction": clip_count / step,
                "longest_clip_streak": longest_clip_streak,
                "dominant_gradient_group": dominant_gradient_group,
                "cancellation_ratio": float(cancellation_ratio.mean()),
                "used_by_group": {
                    spec.name: int(
                        (feature_counts[slice(*sae.ranges[spec.name])] > 0).sum()
                    )
                    for spec in specs
                },
                "rescued": rescued_step,
            }
            history.append(record)
            print(record, flush=True)

    final_valid_mse, valid_used = evaluate_latent(
        sae,
        raw_latents,
        valid_indices,
        device_mean,
        device_std,
        args.batch_size,
        device,
        collect_used=True,
    )
    final_valid_ev = 1.0 - final_valid_mse / valid_variance
    gradient_norms = torch.tensor(gradient_norm_history, dtype=torch.float64)
    lambda_summary = {}
    for name, values in lambda_history.items():
        tensor = torch.tensor(values, dtype=torch.float64)
        if len(tensor):
            lambda_summary[name] = {
                "median": float(tensor.median()),
                "p05": float(torch.quantile(tensor, 0.05)),
                "p95": float(torch.quantile(tensor, 0.95)),
                "final": float(tensor[-1]),
            }
    result = {
        "backend": "spatial_conv_autoencoder",
        "uses_latent_residuals": False,
        "mode": args.mode,
        "examples": len(raw_latents),
        "train_examples": len(train_indices),
        "valid_examples": len(valid_indices),
        "input_shape": input_shape,
        "input_dim": input_dim,
        "dict_size": sae.dict_size,
        "total_k": sae.total_k,
        "groups": [asdict(spec) for spec in specs],
        "valid_mse": final_valid_mse,
        "valid_ev": final_valid_ev,
        "valid_variance": valid_variance,
        "best_valid_ev": best_valid_ev,
        "valid_used_by_group": {
            name: len(indices) for name, indices in valid_used.items()
        },
        "rescued_by_group": rescued_by_group,
        "clip_summary": {
            "count": clip_count,
            "fraction": clip_count / args.steps,
            "longest_streak": longest_clip_streak,
            "dominant_group_counts": dominant_clip_counts,
            "gradient_norm_median": float(gradient_norms.median()),
            "gradient_norm_p95": float(torch.quantile(gradient_norms, 0.95)),
            "gradient_norm_max": float(gradient_norms.max()),
        },
        "component_powers": {
            name: values.cpu().tolist() for name, values in powers.items()
        },
        "objective_lambda_by_group": objective_lambdas,
        "objective_lambda_summary": lambda_summary,
        "gradient_calibration": gradient_calibration,
        "parameter_gradient_target": parameter_gradient_target,
        "validation_history": validation_history,
        "fixed_decoder_bias": True,
        "config": vars(args),
        "history": history,
    }
    last = model_payload(sae, args, mean, std, args.steps, feature_counts, last_fire)
    last.update({
        "input_shape": input_shape,
        "objective_lambda_by_group": objective_lambdas,
        "gradient_calibration": gradient_calibration,
        "parameter_gradient_target": parameter_gradient_target,
        "metrics": result,
    })
    torch.save(last, os.path.join(args.out_dir, "last.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    with open(os.path.join(args.out_dir, "COMPLETED.json"), "w") as handle:
        json.dump({
            "step": args.steps,
            "valid_mse": final_valid_mse,
            "valid_ev": final_valid_ev,
            "best_valid_ev": best_valid_ev,
            "elapsed_seconds": time.time() - started_at,
        }, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
