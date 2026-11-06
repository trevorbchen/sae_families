#!/usr/bin/env python3
"""Train one transformed-objective SAE on a frozen MSE-SAE residual."""

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
    GroupSpec,
    PartitionedTopKSAE,
    append_jsonl,
    rescue_group,
)
from train_grouped_spatial_ae_sae import spatial_geometry_losses


OBJECTIVES = ("mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=OBJECTIVES, required=True)
    parser.add_argument(
        "--loss-mode", choices=("anchored", "pure_transformed"),
        default="anchored",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--base-sae", required=True)
    parser.add_argument(
        "--prior-specialist", action="append", default=[],
        help="Frozen specialist checkpoint applied before the new specialist",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--feature-count", type=int, default=2_048)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-warmup", type=int, default=500)
    parser.add_argument("--lambda-warmup", type=int, default=2_000)
    parser.add_argument("--gradient-target-ratio", type=float, default=0.1)
    parser.add_argument("--calibration-examples", type=int, default=512)
    parser.add_argument("--mse-tolerance", type=float, default=0.02)
    parser.add_argument("--aux-coeff", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--reinit-every", type=int, default=1_000)
    parser.add_argument("--dead-after", type=int, default=3_000)
    parser.add_argument("--max-rescue", type=int, default=64)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_base(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    specs = tuple(GroupSpec(**record) for record in checkpoint["groups"])
    input_dim = checkpoint["model"]["decoder"].shape[1]
    base = PartitionedTopKSAE(input_dim, specs)
    base.load_state_dict(checkpoint["model"])
    base.eval().requires_grad_(False).to(device)
    return base, checkpoint


def load_frozen_specialists(paths, input_dim, device):
    specialists = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        specs = tuple(GroupSpec(**record) for record in checkpoint["groups"])
        specialist = PartitionedTopKSAE(input_dim, specs)
        specialist.load_state_dict(checkpoint["model"])
        specialist.eval().requires_grad_(False).to(device)
        specialists.append(specialist)
    return specialists


def frozen_prefix_reconstruction(base, prior_specialists, standardized):
    with torch.no_grad():
        reconstruction = base(standardized)[0]
        residual = standardized - reconstruction
        for prior in prior_specialists:
            delta = prior(residual)[0]
            reconstruction = reconstruction + delta
            residual = residual - delta
    return reconstruction, residual


def reconstruct(base, prior_specialists, specialist, standardized):
    base_reconstruction, residual = frozen_prefix_reconstruction(
        base, prior_specialists, standardized
    )
    delta, groups = specialist(residual)
    return base_reconstruction, residual, delta, groups


def calibrate_lambda(
    base,
    prior_specialists,
    specialist,
    autoencoder,
    raw_latents,
    indices,
    mean,
    std,
    input_shape,
    objective,
    batch_size,
    target_ratio,
    device,
):
    if objective == "mse":
        return 0.0, {
            "method": "mse_only_control",
            "target_ratio": 0.0,
            "examples": len(indices),
            "transform_weight": 0.0,
        }
    ratios = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        raw = raw_latents[selected].to(device).float()
        standardized = ((raw - mean) / std).flatten(1)
        with torch.no_grad():
            base_reconstruction, residual = frozen_prefix_reconstruction(
                base, prior_specialists, standardized
            )
            delta = specialist(residual)[0]
        combined = (base_reconstruction + delta).detach().requires_grad_(True)
        latent_loss = (standardized - combined).square().mean(1)
        reconstructed_raw = combined.view(-1, *input_shape) * std + mean
        transformed, _ = spatial_geometry_losses(
            autoencoder, raw, reconstructed_raw, powers=None
        )
        mse_gradient = torch.autograd.grad(
            latent_loss.sum(), combined, retain_graph=True
        )[0].flatten(1)
        transformed_gradient = torch.autograd.grad(
            transformed[objective].sum(), combined
        )[0].flatten(1)
        ratios.extend(
            (
                mse_gradient.norm(dim=1)
                / transformed_gradient.norm(dim=1).clamp_min(1e-12)
            ).cpu().tolist()
        )
    values = torch.tensor(ratios, dtype=torch.float64)
    weight = target_ratio * float(values.median())
    return weight, {
        "method": "fixed_initial_output_gradient_ratio",
        "target_ratio": target_ratio,
        "examples": len(indices),
        "mse_to_transformed_norm_ratio": {
            "median": float(values.median()),
            "p05": float(torch.quantile(values, 0.05)),
            "p95": float(torch.quantile(values, 0.95)),
        },
        "transform_weight": weight,
    }


@torch.no_grad()
def evaluate(
    base,
    prior_specialists,
    specialist,
    autoencoder,
    raw_latents,
    indices,
    mean,
    std,
    input_shape,
    objective,
    batch_size,
    device,
):
    base_mse_sum = latent_mse_sum = transformed_sum = pixel_sum = 0.0
    count = 0
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        raw = raw_latents[selected].to(device).float()
        standardized = ((raw - mean) / std).flatten(1)
        base_reconstruction, _, delta, _ = reconstruct(
            base, prior_specialists, specialist, standardized
        )
        combined = base_reconstruction + delta
        base_mse_sum += float(
            (standardized - base_reconstruction).square().mean(1).sum()
        )
        latent_mse_sum += float(
            (standardized - combined).square().mean(1).sum()
        )
        if objective != "mse":
            reconstructed_raw = combined.view(-1, *input_shape) * std + mean
            transformed, pixel_mse = spatial_geometry_losses(
                autoencoder, raw, reconstructed_raw, powers=None
            )
            transformed_sum += float(transformed[objective].sum())
            pixel_sum += float(pixel_mse.sum())
        count += len(selected)
    return {
        "base_mse": base_mse_sum / count,
        "latent_mse": latent_mse_sum / count,
        "transformed_loss": (
            None if objective == "mse" else transformed_sum / count
        ),
        "pixel_mse": None if objective == "mse" else pixel_sum / count,
    }


def payload(specialist, args, step, feature_counts, last_fire, calibration):
    return {
        "model": specialist.state_dict(),
        "groups": [asdict(spec) for spec in specialist.specs],
        "config": vars(args),
        "step": step,
        "feature_counts": feature_counts.cpu(),
        "last_fire": last_fire.cpu(),
        "gradient_calibration": calibration,
    }


def main():
    args = parse_args()
    if args.objective == "mse" and args.loss_mode != "anchored":
        raise ValueError("the MSE control only supports anchored loss")
    os.makedirs(args.out_dir, exist_ok=True)
    trace_path = os.path.join(args.out_dir, "training_trace.jsonl")
    validation_path = os.path.join(args.out_dir, "validation_trace.jsonl")
    for path in (trace_path, validation_path):
        open(path, "w").close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.data, map_location="cpu", weights_only=False)
    raw_latents = cache["latents"]
    train_indices = cache["train_indices"].long()
    valid_indices = cache["valid_indices"].long()
    base, base_checkpoint = load_base(args.base_sae, device)
    prior_specialists = load_frozen_specialists(
        args.prior_specialist, base.input_dim, device
    )
    mean = base_checkpoint["input_mean"].to(device)
    std = base_checkpoint["input_std"].to(device)
    input_shape = tuple(base_checkpoint["input_shape"])
    valid_variance = float(
        ((raw_latents[valid_indices].float() - mean.cpu()) / std.cpu())
        .flatten(1).var().clamp_min(1e-8)
    )
    autoencoder = load_spatial_autoencoder(args.autoencoder, device)
    autoencoder.requires_grad_(False)

    spec = GroupSpec("specialist", args.objective, args.feature_count, args.k)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    specialist = PartitionedTopKSAE(base.input_dim, (spec,)).to(device)
    calibration_indices = train_indices[:min(args.calibration_examples, len(train_indices))]
    transform_weight, calibration = calibrate_lambda(
        base,
        prior_specialists,
        specialist,
        autoencoder,
        raw_latents,
        calibration_indices,
        mean,
        std,
        input_shape,
        args.objective,
        args.batch_size,
        args.gradient_target_ratio,
        device,
    )
    with open(os.path.join(args.out_dir, "gradient_calibration.json"), "w") as handle:
        json.dump(calibration, handle, indent=2)
    print({"gradient_calibration": calibration}, flush=True)

    optimizer = torch.optim.Adam(specialist.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train_indices), batch_size=args.batch_size,
        shuffle=True, drop_last=True, generator=generator,
    )
    iterator = iter(loader)
    feature_counts = torch.zeros(args.feature_count, dtype=torch.long, device=device)
    last_fire = torch.zeros_like(feature_counts)
    best_feasible_selection_loss = float("inf")
    best_objective_loss = float("inf")
    best_valid_ev = float("-inf")
    started = time.time()

    for step in range(1, args.steps + 1):
        try:
            indices = next(iterator)[0]
        except StopIteration:
            iterator = iter(loader)
            indices = next(iterator)[0]
        raw = raw_latents[indices].to(device).float()
        standardized = ((raw - mean) / std).flatten(1)
        base_reconstruction, residual, delta, groups = reconstruct(
            base, prior_specialists, specialist, standardized
        )
        combined = base_reconstruction + delta
        latent_loss = (standardized - combined).square().mean(1)
        if args.objective == "mse":
            objective_loss = latent_loss.new_zeros(latent_loss.shape)
            pixel_mse = None
        else:
            reconstructed_raw = combined.view(-1, *input_shape) * std + mean
            transformed, pixel_mse = spatial_geometry_losses(
                autoencoder, raw, reconstructed_raw, powers=None
            )
            objective_loss = transformed[args.objective]
        lambda_fraction = min(1.0, step / max(args.lambda_warmup, 1))
        live_weight = transform_weight * lambda_fraction
        aux_loss = F.mse_loss(
            groups["specialist"]["aux_contribution"],
            (residual - delta).detach(),
        )
        transformed_term = live_weight * objective_loss.mean()
        primary_loss = (
            transformed_term
            if args.loss_mode == "pure_transformed"
            else latent_loss.mean() + transformed_term
        )
        loss = primary_loss + args.aux_coeff * aux_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            specialist.parameters(), args.grad_clip
        )
        if step <= args.lr_warmup:
            optimizer.param_groups[0]["lr"] = args.lr * step / args.lr_warmup
        optimizer.step()
        specialist.normalize_decoder()

        values = groups["specialist"]["values"]
        selected = groups["specialist"]["indices"]
        active = values > 0
        feature_counts.scatter_add_(
            0, selected[active], torch.ones_like(selected[active], dtype=torch.long)
        )
        last_fire[selected[active].unique()] = step
        rescued = 0
        if args.reinit_every and step % args.reinit_every == 0 and step < args.steps:
            rescued = rescue_group(
                specialist, spec, residual, delta.detach(), latent_loss.detach(),
                last_fire, step, args.dead_after, args.max_rescue,
            )

        if step == 1 or step % 100 == 0:
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "latent_mse": float(latent_loss.mean().detach()),
                "transformed_loss": (
                    None if args.objective == "mse"
                    else float(objective_loss.mean().detach())
                ),
                "pixel_mse": (
                    None if pixel_mse is None else float(pixel_mse.mean().detach())
                ),
                "transform_weight": live_weight,
                "gradient_norm_before_clip": float(gradient_norm),
                "used_features": int((feature_counts > 0).sum()),
                "rescued": rescued,
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(trace_path, [record])
            print(record, flush=True)

        if step == 1 or step % args.validate_every == 0 or step == args.steps:
            metrics = evaluate(
                base, prior_specialists, specialist, autoencoder,
                raw_latents, valid_indices,
                mean, std, input_shape, args.objective, args.batch_size, device,
            )
            metrics.update({
                "step": step,
                "valid_ev": 1.0 - metrics["latent_mse"] / valid_variance,
                "mse_ratio_to_base": metrics["latent_mse"] / metrics["base_mse"],
                "feasible": (
                    metrics["latent_mse"]
                    <= metrics["base_mse"] * (1 + args.mse_tolerance)
                ),
                "elapsed_seconds": time.time() - started,
            })
            append_jsonl(validation_path, [metrics])
            print({"validation": metrics}, flush=True)
            best_valid_ev = max(best_valid_ev, metrics["valid_ev"])
            selection_loss = (
                metrics["latent_mse"] if args.objective == "mse"
                else metrics["transformed_loss"]
            )
            if selection_loss < best_objective_loss:
                best_objective_loss = selection_loss
                torch.save(
                    payload(specialist, args, step, feature_counts, last_fire, calibration),
                    os.path.join(args.out_dir, "best_objective.pt"),
                )
            if metrics["feasible"] and selection_loss < best_feasible_selection_loss:
                best_feasible_selection_loss = selection_loss
                torch.save(
                    payload(specialist, args, step, feature_counts, last_fire, calibration),
                    os.path.join(args.out_dir, "best_feasible.pt"),
                )

    final = evaluate(
        base, prior_specialists, specialist, autoencoder,
        raw_latents, valid_indices,
        mean, std, input_shape, args.objective, args.batch_size, device,
    )
    final.update({
        "objective": args.objective,
        "steps": args.steps,
        "transform_weight": transform_weight,
        "best_feasible_selection_loss": best_feasible_selection_loss,
        "best_objective_loss": best_objective_loss,
        "selection_metric": (
            "latent_mse" if args.objective == "mse" else "transformed_loss"
        ),
        "best_valid_ev": best_valid_ev,
        "valid_ev": 1.0 - final["latent_mse"] / valid_variance,
        "mse_ratio_to_base": final["latent_mse"] / final["base_mse"],
        "used_features": int((feature_counts > 0).sum()),
        "gradient_calibration": calibration,
        "config": vars(args),
    })
    torch.save(
        payload(specialist, args, args.steps, feature_counts, last_fire, calibration),
        os.path.join(args.out_dir, "last.pt"),
    )
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(final, handle, indent=2)
    with open(os.path.join(args.out_dir, "COMPLETED.json"), "w") as handle:
        json.dump({"step": args.steps, "valid_ev": final["valid_ev"]}, handle, indent=2)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
