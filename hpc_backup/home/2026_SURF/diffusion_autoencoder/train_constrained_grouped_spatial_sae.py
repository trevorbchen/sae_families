#!/usr/bin/env python3
"""Joint changed-objective SAE with projected gradients and an MSE trust region."""

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from spatial_conv_autoencoder import load_spatial_autoencoder
from train_grouped_objective_sae import GroupSpec, PartitionedTopKSAE, append_jsonl
from train_grouped_spatial_ae_sae import evaluate_latent, spatial_geometry_losses


OBJECTIVES = ("laplacian", "fourier_low", "fourier_mid", "fourier_high")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVES, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--initial-sae", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--mse-adapt-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-warmup", type=int, default=500)
    parser.add_argument("--transform-warmup", type=int, default=2_000)
    parser.add_argument("--transform-gradient-ratio", type=float, default=0.1)
    parser.add_argument("--mse-tolerance", type=float, default=0.01)
    parser.add_argument("--aux-coeff", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--trust-every", type=int, default=250)
    parser.add_argument("--geometry-every", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def build_specs(objectives):
    if len(objectives) == 1:
        return (
            GroupSpec("mse", "latent_mse", 8_192, 32),
            GroupSpec(objectives[0], objectives[0], 8_192, 32),
        )
    if tuple(objectives) != OBJECTIVES:
        raise ValueError("multi-objective mode requires all four objectives in order")
    return (
        GroupSpec("mse", "latent_mse", 8_192, 32),
        *(GroupSpec(name, name, 2_048, 8) for name in OBJECTIVES),
    )


def load_partitioned(path, specs, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    input_dim = checkpoint["model"]["decoder"].shape[1]
    model = PartitionedTopKSAE(input_dim, specs)
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def restore_state(model, state):
    model.load_state_dict(state)


def group_aux_loss(model, groups, residual):
    return torch.stack([
        F.mse_loss(groups[spec.name]["aux_contribution"], residual)
        for spec in model.specs
    ]).mean()


def projected_parameter_gradients(model, primary_loss, transformed, ratio):
    params = (model.encoder.weight, model.encoder.bias, model.decoder)
    mse_grads = torch.autograd.grad(primary_loss, params, retain_graph=True)
    final_grads = [gradient.detach().clone() for gradient in mse_grads]
    diagnostics = {}
    for objective, loss in transformed.items():
        transformed_grads = torch.autograd.grad(
            loss.mean(), params, retain_graph=True
        )
        start, end = model.ranges[objective]
        mse_slices = [gradient[start:end] for gradient in mse_grads]
        transformed_slices = [gradient[start:end] for gradient in transformed_grads]
        dot = sum((left * right).sum() for left, right in zip(
            mse_slices, transformed_slices
        ))
        mse_norm = torch.sqrt(sum(gradient.square().sum() for gradient in mse_slices))
        raw_norm = torch.sqrt(sum(
            gradient.square().sum() for gradient in transformed_slices
        ))
        cosine = dot / (mse_norm * raw_norm).clamp_min(1e-12)
        projected = [gradient for gradient in transformed_slices]
        was_projected = float(dot) < 0
        if was_projected:
            coefficient = dot / mse_norm.square().clamp_min(1e-12)
            projected = [
                transformed_gradient - coefficient * mse_gradient
                for transformed_gradient, mse_gradient in zip(projected, mse_slices)
            ]
        projected_norm = torch.sqrt(sum(
            gradient.square().sum() for gradient in projected
        ))
        scale = min(1.0, float(ratio * mse_norm / projected_norm.clamp_min(1e-12)))
        for final, gradient in zip(final_grads, projected):
            final[start:end].add_(gradient, alpha=scale)
        diagnostics[objective] = {
            "mse_transform_cosine": float(cosine),
            "raw_transform_gradient_norm": float(raw_norm),
            "projected_transform_gradient_norm": float(projected_norm),
            "applied_scale": scale,
            "projected": was_projected,
        }
    return params, final_grads, diagnostics


@torch.no_grad()
def evaluate_geometry(
    model, autoencoder, raw_latents, indices, mean, std, input_shape,
    objectives, batch_size, device,
):
    totals = {name: 0.0 for name in objectives}
    pixel_total = latent_squared = 0.0
    latent_elements = count = 0
    model.eval()
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        raw = raw_latents[selected].to(device).float()
        standardized = ((raw - mean) / std).flatten(1)
        reconstruction = model(standardized)[0]
        latent_squared += float((standardized - reconstruction).square().sum())
        latent_elements += standardized.numel()
        reconstructed_raw = reconstruction.view(-1, *input_shape) * std + mean
        transformed, pixel_mse = spatial_geometry_losses(
            autoencoder, raw, reconstructed_raw, powers=None
        )
        pixel_total += float(pixel_mse.sum())
        for name in objectives:
            totals[name] += float(transformed[name].sum())
        count += len(selected)
    model.train()
    return {
        "latent_mse": latent_squared / latent_elements,
        "pixel_mse": pixel_total / count,
        "transformed": {name: value / count for name, value in totals.items()},
    }


def save_checkpoint(path, model, args, specs, checkpoint, step, extra):
    payload = {
        "model": model.state_dict(),
        "groups": [asdict(spec) for spec in specs],
        "input_mean": checkpoint["input_mean"],
        "input_std": checkpoint["input_std"],
        "input_shape": checkpoint["input_shape"],
        "config": vars(args),
        "step": step,
        **extra,
    }
    torch.save(payload, path)


def main():
    args = parse_args()
    if not 0 <= args.mse_adapt_steps < args.steps:
        raise ValueError("mse-adapt-steps must be in [0, steps)")
    os.makedirs(args.out_dir, exist_ok=True)
    trace_path = os.path.join(args.out_dir, "training_trace.jsonl")
    validation_path = os.path.join(args.out_dir, "validation_trace.jsonl")
    open(trace_path, "w").close()
    open(validation_path, "w").close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = build_specs(args.objectives)
    model, initial_checkpoint = load_partitioned(args.initial_sae, specs, device)
    cache = torch.load(args.data, map_location="cpu", weights_only=False)
    raw_latents = cache["latents"]
    train_indices = cache["train_indices"].long()
    valid_indices = cache["valid_indices"].long()
    mean = initial_checkpoint["input_mean"].to(device)
    std = initial_checkpoint["input_std"].to(device)
    input_shape = tuple(initial_checkpoint["input_shape"])
    valid_variance = float(
        ((raw_latents[valid_indices].float() - mean.cpu()) / std.cpu())
        .flatten(1).var().clamp_min(1e-8)
    )
    autoencoder = load_spatial_autoencoder(args.autoencoder, device)
    autoencoder.eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train_indices), batch_size=args.batch_size,
        shuffle=True, drop_last=True, generator=generator,
    )
    iterator = iter(loader)
    feature_counts = torch.zeros(model.dict_size, dtype=torch.long, device=device)
    started = time.time()
    trust_baseline = trust_limit = None
    accepted_state = None
    accepted_step = args.mse_adapt_steps
    effective_ratio = args.transform_gradient_ratio
    rollback_count = 0
    best_score = float("inf")
    baseline_geometry = None

    for step in range(1, args.steps + 1):
        try:
            indices = next(iterator)[0]
        except StopIteration:
            iterator = iter(loader)
            indices = next(iterator)[0]
        raw = raw_latents[indices].to(device).float()
        standardized = ((raw - mean) / std).flatten(1)
        reconstruction, groups = model(standardized)
        latent_per_example = (standardized - reconstruction).square().mean(1)
        residual = (standardized - reconstruction).detach()
        aux_loss = group_aux_loss(model, groups, residual)
        primary_loss = latent_per_example.mean() + args.aux_coeff * aux_loss
        phase_step = step - args.mse_adapt_steps
        ratio = 0.0
        gradient_diagnostics = {}
        optimizer.zero_grad(set_to_none=True)
        if phase_step <= 0:
            primary_loss.backward()
        else:
            reconstructed_raw = reconstruction.view(-1, *input_shape) * std + mean
            transformed, _ = spatial_geometry_losses(
                autoencoder, raw, reconstructed_raw, powers=None
            )
            ratio = effective_ratio * min(
                1.0, phase_step / max(args.transform_warmup, 1)
            )
            params, gradients, gradient_diagnostics = projected_parameter_gradients(
                model, primary_loss, {name: transformed[name] for name in args.objectives},
                ratio,
            )
            for parameter, gradient in zip(params, gradients):
                parameter.grad = gradient
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if torch.isfinite(gradient_norm):
            if step <= args.lr_warmup:
                optimizer.param_groups[0]["lr"] = args.lr * step / args.lr_warmup
            optimizer.step()
            model.normalize_decoder()

        for spec in specs:
            selected = groups[spec.name]["indices"]
            active = groups[spec.name]["values"] > 0
            feature_counts.scatter_add_(
                0, selected[active], torch.ones_like(selected[active], dtype=torch.long)
            )

        if step == 1 or step % 100 == 0:
            record = {
                "step": step,
                "phase": "mse_adaptation" if phase_step <= 0 else "constrained",
                "latent_mse": float(latent_per_example.mean()),
                "aux_mse": float(aux_loss),
                "gradient_norm": float(gradient_norm),
                "transform_gradient_ratio": ratio,
                "effective_ratio_cap": effective_ratio,
                "gradient_diagnostics": gradient_diagnostics,
                "rollback_count": rollback_count,
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(trace_path, [record])
            print(record, flush=True)

        needs_trust = step == args.mse_adapt_steps or (
            phase_step > 0 and step % args.trust_every == 0
        ) or step == args.steps
        accepted = None
        valid_mse = valid_ev = None
        if needs_trust:
            valid_mse, _ = evaluate_latent(
                model, raw_latents, valid_indices, mean, std,
                args.batch_size, device,
            )
            valid_ev = 1.0 - valid_mse / valid_variance
            if trust_baseline is None:
                trust_baseline = valid_mse
                trust_limit = trust_baseline * (1 + args.mse_tolerance)
                accepted_state = cpu_state(model)
                accepted = True
            elif math.isfinite(valid_mse) and valid_mse <= trust_limit:
                accepted_state = cpu_state(model)
                accepted_step = step
                accepted = True
            else:
                restore_state(model, accepted_state)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                effective_ratio *= 0.5
                rollback_count += 1
                accepted = False
            validation = {
                "step": step,
                "valid_mse": valid_mse,
                "valid_ev": valid_ev,
                "trust_baseline_mse": trust_baseline,
                "trust_limit_mse": trust_limit,
                "accepted": accepted,
                "accepted_step_after_check": accepted_step,
                "rollback_count": rollback_count,
                "effective_ratio_cap": effective_ratio,
            }
            append_jsonl(validation_path, [validation])
            print({"validation": validation}, flush=True)

        if step == args.mse_adapt_steps:
            baseline_geometry = evaluate_geometry(
                model, autoencoder, raw_latents, valid_indices, mean, std,
                input_shape, args.objectives, args.batch_size, device,
            )
            save_checkpoint(
                os.path.join(args.out_dir, "mse_adapted.pt"), model, args, specs,
                initial_checkpoint, step, {"geometry": baseline_geometry},
            )

        if phase_step > 0 and accepted is not False and (
            step % args.geometry_every == 0 or step == args.steps
        ):
            geometry = evaluate_geometry(
                model, autoencoder, raw_latents, valid_indices, mean, std,
                input_shape, args.objectives, args.batch_size, device,
            )
            if baseline_geometry is None:
                baseline_geometry = geometry
            score = sum(
                geometry["transformed"][name]
                / max(baseline_geometry["transformed"][name], 1e-12)
                for name in args.objectives
            ) / len(args.objectives)
            if score < best_score:
                best_score = score
                save_checkpoint(
                    os.path.join(args.out_dir, "best_feasible.pt"), model, args,
                    specs, initial_checkpoint, step,
                    {"geometry": geometry, "normalized_objective_score": score},
                )

    if accepted_state is not None:
        restore_state(model, accepted_state)
    final_geometry = evaluate_geometry(
        model, autoencoder, raw_latents, valid_indices, mean, std,
        input_shape, args.objectives, args.batch_size, device,
    )
    final_ev = 1.0 - final_geometry["latent_mse"] / valid_variance
    metrics = {
        "objectives": args.objectives,
        "groups": [asdict(spec) for spec in specs],
        "total_dictionary": model.dict_size,
        "total_k": model.total_k,
        "trust_baseline_mse": trust_baseline,
        "trust_limit_mse": trust_limit,
        "final_geometry": final_geometry,
        "final_valid_ev": final_ev,
        "best_normalized_objective_score": best_score,
        "rollback_count": rollback_count,
        "final_effective_ratio_cap": effective_ratio,
        "used_features": int((feature_counts > 0).sum()),
        "config": vars(args),
    }
    save_checkpoint(
        os.path.join(args.out_dir, "last_accepted.pt"), model, args, specs,
        initial_checkpoint, accepted_step, {"metrics": metrics},
    )
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    with open(os.path.join(args.out_dir, "COMPLETED.json"), "w") as handle:
        json.dump({"step": args.steps, "final_valid_ev": final_ev}, handle, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
