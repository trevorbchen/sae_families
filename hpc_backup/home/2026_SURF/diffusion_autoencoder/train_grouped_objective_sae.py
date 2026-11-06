#!/usr/bin/env python3
"""Train capacity-matched regular and gradient-routed grouped TopK SAEs."""

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_zsem_transformed_sae import (
    calibrate,
    load_diffae,
    timestep_batch,
    transform_objective_name,
    transformed_loss,
)


@dataclass(frozen=True)
class GroupSpec:
    name: str
    objective: str
    size: int
    k: int


REGULAR_SPECS = (
    GroupSpec("mse", "latent_mse", 16_384, 64),
)

GROUPED_SPECS = (
    GroupSpec("mse", "latent_mse", 8_192, 32),
    GroupSpec("laplacian", "weighted_laplacian", 2_048, 8),
    GroupSpec("fourier_low", "weighted_fourier_low", 2_048, 8),
    GroupSpec("fourier_mid", "weighted_fourier_mid", 2_048, 8),
    GroupSpec("fourier_high", "weighted_fourier_high", 2_048, 8),
)

GROUPED_PURE_SPECS = (
    GroupSpec("mse", "latent_mse", 8_192, 32),
    GroupSpec("laplacian", "laplacian", 2_048, 8),
    GroupSpec("fourier_low", "fourier_low", 2_048, 8),
    GroupSpec("fourier_mid", "fourier_mid", 2_048, 8),
    GroupSpec("fourier_high", "fourier_high", 2_048, 8),
)


class PartitionedTopKSAE(nn.Module):
    def __init__(self, input_dim, specs):
        super().__init__()
        self.input_dim = input_dim
        self.specs = tuple(specs)
        self.dict_size = sum(spec.size for spec in self.specs)
        self.total_k = sum(spec.k for spec in self.specs)
        self.encoder = nn.Linear(input_dim, self.dict_size)
        self.decoder = nn.Parameter(torch.empty(self.dict_size, input_dim))
        # Standardized training codes have zero mean. Keeping this fixed avoids
        # an unowned shared parameter receiving gradients from every group.
        self.register_buffer("dec_bias", torch.zeros(input_dim))
        nn.init.normal_(self.decoder, std=input_dim ** -0.5)
        self.normalize_decoder()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder)
            self.encoder.bias.zero_()
        self.ranges = {}
        start = 0
        for spec in self.specs:
            self.ranges[spec.name] = (start, start + spec.size)
            start += spec.size

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.div_(self.decoder.norm(dim=1, keepdim=True).clamp_min(1e-6))

    def forward(self, inputs):
        activations = F.relu(self.encoder(inputs - self.dec_bias))
        groups = {}
        contributions = []
        for spec in self.specs:
            start, end = self.ranges[spec.name]
            group_activations = activations[:, start:end]
            values, local_indices = torch.topk(group_activations, spec.k, dim=-1)
            indices = local_indices + start
            contribution = (
                values.unsqueeze(-1) * self.decoder[indices]
            ).sum(dim=1)
            selected = torch.zeros_like(group_activations, dtype=torch.bool)
            selected.scatter_(1, local_indices, True)
            residual_activations = group_activations.masked_fill(selected, 0)
            aux_values, aux_local_indices = torch.topk(
                residual_activations, spec.k, dim=-1
            )
            aux_indices = aux_local_indices + start
            aux_contribution = (
                aux_values.unsqueeze(-1) * self.decoder[aux_indices]
            ).sum(dim=1)
            groups[spec.name] = {
                "values": values,
                "indices": indices,
                "contribution": contribution,
                "aux_contribution": aux_contribution,
            }
            contributions.append(contribution)
        reconstruction = torch.stack(contributions).sum(0) + self.dec_bias
        return reconstruction, groups


@torch.no_grad()
def rescue_group(
    sae,
    spec,
    inputs,
    reconstruction,
    per_example_loss,
    last_fire,
    step,
    dead_after,
    max_rescue,
):
    start, end = sae.ranges[spec.name]
    local_dead = torch.where(last_fire[start:end] < step - dead_after)[0]
    if not len(local_dead):
        return 0
    count = min(len(local_dead), len(inputs), max_rescue)
    examples = torch.argsort(per_example_loss, descending=True)[:count]
    residual = inputs[examples] - reconstruction[examples]
    directions = F.normalize(residual, dim=1)
    features = local_dead[:count] + start
    sae.decoder[features].copy_(directions)
    sae.encoder.weight[features].copy_(directions)
    response = (directions * (inputs[examples] - sae.dec_bias)).sum(1)
    sae.encoder.bias[features].copy_(-response + 0.1)
    last_fire[features] = step
    return count


def geometry_losses(
    lit,
    image,
    raw_code,
    reconstructed_code,
    step,
    seed,
    powers,
):
    bin_ids, timesteps, noise = timestep_batch(
        step, len(image), seed, image.shape, image.device
    )
    x_t = lit.sampler.q_sample(image.mul(2).sub(1), timesteps, noise=noise)
    with torch.no_grad():
        clean_prediction = lit.sampler.p_mean_variance(
            lit.ema_model,
            x_t,
            timesteps,
            clip_denoised=True,
            model_kwargs={"cond": raw_code},
        )["pred_xstart"].add(1).div(2)
    reconstructed_prediction = lit.sampler.p_mean_variance(
        lit.ema_model,
        x_t,
        timesteps,
        clip_denoised=True,
        model_kwargs={"cond": reconstructed_code},
    )["pred_xstart"].add(1).div(2)
    error = clean_prediction - reconstructed_prediction
    losses = {}
    for name, objective_powers in powers.items():
        objective = f"weighted_{name}"
        losses[name] = transformed_loss(
            transform_objective_name(objective), error, bin_ids, objective_powers
        )
    pixel_mse = error.square().flatten(1).mean(1)
    return losses, pixel_mse, timesteps


def routed_gradient_proxy(sae, groups, gradients):
    """Return a scalar whose parameter gradients implement group routing."""
    proxy = torch.zeros((), device=sae.decoder.device)
    for spec in sae.specs:
        proxy = proxy + (
            groups[spec.name]["contribution"] * gradients[spec.name].detach()
        ).sum()
    return proxy


def group_objective_losses(mode, specs, latent_loss, transformed, weight):
    """Build each group's primary objective before routing its gradient."""
    losses = {"mse": latent_loss}
    for spec in specs[1:]:
        transform_name = transform_objective_name(spec.objective)
        geometry_loss = weight * transformed[transform_name]
        losses[spec.name] = (
            latent_loss + geometry_loss if mode == "grouped" else geometry_loss
        )
    return losses


def append_jsonl(path, rows):
    if not rows:
        return
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), allow_nan=False) + "\n")


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command, cwd=None):
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {"returncode": result.returncode, "output": result.stdout.strip()}


def build_run_manifest(args, extra_source_files=()):
    root = os.path.dirname(os.path.abspath(__file__))
    source_files = (
        os.path.abspath(__file__),
        os.path.join(root, "train_zsem_transformed_sae.py"),
        *tuple(os.path.abspath(path) for path in extra_source_files),
    )
    manifest = {
        "command": [sys.executable, *sys.argv],
        "config": vars(args),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source_sha256": {
            os.path.basename(path): sha256_file(path) for path in source_files
        },
        "git_revision": command_output(["git", "rev-parse", "HEAD"], root),
        "git_status": command_output(["git", "status", "--short"], root),
        "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]),
    }
    if not args.skip_file_hashes:
        manifest["input_sha256"] = {
            "data": sha256_file(args.data),
            "checkpoint": sha256_file(args.checkpoint),
        }
    return manifest


@torch.no_grad()
def evaluate_latent(sae, codes, mean, std, batch_size, device):
    was_training = sae.training
    sae.eval()
    squared_error = 0.0
    count = 0
    for start in range(0, len(codes), batch_size):
        raw = codes[start:start + batch_size].to(device)
        standardized = (raw - mean) / std
        reconstruction, _ = sae(standardized)
        squared_error += float((standardized - reconstruction).square().sum())
        count += standardized.numel()
    sae.train(was_training)
    return squared_error / count


@torch.no_grad()
def write_diagnostics(specs, groups):
    writes = torch.stack(
        [groups[spec.name]["contribution"] for spec in specs], dim=1
    )
    norms = writes.norm(dim=-1)
    combined_norm = writes.sum(dim=1).norm(dim=-1)
    cancellation_ratio = combined_norm / norms.sum(dim=1).clamp_min(1e-12)
    pairwise = {}
    for left in range(len(specs)):
        for right in range(left + 1, len(specs)):
            name = f"{specs[left].name}__{specs[right].name}"
            pairwise[name] = F.cosine_similarity(
                writes[:, left], writes[:, right], dim=-1
            )
    return norms, cancellation_ratio, pairwise


def group_parameter_gradient_norms(sae):
    norms = {}
    for spec in sae.specs:
        start, end = sae.ranges[spec.name]
        squared = torch.zeros((), device=sae.decoder.device)
        for gradient in (
            sae.decoder.grad[start:end] if sae.decoder.grad is not None else None,
            sae.encoder.weight.grad[start:end]
            if sae.encoder.weight.grad is not None else None,
            sae.encoder.bias.grad[start:end]
            if sae.encoder.bias.grad is not None else None,
        ):
            if gradient is not None:
                squared = squared + gradient.square().sum()
        norms[spec.name] = squared.sqrt()
    return norms


def model_payload(sae, args, mean, std, step, feature_counts, last_fire):
    return {
        "model": sae.state_dict(),
        "input_mean": mean,
        "input_std": std,
        "groups": [asdict(spec) for spec in sae.specs],
        "config": vars(args),
        "step": step,
        "feature_counts": feature_counts.cpu(),
        "last_fire": last_fire.cpu(),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("regular", "grouped", "grouped_pure"), required=True
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--transform-weight", type=float, default=1.0)
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
    manifest = build_run_manifest(args)
    with open(os.path.join(args.out_dir, "run_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    images = data["images_uint8"]
    codes = data["codes"].float()
    split_generator = torch.Generator().manual_seed(args.split_seed)
    permutation = torch.randperm(len(codes), generator=split_generator)
    split = int(0.9 * len(codes))
    train_indices, valid_indices = permutation[:split], permutation[split:]
    mean = codes[train_indices].mean(0)
    std = codes[train_indices].std(0).clamp_min(1e-6)
    valid_codes = codes[valid_indices]
    valid_variance = float(((valid_codes - mean) / std).var().clamp_min(1e-8))
    device_mean, device_std = mean.to(device), std.to(device)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(images[train_indices], codes[train_indices]),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
    )
    iterator = iter(loader)

    specs_by_mode = {
        "regular": REGULAR_SPECS,
        "grouped": GROUPED_SPECS,
        "grouped_pure": GROUPED_PURE_SPECS,
    }
    specs = specs_by_mode[args.mode]
    lit = None
    powers = {}
    power_counts = {}
    if args.mode != "regular":
        lit = load_diffae(args.repo, args.checkpoint, device)
        calibration_indices = train_indices[: min(
            len(train_indices), args.calibration_examples
        )]
        for offset, spec in enumerate(specs[1:]):
            objective = transform_objective_name(spec.objective)
            objective_powers, counts = calibrate(
                lit,
                objective,
                images[calibration_indices],
                codes[calibration_indices],
                args.batch_size,
                args.seed + 100_000 + 10_000 * offset,
                device,
            )
            powers[objective] = objective_powers
            power_counts[objective] = counts

    # Calibration and DiffAE construction consume RNG state. Re-seeding here
    # guarantees identical initial encoder/decoder tensors across both modes.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    sae = PartitionedTopKSAE(512, specs).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    feature_counts = torch.zeros(sae.dict_size, dtype=torch.long, device=device)
    last_fire = torch.zeros_like(feature_counts)
    rescued_by_group = {spec.name: 0 for spec in specs}
    history = []
    trace_buffer = []
    validation_history = []
    clip_count = 0
    clip_streak = 0
    longest_clip_streak = 0
    gradient_norm_history = []
    dominant_clip_counts = {spec.name: 0 for spec in specs}
    best_valid_ev = float("-inf")
    started_at = time.time()

    for step in range(1, args.steps + 1):
        try:
            image_u8, raw_code = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            image_u8, raw_code = next(iterator)
        image = image_u8.to(device).float().div(255)
        raw_code = raw_code.to(device)
        standardized = (raw_code - device_mean) / device_std
        reconstruction, groups = sae(standardized)
        latent_per_example = (reconstruction - standardized).square().mean(1)
        per_group_losses = {"mse": latent_per_example}
        transformed = {}
        pixel_mse = torch.full_like(latent_per_example, float("nan"))
        timesteps = None

        if args.mode != "regular":
            reconstructed_code = reconstruction * device_std + device_mean
            transformed, pixel_mse, timesteps = geometry_losses(
                lit,
                image,
                raw_code,
                reconstructed_code,
                step,
                args.seed,
                powers,
            )
            per_group_losses = group_objective_losses(
                args.mode,
                specs,
                latent_per_example,
                transformed,
                args.transform_weight,
            )

        residual = (standardized - reconstruction).detach()
        aux_losses = [
            F.mse_loss(groups[spec.name]["aux_contribution"], residual)
            for spec in specs
        ]
        aux_loss = torch.stack(aux_losses).mean()
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
            routed_gradients = {}
            for spec in specs:
                routed_gradients[spec.name] = torch.autograd.grad(
                    per_group_losses[spec.name].mean(),
                    reconstruction,
                    retain_graph=True,
                )[0]
            proxy = routed_gradient_proxy(sae, groups, routed_gradients)
            training_loss = (
                torch.stack([loss.mean() for loss in per_group_losses.values()]).mean()
                + args.aux_coeff * aux_loss.detach()
            )
            (proxy + args.aux_coeff * aux_loss).backward()

        parameter_gradient_norms = group_parameter_gradient_norms(sae)
        gradient_norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip)
        gradient_norm_value = float(gradient_norm)
        clipped = gradient_norm_value > args.grad_clip
        if clipped:
            clip_count += 1
            clip_streak += 1
            longest_clip_streak = max(longest_clip_streak, clip_streak)
        else:
            clip_streak = 0
        if args.warmup and step <= args.warmup:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = args.lr * step / args.warmup
        optimizer.step()
        sae.normalize_decoder()

        for spec in specs:
            values = groups[spec.name]["values"]
            indices = groups[spec.name]["indices"]
            active = values > 0
            feature_counts.scatter_add_(
                0, indices[active], torch.ones_like(indices[active], dtype=torch.long)
            )
            fired = indices[active].unique()
            last_fire[fired] = step

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

        dominant_gradient_group = max(
            parameter_gradient_norms,
            key=lambda name: float(parameter_gradient_norms[name]),
        )
        gradient_norm_history.append(gradient_norm_value)
        if clipped:
            dominant_clip_counts[dominant_gradient_group] += 1
        trace_record = {
            "step": step,
            "elapsed_seconds": time.time() - started_at,
            "timesteps": None if timesteps is None else timesteps.detach().cpu().tolist(),
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
            "routed_gradient_norm_by_group": {
                name: float(gradient.detach().norm())
                for name, gradient in routed_gradients.items()
            },
            "parameter_gradient_norm_by_group": {
                name: float(value) for name, value in parameter_gradient_norms.items()
            },
            "dominant_gradient_group": dominant_gradient_group,
            "gradient_norm_before_clip": gradient_norm_value,
            "gradient_clip_threshold": args.grad_clip,
            "clipped": clipped,
            "clip_count": clip_count,
            "clip_fraction": clip_count / step,
            "current_clip_streak": clip_streak,
            "longest_clip_streak": longest_clip_streak,
            "write_norm_mean_by_group": {
                spec.name: float(write_norms[:, index].mean())
                for index, spec in enumerate(specs)
            },
            "cancellation_ratio_mean": float(cancellation_ratio.mean()),
            "cancellation_ratio_per_example": cancellation_ratio.cpu().tolist(),
            "pairwise_write_cosine_mean": {
                name: float(values.mean()) for name, values in pairwise_cosines.items()
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
                        .norm(dim=1)
                        .mean()
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

        validation_record = None
        if step == 1 or step % args.validate_every == 0 or step == args.steps:
            current_valid_mse = evaluate_latent(
                sae,
                valid_codes,
                device_mean,
                device_std,
                args.batch_size,
                device,
            )
            current_valid_ev = 1.0 - current_valid_mse / valid_variance
            validation_record = {
                "step": step,
                "valid_mse": current_valid_mse,
                "valid_ev": current_valid_ev,
                "elapsed_seconds": time.time() - started_at,
                "clip_count": clip_count,
                "clip_fraction": clip_count / step,
                "longest_clip_streak": longest_clip_streak,
            }
            validation_history.append(validation_record)
            append_jsonl(validation_path, [validation_record])
            if current_valid_ev > best_valid_ev:
                best_valid_ev = current_valid_ev
                best_payload = model_payload(
                    sae, args, mean, std, step, feature_counts, last_fire
                )
                best_payload["validation"] = validation_record
                torch.save(best_payload, os.path.join(args.out_dir, "best_valid.pt"))
            print({"validation": validation_record}, flush=True)

        if step % args.checkpoint_every == 0 or step == args.steps:
            torch.save(
                model_payload(sae, args, mean, std, step, feature_counts, last_fire),
                os.path.join(checkpoint_dir, f"step_{step:06d}.pt"),
            )
        if step % args.state_every == 0 or step == args.steps:
            state = model_payload(
                sae, args, mean, std, step, feature_counts, last_fire
            )
            state.update({
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
                "active_by_group": {
                    spec.name: float(
                        (groups[spec.name]["values"] > 0).sum(1).float().mean()
                    )
                    for spec in specs
                },
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

    sae.eval()
    valid_loader = DataLoader(
        TensorDataset(images[valid_indices], codes[valid_indices]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    valid_squared_error = 0.0
    valid_count = 0
    valid_used = {spec.name: set() for spec in specs}
    with torch.no_grad():
        for _, raw_code in valid_loader:
            raw_code = raw_code.to(device)
            standardized = (raw_code - device_mean) / device_std
            reconstruction, groups = sae(standardized)
            valid_squared_error += float(
                (standardized - reconstruction).square().sum()
            )
            valid_count += standardized.numel()
            for spec in specs:
                valid_used[spec.name].update(
                    groups[spec.name]["indices"].unique().cpu().tolist()
                )
    valid_mse = valid_squared_error / valid_count
    valid_variance = float(
        ((codes[valid_indices] - mean) / std).var().clamp_min(1e-8)
    )
    gradient_norm_tensor = torch.tensor(gradient_norm_history, dtype=torch.float64)
    result = {
        "mode": args.mode,
        "examples": len(codes),
        "train_examples": len(train_indices),
        "valid_examples": len(valid_indices),
        "input_dim": 512,
        "dict_size": sae.dict_size,
        "total_k": sae.total_k,
        "groups": [asdict(spec) for spec in specs],
        "valid_mse": valid_mse,
        "valid_ev": 1.0 - valid_mse / valid_variance,
        "valid_variance": valid_variance,
        "valid_used_by_group": {
            name: len(indices) for name, indices in valid_used.items()
        },
        "rescued_by_group": rescued_by_group,
        "clip_summary": {
            "count": clip_count,
            "fraction": clip_count / args.steps,
            "longest_streak": longest_clip_streak,
            "dominant_group_counts": dominant_clip_counts,
            "gradient_norm_median": float(gradient_norm_tensor.median()),
            "gradient_norm_p95": float(torch.quantile(gradient_norm_tensor, 0.95)),
            "gradient_norm_max": float(gradient_norm_tensor.max()),
        },
        "validation_history": validation_history,
        "best_valid_ev": best_valid_ev,
        "component_powers": {
            name: value.cpu().tolist() for name, value in powers.items()
        },
        "power_counts": {
            name: value.tolist() for name, value in power_counts.items()
        },
        "fixed_decoder_bias": True,
        "config": vars(args),
        "history": history,
    }
    torch.save(
        {
            "model": sae.state_dict(),
            "input_mean": mean,
            "input_std": std,
            "groups": [asdict(spec) for spec in specs],
            "metrics": result,
            "config": vars(args),
        },
        os.path.join(args.out_dir, "last.pt"),
    )
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    with open(os.path.join(args.out_dir, "COMPLETED.json"), "w") as handle:
        json.dump(
            {
                "step": args.steps,
                "valid_mse": valid_mse,
                "valid_ev": 1.0 - valid_mse / valid_variance,
                "best_valid_ev": best_valid_ev,
                "elapsed_seconds": time.time() - started_at,
            },
            handle,
            indent=2,
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
