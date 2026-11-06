#!/usr/bin/env python3
"""Train branch-local SAEs and union their independently learned vocabularies."""

import argparse
from dataclasses import asdict, dataclass
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spatial_conv_autoencoder import load_spatial_autoencoder
from vector_conv_autoencoder import load_vector_autoencoder
from train_spatial_map_sae import (
    FlatTopKSAE,
    calibrate,
    rescue_dead,
    transformed_loss,
)


OBJECTIVES = ("laplacian", "fourier_low", "fourier_mid", "fourier_high")


@dataclass(frozen=True)
class BranchSpec:
    name: str
    objective: str
    feature_count: int
    k: int


class IndependentRoutedUnionSAE(nn.Module):
    """Partitioned TopK dictionaries with branch-local routing and one output."""

    def __init__(self, input_shape, specs, branch_weights=None):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.specs = tuple(specs)
        self.branches = nn.ModuleDict({
            spec.name: FlatTopKSAE(self.input_shape, spec.feature_count, spec.k)
            for spec in self.specs
        })
        if branch_weights is None:
            branch_weights = torch.ones(len(self.specs))
        weights = torch.as_tensor(branch_weights, dtype=torch.float32)
        if weights.shape != (len(self.specs),) or (weights < 0).any():
            raise ValueError("branch weights must be one nonnegative value per branch")
        self.register_buffer("branch_weights", weights / weights.sum().clamp_min(1e-12))

    def forward(self, inputs):
        outputs = {}
        reconstructions = []
        for spec in self.specs:
            reconstruction, values, indices = self.branches[spec.name](inputs)
            outputs[spec.name] = {
                "reconstruction": reconstruction,
                "values": values,
                "indices": indices,
            }
            reconstructions.append(reconstruction)
        stacked = torch.stack(reconstructions, dim=1)
        shape = (1, len(self.specs)) + (1,) * (stacked.ndim - 2)
        union = (stacked * self.branch_weights.view(shape)).sum(1)
        return union, outputs


def load_independent_routed_union(path, device="cpu"):
    """Load a trained routed-union checkpoint as a directly callable SAE."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    specs = tuple(BranchSpec(**record) for record in payload["specs"])
    weights = payload.get("combination", {}).get("branch_weights")
    model = IndependentRoutedUnionSAE(payload["input_shape"], specs, weights)
    for spec in specs:
        model.branches[spec.name].load_state_dict(payload["models"][spec.name])
    model.eval().requires_grad_(False).to(device)
    return model, payload


def split_budget(total, count):
    base, remainder = divmod(total, count)
    return [base + int(index < remainder) for index in range(count)]


def build_specs(args):
    objectives = [value for value in args.objectives.split(",") if value]
    if not objectives:
        raise ValueError("at least one specialist objective is required")
    if any(value not in OBJECTIVES for value in objectives):
        raise ValueError(f"unknown objectives: {objectives}")
    if len(set(objectives)) != len(objectives):
        raise ValueError("specialist objectives must be unique")
    widths = split_budget(args.specialist_dict_size, len(objectives))
    ks = split_budget(args.specialist_k, len(objectives))
    specs = [BranchSpec("mse", "latent_mse", args.mse_dict_size, args.mse_k)]
    for index, (objective, width, k) in enumerate(zip(objectives, widths, ks)):
        specs.append(BranchSpec(
            f"control_{index + 1}" if args.all_mse_control else objective,
            "latent_mse" if args.all_mse_control else objective,
            width,
            k,
        ))
    return objectives, specs


@torch.no_grad()
def evaluate_latent(branches, specs, raw_latents, indices, mean, std,
                    batch_size, device, valid_variance):
    mse_sums = {spec.name: 0.0 for spec in specs}
    union_mse_sum = 0.0
    count = 0
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        raw = raw_latents[selected].to(device).float()
        standardized = (raw - mean) / std
        reconstructions = []
        for spec in specs:
            reconstruction, _, _ = branches[spec.name](standardized)
            reconstructions.append(reconstruction)
            mse_sums[spec.name] += float(
                (standardized - reconstruction).square().flatten(1).mean(1).sum()
            )
        union = torch.stack(reconstructions).mean(0)
        union_mse_sum += float(
            (standardized - union).square().flatten(1).mean(1).sum()
        )
        count += len(raw)
    branch_mse = {name: value / count for name, value in mse_sums.items()}
    union_mse = union_mse_sum / count
    return {
        "branch_mse": branch_mse,
        "branch_ev": {
            name: 1 - value / valid_variance for name, value in branch_mse.items()
        },
        "union_mse": union_mse,
        "union_ev": 1 - union_mse / valid_variance,
    }


@torch.no_grad()
def evaluate_targets(branches, specs, powers, autoencoder, raw_latents,
                     indices, mean, std, batch_size, device):
    sums = {spec.name: 0.0 for spec in specs if spec.objective != "latent_mse"}
    count = 0
    if not sums:
        return sums
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        raw = raw_latents[selected].to(device).float()
        standardized = (raw - mean) / std
        clean = autoencoder.decode(raw).add(1).div(2)
        for spec in specs:
            if spec.objective == "latent_mse":
                continue
            reconstruction, _, _ = branches[spec.name](standardized)
            reconstructed_raw = reconstruction * std + mean
            image = autoencoder.decode(reconstructed_raw).add(1).div(2)
            loss = transformed_loss(
                spec.objective, clean - image, powers[spec.objective]
            )
            sums[spec.name] += float(loss.sum())
        count += len(raw)
    return {name: value / count for name, value in sums.items()}


def archival_payload(branches, specs, mean, std, input_shape, args, step,
                     metrics=None):
    models = {}
    for spec in specs:
        models[spec.name] = {
            key: value.detach().half().cpu()
            for key, value in branches[spec.name].state_dict().items()
        }
    return {
        "models": models,
        "specs": [asdict(spec) for spec in specs],
        "combination": {
            "kind": "uniform_mean",
            "branch_weights": [1 / len(specs)] * len(specs),
        },
        "channel_mean": mean.detach().cpu(),
        "channel_std": std.detach().cpu(),
        "input_shape": input_shape,
        "step": step,
        "config": vars(args),
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objectives", required=True, help="Comma-separated subset")
    parser.add_argument("--all-mse-control", action="store_true")
    parser.add_argument("--data", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument(
        "--autoencoder-kind", choices=("spatial", "vector"), default="spatial",
        help="spatial = load_spatial_autoencoder (4D N,C,H,W latents); "
             "vector = load_vector_autoencoder (2D N,D latents, e.g. the "
             "512-dim vector_4x_k8-style representation)",
    )
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
    # Channel dim is always dim 1; reduce over every other dim (batch, plus
    # spatial dims if present) so this works for both 4D (N,C,H,W) spatial
    # latents and 2D (N,D) vector latents without duplicating this function.
    reduce_dims = tuple(d for d in range(raw_latents.ndim) if d != 1)
    broadcast_shape = (1, -1) + (1,) * (raw_latents.ndim - 2)
    train_latents = raw_latents[train_indices].float()
    channel_mean = train_latents.mean(dim=reduce_dims)
    channel_std = train_latents.std(dim=reduce_dims).clamp_min(1e-6)
    del train_latents
    mean = channel_mean.to(device).view(broadcast_shape)
    std = channel_std.to(device).view(broadcast_shape)
    valid_standardized = (
        (raw_latents[valid_indices].float() - channel_mean.view(broadcast_shape))
        / channel_std.view(broadcast_shape)
    )
    valid_variance = float(valid_standardized.var().clamp_min(1e-8))
    del valid_standardized

    autoencoder = (
        load_vector_autoencoder(args.autoencoder, device)
        if args.autoencoder_kind == "vector"
        else load_spatial_autoencoder(args.autoencoder, device)
    )
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
                loss = (latent_mse + transformed).mean()
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
