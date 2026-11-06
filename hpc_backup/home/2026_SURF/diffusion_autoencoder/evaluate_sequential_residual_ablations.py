#!/usr/bin/env python3
"""Evaluate subset and write-level ablations of a sequential residual-SAE stack."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import torch

from spatial_conv_autoencoder import load_spatial_autoencoder
from train_grouped_objective_sae import GroupSpec, PartitionedTopKSAE
from train_grouped_spatial_ae_sae import spatial_geometry_losses
from train_spatial_residual_specialist import load_base


STAGES = ("laplacian", "fourier_low", "fourier_mid", "fourier_high")
STAGE_DIRS = (
    "stage_1_laplacian",
    "stage_2_fourier_low",
    "stage_3_fourier_mid",
    "stage_4_fourier_high",
)
METRICS = ("latent_mse", "pixel_mse", *STAGES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--base-sae", required=True)
    parser.add_argument("--stack-root", required=True)
    parser.add_argument("--sequential-mse-control", required=True)
    parser.add_argument("--direct-mse-control", required=True)
    parser.add_argument("--pure-stack-root", default="")
    parser.add_argument("--mse-stack-root", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mode-chunk-size", type=int, default=2)
    parser.add_argument(
        "--max-examples", type=int, default=0,
        help="Optional validation subset for smoke testing; 0 uses all examples.",
    )
    return parser.parse_args()


def load_specialist(path: Path, input_dim: int, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    specs = tuple(GroupSpec(**record) for record in checkpoint["groups"])
    model = PartitionedTopKSAE(input_dim, specs)
    model.load_state_dict(checkpoint["model"])
    return model.eval().requires_grad_(False).to(device)


def subset_mode_name(subset: tuple[str, ...]) -> str:
    if not subset:
        return "base_only"
    if len(subset) == 1:
        return f"only_{subset[0]}"
    if len(subset) == len(STAGES):
        return "full_stack"
    return "subset__" + "__".join(subset)


def build_mode_metadata(include_pure_stack: bool, include_mse_stack: bool) -> list[dict]:
    modes = []
    for size in range(len(STAGES) + 1):
        for subset in itertools.combinations(STAGES, size):
            modes.append(
                {
                    "mode": subset_mode_name(subset),
                    "family": "recomputed_subset",
                    "subset": list(subset),
                    "subset_size": size,
                    "removed_stage": "",
                }
            )
    for stage in STAGES:
        modes.append(
            {
                "mode": f"full_minus_{stage}_frozen_write",
                "family": "frozen_write_ablation",
                "subset": [name for name in STAGES if name != stage],
                "subset_size": len(STAGES) - 1,
                "removed_stage": stage,
            }
        )
    modes.append(
        {
            "mode": "reverse_order",
            "family": "order_diagnostic",
            "subset": list(reversed(STAGES)),
            "subset_size": len(STAGES),
            "removed_stage": "",
        }
    )
    modes.extend(
        (
            {
                "mode": "sequential_mse_residual_control",
                "family": "mse_control",
                "subset": [],
                "subset_size": 1,
                "removed_stage": "",
            },
            {
                "mode": "direct_mse_control",
                "family": "mse_control",
                "subset": [],
                "subset_size": 1,
                "removed_stage": "",
            },
        )
    )
    for stack_type, include in (
        ("pure_transformed", include_pure_stack),
        ("mse_four_stage", include_mse_stack),
    ):
        if not include:
            continue
        for prefix_size in range(1, len(STAGES) + 1):
            modes.append(
                {
                    "mode": f"{stack_type}_prefix_{prefix_size}",
                    "family": "comparison_stack",
                    "subset": list(STAGES[:prefix_size]),
                    "subset_size": prefix_size,
                    "removed_stage": "",
                    "stack_type": stack_type,
                }
            )
    for record in modes:
        record.setdefault("stack_type", "proper")
    return modes


def apply_sequence(
    standardized: torch.Tensor,
    base_reconstruction: torch.Tensor,
    specialists: dict[str, torch.nn.Module],
    sequence: list[str] | tuple[str, ...],
) -> torch.Tensor:
    reconstruction = base_reconstruction
    for stage in sequence:
        residual = standardized - reconstruction
        reconstruction = reconstruction + specialists[stage](residual)[0]
    return reconstruction


def apply_model_sequence(
    standardized: torch.Tensor,
    base_reconstruction: torch.Tensor,
    models: list[torch.nn.Module],
) -> torch.Tensor:
    reconstruction = base_reconstruction
    for model in models:
        residual = standardized - reconstruction
        reconstruction = reconstruction + model(residual)[0]
    return reconstruction


def load_stack(
    root: str,
    directories: tuple[str, ...],
    input_dim: int,
    device: torch.device,
) -> list[torch.nn.Module]:
    return [
        load_specialist(Path(root) / directory / "best_feasible.pt", input_dim, device)
        for directory in directories
    ]


def mean_ci(values: torch.Tensor) -> tuple[float, float, float]:
    values = values.double()
    mean = float(values.mean())
    half_width = 1.96 * float(values.std(unbiased=True)) / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = torch.load(args.data, map_location="cpu", weights_only=False)
    raw_latents = cache["latents"]
    valid_indices = cache["valid_indices"].long()
    if args.max_examples:
        valid_indices = valid_indices[:args.max_examples]
    base, base_checkpoint = load_base(args.base_sae, device)
    mean = base_checkpoint["input_mean"].to(device)
    std = base_checkpoint["input_std"].to(device)
    input_shape = tuple(base_checkpoint["input_shape"])
    specialists = {
        stage: load_specialist(
            Path(args.stack_root) / directory / "best_feasible.pt",
            base.input_dim,
            device,
        )
        for stage, directory in zip(STAGES, STAGE_DIRS)
    }
    sequential_mse_control = load_specialist(
        Path(args.sequential_mse_control), base.input_dim, device
    )
    direct_mse_control = load_specialist(
        Path(args.direct_mse_control), base.input_dim, device
    )
    comparison_stacks = {}
    if args.pure_stack_root:
        comparison_stacks["pure_transformed"] = load_stack(
            args.pure_stack_root, STAGE_DIRS, base.input_dim, device
        )
    if args.mse_stack_root:
        comparison_stacks["mse_four_stage"] = load_stack(
            args.mse_stack_root,
            tuple(f"stage_{index}_mse" for index in range(1, 5)),
            base.input_dim,
            device,
        )
    autoencoder = load_spatial_autoencoder(args.autoencoder, device)
    autoencoder.eval().requires_grad_(False)

    mode_metadata = build_mode_metadata(
        include_pure_stack=bool(args.pure_stack_root),
        include_mse_stack=bool(args.mse_stack_root),
    )
    mode_by_name = {record["mode"]: record for record in mode_metadata}
    values = {
        record["mode"]: {metric: [] for metric in METRICS}
        for record in mode_metadata
    }
    stage_values = {
        stage: {
            "incoming_residual_mse": [],
            "outgoing_residual_mse": [],
            "write_rms": [],
            "write_residual_cosine": [],
        }
        for stage in STAGES
    }
    target_sum = 0.0
    target_squared_sum = 0.0
    target_elements = 0

    for start in range(0, len(valid_indices), args.batch_size):
        selected = valid_indices[start:start + args.batch_size]
        raw = raw_latents[selected].to(device).float()
        standardized = ((raw - mean) / std).flatten(1)
        base_reconstruction = base(standardized)[0]
        target_sum += float(standardized.sum())
        target_squared_sum += float(standardized.square().sum())
        target_elements += standardized.numel()

        clean_prefixes = {0: base_reconstruction}
        clean_deltas = {}
        reconstruction = base_reconstruction
        for index, stage in enumerate(STAGES, start=1):
            incoming = standardized - reconstruction
            delta = specialists[stage](incoming)[0]
            reconstruction = reconstruction + delta
            outgoing = standardized - reconstruction
            clean_deltas[stage] = delta
            clean_prefixes[index] = reconstruction
            incoming_flat = incoming.flatten(1)
            delta_flat = delta.flatten(1)
            stage_values[stage]["incoming_residual_mse"].append(
                incoming_flat.square().mean(1).cpu()
            )
            stage_values[stage]["outgoing_residual_mse"].append(
                outgoing.flatten(1).square().mean(1).cpu()
            )
            stage_values[stage]["write_rms"].append(
                delta_flat.square().mean(1).sqrt().cpu()
            )
            stage_values[stage]["write_residual_cosine"].append(
                torch.nn.functional.cosine_similarity(
                    delta_flat, incoming_flat, dim=1, eps=1e-12
                ).cpu()
            )
        full_reconstruction = clean_prefixes[len(STAGES)]

        reconstructions = {}
        for record in mode_metadata:
            name = record["mode"]
            if record["family"] == "recomputed_subset":
                subset = tuple(record["subset"])
                if not subset:
                    reconstructions[name] = base_reconstruction
                elif subset == STAGES:
                    reconstructions[name] = full_reconstruction
                elif subset == STAGES[:len(subset)]:
                    reconstructions[name] = clean_prefixes[len(subset)]
                else:
                    reconstructions[name] = apply_sequence(
                        standardized, base_reconstruction, specialists, subset
                    )
            elif record["family"] == "frozen_write_ablation":
                reconstructions[name] = (
                    full_reconstruction - clean_deltas[record["removed_stage"]]
                )
            elif record["family"] == "order_diagnostic":
                reconstructions[name] = apply_sequence(
                    standardized,
                    base_reconstruction,
                    specialists,
                    tuple(reversed(STAGES)),
                )
            elif name == "sequential_mse_residual_control":
                base_residual = standardized - base_reconstruction
                reconstructions[name] = (
                    base_reconstruction + sequential_mse_control(base_residual)[0]
                )
            elif record["family"] == "comparison_stack":
                reconstructions[name] = apply_model_sequence(
                    standardized,
                    base_reconstruction,
                    comparison_stacks[record["stack_type"]][
                        :record["subset_size"]
                    ],
                )
            else:
                reconstructions[name] = direct_mse_control(standardized)[0]

        mode_names = list(reconstructions)
        for chunk_start in range(0, len(mode_names), args.mode_chunk_size):
            chunk_names = mode_names[chunk_start:chunk_start + args.mode_chunk_size]
            combined = torch.cat([reconstructions[name] for name in chunk_names])
            repeated_target = standardized.repeat(len(chunk_names), 1)
            latent_mse = (repeated_target - combined).square().mean(1)
            reconstructed_raw = combined.view(-1, *input_shape) * std + mean
            repeated_raw = raw.repeat((len(chunk_names), 1, 1, 1))
            transformed, pixel_mse = spatial_geometry_losses(
                autoencoder, repeated_raw, reconstructed_raw, powers=None
            )
            batch_size = len(selected)
            for chunk_index, name in enumerate(chunk_names):
                section = slice(
                    chunk_index * batch_size, (chunk_index + 1) * batch_size
                )
                values[name]["latent_mse"].append(latent_mse[section].cpu())
                values[name]["pixel_mse"].append(pixel_mse[section].cpu())
                for objective in STAGES:
                    values[name][objective].append(
                        transformed[objective][section].cpu()
                    )

        if start == 0 or (start // args.batch_size + 1) % 25 == 0:
            print(
                f"evaluated {min(start + args.batch_size, len(valid_indices))}"
                f"/{len(valid_indices)} validation examples",
                flush=True,
            )

    target_mean = target_sum / target_elements
    target_variance = target_squared_sum / target_elements - target_mean ** 2
    summaries = []
    per_example_rows = []
    concatenated = {}
    ev_by_mode = {}
    for record in mode_metadata:
        name = record["mode"]
        concatenated[name] = {
            metric: torch.cat(parts) for metric, parts in values[name].items()
        }
        ev_values = 1.0 - concatenated[name]["latent_mse"] / target_variance
        ev_by_mode[name] = ev_values
        ev_mean, ev_low, ev_high = mean_ci(ev_values)
        summary = {
            **record,
            "latent_ev": ev_mean,
            "latent_ev_ci95_low": ev_low,
            "latent_ev_ci95_high": ev_high,
        }
        for metric in METRICS:
            metric_mean, metric_low, metric_high = mean_ci(
                concatenated[name][metric]
            )
            summary[metric] = metric_mean
            summary[f"{metric}_ci95_low"] = metric_low
            summary[f"{metric}_ci95_high"] = metric_high
        summaries.append(summary)
        for sample_offset, sample_index in enumerate(valid_indices.tolist()):
            per_example_rows.append(
                {
                    "sample_index": sample_index,
                    "mode": name,
                    "family": record["family"],
                    "latent_ev": float(ev_values[sample_offset]),
                    **{
                        metric: float(concatenated[name][metric][sample_offset])
                        for metric in METRICS
                    },
                }
            )

    full_ev_values = ev_by_mode["full_stack"]
    for summary in summaries:
        paired_delta = 100.0 * (
            ev_by_mode[summary["mode"]] - full_ev_values
        )
        delta_mean, delta_low, delta_high = mean_ci(paired_delta)
        summary["ev_change_vs_full_stack_pp"] = delta_mean
        summary["ev_change_vs_full_stack_pp_ci95_low"] = delta_low
        summary["ev_change_vs_full_stack_pp_ci95_high"] = delta_high

    stage_summaries = []
    for stage in STAGES:
        stage_record = {"stage": stage}
        stage_data = {
            metric: torch.cat(parts)
            for metric, parts in stage_values[stage].items()
        }
        for metric, metric_values in stage_data.items():
            metric_mean, metric_low, metric_high = mean_ci(metric_values)
            stage_record[metric] = metric_mean
            stage_record[f"{metric}_ci95_low"] = metric_low
            stage_record[f"{metric}_ci95_high"] = metric_high
        stage_record["marginal_residual_mse_reduction"] = (
            stage_record["incoming_residual_mse"]
            - stage_record["outgoing_residual_mse"]
        )
        stage_summaries.append(stage_record)

    summary_fields = list(summaries[0])
    with (out_dir / "ablation_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    with (out_dir / "per_example_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_example_rows[0]))
        writer.writeheader()
        writer.writerows(per_example_rows)
    with (out_dir / "stage_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stage_summaries[0]))
        writer.writeheader()
        writer.writerows(stage_summaries)

    payload = {
        "validation_examples": len(valid_indices),
        "target_variance": target_variance,
        "stage_order": list(STAGES),
        "ablation_semantics": {
            "recomputed_subset": (
                "Included stages run in original relative order and receive the "
                "residual produced by the ablated prefix."
            ),
            "frozen_write_ablation": (
                "The selected clean-path stage write is removed without "
                "recomputing downstream writes."
            ),
            "order_diagnostic": (
                "All stages run in reverse order; this is intentionally "
                "out-of-distribution and only measures order sensitivity."
            ),
            "mse_control": (
                "Capacity-matched sequential residual and direct MSE controls "
                "evaluated on the same validation examples."
            ),
            "comparison_stack": (
                "Prefix reconstructions from architecture-matched pure-"
                "transformed and four-stage MSE stacks."
            ),
        },
        "summaries": summaries,
        "stage_diagnostics": stage_summaries,
        "config": vars(args),
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out_dir / "COMPLETED.json").write_text(
        json.dumps({"status": "complete", "modes": len(summaries)}, indent=2)
        + "\n"
    )
    print(json.dumps({"status": "complete", "modes": len(summaries)}), flush=True)


if __name__ == "__main__":
    main()
