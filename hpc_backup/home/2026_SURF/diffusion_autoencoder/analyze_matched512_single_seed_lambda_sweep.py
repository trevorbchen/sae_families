#!/usr/bin/env python3
"""Render same-seed vocabulary heatmaps across a matched lambda sweep."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_spatial_sae_vocabulary_alignment import (
    align_pair,
    assigned_mean,
    linear_cka,
    load_vocabulary,
    shared_mask,
)


LAMBDAS = (
    ("0p1", "0.1"), ("0p3", "0.3"), ("1p0", "1"),
    ("3p0", "3"), ("10p0", "10"), ("30p0", "30"),
    ("100p0", "100"),
)
OBJECTIVES = (
    ("latent_mse", "MSE", None),
    ("weighted_laplacian", "Laplacian", "laplacian"),
    ("weighted_fourier_low", "Fourier low", "fourier_low"),
    ("weighted_fourier_mid", "Fourier mid", "fourier_mid"),
    ("weighted_fourier_high", "Fourier high", "fourier_high"),
)
METRICS = (
    ("decoder_cosine", "Decoder cosine"),
    ("activation_correlation", "Activation correlation"),
    ("activation_cka", "Activation CKA"),
    ("shared_atoms_0p5", "Shared atoms (cosine >= 0.5)"),
)


def checkpoint_path(root, lambda_tag, directory, seed):
    if directory is None:
        return root / "checkpoints" / "mse_control" / "latent_mse" / f"seed_{seed}" / "last.pt"
    return root / "checkpoints" / f"lambda_{lambda_tag}" / directory / f"seed_{seed}" / "last.pt"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair_metrics(vocabulary_a, vocabulary_b):
    pair = align_pair(vocabulary_a, vocabulary_b)
    shared_a = shared_mask(
        pair["decoder_similarity"], pair["encoder_similarity"],
        pair["decoder_map_a"], pair["encoder_map_a"], 0.5,
    )
    shared_b = shared_mask(
        pair["decoder_similarity"].T, pair["encoder_similarity"].T,
        pair["decoder_map_b"], pair["encoder_map_b"], 0.5,
    )
    return {
        "decoder_cosine": assigned_mean(
            pair["decoder_similarity"], pair["decoder_map_a"]
        ),
        "activation_correlation": assigned_mean(
            pair["activation_eval"], pair["activation_map_a"]
        ),
        "activation_cka": linear_cka(
            vocabulary_a["eval_activations"],
            vocabulary_b["eval_activations"],
        ),
        "shared_atoms_0p5": 0.5 * (
            float(shared_a.float().mean()) + float(shared_b.float().mean())
        ),
    }


def draw_heatmap(axis, matrix, title, vmin, vmax, labels, colorbar=False):
    masked = np.ma.masked_invalid(matrix)
    image = axis.imshow(masked, cmap="magma", vmin=vmin, vmax=vmax)
    image.cmap.set_bad("#eceff1")
    axis.set_xticks(range(len(labels)), labels, rotation=28, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title(title, fontweight="bold")
    for row in range(len(labels)):
        for column in range(len(labels)):
            if row == column:
                axis.text(column, row, "self", ha="center", va="center", fontsize=8)
            else:
                value = matrix[row, column]
                rgba = image.cmap(image.norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                axis.text(
                    column, row, f"{value:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if luminance < 0.48 else "#16191d",
                )
    axis.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.4)
    axis.tick_params(which="minor", bottom=False, left=False)
    if colorbar:
        axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.035)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--top-features", type=int, default=64)
    parser.add_argument("--min-active", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
    raw_latents = cache["latents"].float()
    valid_indices = cache["valid_indices"]
    mse_path = checkpoint_path(args.experiment_root, "", None, args.seed)
    mse_vocabulary = load_vocabulary(
        mse_path, raw_latents, valid_indices, args.top_features,
        args.min_active, args.batch_size, device,
    )

    results = {}
    all_rows = []
    for lambda_tag, lambda_label in LAMBDAS:
        print(f"analyzing lambda={lambda_label}", flush=True)
        vocabularies = {"latent_mse": mse_vocabulary}
        for objective, _, directory in OBJECTIVES[1:]:
            path = checkpoint_path(
                args.experiment_root, lambda_tag, directory, args.seed
            )
            vocabularies[objective] = load_vocabulary(
                path, raw_latents, valid_indices, args.top_features,
                args.min_active, args.batch_size, device,
            )
        matrices = {
            metric: np.full((len(OBJECTIVES), len(OBJECTIVES)), np.nan)
            for metric, _ in METRICS
        }
        for row, (objective_a, _, _) in enumerate(OBJECTIVES):
            for column in range(row + 1, len(OBJECTIVES)):
                objective_b = OBJECTIVES[column][0]
                values = pair_metrics(
                    vocabularies[objective_a], vocabularies[objective_b]
                )
                for metric, value in values.items():
                    matrices[metric][row, column] = value
                    matrices[metric][column, row] = value
                    all_rows.append({
                        "lambda": lambda_label,
                        "lambda_tag": lambda_tag,
                        "seed": args.seed,
                        "objective_a": objective_a,
                        "objective_b": objective_b,
                        "metric": metric,
                        "value": value,
                    })
        results[lambda_tag] = matrices

    write_csv(args.out_dir / "same_seed_pairwise_similarity.csv", all_rows)
    scales = {}
    for metric, _ in METRICS:
        values = np.concatenate([
            matrices[metric][np.isfinite(matrices[metric])]
            for matrices in results.values()
        ])
        padding = max(1e-4, 0.04 * (values.max() - values.min()))
        scales[metric] = (max(0.0, float(values.min() - padding)),
                          float(values.max() + padding))

    labels = [label for _, label, _ in OBJECTIVES]
    for lambda_tag, lambda_label in LAMBDAS:
        target = args.out_dir / f"lambda_{lambda_tag}"
        target.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
        for axis, (metric, title) in zip(axes.flat, METRICS):
            draw_heatmap(
                axis, results[lambda_tag][metric], title,
                *scales[metric], labels, colorbar=True,
            )
        figure.suptitle(
            f"Same-seed vocabulary comparison - lambda={lambda_label}",
            fontsize=17, fontweight="bold",
        )
        figure.savefig(target / "same_seed_vocabulary_heatmaps.png", dpi=220)
        figure.savefig(target / "same_seed_vocabulary_heatmaps.pdf")
        plt.close(figure)

        for metric, title in METRICS:
            figure, axis = plt.subplots(figsize=(8.4, 7.3), constrained_layout=True)
            draw_heatmap(
                axis, results[lambda_tag][metric], title,
                *scales[metric], labels, colorbar=True,
            )
            axis.set_title(f"{title} - lambda={lambda_label}", fontweight="bold")
            figure.savefig(target / f"same_seed_{metric}.png", dpi=220)
            figure.savefig(target / f"same_seed_{metric}.svg")
            plt.close(figure)

    for metric, title in METRICS:
        figure, axes = plt.subplots(2, 4, figsize=(18, 9.5), constrained_layout=True)
        image = None
        for axis, (lambda_tag, lambda_label) in zip(axes.flat, LAMBDAS):
            image = draw_heatmap(
                axis, results[lambda_tag][metric], f"lambda={lambda_label}",
                *scales[metric], labels,
            )
        axes.flat[-1].axis("off")
        figure.colorbar(image, ax=list(axes.flat[:-1]), fraction=0.025, pad=0.02)
        figure.suptitle(
            f"{title} across transform weights - seed {args.seed}",
            fontsize=17, fontweight="bold",
        )
        figure.savefig(args.out_dir / f"overview_{metric}.png", dpi=220)
        figure.savefig(args.out_dir / f"overview_{metric}.pdf")
        plt.close(figure)

    (args.out_dir / "README.md").write_text(
        "# Same-seed vocabulary heatmaps\n\n"
        "These matrices compare seed-2030 vocabularies across objectives for each "
        "transform weight. Diagonal cells are omitted because self-similarity is "
        "trivially one. Color limits are shared across all seven lambdas for each "
        "metric. These are exploratory objective comparisons, not substitutes for "
        "the three-seed cross-seed confidence-interval analysis.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
