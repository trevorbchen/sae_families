#!/usr/bin/env python3
"""Aggregate matched-seed feature geometry measurements and render comparisons."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OBJECTIVES = (
    "latent_mse",
    "weighted_laplacian",
    "weighted_fourier_low",
    "weighted_fourier_mid",
    "weighted_fourier_high",
)
LABELS = ("MSE", "Laplacian", "Fourier low", "Fourier mid", "Fourier high")
METRICS = (
    "laplacian_norm",
    "fourier_low_fraction",
    "fourier_mid_fraction",
    "fourier_high_fraction",
)
METRIC_LABELS = ("Laplacian", "Fourier low", "Fourier mid", "Fourier high")


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def bootstrap(values, samples=5000, seed=2026):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0]), float(values[0])
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(samples, len(values)), replace=True).mean(1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root")
    parser.add_argument("--experiment-root")
    parser.add_argument("--lambda-tag")
    parser.add_argument("--seeds", type=int, nargs="+", default=(2030, 2031, 2032))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if bool(args.input_root) == bool(args.experiment_root):
        parser.error("provide exactly one of --input-root or --experiment-root")
    if args.experiment_root and not args.lambda_tag:
        parser.error("--lambda-tag is required with --experiment-root")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.input_root:
        for seed in args.seeds:
            rows.extend(read_csv(
                Path(args.input_root) / f"seed_{seed}" / "feature_means.csv"
            ))
    else:
        root = Path(args.experiment_root)
        directories = {
            "latent_mse": root / "geometry" / "mse_control",
            "weighted_laplacian": root / "geometry" / f"lambda_{args.lambda_tag}" / "laplacian",
            "weighted_fourier_low": root / "geometry" / f"lambda_{args.lambda_tag}" / "fourier_low",
            "weighted_fourier_mid": root / "geometry" / f"lambda_{args.lambda_tag}" / "fourier_mid",
            "weighted_fourier_high": root / "geometry" / f"lambda_{args.lambda_tag}" / "fourier_high",
        }
        for objective, directory in directories.items():
            for seed in args.seeds:
                loaded = read_csv(directory / f"seed_{seed}" / "feature_means.csv")
                for row in loaded:
                    row["objective"] = objective
                rows.extend(loaded)

    summaries = []
    for measurement in ("edited_output", "causal_delta"):
        for objective, label in zip(OBJECTIVES, LABELS):
            selected = [row for row in rows if row["measurement"] == measurement and row["objective"] == objective]
            for metric, metric_label in zip(METRICS, METRIC_LABELS):
                seed_means = []
                for seed in args.seeds:
                    values = [float(row[metric]) for row in selected if int(row["seed"]) == seed]
                    seed_means.append(float(np.mean(values)))
                mean, low, high = bootstrap(seed_means)
                summaries.append({
                    "measurement": measurement,
                    "objective": objective,
                    "objective_label": label,
                    "metric": metric,
                    "metric_label": metric_label,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "seed_means": json.dumps(seed_means),
                })

    with (out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    for measurement in ("edited_output", "causal_delta"):
        matrix = np.array([
            [next(float(row["mean"]) for row in summaries if row["measurement"] == measurement and row["objective"] == objective and row["metric"] == metric) for metric in METRICS]
            for objective in OBJECTIVES
        ])
        # Column-relative percent differences reveal vocabulary alignment despite differing metric scales.
        relative = 100 * (matrix / matrix.mean(0, keepdims=True) - 1)
        limit = max(1.0, float(np.abs(relative).max()))
        fig, ax = plt.subplots(figsize=(9.2, 5.6))
        image = ax.imshow(relative, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(METRIC_LABELS)), METRIC_LABELS)
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_xlabel("Measured output geometry")
        ax.set_ylabel("SAE training objective")
        title = "Edited output geometry" if measurement == "edited_output" else "Causal edit geometry: edited minus original"
        ax.set_title(title)
        for row in range(relative.shape[0]):
            for column in range(relative.shape[1]):
                ax.text(column, row, f"{relative[row, column]:+.1f}%", ha="center", va="center", fontsize=9)
        colorbar = fig.colorbar(image, ax=ax, pad=0.02)
        colorbar.set_label("Relative to the five-vocabulary mean")
        fig.tight_layout()
        fig.savefig(out_dir / f"{measurement}_geometry_alignment.png", dpi=220)
        fig.savefig(out_dir / f"{measurement}_geometry_alignment.pdf")
        plt.close(fig)

    aligned = []
    expected = {
        "weighted_laplacian": "laplacian_norm",
        "weighted_fourier_low": "fourier_low_fraction",
        "weighted_fourier_mid": "fourier_mid_fraction",
        "weighted_fourier_high": "fourier_high_fraction",
    }
    for measurement in ("edited_output", "causal_delta"):
        for objective, metric in expected.items():
            values = {
                row["objective"]: float(row["mean"])
                for row in summaries
                if row["measurement"] == measurement and row["metric"] == metric
            }
            aligned.append({
                "measurement": measurement,
                "objective": objective,
                "metric": metric,
                "aligned_value": values[objective],
                "other_mean": float(np.mean([value for key, value in values.items() if key != objective])),
                "aligned_rank": 1 + sum(value > values[objective] for value in values.values()),
            })
    with (out_dir / "aligned_objective_test.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aligned[0]))
        writer.writeheader()
        writer.writerows(aligned)
    (out_dir / "README.md").write_text(
        "# SAE feature output geometry\n\n"
        "Each row vocabulary is evaluated with four normalized image-space metrics. "
        "The edited-output view measures the geometry of the resulting image. The causal-delta "
        "view measures only `decoded(edited latent) - decoded(original latent)`. Fourier cells "
        "are fractions of non-DC power; Laplacian cells are squared discrete-Laplacian energy "
        "divided by image energy. Heatmaps show each cell relative to the mean across the five "
        "matched vocabularies for that metric.\n"
    )


if __name__ == "__main__":
    main()
