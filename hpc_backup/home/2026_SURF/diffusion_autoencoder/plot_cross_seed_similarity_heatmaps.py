#!/usr/bin/env python3
"""Render cross-seed SAE vocabulary similarity as triangular heatmaps."""

import argparse
import csv
from itertools import combinations_with_replacement, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from plot_zsem_pairwise_category_matrices import (
    METRICS,
    OBJECTIVES,
    OBJECTIVE_LABELS,
    metric_value,
    write_csv,
)


CMAP = "coolwarm"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def pair_rows(rows, objective_a, objective_b):
    if objective_a == objective_b:
        return [
            row for row in rows
            if row["objective_a"] == objective_a
            and row["objective_b"] == objective_b
            and row["seed_a"] != row["seed_b"]
        ]
    return [
        row for row in rows
        if {row["objective_a"], row["objective_b"]}
        == {objective_a, objective_b}
        and row["seed_a"] != row["seed_b"]
    ]


def oriented_values(rows, objective_a, objective_b, metric):
    values = {}
    for row in pair_rows(rows, objective_a, objective_b):
        if row["objective_a"] == objective_a:
            seed_a, seed_b = int(row["seed_a"]), int(row["seed_b"])
        else:
            seed_a, seed_b = int(row["seed_b"]), int(row["seed_a"])
        values[(seed_a, seed_b)] = metric_value(row, metric)
    return values


def seed_bootstrap_ci(rows, objective_a, objective_b, metric):
    """Exact run-level bootstrap; pair scores sharing a seed stay dependent."""
    lookup = oriented_values(rows, objective_a, objective_b, metric)
    if objective_a == objective_b:
        lookup.update({(b, a): value for (a, b), value in list(lookup.items())})
        seeds = sorted({seed for pair in lookup for seed in pair})
        means = []
        for sample in product(seeds, repeat=len(seeds)):
            sample_values = [
                lookup[(sample[i], sample[j])]
                for i in range(len(sample))
                for j in range(i + 1, len(sample))
                if sample[i] != sample[j]
            ]
            if sample_values:
                means.append(np.mean(sample_values))
    else:
        seeds_a = sorted({a for a, _ in lookup})
        seeds_b = sorted({b for _, b in lookup})
        samples_a = tuple(product(seeds_a, repeat=len(seeds_a)))
        samples_b = tuple(product(seeds_b, repeat=len(seeds_b)))
        means = []
        for sample_a in samples_a:
            for sample_b in samples_b:
                sample_values = [
                    lookup[(seed_a, seed_b)]
                    for seed_a in sample_a
                    for seed_b in sample_b
                    if seed_a != seed_b
                ]
                if sample_values:
                    means.append(np.mean(sample_values))
    return np.quantile(np.asarray(means), (0.025, 0.975))


def summarize(rows, objectives):
    records = []
    by_metric = {metric: {} for metric in METRICS}
    for objective_a, objective_b in combinations_with_replacement(objectives, 2):
        relevant = pair_rows(rows, objective_a, objective_b)
        for metric in METRICS:
            sample = np.asarray([metric_value(row, metric) for row in relevant])
            ci_low, ci_high = seed_bootstrap_ci(
                rows, objective_a, objective_b, metric
            )
            record = {
                "objective_a": objective_a,
                "objective_b": objective_b,
                "metric": metric,
                "comparison": (
                    "within_objective_cross_seed"
                    if objective_a == objective_b
                    else "between_objective_cross_seed"
                ),
                "count": len(sample),
                "mean": sample.mean(),
                "sd": sample.std(ddof=1),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "minimum": sample.min(),
                "maximum": sample.max(),
            }
            records.append(record)
            by_metric[metric][(objective_a, objective_b)] = record
            by_metric[metric][(objective_b, objective_a)] = record
    return records, by_metric


def plot_heatmap(output_dir, objectives, metric, records):
    size = len(objectives)
    full_matrix = np.empty((size, size), dtype=float)
    for row, objective_a in enumerate(objectives):
        for column, objective_b in enumerate(objectives):
            full_matrix[row, column] = records[(objective_a, objective_b)]["mean"]

    baseline = float(full_matrix.mean())
    relative_matrix = 100.0 * (
        full_matrix / max(abs(baseline), 1e-12) - 1.0
    )
    matrix = np.where(np.tri(size, dtype=bool), relative_matrix, np.nan)
    visible_values = matrix[np.isfinite(matrix)]
    limit = max(1.0, float(np.abs(visible_values).max()))

    figure, axis = plt.subplots(figsize=(10.8, 9.2), constrained_layout=True)
    image = axis.imshow(
        np.ma.masked_invalid(matrix), cmap=CMAP, vmin=-limit, vmax=limit
    )
    labels = [OBJECTIVE_LABELS[objective] for objective in objectives]
    axis.set_xticks(range(size), labels)
    axis.set_yticks(range(size), labels)
    axis.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
    axis.tick_params(length=0, labelsize=10)
    plt.setp(axis.get_xticklabels(), rotation=28, ha="left", rotation_mode="anchor")
    axis.set_xticks(np.arange(-0.5, size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, size, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2.2)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    for row, objective_a in enumerate(objectives):
        for column, objective_b in enumerate(objectives):
            if row < column:
                continue
            record = records[(objective_a, objective_b)]
            red, green, blue, _ = image.cmap(image.norm(relative_matrix[row, column]))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = "white" if luminance < 0.48 else "#20262e"
            axis.text(
                column, row - 0.10, f'{record["mean"]:.3f}',
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=color,
            )
            axis.text(
                column, row + 0.20,
                f'[{record["bootstrap_ci_low"]:.3f}, '
                f'{record["bootstrap_ci_high"]:.3f}]',
                ha="center", va="center", fontsize=7.4, color=color,
            )
            if row == column:
                axis.add_patch(Rectangle(
                    (column - 0.5, row - 0.5), 1, 1, fill=False,
                    edgecolor="#20262e", linewidth=2.3,
                ))

    axis.set_title(
        f"{METRICS[metric][1]}\n"
        "Diagonal: same objective | Off-diagonal: different objectives | "
        "Different seeds throughout\nColor is relative to the five-vocabulary mean; text reports raw similarity",
        fontsize=14, pad=18,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.035)
    colorbar.set_label("Relative to the five-vocabulary mean (%)", fontsize=10)
    output = output_dir / f"pairwise_categories_{metric}.png"
    figure.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    present = {
        row[field] for row in rows for field in ("objective_a", "objective_b")
    }
    excluded = {"fourier_bands", "weighted_fourier_bands"}
    objectives = tuple(
        objective for objective in OBJECTIVES
        if objective in present and objective not in excluded
    )
    summary, by_metric = summarize(rows, objectives)
    write_csv(args.output_dir / "pairwise_category_summary.csv", summary)
    for metric in METRICS:
        plot_heatmap(args.output_dir, objectives, metric, by_metric[metric])


if __name__ == "__main__":
    main()
