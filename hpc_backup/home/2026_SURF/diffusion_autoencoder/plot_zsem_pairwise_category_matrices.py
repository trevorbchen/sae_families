#!/usr/bin/env python3
"""Plot objective-specific alignment distributions for every objective pair."""

import argparse
import csv
from functools import lru_cache
from itertools import combinations
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


OBJECTIVES = (
    "latent_mse",
    "laplacian",
    "fourier_bands",
    "fourier_low",
    "fourier_mid",
    "fourier_high",
    "weighted_laplacian",
    "weighted_fourier_bands",
    "weighted_fourier_low",
    "weighted_fourier_mid",
    "weighted_fourier_high",
)
OBJECTIVE_LABELS = {
    "latent_mse": "MSE",
    "laplacian": "Laplacian",
    "fourier_bands": "Bands",
    "fourier_low": "Low",
    "fourier_mid": "Mid",
    "fourier_high": "High",
    "weighted_laplacian": "MSE + Lap",
    "weighted_fourier_bands": "MSE + Bands",
    "weighted_fourier_low": "MSE + Low",
    "weighted_fourier_mid": "MSE + Mid",
    "weighted_fourier_high": "MSE + High",
}
METRICS = {
    "decoder_cosine": (
        "hungarian_decoder_cosine",
        "Decoder-vector alignment by objective pair",
    ),
    "activation_correlation": (
        "hungarian_activation_correlation",
        "Feature firing-pattern alignment by objective pair (exploratory)",
    ),
    "activation_cka": (
        "linear_cka",
        "Vocabulary geometry (activation CKA) by objective pair",
    ),
    "shared_atoms_0p5": (
        None,
        "Strict atom recurrence by objective pair (cosine >= 0.5)",
    ),
}
CATEGORIES = (
    ("within_a", "A", "#d95f02"),
    ("within_b", "B", "#7570b3"),
    ("same_seed", "Seed", "#1b9e77"),
    ("cross_seed", "Base", "#6b7280"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def metric_value(row, metric):
    source = METRICS[metric][0]
    if source is not None:
        return float(row[source])
    return 0.5 * (
        float(row["shared_fraction_a_0p5"])
        + float(row["shared_fraction_b_0p5"])
    )


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def category_values(rows, objective_a, objective_b, metric):
    between = [
        row for row in rows
        if {row["objective_a"], row["objective_b"]}
        == {objective_a, objective_b}
    ]
    same_seed = [
        metric_value(row, metric)
        for row in between
        if int(row["seed_a"]) == int(row["seed_b"])
    ]
    cross_seed = [
        metric_value(row, metric)
        for row in between
        if int(row["seed_a"]) != int(row["seed_b"])
    ]
    return {
        "within_a": diagonal_values(rows, objective_a, metric),
        "within_b": diagonal_values(rows, objective_b, metric),
        "same_seed": np.asarray(same_seed),
        "cross_seed": np.asarray(cross_seed),
    }


def diagonal_values(rows, objective, metric):
    return np.asarray([
        metric_value(row, metric)
        for row in rows
        if row["objective_a"] == objective
        and row["objective_b"] == objective
    ])


@lru_cache(maxsize=None)
def exact_bootstrap_interval(values):
    """Exact percentile bootstrap for the small samples used in each cell."""
    sample = np.asarray(values, dtype=float)
    indices = np.asarray(list(product(range(len(sample)), repeat=len(sample))))
    bootstrap_means = sample[indices].mean(axis=1)
    return tuple(np.quantile(bootstrap_means, (0.025, 0.975)))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    present = {
        row[field]
        for row in rows
        for field in ("objective_a", "objective_b")
    }
    objectives = tuple(objective for objective in OBJECTIVES if objective in present)

    summary_rows = []
    for objective_a, objective_b in combinations(objectives, 2):
        for metric in METRICS:
            values = category_values(rows, objective_a, objective_b, metric)
            for category, _, _ in CATEGORIES:
                sample = values[category]
                summary_rows.append({
                    "objective_a": objective_a,
                    "objective_b": objective_b,
                    "metric": metric,
                    "category": category,
                    "count": len(sample),
                    "mean": sample.mean(),
                    "sd": sample.std(ddof=1),
                    "bootstrap_ci_low": exact_bootstrap_interval(tuple(sample))[0],
                    "bootstrap_ci_high": exact_bootstrap_interval(tuple(sample))[1],
                    "minimum": sample.min(),
                    "maximum": sample.max(),
                })
    write_csv(args.output_dir / "pairwise_category_summary.csv", summary_rows)

    rng = np.random.default_rng(args.seed)
    for metric, (_, metric_label) in METRICS.items():
        all_values = np.asarray([metric_value(row, metric) for row in rows])
        observed_min = float(all_values.min())
        observed_max = float(all_values.max())
        padding = max((observed_max - observed_min) * 0.06, 1e-4)
        lower = max(0.0, observed_min - padding)
        upper = observed_max + padding

        figure, axes = plt.subplots(
            len(objectives), len(objectives), figsize=(17, 16),
            sharex=True, sharey=True, constrained_layout=True,
        )
        for row_index, objective_a in enumerate(objectives):
            for column_index, objective_b in enumerate(objectives):
                axis = axes[row_index, column_index]
                axis.set_ylim(lower, upper)
                axis.grid(axis="y", alpha=0.13, linewidth=0.6)
                if row_index == column_index:
                    sample = diagonal_values(rows, objective_a, metric)
                    jitter = rng.uniform(-0.08, 0.08, len(sample))
                    axis.scatter(
                        jitter, sample, s=18, color=CATEGORIES[0][2],
                        alpha=0.7, edgecolors="none",
                    )
                    axis.plot(
                        [-0.22, 0.22], [sample.mean(), sample.mean()],
                        color="black", linewidth=1.8,
                    )
                    ci_low, ci_high = exact_bootstrap_interval(tuple(sample))
                    axis.errorbar(
                        0, sample.mean(),
                        yerr=[[sample.mean() - ci_low], [ci_high - sample.mean()]],
                        fmt="none", ecolor="black", capsize=3, linewidth=1.2,
                    )
                else:
                    values = category_values(
                        rows, objective_a, objective_b, metric
                    )
                    for position, (category, _, color) in enumerate(CATEGORIES):
                        sample = values[category]
                        jitter = rng.uniform(-0.10, 0.10, len(sample))
                        axis.scatter(
                            position + jitter, sample, s=14, color=color,
                            alpha=0.55, edgecolors="none",
                        )
                        axis.plot(
                            [position - 0.22, position + 0.22],
                            [sample.mean(), sample.mean()],
                            color="black", linewidth=1.5,
                        )
                        ci_low, ci_high = exact_bootstrap_interval(tuple(sample))
                        axis.errorbar(
                            position, sample.mean(),
                            yerr=[
                                [sample.mean() - ci_low],
                                [ci_high - sample.mean()],
                            ],
                            fmt="none", ecolor="black", capsize=2,
                            linewidth=1.0,
                        )
                axis.set_xlim(-0.45, 3.45)
                if row_index == column_index:
                    tick_labels = (OBJECTIVE_LABELS[objective_a], "", "", "")
                else:
                    tick_labels = (
                        OBJECTIVE_LABELS[objective_a],
                        OBJECTIVE_LABELS[objective_b],
                        "Seed",
                        "Base",
                    )
                axis.set_xticks((0, 1, 2, 3), tick_labels, fontsize=5)
                axis.tick_params(axis="x", labelbottom=True)
                if column_index != 0:
                    axis.tick_params(axis="y", labelleft=False)
                else:
                    axis.tick_params(axis="y", labelsize=7)
                    axis.set_ylabel(OBJECTIVE_LABELS[objective_a], fontsize=10)
                if row_index == 0:
                    axis.set_title(OBJECTIVE_LABELS[objective_b], fontsize=10)

        handles = [
            Line2D(
                [0], [0], marker="o", linestyle="none", color=color,
                label={
                    "within_a": "A: row objective, different seeds",
                    "within_b": "B: column objective, different seeds",
                    "same_seed": "Seed: different objective, same seed",
                    "cross_seed": "Base: different objective, different seed",
                }[category],
            )
            for category, _, color in CATEGORIES
        ]
        figure.legend(
            handles=handles, loc="upper center", ncol=4,
            bbox_to_anchor=(0.5, 1.025), frameon=False, fontsize=10,
        )
        figure.suptitle(metric_label, fontsize=15)
        output = args.output_dir / f"pairwise_categories_{metric}.png"
        figure.savefig(output, dpi=220, bbox_inches="tight")
        figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
