#!/usr/bin/env python3
"""Test objective and initialization-seed effects in SAE vocabulary alignment."""

import argparse
import csv
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OBJECTIVES = (
    "latent_mse",
    "laplacian",
    "fourier_bands",
    "fourier_low",
    "fourier_mid",
    "fourier_high",
)
SEEDS = (2030, 2031, 2032)
METRICS = {
    "decoder_cosine": "hungarian_decoder_cosine",
    "activation_correlation": "hungarian_activation_correlation",
    "activation_cka": "linear_cka",
    "shared_atoms_0p5": None,
}
METRIC_LABELS = {
    "decoder_cosine": "Decoder cosine",
    "activation_correlation": "Activation correlation",
    "activation_cka": "Activation CKA",
    "shared_atoms_0p5": "Shared atoms (cosine >= 0.5)",
}
CATEGORY_LABELS = {
    "cross_objective_cross_seed": "Different objective\ndifferent seed",
    "same_objective_cross_seed": "Same objective\ndifferent seed",
    "cross_objective_same_seed": "Different objective\nsame seed",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--bootstraps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def metric_value(row, metric):
    source = METRICS[metric]
    if source is not None:
        return float(row[source])
    return 0.5 * (
        float(row["shared_fraction_a_0p5"])
        + float(row["shared_fraction_b_0p5"])
    )


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile_interval(values):
    return tuple(np.quantile(values, (0.025, 0.975)))


def bootstrap_mean(units, count, rng):
    units = np.asarray(units, dtype=float)
    indices = rng.integers(0, len(units), size=(count, len(units)))
    return units[indices].mean(axis=1)


def bh_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(p_values) - reverse_rank + 1
        running = min(running, p_values[index] * len(p_values) / rank)
        adjusted[index] = running
    return np.clip(adjusted, 0, 1)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with args.input.open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))

    rows = []
    for raw in raw_rows:
        objective_a = raw["objective_a"]
        objective_b = raw["objective_b"]
        seed_a = int(raw["seed_a"])
        seed_b = int(raw["seed_b"])
        same_objective = objective_a == objective_b
        same_seed = seed_a == seed_b
        if same_objective:
            category = "same_objective_cross_seed"
        elif same_seed:
            category = "cross_objective_same_seed"
        else:
            category = "cross_objective_cross_seed"
        row = {
            "objective_a": objective_a,
            "seed_a": seed_a,
            "objective_b": objective_b,
            "seed_b": seed_b,
            "category": category,
        }
        row.update({metric: metric_value(raw, metric) for metric in METRICS})
        rows.append(row)
    write_csv(args.output_dir / "categorized_pairs.csv", rows)

    categories = tuple(CATEGORY_LABELS)
    category_rows = []
    for metric in METRICS:
        for category in categories:
            values = np.array([
                row[metric] for row in rows if row["category"] == category
            ])
            category_rows.append({
                "metric": metric,
                "category": category,
                "pairs": len(values),
                "mean": values.mean(),
                "sd": values.std(ddof=1),
                "minimum": values.min(),
                "maximum": values.max(),
            })
    write_csv(args.output_dir / "category_summary.csv", category_rows)

    # Per-objective units: each within-objective mean uses its three cross-seed
    # pairs. Its baseline excludes all matched-seed cross-objective pairs.
    objective_units = {metric: [] for metric in METRICS}
    per_objective_rows = []
    for objective in OBJECTIVES:
        within = [
            row for row in rows
            if row["objective_a"] == objective
            and row["objective_b"] == objective
        ]
        cross = [
            row for row in rows
            if row["seed_a"] != row["seed_b"]
            and row["objective_a"] != row["objective_b"]
            and objective in (row["objective_a"], row["objective_b"])
        ]
        for metric in METRICS:
            within_mean = np.mean([row[metric] for row in within])
            baseline_mean = np.mean([row[metric] for row in cross])
            effect = within_mean - baseline_mean
            objective_units[metric].append(effect)
            per_objective_rows.append({
                "objective": objective,
                "metric": metric,
                "within_objective_cross_seed_pairs": len(within),
                "cross_objective_cross_seed_pairs": len(cross),
                "within_objective_mean": within_mean,
                "cross_objective_cross_seed_mean": baseline_mean,
                "effect": effect,
            })
    write_csv(args.output_dir / "per_objective_effects.csv", per_objective_rows)

    # Per-objective-pair seed units: exactly three matched-seed cells versus
    # the six off-diagonal seed cells in each 3x3 objective-pair block.
    seed_units = {metric: [] for metric in METRICS}
    per_objective_pair_rows = []
    for objective_a, objective_b in combinations(OBJECTIVES, 2):
        relevant = [
            row for row in rows
            if {row["objective_a"], row["objective_b"]}
            == {objective_a, objective_b}
        ]
        matched = [row for row in relevant if row["seed_a"] == row["seed_b"]]
        mismatched = [row for row in relevant if row["seed_a"] != row["seed_b"]]
        for metric in METRICS:
            matched_mean = np.mean([row[metric] for row in matched])
            mismatched_mean = np.mean([row[metric] for row in mismatched])
            effect = matched_mean - mismatched_mean
            seed_units[metric].append(effect)
            per_objective_pair_rows.append({
                "objective_a": objective_a,
                "objective_b": objective_b,
                "metric": metric,
                "same_seed_pairs": len(matched),
                "different_seed_pairs": len(mismatched),
                "same_seed_mean": matched_mean,
                "different_seed_mean": mismatched_mean,
                "effect": effect,
            })
    write_csv(
        args.output_dir / "per_objective_pair_seed_effects.csv",
        per_objective_pair_rows,
    )

    keys = [(objective, seed) for objective in OBJECTIVES for seed in SEEDS]
    key_index = {key: index for index, key in enumerate(keys)}
    pair_i = np.array([
        key_index[(row["objective_a"], row["seed_a"])] for row in rows
    ])
    pair_j = np.array([
        key_index[(row["objective_b"], row["seed_b"])] for row in rows
    ])
    metric_matrix = np.array([
        [row[metric] for row in rows] for metric in METRICS
    ])
    objective_labels = np.array([key[0] for key in keys], dtype=object)
    seed_labels = np.array([key[1] for key in keys])

    observed = {}
    baseline_mask = (
        (objective_labels[pair_i] != objective_labels[pair_j])
        & (seed_labels[pair_i] != seed_labels[pair_j])
    )
    objective_mask = (
        (objective_labels[pair_i] == objective_labels[pair_j])
        & (seed_labels[pair_i] != seed_labels[pair_j])
    )
    seed_mask = (
        (objective_labels[pair_i] != objective_labels[pair_j])
        & (seed_labels[pair_i] == seed_labels[pair_j])
    )
    observed["objective"] = (
        metric_matrix[:, objective_mask].mean(1)
        - metric_matrix[:, baseline_mask].mean(1)
    )
    observed["seed"] = (
        metric_matrix[:, seed_mask].mean(1)
        - metric_matrix[:, baseline_mask].mean(1)
    )

    null_objective = np.empty((args.permutations, len(METRICS)))
    null_seed = np.empty_like(null_objective)
    for iteration in range(args.permutations):
        permuted_objectives = objective_labels.copy()
        for seed in SEEDS:
            indices = np.where(seed_labels == seed)[0]
            permuted_objectives[indices] = rng.permutation(
                permuted_objectives[indices]
            )
        same_objective = (
            permuted_objectives[pair_i] == permuted_objectives[pair_j]
        )
        objective_null_mask = same_objective & (
            seed_labels[pair_i] != seed_labels[pair_j]
        )
        objective_null_baseline = (~same_objective) & (
            seed_labels[pair_i] != seed_labels[pair_j]
        )
        null_objective[iteration] = (
            metric_matrix[:, objective_null_mask].mean(1)
            - metric_matrix[:, objective_null_baseline].mean(1)
        )

        permuted_seeds = seed_labels.copy()
        for objective in OBJECTIVES:
            indices = np.where(objective_labels == objective)[0]
            permuted_seeds[indices] = rng.permutation(permuted_seeds[indices])
        same_seed = permuted_seeds[pair_i] == permuted_seeds[pair_j]
        seed_null_mask = same_seed & (
            objective_labels[pair_i] != objective_labels[pair_j]
        )
        seed_null_baseline = (~same_seed) & (
            objective_labels[pair_i] != objective_labels[pair_j]
        )
        null_seed[iteration] = (
            metric_matrix[:, seed_null_mask].mean(1)
            - metric_matrix[:, seed_null_baseline].mean(1)
        )

    effect_rows = []
    p_values = []
    for metric_index, metric in enumerate(METRICS):
        for effect_name, units, null in (
            ("objective", objective_units[metric], null_objective[:, metric_index]),
            ("seed", seed_units[metric], null_seed[:, metric_index]),
        ):
            bootstrap = bootstrap_mean(units, args.bootstraps, rng)
            ci_low, ci_high = percentile_interval(bootstrap)
            effect = observed[effect_name][metric_index]
            p_value = (
                1 + np.count_nonzero(np.abs(null) >= abs(effect))
            ) / (args.permutations + 1)
            effect_rows.append({
                "metric": metric,
                "effect": effect_name,
                "estimate": effect,
                "cluster_units": len(units),
                "unit_sd": np.std(units, ddof=1),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "permutation_null_mean": null.mean(),
                "permutation_null_sd": null.std(ddof=1),
                "permutation_p": p_value,
            })
            p_values.append(p_value)
    adjusted = bh_adjust(p_values)
    for row, q_value in zip(effect_rows, adjusted):
        row["fdr_q"] = q_value
    write_csv(args.output_dir / "effect_significance.csv", effect_rows)

    colors = ("#6b7280", "#d95f02", "#1b9e77")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    jitter_rng = np.random.default_rng(args.seed + 1)
    for axis, metric in zip(axes.flat, METRICS):
        for position, (category, color) in enumerate(zip(categories, colors)):
            values = np.array([
                row[metric] for row in rows if row["category"] == category
            ])
            jitter = jitter_rng.uniform(-0.12, 0.12, len(values))
            axis.scatter(
                position + jitter, values, s=13, alpha=0.35, color=color,
                edgecolors="none",
            )
            axis.errorbar(
                position, values.mean(), yerr=values.std(ddof=1), fmt="o",
                color="black", markersize=5, capsize=4, linewidth=1.4,
            )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xticks(range(3), [CATEGORY_LABELS[item] for item in categories])
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Alignment distributions (points are dependent pairwise cells)")
    fig.savefig(args.output_dir / "category_distributions.png", dpi=200)
    fig.savefig(args.output_dir / "category_distributions.svg")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, METRICS):
        selected = [row for row in effect_rows if row["metric"] == metric]
        for position, row in enumerate(selected):
            estimate = row["estimate"]
            lower = estimate - row["bootstrap_ci_low"]
            upper = row["bootstrap_ci_high"] - estimate
            axis.errorbar(
                position, estimate, yerr=[[lower], [upper]], fmt="o",
                color=colors[position + 1], capsize=5, markersize=7,
            )
            axis.text(
                position, row["bootstrap_ci_high"],
                f"q={row['fdr_q']:.3g}", ha="center", va="bottom", fontsize=9,
            )
        axis.axhline(0, color="black", linewidth=1, alpha=0.6)
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xticks((0, 1), ("Objective effect", "Seed effect"))
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Mean alignment effects with cluster-bootstrap 95% intervals")
    fig.savefig(args.output_dir / "effect_significance.png", dpi=200)
    fig.savefig(args.output_dir / "effect_significance.svg")


if __name__ == "__main__":
    main()
