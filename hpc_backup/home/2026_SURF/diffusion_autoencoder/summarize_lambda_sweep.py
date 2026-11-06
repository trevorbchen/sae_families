#!/usr/bin/env python3
"""Select lambda using reconstruction, output geometry, and vocabulary focus."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LAMBDAS = (("0p1", 0.1), ("0p3", 0.3), ("1p0", 1.0), ("3p0", 3.0), ("10p0", 10.0))
OBJECTIVES = (
    "weighted_laplacian",
    "weighted_fourier_low",
    "weighted_fourier_mid",
    "weighted_fourier_high",
)
DIRECTORIES = {
    "weighted_laplacian": "laplacian",
    "weighted_fourier_low": "fourier_low",
    "weighted_fourier_mid": "fourier_mid",
    "weighted_fourier_high": "fourier_high",
}
ALIGNED_METRICS = {
    "weighted_laplacian": "laplacian_norm",
    "weighted_fourier_low": "fourier_low_fraction",
    "weighted_fourier_mid": "fourier_mid_fraction",
    "weighted_fourier_high": "fourier_high_fraction",
}
PAIR_METRICS = {
    "decoder_cosine": "hungarian_decoder_cosine",
    "activation_correlation": "hungarian_activation_correlation",
    "activation_cka": "linear_cka",
    "shared_atoms_0p5": "shared_fraction_a_0p5",
}
SEEDS = (2030, 2031, 2032)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return float(np.mean(np.asarray(list(values), dtype=float)))


def bootstrap_gap(diagonal, off_diagonal, samples=5000, seed=2026):
    diagonal = np.asarray(diagonal, dtype=float)
    off_diagonal = np.asarray(off_diagonal, dtype=float)
    generator = np.random.default_rng(seed)
    diagonal_draws = generator.choice(
        diagonal, size=(samples, len(diagonal)), replace=True
    ).mean(1)
    off_diagonal_draws = generator.choice(
        off_diagonal, size=(samples, len(off_diagonal)), replace=True
    ).mean(1)
    gaps = diagonal_draws - off_diagonal_draws
    return float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975))


def rank01(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks / max(1, len(values) - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-mean-ev-drop-pp", type=float, default=2.0)
    parser.add_argument("--max-objective-ev-drop-pp", type=float, default=3.0)
    args = parser.parse_args()
    root = Path(args.experiment_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    control_metrics = [
        json.loads((root / "geometry" / "mse_control" / f"seed_{seed}" / "model_metrics.json").read_text())
        for seed in SEEDS
    ]
    control_ev = mean(row["latent_explained_variance"] for row in control_metrics)
    control_geometry = []
    for seed in SEEDS:
        control_geometry.extend(read_csv(
            root / "geometry" / "mse_control" / f"seed_{seed}" / "feature_means.csv"
        ))

    reconstruction_rows = []
    geometry_rows = []
    pairwise_rows = []
    summary_rows = []

    for tag, lambda_value in LAMBDAS:
        objective_evs = {}
        geometry_by_objective = {"latent_mse": control_geometry}
        for objective in OBJECTIVES:
            directory = DIRECTORIES[objective]
            model_metrics = []
            feature_rows = []
            for seed in SEEDS:
                model_metrics.append(json.loads((
                    root / "geometry" / f"lambda_{tag}" / directory /
                    f"seed_{seed}" / "model_metrics.json"
                ).read_text()))
                feature_rows.extend(read_csv(
                    root / "geometry" / f"lambda_{tag}" / directory /
                    f"seed_{seed}" / "feature_means.csv"
                ))
            values = [row["latent_explained_variance"] for row in model_metrics]
            objective_evs[objective] = mean(values)
            geometry_by_objective[objective] = feature_rows
            reconstruction_rows.append({
                "lambda": lambda_value,
                "objective": objective,
                "mean_ev": mean(values),
                "sd_ev": float(np.std(values, ddof=1)),
                "control_mean_ev": control_ev,
                "ev_drop_pp": 100 * (control_ev - mean(values)),
            })

        geometry_gains = {"edited_output": [], "causal_delta": []}
        for measurement in geometry_gains:
            for objective in OBJECTIVES:
                metric = ALIGNED_METRICS[objective]
                aligned = mean(
                    float(row[metric]) for row in geometry_by_objective[objective]
                    if row["measurement"] == measurement
                )
                other_values = []
                for other, rows in geometry_by_objective.items():
                    if other == objective:
                        continue
                    other_values.extend(
                        float(row[metric]) for row in rows
                        if row["measurement"] == measurement
                    )
                other_mean = mean(other_values)
                seed_gains = []
                for seed in SEEDS:
                    aligned_seed = mean(
                        float(row[metric]) for row in geometry_by_objective[objective]
                        if row["measurement"] == measurement and int(row["seed"]) == seed
                    )
                    other_seed_values = []
                    for other, rows in geometry_by_objective.items():
                        if other == objective:
                            continue
                        other_seed_values.extend(
                            float(row[metric]) for row in rows
                            if row["measurement"] == measurement and int(row["seed"]) == seed
                        )
                    seed_gains.append(
                        aligned_seed / max(mean(other_seed_values), 1e-12) - 1.0
                    )
                gain = mean(seed_gains)
                geometry_gains[measurement].append(gain)
                geometry_rows.append({
                    "lambda": lambda_value,
                    "measurement": measurement,
                    "objective": objective,
                    "metric": metric,
                    "aligned_value": aligned,
                    "other_vocabulary_mean": other_mean,
                    "relative_alignment_gain": gain,
                    "seed_gain_sd": float(np.std(seed_gains, ddof=1)),
                    "seed_gains": json.dumps(seed_gains),
                })

        pairs = read_csv(root / "alignment" / f"lambda_{tag}" / "pairwise_alignment.csv")
        focus_gaps = []
        all_objectives = ("latent_mse",) + OBJECTIVES
        for metric_label, column in PAIR_METRICS.items():
            for objective in all_objectives:
                def pair_value(row):
                    if metric_label == "shared_atoms_0p5":
                        return 0.5 * (
                            float(row["shared_fraction_a_0p5"])
                            + float(row["shared_fraction_b_0p5"])
                        )
                    return float(row[column])

                diagonal = [
                    pair_value(row) for row in pairs
                    if row["objective_a"] == objective and row["objective_b"] == objective
                ]
                off_diagonal = [
                    pair_value(row) for row in pairs
                    if row["objective_a"] != row["objective_b"]
                    and objective in (row["objective_a"], row["objective_b"])
                    and row["seed_a"] != row["seed_b"]
                ]
                diagonal_mean = mean(diagonal)
                off_diagonal_mean = mean(off_diagonal)
                absolute_gap = diagonal_mean - off_diagonal_mean
                relative_gap = absolute_gap / max(
                    abs(diagonal_mean), abs(off_diagonal_mean), 1e-3
                )
                gap_ci_low, gap_ci_high = bootstrap_gap(diagonal, off_diagonal)
                focus_gaps.append(absolute_gap)
                pairwise_rows.append({
                    "lambda": lambda_value,
                    "metric": metric_label,
                    "objective": objective,
                    "diagonal_cross_seed_mean": diagonal_mean,
                    "off_diagonal_cross_seed_mean": off_diagonal_mean,
                    "absolute_gap": absolute_gap,
                    "absolute_gap_ci_low": gap_ci_low,
                    "absolute_gap_ci_high": gap_ci_high,
                    "relative_gap": relative_gap,
                    "diagonal_pairs": len(diagonal),
                    "off_diagonal_pairs": len(off_diagonal),
                })

        evs = list(objective_evs.values())
        mean_ev = mean(evs)
        mean_drop = 100 * (control_ev - mean_ev)
        worst_drop = max(100 * (control_ev - value) for value in evs)
        summary_rows.append({
            "lambda": lambda_value,
            "control_mean_ev": control_ev,
            "specialist_mean_ev": mean_ev,
            "mean_ev_drop_pp": mean_drop,
            "worst_objective_ev_drop_pp": worst_drop,
            "edited_output_alignment_gain": mean(geometry_gains["edited_output"]),
            "causal_delta_alignment_gain": mean(geometry_gains["causal_delta"]),
            "pairwise_diagonal_focus": mean(focus_gaps),
            "reconstruction_feasible": (
                mean_drop <= args.max_mean_ev_drop_pp
                and worst_drop <= args.max_objective_ev_drop_pp
            ),
        })

    geometry_rank = rank01([row["causal_delta_alignment_gain"] for row in summary_rows])
    focus_rank = rank01([row["pairwise_diagonal_focus"] for row in summary_rows])
    for index, row in enumerate(summary_rows):
        row["geometry_rank_score"] = float(geometry_rank[index])
        row["diagonal_focus_rank_score"] = float(focus_rank[index])
        row["selection_score"] = float((geometry_rank[index] + focus_rank[index]) / 2)
    feasible = [row for row in summary_rows if row["reconstruction_feasible"]]
    selected = max(feasible, key=lambda row: row["selection_score"]) if feasible else min(
        summary_rows, key=lambda row: row["mean_ev_drop_pp"]
    )

    write_csv(out_dir / "lambda_summary.csv", summary_rows)
    write_csv(out_dir / "objective_reconstruction.csv", reconstruction_rows)
    write_csv(out_dir / "feature_geometry_alignment.csv", geometry_rows)
    write_csv(out_dir / "pairwise_diagonal_focus.csv", pairwise_rows)
    selection = {
        "selected_lambda": selected["lambda"],
        "selected": selected,
        "selection_rule": {
            "reconstruction_constraint": (
                f"mean EV drop <= {args.max_mean_ev_drop_pp} pp and every objective "
                f"EV drop <= {args.max_objective_ev_drop_pp} pp"
            ),
            "within_feasible_set": (
                "equal-weight mean of rank-normalized causal-delta alignment gain and "
                "cross-seed diagonal-focus gap"
            ),
            "off_diagonal_control": "different objective and different seed only",
        },
    }
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (out_dir / "README.md").write_text(
        "# SAE transform-weight sweep\n\n"
        "This sweep uses three paired seeds for each of five transform weights. A weight is "
        "eligible only when its mean held-out latent-EV loss is at most 2 percentage points "
        "and no objective loses more than 3 points versus the matched MSE control. Among "
        "eligible weights, selection equally ranks (1) objective-aligned causal-delta image "
        "geometry and (2) vocabulary diagonal focus. Diagonal focus compares the same "
        "objective across different seeds against different objectives across different seeds; "
        "matched-seed off-diagonal comparisons are excluded to avoid initialization leakage.\n"
    )

    x = np.array([row["lambda"] for row in summary_rows])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].axhline(100 * control_ev, color="#555555", linestyle="--", label="MSE control")
    axes[0].plot(x, [100 * row["specialist_mean_ev"] for row in summary_rows], marker="o")
    axes[0].set_ylabel("Held-out latent EV (%)")
    axes[0].legend(frameon=False)
    axes[1].plot(x, [100 * row["causal_delta_alignment_gain"] for row in summary_rows], marker="o")
    axes[1].set_ylabel("Aligned causal-geometry gain (%)")
    axes[2].plot(x, [100 * row["pairwise_diagonal_focus"] for row in summary_rows], marker="o")
    axes[2].set_ylabel("Relative diagonal-focus gap (%)")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Transform weight lambda")
        axis.grid(alpha=0.2)
        axis.axvline(selected["lambda"], color="#008687", linestyle=":")
    fig.tight_layout()
    fig.savefig(out_dir / "lambda_tradeoff.png", dpi=220)
    fig.savefig(out_dir / "lambda_tradeoff.pdf")
    plt.close(fig)
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
