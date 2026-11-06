#!/usr/bin/env python3
"""Create cross-substrate/capacity overview plots for the matched 512-D sweep."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARMS = (
    ("spatial_4x_k8", "Spatial 4x / k8"),
    ("spatial_16x_k32", "Spatial 16x / k32"),
    ("vector_4x_k8", "Vector 4x / k8"),
    ("vector_16x_k32", "Vector 16x / k32"),
)
LAMBDAS = (("0p1", 0.1), ("0p3", 0.3), ("1p0", 1.0), ("3p0", 3.0), ("10p0", 10.0))
OBJECTIVES = (
    ("laplacian", "Laplacian"),
    ("fourier_low", "Fourier low"),
    ("fourier_mid", "Fourier mid"),
    ("fourier_high", "Fourier high"),
)
SEEDS = (2030, 2031, 2032)
COLORS = ("#313640", "#17878b", "#6f7683", "#b73545")


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    controls = []
    diagnostics = []
    for arm, label in ARMS:
        root = args.family_root / arm
        diagnostics.extend({**row, "arm": arm, "arm_label": label} for row in read_csv(
            root / "summary" / "training_diagnostics" / "run_diagnostics.csv"
        ))
        control_values = []
        for seed in SEEDS:
            metrics = json.loads((
                root / "geometry" / "mse_control" / f"seed_{seed}" / "model_metrics.json"
            ).read_text())
            control_values.append(float(metrics["latent_explained_variance"]))
        controls.append({
            "arm": arm,
            "arm_label": label,
            "objective": "latent_mse",
            "lambda": 0.0,
            "mean_ev": float(np.mean(control_values)),
            "sd_ev": float(np.std(control_values, ddof=1)),
        })
        for tag, value in LAMBDAS:
            for directory, objective_label in OBJECTIVES:
                values = []
                for seed in SEEDS:
                    metrics = json.loads((
                        root / "geometry" / f"lambda_{tag}" / directory /
                        f"seed_{seed}" / "model_metrics.json"
                    ).read_text())
                    values.append(float(metrics["latent_explained_variance"]))
                rows.append({
                    "arm": arm,
                    "arm_label": label,
                    "objective": directory,
                    "objective_label": objective_label,
                    "lambda": value,
                    "mean_ev": float(np.mean(values)),
                    "sd_ev": float(np.std(values, ddof=1)),
                })
    write_csv(args.out_dir / "family_reconstruction_overview.csv", controls + rows)
    write_csv(args.out_dir / "family_training_diagnostics.csv", diagnostics)

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True, sharey=True)
    for axis, (directory, objective_label) in zip(axes.flat, OBJECTIVES):
        for color, (arm, arm_label) in zip(COLORS, ARMS):
            selected = [row for row in rows if row["arm"] == arm and row["objective"] == directory]
            x = np.asarray([row["lambda"] for row in selected])
            y = 100 * np.asarray([row["mean_ev"] for row in selected])
            sd = 100 * np.asarray([row["sd_ev"] for row in selected])
            axis.plot(x, y, marker="o", color=color, label=arm_label)
            axis.fill_between(x, y - sd, y + sd, color=color, alpha=0.10)
        axis.set_xscale("log")
        axis.set_title(objective_label)
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Held-out latent EV (%)")
    axes[1, 0].set_ylabel("Held-out latent EV (%)")
    axes[1, 0].set_xlabel("Transform weight lambda")
    axes[1, 1].set_xlabel("Transform weight lambda")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(args.out_dir / "family_ev_overview.png", dpi=240, facecolor="white")
    figure.savefig(args.out_dir / "family_ev_overview.pdf", facecolor="white")
    plt.close(figure)

    (args.out_dir / "README.md").write_text(
        "# Matched 512-dimensional SAE family\n\n"
        "Both latent sources were trained on the same 32,768 FFHQ images with split seed "
        "2030. Every SAE arm uses seeds 2030--2032, proper MSE-anchored transformed losses, "
        "and the same lambda grid. `family_ev_overview` compares held-out latent explained "
        "variance; per-arm folders contain loss, latent-MSE, gradient-norm, pairwise-vocabulary, "
        "edited-output geometry, and causal-delta geometry diagnostics.\n"
    )


if __name__ == "__main__":
    main()
