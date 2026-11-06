#!/usr/bin/env python3
"""Aggregate SAE training histories and validation metrics for one lambda sweep."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ALL_LAMBDAS = {
    "0p1": 0.1, "0p3": 0.3, "1p0": 1.0, "3p0": 3.0,
    "10p0": 10.0, "30p0": 30.0, "100p0": 100.0,
}
OBJECTIVES = (
    ("laplacian", "Laplacian"),
    ("fourier_low", "Fourier low"),
    ("fourier_mid", "Fourier mid"),
    ("fourier_high", "Fourier high"),
)
SEEDS = (2030, 2031, 2032)


def checkpoint_metrics(path, target_variance=None):
    metrics = json.loads((path / "metrics.json").read_text())
    if target_variance is not None:
        for step in metrics["history"]:
            step["latent_ev_monitor"] = 1.0 - step["latent_mse_monitor"] / target_variance
    return metrics


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_curve(histories, key):
    common = sorted(set.intersection(*(
        {int(row["step"]) for row in history} for history in histories
    )))
    lookup = [
        {int(row["step"]): float(row[key]) for row in history}
        for history in histories
    ]
    values = np.asarray([[run[step] for step in common] for run in lookup])
    return np.asarray(common), values.mean(0), values.std(0, ddof=1)


def plot_curves(root, out_dir, key, ylabel, filename, lambdas, colors, target_variance=None):
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
    for axis, (directory, label) in zip(axes.flat, OBJECTIVES):
        for color, (tag, value) in zip(colors, lambdas):
            histories = [
                checkpoint_metrics(
                    root / "checkpoints" / f"lambda_{tag}" / directory / f"seed_{seed}",
                    target_variance,
                )["history"]
                for seed in SEEDS
            ]
            steps, mean, sd = mean_curve(histories, key)
            axis.plot(steps, mean, color=color, label=f"lambda={value:g}")
            axis.fill_between(steps, mean - sd, mean + sd, color=color, alpha=0.12)
        axis.set_title(label)
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    axes[1, 0].set_xlabel("Training step")
    axes[1, 1].set_xlabel("Training step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(out_dir / filename, dpi=220, facecolor="white")
    figure.savefig((out_dir / filename).with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lambda-tags", default="0p1,0p3,1p0,3p0,10p0",
                         help="Comma-separated lambda tags to include (must have all 3 seeds).")
    args = parser.parse_args()
    root = args.experiment_root
    lambda_tags = args.lambda_tags.split(",")
    LAMBDAS = tuple((tag, ALL_LAMBDAS[tag]) for tag in lambda_tags)
    COLORS = plt.get_cmap("coolwarm")(np.linspace(0.05, 0.95, len(LAMBDAS)))

    target_variance = None
    variance_path = root / "target_variance.json"
    if variance_path.exists():
        target_variance = json.loads(variance_path.read_text())["target_variance"]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    controls = []
    for seed in SEEDS:
        metrics = checkpoint_metrics(
            root / "checkpoints" / "mse_control" / "latent_mse" / f"seed_{seed}"
        )
        controls.append(metrics)
        rows.append({
            "lambda": 0.0,
            "objective": "latent_mse",
            "seed": seed,
            "valid_objective_loss": metrics["valid_objective_loss"],
            "valid_latent_mse": metrics["valid_latent_mse_monitor"],
            "valid_pixel_mse": metrics["valid_pixel_mse_monitor"],
            "used_features": metrics["used_features"],
            "dead_features": metrics["dead_features_at_end"],
            "final_gradient_norm": metrics["history"][-1]["gradient_norm_before_clip"],
        })
    for tag, value in LAMBDAS:
        for directory, _ in OBJECTIVES:
            for seed in SEEDS:
                metrics = checkpoint_metrics(
                    root / "checkpoints" / f"lambda_{tag}" / directory / f"seed_{seed}"
                )
                rows.append({
                    "lambda": value,
                    "objective": directory,
                    "seed": seed,
                    "valid_objective_loss": metrics["valid_objective_loss"],
                    "valid_latent_mse": metrics["valid_latent_mse_monitor"],
                    "valid_pixel_mse": metrics["valid_pixel_mse_monitor"],
                    "used_features": metrics["used_features"],
                    "dead_features": metrics["dead_features_at_end"],
                    "final_gradient_norm": metrics["history"][-1]["gradient_norm_before_clip"],
                })
    write_csv(out_dir / "run_diagnostics.csv", rows)
    plot_curves(root, out_dir, "loss", "Training objective", "objective_loss_curves.png", LAMBDAS, COLORS)
    plot_curves(root, out_dir, "latent_mse_monitor", "Latent MSE monitor", "latent_mse_curves.png", LAMBDAS, COLORS)
    plot_curves(root, out_dir, "gradient_norm_before_clip", "Gradient norm before clipping", "gradient_norm_curves.png", LAMBDAS, COLORS)
    if target_variance is not None:
        plot_curves(root, out_dir, "latent_ev_monitor", "Explained variance (approx.)",
                    "explained_variance_curves.png", LAMBDAS, COLORS, target_variance=target_variance)

    variance_note = (
        f"`explained_variance_curves.png` derives an approximate EV per training step as "
        f"`1 - latent_mse_monitor / target_variance` (target_variance={target_variance:.6g}, "
        "computed once per arm from a checkpoint's own channel_mean/channel_std and the "
        "held-out validation latents -- see compute_target_variance.py). This is an "
        "approximation of the rigorous held-out `latent_explained_variance` in "
        "lambda_summary.csv (which uses variance-of-residual, not mean-square-of-residual, "
        "and is measured at the very end of training only) -- the two agree closely when "
        "the residual is close to zero-mean, which is expected but not guaranteed early in "
        "training.\n"
        if target_variance is not None else ""
    )
    (out_dir / "README.md").write_text(
        "# Training diagnostics\n\n"
        "Curves show the mean across three SAE seeds; shading is one standard deviation. "
        "`run_diagnostics.csv` records final validation objective loss, latent and pixel "
        "MSE monitors, used/dead features, and final pre-clipping gradient norm.\n\n"
        + variance_note
    )


if __name__ == "__main__":
    main()
