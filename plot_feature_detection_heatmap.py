#!/usr/bin/env python3
"""Render feature_detection_summary.csv as a colored table (PNG), one row per
representation (lambda x objective, plus a single mse_control reference row
averaged across lambdas since it's the same symlinked checkpoint every time),
one column per headline detection metric (top1/top5/top10 fraction of the
target signal explained). Each row is first averaged across the 20 planted
conditions and 3 seeds. Mirrors plot_linear_probe_heatmap.py's per-column
color scaling and reference-row treatment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRIC_ORDER = (
    "top1_fraction_of_target_explained",
    "top5_fraction_of_target_explained",
    "top10_fraction_of_target_explained",
)
METRIC_LABELS = {
    "top1_fraction_of_target_explained": "Top-1 feature\n(fraction of target explained)",
    "top5_fraction_of_target_explained": "Top-5 features\n(fraction of target explained)",
    "top10_fraction_of_target_explained": "Top-10 features\n(fraction of target explained)",
}
OBJECTIVE_ORDER = ("laplacian", "fourier_low", "fourier_mid", "fourier_high")
OBJECTIVE_LABELS = {
    "laplacian": "Laplacian",
    "fourier_low": "Fourier low",
    "fourier_mid": "Fourier mid",
    "fourier_high": "Fourier high",
}
REFERENCE_ROW = "mse_control"


def objective_dir_name(objective: str) -> str:
    # feature_detection_summary.csv stores training-config names
    # (weighted_laplacian, weighted_fourier_low, ...); normalize to the
    # short form used elsewhere (laplacian, fourier_low, ...).
    return objective.replace("weighted_", "")


def build_table(summary_csv: Path) -> tuple[list[str], pd.DataFrame]:
    df = pd.read_csv(summary_csv)
    df["objective_short"] = df["objective"].map(objective_dir_name)

    variant = df[df["objective_short"] != "latent_mse"]
    per_row = variant.groupby(["lambda", "objective_short"])[list(METRIC_ORDER)].mean()
    per_row.index = per_row.index.set_names(["lambda", "objective"])

    reps: list[str] = []
    rows = []
    lambdas = sorted(variant["lambda"].unique())
    for lam in lambdas:
        tag = f"{lam:g}".replace(".", "p")
        for objective in OBJECTIVE_ORDER:
            key = (lam, objective)
            if key not in per_row.index:
                continue
            reps.append(f"lambda_{tag}_{objective}")
            rows.append(per_row.loc[key])

    control = df[df["objective_short"] == "latent_mse"]
    if len(control):
        control_row = control[list(METRIC_ORDER)].mean()
        reps.insert(0, REFERENCE_ROW)
        rows.insert(0, control_row)

    table = pd.DataFrame(rows, index=reps)[list(METRIC_ORDER)]
    return reps, table


def representation_label(rep: str) -> str:
    if rep == REFERENCE_ROW:
        return "MSE control"
    _, tag, objective = rep.split("_", 2)
    lam = float(tag.replace("p", ".", 1))
    return f"lambda={lam:g}, {OBJECTIVE_LABELS.get(objective, objective)}"


def plot_heatmap(reps: list[str], table: pd.DataFrame, output: Path, title: str) -> None:
    n_rows, n_cols = table.shape
    fig, ax = plt.subplots(figsize=(3.3 * n_cols + 2.6, 0.55 * n_rows + 2.0))

    is_reference = [rep == REFERENCE_ROW for rep in reps]
    display = np.zeros((n_rows, n_cols))
    for col_index, column in enumerate(table.columns):
        values = table[column].to_numpy(dtype=float)
        variant_values = values[~np.array(is_reference)]
        lo, hi = np.nanmin(variant_values), np.nanmax(variant_values)
        span = hi - lo if hi > lo else 1.0
        display[:, col_index] = np.clip((values - lo) / span, 0, 1)

    cmap = plt.get_cmap("RdYlGn")
    ax.imshow(display, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_cols), [METRIC_LABELS.get(c, c) for c in table.columns], fontsize=14)
    ax.set_yticks(range(n_rows), [representation_label(r) for r in reps], fontsize=14)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index, rep in enumerate(reps):
        for col_index, column in enumerate(table.columns):
            value = table.iloc[row_index, col_index]
            if pd.isna(value):
                continue
            red, green, blue, _ = cmap(display[row_index, col_index])
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = "white" if luminance < 0.6 else "#1f2430"
            weight = "bold" if rep == REFERENCE_ROW else "normal"
            ax.text(col_index, row_index, f"{value * 100:.3f}%", ha="center", va="center",
                     fontsize=15, color=color, fontweight=weight)

    for row_index, rep in enumerate(reps):
        if rep == REFERENCE_ROW:
            ax.axhline(row_index + 0.5, color="#1f2430", linewidth=1.2)

    ax.set_title(title, fontsize=17, pad=14)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Planted-feature detection by representation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reps, table = build_table(args.summary_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_heatmap(reps, table, args.output, args.title)
    print(f"wrote {args.output}: {len(reps)} rows")


if __name__ == "__main__":
    main()
