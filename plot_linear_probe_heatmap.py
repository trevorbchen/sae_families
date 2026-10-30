#!/usr/bin/env python3
"""Render the linear-probe summary.csv as a colored table (PNG), one row per
representation (lambda x objective, plus mse_control/raw_latent references),
one column per probe family. Each column is colored independently (its own
sequential colormap scaled to that column's min/max) since the four families
use different metrics on different scales.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILY_ORDER = ("semantic", "geometry", "source_geometry", "controlled")
FAMILY_LABELS = {
    "semantic": "CelebA attributes\n(macro AUROC)",
    "geometry": "Decoder-output geometry\n(macro R2)",
    "source_geometry": "Source-image geometry\n(macro R2)",
    "controlled": "Controlled transforms\n(balanced accuracy)",
}
OBJECTIVE_ORDER = ("laplacian", "fourier_low", "fourier_mid", "fourier_high")
OBJECTIVE_LABELS = {
    "laplacian": "Laplacian",
    "fourier_low": "Fourier low",
    "fourier_mid": "Fourier mid",
    "fourier_high": "Fourier high",
}


def representation_order(representations: set[str]) -> list[str]:
    ordered: list[str] = []
    if "raw_latent" in representations:
        ordered.append("raw_latent")
    if "mse_control" in representations:
        ordered.append("mse_control")
    lambda_tags = sorted(
        {rep.split("_", 2)[1] for rep in representations if rep.startswith("lambda_")},
        key=lambda tag: float(tag.replace("p", ".", 1)),
    )
    for tag in lambda_tags:
        for objective in OBJECTIVE_ORDER:
            rep = f"lambda_{tag}_{objective}"
            if rep in representations:
                ordered.append(rep)
    return ordered


def representation_label(rep: str) -> str:
    if rep in ("raw_latent", "mse_control"):
        return {"raw_latent": "Raw ConvAE", "mse_control": "MSE control"}[rep]
    _, tag, objective = rep.split("_", 2)
    lam = float(tag.replace("p", ".", 1))
    return f"lambda={lam:g}, {OBJECTIVE_LABELS.get(objective, objective)}"


def build_table(summary_csv: Path) -> tuple[list[str], pd.DataFrame]:
    df = pd.read_csv(summary_csv)
    reps = representation_order(set(df["representation"]))
    pivot = df.pivot_table(index="representation", columns="family", values="score")
    pivot = pivot.reindex(index=reps, columns=[f for f in FAMILY_ORDER if f in pivot.columns])
    return reps, pivot


def plot_heatmap(reps: list[str], table: pd.DataFrame, output: Path, title: str) -> None:
    n_rows, n_cols = table.shape
    fig, ax = plt.subplots(figsize=(3.1 * n_cols + 2.6, 0.55 * n_rows + 2.0))

    # Scale each column's color range off the SAE variant rows only, not the
    # reference rows (raw_latent/mse_control) -- those are usually a ceiling
    # or floor far outside the SAE spread and would otherwise wash out the
    # variation that matters (SAE variants compared against each other).
    is_reference = [rep in ("raw_latent", "mse_control") for rep in reps]
    display = np.zeros((n_rows, n_cols))
    for col_index, column in enumerate(table.columns):
        values = table[column].to_numpy(dtype=float)
        variant_values = values[~np.array(is_reference)]
        lo, hi = np.nanmin(variant_values), np.nanmax(variant_values)
        span = hi - lo if hi > lo else 1.0
        display[:, col_index] = np.clip((values - lo) / span, 0, 1)

    cmap = plt.get_cmap("RdYlGn")
    ax.imshow(display, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_cols), [FAMILY_LABELS.get(c, c) for c in table.columns], fontsize=14)
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
            weight = "bold" if rep in ("raw_latent", "mse_control") else "normal"
            ax.text(col_index, row_index, f"{value:.3f}", ha="center", va="center",
                     fontsize=15, color=color, fontweight=weight)

    for row_index, rep in enumerate(reps):
        if rep in ("raw_latent", "mse_control"):
            ax.axhline(row_index + 0.5, color="#1f2430", linewidth=1.2)

    ax.set_title(title, fontsize=17, pad=14)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Linear-probe accessibility by representation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reps, table = build_table(args.summary_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_heatmap(reps, table, args.output, args.title)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
