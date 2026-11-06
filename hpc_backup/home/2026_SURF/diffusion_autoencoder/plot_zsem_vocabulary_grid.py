#!/usr/bin/env python3
"""Render pairwise SAE vocabulary-alignment matrices from the analysis CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    keys = list(
        {
            (row["objective_a"], int(row["seed_a"]))
            for row in rows
        }
        | {
            (row["objective_b"], int(row["seed_b"]))
            for row in rows
        }
    )
    objective_order = {
        "latent_mse": 0,
        "laplacian": 1,
        "fourier_bands": 2,
        "fourier_low": 3,
        "fourier_mid": 4,
        "fourier_high": 5,
        "weighted_laplacian": 1,
        "weighted_fourier_bands": 2,
        "weighted_fourier_low": 3,
        "weighted_fourier_mid": 4,
        "weighted_fourier_high": 5,
    }
    objective_labels = {
        "latent_mse": "MSE",
        "laplacian": "Lap",
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
    keys.sort(key=lambda item: (objective_order[item[0]], item[1]))
    labels = [f"{objective_labels[objective]}\n{seed}" for objective, seed in keys]
    index = {key: i for i, key in enumerate(keys)}
    metrics = (
        ("hungarian_decoder_cosine", "One-to-one decoder cosine", None, None, ".2f"),
        ("hungarian_activation_correlation", "One-to-one activation correlation", None, None, ".2f"),
        ("linear_cka", "Vocabulary activation CKA", 0.0, 1.0, ".2f"),
        ("shared_fraction", "Strict shared atoms at cosine >= 0.5", 0.0, 0.02, ".1%"),
    )

    matrices = {name: np.eye(len(keys), dtype=float) for name, *_ in metrics}
    for row in rows:
        i = index[(row["objective_a"], int(row["seed_a"]))]
        j = index[(row["objective_b"], int(row["seed_b"]))]
        values = {
            "hungarian_decoder_cosine": float(row["hungarian_decoder_cosine"]),
            "hungarian_activation_correlation": float(row["hungarian_activation_correlation"]),
            "linear_cka": float(row["linear_cka"]),
            "shared_fraction": 0.5
            * (float(row["shared_fraction_a_0p5"]) + float(row["shared_fraction_b_0p5"])),
        }
        for name, value in values.items():
            matrices[name][i, j] = matrices[name][j, i] = value

    feature_sets = len(keys)
    figure_size = (max(13, feature_sets * 1.4), max(11, feature_sets * 1.25))
    cell_font_size = max(5, 11 - feature_sets // 3)
    fig, axes = plt.subplots(2, 2, figsize=figure_size, constrained_layout=True)
    for axis, (name, title, vmin, vmax, number_format) in zip(axes.flat, metrics):
        matrix = matrices[name]
        off_diagonal = matrix[~np.eye(len(keys), dtype=bool)]
        if vmin is None or vmax is None:
            observed_min = float(off_diagonal.min())
            observed_max = float(off_diagonal.max())
            padding = max((observed_max - observed_min) * 0.02, 1e-6)
            vmin = observed_min - padding
            vmax = observed_max + padding
        if name == "shared_fraction" and len(keys) > 1:
            vmax = max(vmax, float(off_diagonal.max()))
        image = axis.imshow(matrix, cmap="magma", vmin=vmin, vmax=vmax)
        axis.set_title(title, fontsize=13)
        axis.set_xticks(range(len(keys)), labels, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(range(len(keys)), labels, fontsize=8)
        for i in range(len(keys)):
            for j in range(len(keys)):
                text = "self" if i == j else format(matrix[i, j], number_format)
                color = "black" if matrix[i, j] > (vmin + vmax) / 2 else "white"
                axis.text(
                    j, i, text, ha="center", va="center", color=color,
                    fontsize=cell_font_size,
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle("Corrected cross-seed SAE dictionary alignment", fontsize=16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")


if __name__ == "__main__":
    main()
