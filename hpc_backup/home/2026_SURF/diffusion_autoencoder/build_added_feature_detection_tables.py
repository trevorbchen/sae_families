#!/usr/bin/env python3
"""Build readable tables for the sparse planted-feature detection experiment."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_SOURCE = Path("results/sparse_planted_residual_suite/comparison.csv")
DEFAULT_OUTPUT = Path("results/added_feature_detection")
METRICS = ("top1", "top5", "top10", "full_sae")
SOURCE_COLUMNS = {
    "top1": "top1_residual_variance_explained_percent",
    "top5": "top5_residual_variance_explained_percent",
    "top10": "top10_residual_variance_explained_percent",
    "full_sae": "full_sae_residual_variance_explained_percent",
}
OBJECTIVE_ORDER = (
    "latent_mse",
    "weighted_laplacian",
    "weighted_fourier_bands",
    "weighted_fourier_low",
    "weighted_fourier_mid",
    "weighted_fourier_high",
)
PURE_OBJECTIVE_ORDER = (
    "latent_mse",
    "laplacian",
    "fourier_bands",
    "fourier_low",
    "fourier_mid",
    "fourier_high",
)


def read_rows(source):
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for metric, column in SOURCE_COLUMNS.items():
            row[metric] = float(row[column])
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_objective_matrices(rows, output, objective_order):
    image_rows = [row for row in rows if row["feature"] != "latent_random"]
    configurations = sorted({row["configuration"] for row in image_rows})
    lookup = {
        (row["objective"], row["configuration"]): row
        for row in image_rows
    }
    for metric in METRICS:
        matrix_rows = []
        for objective in objective_order:
            matrix_row = {"objective": objective}
            for configuration in configurations:
                run = lookup.get((objective, configuration))
                matrix_row[configuration] = "" if run is None else run[metric]
            matrix_rows.append(matrix_row)
        write_csv(
            output / f"objective_by_feature_{metric}_percent_explained.csv",
            matrix_rows,
        )


def effect_label(delta):
    if delta >= 5:
        return "large improvement"
    if delta >= 1:
        return "moderate improvement"
    if delta > 0:
        return "marginal improvement"
    return "no improvement"


def interpretation(top1_delta, top10_delta, alternative_label="alternative loss"):
    if top1_delta > 0 and top10_delta > 0:
        return f"{alternative_label} improves both single-feature concentration and top-10 coverage"
    if top1_delta > 0:
        return f"{alternative_label} concentrates more signal in one feature but reduces top-10 coverage"
    if top10_delta > 0:
        return f"{alternative_label} spreads the signal more but improves top-10 coverage"
    return "ordinary latent MSE explains more with both one and ten features"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pure-transformed", action="store_true")
    args = parser.parse_args()
    objective_order = PURE_OBJECTIVE_ORDER if args.pure_transformed else OBJECTIVE_ORDER
    alternative_label = "pure transformed loss" if args.pure_transformed else "weighted loss"
    rows = read_rows(args.source)
    args.output.mkdir(parents=True, exist_ok=True)

    run_rows = []
    for row in rows:
        run_rows.append({
            "configuration": row["configuration"],
            "feature": row["feature"],
            "is_latent_control": row["feature"] == "latent_random",
            "objective": row["objective"],
            "top1_percent_explained": row["top1"],
            "top5_percent_explained": row["top5"],
            "top10_percent_explained": row["top10"],
            "full_sae_percent_explained": row["full_sae"],
        })
    write_csv(args.output / "all_objective_results.csv", run_rows)
    write_objective_matrices(rows, args.output, objective_order)

    grouped = defaultdict(list)
    for row in rows:
        if row["feature"] != "latent_random":
            grouped[row["configuration"]].append(row)

    summary_rows = []
    for configuration in sorted(grouped):
        group = grouped[configuration]
        baseline = next(row for row in group if row["objective"] == "latent_mse")
        weighted = [row for row in group if row["objective"] != "latent_mse"]
        best = {
            metric: max(weighted, key=lambda row: row[metric])
            for metric in METRICS
        }
        top1_delta = best["top1"]["top1"] - baseline["top1"]
        top10_delta = best["top10"]["top10"] - baseline["top10"]
        summary_rows.append({
            "configuration": configuration,
            "feature": baseline["feature"],
            "mse_top1_percent": baseline["top1"],
            "best_weighted_top1_objective": best["top1"]["objective"],
            "best_weighted_top1_percent": best["top1"]["top1"],
            "top1_change_percentage_points": top1_delta,
            "top1_effect": effect_label(top1_delta),
            "mse_top5_percent": baseline["top5"],
            "best_weighted_top5_objective": best["top5"]["objective"],
            "best_weighted_top5_percent": best["top5"]["top5"],
            "top5_change_percentage_points": best["top5"]["top5"] - baseline["top5"],
            "mse_top10_percent": baseline["top10"],
            "best_weighted_top10_objective": best["top10"]["objective"],
            "best_weighted_top10_percent": best["top10"]["top10"],
            "top10_change_percentage_points": top10_delta,
            "mse_full_sae_percent": baseline["full_sae"],
            "best_weighted_full_objective": best["full_sae"]["objective"],
            "best_weighted_full_sae_percent": best["full_sae"]["full_sae"],
            "interpretation": interpretation(top1_delta, top10_delta, alternative_label),
        })
    write_csv(args.output / "feature_detection_summary.csv", summary_rows)

    control = next((row for row in rows if row["feature"] == "latent_random"), None)
    if control is not None:
        write_csv(args.output / "latent_control.csv", [{
            "configuration": control["configuration"],
            "feature": control["feature"],
            "objective": control["objective"],
            "top1_percent_explained": control["top1"],
            "top5_percent_explained": control["top5"],
            "top10_percent_explained": control["top10"],
            "full_sae_percent_explained": control["full_sae"],
        }])

    improved_top1 = sum(row["top1_change_percentage_points"] > 0 for row in summary_rows)
    improved_top10 = sum(row["top10_change_percentage_points"] > 0 for row in summary_rows)
    markdown = [
        "# Added Feature Detection",
        "",
        "This report compares ordinary latent-MSE SAEs against SAEs trained with ",
        (
            "only a normalized Laplacian or Fourier reconstruction objective. "
            if args.pure_transformed else
            "latent MSE plus a normalized Laplacian or Fourier reconstruction term. "
        ),
        "The DiffAE/spatial-AE latent representation is frozen; only the SAE is retrained.",
        "",
        "Each image-space feature is planted in 10% of examples. The reported percentage is:",
        "",
        "```text",
        "100 * (bias-only residual MSE - reconstructed residual MSE)",
        "    / bias-only residual MSE",
        "```",
        "",
        "Top-1 is the percentage explained by the single most useful SAE feature; top-10",
        "uses the ten most useful features jointly. Higher is better.",
        "",
        f"{alternative_label.capitalize()} improves top-1 percentage for **{improved_top1}/20** planted ",
        f"conditions and top-10 percentage for **{improved_top10}/20** conditions.",
        "",
        f"| Condition | MSE top-1 | Best {alternative_label} top-1 | Change | MSE top-10 | Best {alternative_label} top-10 | Change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        markdown.append(
            "| {configuration} | {mse_top1_percent:.2f}% | {best_weighted_top1_percent:.2f}% "
            "({best_weighted_top1_objective}) | {top1_change_percentage_points:+.2f} pp | "
            "{mse_top10_percent:.2f}% | {best_weighted_top10_percent:.2f}% "
            "({best_weighted_top10_objective}) | {top10_change_percentage_points:+.2f} pp |".format(**row)
        )
    markdown.extend([
        "",
        "## Files",
        "",
        "- `feature_detection_summary.csv`: one comparison row per planted image feature.",
        "- `all_objective_results.csv`: every objective run, including the latent control.",
        "- `objective_by_feature_top1_percent_explained.csv`: 6 x 20 top-1 matrix.",
        "- `objective_by_feature_top5_percent_explained.csv`: 6 x 20 top-5 matrix.",
        "- `objective_by_feature_top10_percent_explained.csv`: 6 x 20 top-10 matrix.",
        "- `objective_by_feature_full_sae_percent_explained.csv`: 6 x 20 full-SAE matrix.",
        "- `latent_control.csv`: fixed random latent-direction positive control, when included.",
        "- Source visualizations are stored in the suite results directory.",
        "",
        "The latent-random control is excluded from the 20-feature win counts because it",
        "was trained only with ordinary MSE and has no matched image-space objective.",
    ])
    (args.output / "README.md").write_text("\n".join(markdown) + "\n")


if __name__ == "__main__":
    main()
