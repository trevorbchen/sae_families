#!/usr/bin/env python3
"""Summarize matched routed-union replicates with seed-level uncertainty."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


T_CRITICAL_95 = {2: 12.7062047364, 3: 4.3026527299}


def load_metrics(path: Path):
    metrics = json.loads(path.read_text())
    peak = max(metrics["validation_history"], key=lambda row: row["union_ev"])
    return {
        "peak_ev": float(peak["union_ev"]),
        "peak_step": int(peak["step"]),
        "final_ev": float(metrics["final_latent"]["union_ev"]),
    }


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1)) if count > 1 else float("nan")
    critical = T_CRITICAL_95.get(count)
    if critical is None:
        raise ValueError(f"no t critical value configured for n={count}")
    half_width = critical * standard_deviation / math.sqrt(count)
    return {
        "n": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "ci_half_width": half_width,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("SEED", "GROUPED_METRICS", "CONTROL_METRICS"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed_text, grouped_path, control_path in args.pair:
        seed = int(seed_text)
        for condition, path in (
            ("grouped_all_four", Path(grouped_path)),
            ("five_branch_mse", Path(control_path)),
        ):
            values = load_metrics(path)
            rows.append({"seed": seed, "condition": condition, **values})

    with (args.output_dir / "seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for condition in ("five_branch_mse", "grouped_all_four"):
        selected = [row for row in rows if row["condition"] == condition]
        for metric in ("peak_ev", "final_ev"):
            summaries.append({
                "condition": condition,
                "metric": metric,
                **summarize([row[metric] for row in selected]),
            })
    grouped_by_seed = {
        row["seed"]: row for row in rows if row["condition"] == "grouped_all_four"
    }
    control_by_seed = {
        row["seed"]: row for row in rows if row["condition"] == "five_branch_mse"
    }
    paired = []
    for metric in ("peak_ev", "final_ev"):
        differences = [
            grouped_by_seed[seed][metric] - control_by_seed[seed][metric]
            for seed in sorted(set(grouped_by_seed) & set(control_by_seed))
        ]
        paired.append({
            "condition": "grouped_minus_control",
            "metric": metric,
            **summarize(differences),
        })
    summaries.extend(paired)
    with (args.output_dir / "seed_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    peak_rows = {
        row["condition"]: row for row in summaries
        if row["metric"] == "peak_ev" and row["condition"] != "grouped_minus_control"
    }
    conditions = ("five_branch_mse", "grouped_all_four")
    labels = ("Five-branch\nMSE control", "MSE + all four\nchanged objectives")
    colors = ("#6b7280", "#d95f02")
    means = np.array([peak_rows[name]["mean"] for name in conditions])
    errors = np.array([peak_rows[name]["ci_half_width"] for name in conditions])

    figure, axis = plt.subplots(figsize=(8.5, 5.8))
    figure.subplots_adjust(left=0.12, right=0.96, bottom=0.20, top=0.84)
    bars = axis.bar(labels, means, yerr=errors, color=colors, width=0.58,
                    capsize=6, ecolor="#202020")
    for bar, mean, color in zip(bars, means, colors):
        axis.text(bar.get_x() + bar.get_width() / 2, mean + 0.004,
                  f"{mean:.2%}", ha="center", va="bottom", fontsize=12,
                  color=color)
    axis.set_title("Peak validation explained variance across three seeds", pad=12)
    axis.set_ylabel("Mean peak validation explained variance")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylim(0, 0.57)
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    stem = args.output_dir / "non_residual_all_four_union_ev_seed_ci"
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({"rows": rows, "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
