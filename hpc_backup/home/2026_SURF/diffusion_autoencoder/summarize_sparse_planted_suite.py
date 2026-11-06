#!/usr/bin/env python3
"""Consolidate sparse planted-residual discovery metrics."""

import argparse
import csv
import json
from pathlib import Path


def explained_percent(bias, loss):
    return 100 * (bias - loss) / bias if bias > 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.glob("*/summary.json")):
        payload = json.loads(path.read_text())
        for run in payload["runs"]:
            bias = run["evaluation_bias_only_mse"]
            row = {
                "configuration": path.parent.name,
                "feature": run["processing_feature"],
                "objective": run["objective"],
                "top1_residual_variance_explained_percent": explained_percent(
                    bias, run["evaluation_top1_mse"]
                ),
                "top5_residual_variance_explained_percent": explained_percent(
                    bias, run["evaluation_top5_mse"]
                ),
                "top10_residual_variance_explained_percent": explained_percent(
                    bias, run["evaluation_top10_mse"]
                ),
                "full_sae_residual_variance_explained_percent": explained_percent(
                    bias, run["evaluation_full_sae_mse"]
                ),
                "best_atom_mean_positive_direction_abs_cosine": run[
                    "mean_positive_direction_best_abs_decoder_cosine"
                ],
                "best_direction_atom_ablation_rank": run[
                    "mean_positive_direction_feature_ablation_rank"
                ],
                "best_direction_atom": run[
                    "mean_positive_direction_best_feature"
                ],
            }
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no summaries found below {args.root}")
    with (args.root / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.root / "comparison.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
