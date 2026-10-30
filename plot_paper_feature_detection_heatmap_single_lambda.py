#!/usr/bin/env python3
"""Main-text planted-feature-detection heatmap, scoped to one headline
lambda instead of all 7 (25 rows was too dense to read) -- reuses
plot_feature_detection_heatmap.py's own table-building and coloring code
directly so the visual style matches the full (appendix) version exactly,
just with fewer rows.
"""

import argparse
from pathlib import Path

import pandas as pd

from plot_feature_detection_heatmap import build_table, plot_heatmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--lambda-value", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Planted-feature detection by representation")
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    filtered = df[df["lambda"] == args.lambda_value]

    tmp_csv = args.output.with_suffix(".filtered.csv")
    tmp_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(tmp_csv, index=False)

    reps, table = build_table(tmp_csv)
    plot_heatmap(reps, table, args.output, f"{args.title} (lambda={args.lambda_value:g})")
    tmp_csv.unlink()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
