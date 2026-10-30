#!/usr/bin/env python3
"""Main-text linear-probe heatmap, scoped to one headline lambda instead of
all 7 -- reuses plot_linear_probe_heatmap.py's own table-building and
coloring code directly so the visual style matches the full (appendix)
version exactly, just with fewer rows.
"""

import argparse
from pathlib import Path

import pandas as pd

from plot_linear_probe_heatmap import build_table, plot_heatmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--lambda-value", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    tag = {0.1: "0p1", 0.3: "0p3", 1.0: "1p0", 3.0: "3p0", 10.0: "10p0", 30.0: "30p0", 100.0: "100p0"}[args.lambda_value]
    keep = df["representation"].isin(["raw_latent", "mse_control"]) | df["representation"].str.startswith(f"lambda_{tag}_")
    filtered = df[keep]

    tmp_csv = args.output.with_suffix(".filtered.csv")
    tmp_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(tmp_csv, index=False)

    reps, table = build_table(tmp_csv)
    plot_heatmap(reps, table, args.output, f"Linear-probe accessibility by representation (lambda={args.lambda_value:g})")
    tmp_csv.unlink()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
