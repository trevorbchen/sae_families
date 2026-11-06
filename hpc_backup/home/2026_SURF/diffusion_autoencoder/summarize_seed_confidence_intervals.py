#!/usr/bin/env python3
"""Summarize linear-probe and planted-feature results across SAE seeds."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


PAIRINGS = {
    "plus_laplacian": "all_mse_2branch",
    "plus_fourier_low": "all_mse_2branch",
    "plus_fourier_mid": "all_mse_2branch",
    "plus_fourier_high": "all_mse_2branch",
    "tandem_all4": "all_mse_5branch",
}
T_CRITICAL_95 = {2: 12.706204736, 3: 4.302652730}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def t_interval(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, float("nan"), float("nan"), float("nan")
    critical = T_CRITICAL_95.get(n)
    if critical is None:
        from scipy.stats import t
        critical = float(t.ppf(0.975, n - 1))
    sem = float(values.std(ddof=1) / math.sqrt(n))
    return mean, mean - critical * sem, mean + critical * sem, sem


def interval_row(values):
    mean, low, high, sem = t_interval(values)
    return {
        "mean": mean, "ci_low": low, "ci_high": high, "sem": sem,
        "seed_count": len(values), "ci_unit": "SAE initialization seed",
    }


def summarize_linear(root, output):
    by_seed = {}
    for seed_dir in sorted(root.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[-1])
        by_seed[seed] = {}
        for path in sorted(seed_dir.glob("*/results.csv")):
            by_seed[seed][path.parent.name] = read_csv(path)
    if len(by_seed) < 2:
        raise RuntimeError(f"fewer than two completed linear-probe seeds under {root}")

    task_values = defaultdict(dict)
    family_values = defaultdict(dict)
    for seed, representations in by_seed.items():
        for name, rows in representations.items():
            grouped = defaultdict(list)
            for row in rows:
                key = (name, row["family"], row["metric"], row["task"])
                task_values[key][seed] = float(row["score"])
                grouped[(row["family"], row["metric"])].append(float(row["score"]))
            for (family, metric), values in grouped.items():
                family_values[(name, family, metric)][seed] = float(np.mean(values))

    task_rows = []
    for (name, family, metric, task), seed_scores in sorted(task_values.items()):
        row = {"representation": name, "family": family, "metric": metric,
               "task": task, **interval_row(list(seed_scores.values()))}
        task_rows.append(row)
    family_rows = []
    for (name, family, metric), seed_scores in sorted(family_values.items()):
        family_rows.append({
            "representation": name, "family": family, "metric": metric,
            **interval_row(list(seed_scores.values())),
        })

    paired_task, paired_family = [], []
    for changed, control in PAIRINGS.items():
        common_seeds = sorted(
            seed for seed, reps in by_seed.items()
            if changed in reps and control in reps
        )
        per_family_seed = defaultdict(lambda: defaultdict(list))
        per_task = defaultdict(dict)
        for seed in common_seeds:
            changed_rows = {
                (r["family"], r["metric"], r["task"]): float(r["score"])
                for r in by_seed[seed][changed]
            }
            control_rows = {
                (r["family"], r["metric"], r["task"]): float(r["score"])
                for r in by_seed[seed][control]
            }
            for key in sorted(changed_rows.keys() & control_rows.keys()):
                delta = changed_rows[key] - control_rows[key]
                per_task[key][seed] = delta
                per_family_seed[key[:2]][seed].append(delta)
        for (family, metric, task), seed_deltas in sorted(per_task.items()):
            paired_task.append({
                "changed": changed, "matched_control": control,
                "family": family, "metric": metric, "task": task,
                **interval_row(list(seed_deltas.values())),
            })
        for (family, metric), values_by_seed in sorted(per_family_seed.items()):
            seed_means = [float(np.mean(values)) for values in values_by_seed.values()]
            paired_family.append({
                "changed": changed, "matched_control": control,
                "family": family, "metric": metric,
                **interval_row(seed_means),
            })

    write_csv(output / "linear_probe_task_seed_ci.csv", task_rows)
    write_csv(output / "linear_probe_family_seed_ci.csv", family_rows)
    write_csv(output / "linear_probe_paired_task_seed_ci.csv", paired_task)
    write_csv(output / "linear_probe_paired_family_seed_ci.csv", paired_family)
    return {"seeds": sorted(by_seed), "representations": sorted(set(
        name for reps in by_seed.values() for name in reps
    ))}


def explained_percent(bias, loss):
    return 100.0 * (bias - loss) / bias if bias > 0 else float("nan")


def summarize_planted(root, output):
    values = defaultdict(dict)
    raw_rows = []
    metrics = (
        "top1_residual_variance_explained_percent",
        "top5_residual_variance_explained_percent",
        "top10_residual_variance_explained_percent",
        "full_sae_residual_variance_explained_percent",
        "top1_fraction_of_sae_improvement",
        "top5_fraction_of_sae_improvement",
        "top10_fraction_of_sae_improvement",
        "mean_positive_direction_best_abs_decoder_cosine",
        "mean_positive_direction_feature_ablation_rank",
    )
    for path in sorted(root.glob("*/summary.json")):
        payload = json.loads(path.read_text())
        configuration = path.parent.name
        for run in payload["runs"]:
            bias = run["evaluation_bias_only_mse"]
            row = {
                "configuration": configuration, "objective": run["objective"],
                "seed": int(run["seed"]),
                "top1_residual_variance_explained_percent": explained_percent(
                    bias, run["evaluation_top1_mse"]),
                "top5_residual_variance_explained_percent": explained_percent(
                    bias, run["evaluation_top5_mse"]),
                "top10_residual_variance_explained_percent": explained_percent(
                    bias, run["evaluation_top10_mse"]),
                "full_sae_residual_variance_explained_percent": explained_percent(
                    bias, run["evaluation_full_sae_mse"]),
            }
            for metric in metrics[4:]:
                row[metric] = run[metric]
            raw_rows.append(row)
            for metric in metrics:
                values[(configuration, run["objective"], metric)][int(run["seed"])] = float(row[metric])
    if not raw_rows:
        raise RuntimeError(f"no planted-feature summaries under {root}")

    summary_rows = []
    for (configuration, objective, metric), seed_values in sorted(values.items()):
        summary_rows.append({
            "configuration": configuration, "objective": objective,
            "metric": metric, **interval_row(list(seed_values.values())),
        })

    paired_rows = []
    objectives = sorted({row["objective"] for row in raw_rows if row["objective"] != "latent_mse"})
    for configuration in sorted({row["configuration"] for row in raw_rows}):
        for objective in objectives:
            for metric in metrics:
                changed = values.get((configuration, objective, metric), {})
                control = values.get((configuration, "latent_mse", metric), {})
                seeds = sorted(changed.keys() & control.keys())
                if not seeds:
                    continue
                deltas = [changed[seed] - control[seed] for seed in seeds]
                paired_rows.append({
                    "configuration": configuration, "changed": objective,
                    "matched_control": "latent_mse", "metric": metric,
                    **interval_row(deltas),
                })

    write_csv(output / "planted_feature_seed_results.csv", raw_rows)
    write_csv(output / "planted_feature_seed_ci.csv", summary_rows)
    write_csv(output / "planted_feature_paired_seed_ci.csv", paired_rows)
    return {
        "configurations": len({row["configuration"] for row in raw_rows}),
        "objectives": sorted({row["objective"] for row in raw_rows}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear-runs", type=Path)
    parser.add_argument("--planted-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    if args.linear_runs:
        manifest["linear_probe"] = summarize_linear(args.linear_runs, args.output_dir)
    if args.planted_root:
        manifest["planted_features"] = summarize_planted(args.planted_root, args.output_dir)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
