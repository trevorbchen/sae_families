#!/usr/bin/env python3
"""Aggregate SAE probe runs and compute paired uncertainty estimates."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, r2_score


PAIRINGS = {
    "plus_laplacian": "all_mse_2branch",
    "plus_fourier_low": "all_mse_2branch",
    "plus_fourier_mid": "all_mse_2branch",
    "plus_fourier_high": "all_mse_2branch",
    "tandem_all4": "all_mse_5branch",
}

ALIGNED_TASKS = {
    "plus_laplacian": ("laplacian_log_power",),
    "plus_fourier_low": ("fourier_low_log_power",),
    "plus_fourier_mid": ("fourier_mid_log_power",),
    "plus_fourier_high": ("fourier_high_log_power",),
    "tandem_all4": (
        "laplacian_log_power", "fourier_low_log_power",
        "fourier_mid_log_power", "fourier_high_log_power",
    ),
}

LABELS = {
    "raw_latent": "Raw ConvAE",
    "direct_mse": "Direct MSE",
    "all_mse_2branch": "2-branch MSE",
    "plus_laplacian": "+ Laplacian",
    "plus_fourier_low": "+ Fourier low",
    "plus_fourier_mid": "+ Fourier mid",
    "plus_fourier_high": "+ Fourier high",
    "all_mse_5branch": "5-branch MSE",
    "tandem_all4": "+ All four",
}


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def interval(values, generator, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    samples = generator.choice(values, size=(draws, len(values)), replace=True).mean(1)
    return float(values.mean()), *np.quantile(samples, (0.025, 0.975)).tolist()


def cluster_control_interval(prediction, labels, identities, generator, draws=1000):
    unique = np.unique(identities)
    grouped = {identity: np.where(identities == identity)[0] for identity in unique}
    scores = []
    for _ in range(draws):
        sampled = generator.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([grouped[identity] for identity in sampled])
        scores.append(balanced_accuracy_score(labels[indices], prediction[indices]))
    score = balanced_accuracy_score(labels, prediction)
    return float(score), *np.quantile(scores, (0.025, 0.975)).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=260818)
    args = parser.parse_args()
    generator = np.random.default_rng(args.seed)
    runs_dir = Path(args.runs_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets = torch.load(args.targets, map_location="cpu", weights_only=False)

    runs, all_rows, all_efficiency_rows = {}, [], []
    for result_path in sorted(runs_dir.glob("*/results.csv")):
        name = result_path.parent.name
        rows = read_csv(result_path)
        prediction_path = result_path.parent / "predictions.pt"
        if not prediction_path.is_file():
            continue
        predictions = torch.load(prediction_path, map_location="cpu", weights_only=False)
        efficiency_path = result_path.parent / "semantic_feature_efficiency.csv"
        efficiency_rows = (
            read_csv(efficiency_path) if efficiency_path.is_file() else []
        )
        runs[name] = {
            "rows": rows, "predictions": predictions,
            "efficiency": efficiency_rows,
        }
        all_rows.extend(rows)
        all_efficiency_rows.extend(efficiency_rows)
    if not runs:
        raise RuntimeError(f"no complete probe runs under {runs_dir}")
    write_csv(output / "all_task_results.csv", all_rows)
    write_csv(output / "all_semantic_feature_efficiency.csv", all_efficiency_rows)

    summaries = []
    for name, run in runs.items():
        rows = run["rows"]
        semantic = [float(row["score"]) for row in rows
                    if row["family"] == "semantic" and row["metric"] == "auroc"]
        geometry = [float(row["score"]) for row in rows
                    if row["family"] == "geometry" and row["metric"] == "r2"]
        source_geometry = [float(row["score"]) for row in rows
                           if row["family"] == "source_geometry"
                           and row["metric"] == "r2"]
        for family, metric, values in (
            ("semantic", "macro_auroc", semantic),
            ("geometry", "macro_r2", geometry),
            ("source_geometry", "macro_r2", source_geometry),
        ):
            score, low, high = interval(values, generator)
            summaries.append({
                "representation": name, "family": family, "metric": metric,
                "score": score, "ci_low": low, "ci_high": high,
                "ci_unit": "task", "task_count": len(values),
            })
        prediction = run["predictions"]
        test_indices = prediction["test_indices"].long()
        labels = targets["control_labels"][test_indices].numpy()
        identities = prediction["test_identities"].numpy()
        score, low, high = cluster_control_interval(
            prediction["control_prediction"].numpy(), labels, identities, generator
        )
        summaries.append({
            "representation": name, "family": "controlled",
            "metric": "balanced_accuracy", "score": score,
            "ci_low": low, "ci_high": high, "ci_unit": "identity",
            "task_count": 1,
        })
    write_csv(output / "summary.csv", summaries)

    comparisons = []
    for changed, control in PAIRINGS.items():
        if changed not in runs or control not in runs:
            continue
        changed_rows = runs[changed]["rows"]
        control_rows = runs[control]["rows"]
        for family, metric in (
            ("semantic", "auroc"), ("geometry", "r2"),
            ("source_geometry", "r2"),
        ):
            changed_by_task = {
                row["task"]: float(row["score"]) for row in changed_rows
                if row["family"] == family and row["metric"] == metric
            }
            control_by_task = {
                row["task"]: float(row["score"]) for row in control_rows
                if row["family"] == family and row["metric"] == metric
            }
            tasks = sorted(set(changed_by_task) & set(control_by_task))
            deltas = [changed_by_task[task] - control_by_task[task] for task in tasks]
            score, low, high = interval(deltas, generator)
            comparisons.append({
                "changed": changed, "matched_control": control,
                "family": family, "metric": metric, "mean_delta": score,
                "ci_low": low, "ci_high": high, "ci_unit": "paired_task",
                "task_count": len(tasks),
            })
        changed_prediction = runs[changed]["predictions"]
        control_prediction = runs[control]["predictions"]
        test_indices = changed_prediction["test_indices"].long()
        labels = targets["control_labels"][test_indices].numpy()
        identities = changed_prediction["test_identities"].numpy()
        changed_labels = changed_prediction["control_prediction"].numpy()
        control_labels = control_prediction["control_prediction"].numpy()
        unique = np.unique(identities)
        grouped = {identity: np.where(identities == identity)[0] for identity in unique}
        deltas = []
        for _ in range(1000):
            sampled = generator.choice(unique, size=len(unique), replace=True)
            indices = np.concatenate([grouped[identity] for identity in sampled])
            deltas.append(
                balanced_accuracy_score(labels[indices], changed_labels[indices])
                - balanced_accuracy_score(labels[indices], control_labels[indices])
            )
        observed = (
            balanced_accuracy_score(labels, changed_labels)
            - balanced_accuracy_score(labels, control_labels)
        )
        low, high = np.quantile(deltas, (0.025, 0.975))
        comparisons.append({
            "changed": changed, "matched_control": control,
            "family": "controlled", "metric": "balanced_accuracy",
            "mean_delta": float(observed), "ci_low": float(low),
            "ci_high": float(high), "ci_unit": "paired_identity",
            "task_count": 1,
        })
    write_csv(output / "paired_comparisons.csv", comparisons)

    aligned_geometry = []
    for changed, tasks in ALIGNED_TASKS.items():
        control = PAIRINGS[changed]
        if changed not in runs or control not in runs:
            continue
        changed_prediction = runs[changed]["predictions"]
        control_prediction = runs[control]["predictions"]
        test_indices = changed_prediction["test_indices"].long()
        identities = changed_prediction["test_identities"].numpy()
        unique = np.unique(identities)
        grouped = {identity: np.where(identities == identity)[0] for identity in unique}
        geometry_names = list(changed_prediction["geometry_names"])
        for family, target_key, prediction_key in (
            ("decoder_geometry", "geometry_targets", "geometry_prediction"),
            ("source_geometry", "source_geometry_targets",
             "source_geometry_prediction"),
        ):
            labels = targets[target_key][test_indices].numpy()
            changed_values = changed_prediction[prediction_key].numpy()
            control_values = control_prediction[prediction_key].numpy()
            for task in tasks:
                index = geometry_names.index(task)
                observed = (
                    r2_score(labels[:, index], changed_values[:, index])
                    - r2_score(labels[:, index], control_values[:, index])
                )
                deltas = []
                for _ in range(1000):
                    sampled = generator.choice(unique, size=len(unique), replace=True)
                    selected = np.concatenate([grouped[value] for value in sampled])
                    deltas.append(
                        r2_score(labels[selected, index], changed_values[selected, index])
                        - r2_score(labels[selected, index], control_values[selected, index])
                    )
                low, high = np.quantile(deltas, (0.025, 0.975))
                aligned_geometry.append({
                    "changed": changed, "matched_control": control,
                    "family": family, "task": task,
                    "r2_delta": float(observed), "ci_low": float(low),
                    "ci_high": float(high), "ci_unit": "paired_identity",
                })
    write_csv(output / "aligned_geometry_comparisons.csv", aligned_geometry)

    controlled_by_class, controlled_class_paired = [], []
    for name, run in runs.items():
        prediction = run["predictions"]
        test_indices = prediction["test_indices"].long()
        labels = targets["control_labels"][test_indices].numpy()
        predicted = prediction["control_prediction"].numpy()
        class_names = list(prediction["control_names"])
        for index, class_name in enumerate(class_names):
            selected = labels == index
            controlled_by_class.append({
                "representation": name, "transform": class_name,
                "recall": float((predicted[selected] == labels[selected]).mean()),
                "examples": int(selected.sum()),
            })
    for changed, control in PAIRINGS.items():
        if changed not in runs or control not in runs:
            continue
        changed_prediction = runs[changed]["predictions"]
        control_prediction = runs[control]["predictions"]
        test_indices = changed_prediction["test_indices"].long()
        labels = targets["control_labels"][test_indices].numpy()
        identities = changed_prediction["test_identities"].numpy()
        changed_labels = changed_prediction["control_prediction"].numpy()
        control_labels = control_prediction["control_prediction"].numpy()
        unique = np.unique(identities)
        grouped = {identity: np.where(identities == identity)[0] for identity in unique}
        for index, class_name in enumerate(changed_prediction["control_names"]):
            selected = labels == index
            observed = (
                (changed_labels[selected] == labels[selected]).mean()
                - (control_labels[selected] == labels[selected]).mean()
            )
            deltas = []
            for _ in range(1000):
                sampled = generator.choice(unique, size=len(unique), replace=True)
                sampled_indices = np.concatenate([grouped[value] for value in sampled])
                class_indices = sampled_indices[labels[sampled_indices] == index]
                if len(class_indices):
                    deltas.append(
                        (changed_labels[class_indices] == labels[class_indices]).mean()
                        - (control_labels[class_indices] == labels[class_indices]).mean()
                    )
            low, high = np.quantile(deltas, (0.025, 0.975))
            controlled_class_paired.append({
                "changed": changed, "matched_control": control,
                "transform": class_name, "recall_delta": float(observed),
                "ci_low": float(low), "ci_high": float(high),
                "ci_unit": "paired_identity",
            })
    write_csv(output / "controlled_transform_per_class.csv", controlled_by_class)
    write_csv(output / "controlled_transform_paired.csv", controlled_class_paired)

    efficiency_summary, concentration_summary = [], []
    for name, run in runs.items():
        rows = run["efficiency"]
        if not rows:
            continue
        budgets = sorted({int(row["feature_budget"]) for row in rows})
        for budget in budgets:
            selected = [row for row in rows if int(row["feature_budget"]) == budget]
            auroc = [float(row["test_auroc"]) for row in selected]
            retention = [
                float(row["above_chance_retention"]) for row in selected
                if np.isfinite(float(row["above_chance_retention"]))
            ]
            score, low, high = interval(auroc, generator)
            ret_score, ret_low, ret_high = interval(retention, generator)
            efficiency_summary.append({
                "representation": name, "feature_budget": budget,
                "macro_auroc": score, "auroc_ci_low": low,
                "auroc_ci_high": high,
                "mean_above_chance_retention": ret_score,
                "retention_ci_low": ret_low, "retention_ci_high": ret_high,
                "task_count": len(auroc),
            })
        by_task = {}
        for row in rows:
            by_task.setdefault(row["task"], []).append(row)
        effective_support, minimum_90, minimum_95 = [], [], []
        eligible = 0
        for task_rows in by_task.values():
            task_rows.sort(key=lambda row: int(row["feature_budget"]))
            effective_support.append(float(task_rows[0]["effective_support"]))
            if float(task_rows[0]["full_test_auroc"]) < 0.55:
                continue
            eligible += 1
            for threshold, values in ((0.90, minimum_90), (0.95, minimum_95)):
                reached = [
                    int(row["feature_budget"]) for row in task_rows
                    if float(row["above_chance_retention"]) >= threshold
                ]
                if reached:
                    values.append(min(reached))
        concentration_summary.append({
            "representation": name,
            "median_effective_support": float(np.median(effective_support)),
            "eligible_tasks": eligible,
            "median_m90": float(np.median(minimum_90)) if minimum_90 else float("nan"),
            "m90_reached_fraction": len(minimum_90) / max(eligible, 1),
            "median_m95": float(np.median(minimum_95)) if minimum_95 else float("nan"),
            "m95_reached_fraction": len(minimum_95) / max(eligible, 1),
        })
    write_csv(output / "feature_efficiency_summary.csv", efficiency_summary)
    write_csv(output / "feature_concentration_summary.csv", concentration_summary)

    efficiency_paired = []
    for changed, control in PAIRINGS.items():
        if changed not in runs or control not in runs:
            continue
        changed_rows = runs[changed]["efficiency"]
        control_rows = runs[control]["efficiency"]
        changed_scores = {
            (row["task"], int(row["feature_budget"])): float(row["test_auroc"])
            for row in changed_rows
        }
        control_scores = {
            (row["task"], int(row["feature_budget"])): float(row["test_auroc"])
            for row in control_rows
        }
        budgets = sorted({budget for _, budget in set(changed_scores) & set(control_scores)})
        for budget in budgets:
            keys = sorted(
                key for key in set(changed_scores) & set(control_scores)
                if key[1] == budget
            )
            deltas = [changed_scores[key] - control_scores[key] for key in keys]
            score, low, high = interval(deltas, generator)
            efficiency_paired.append({
                "changed": changed, "matched_control": control,
                "feature_budget": budget, "mean_auroc_delta": score,
                "ci_low": low, "ci_high": high, "task_count": len(keys),
            })
    write_csv(output / "feature_efficiency_paired.csv", efficiency_paired)

    primary = [name for name in LABELS if name in runs]
    figure, axes = plt.subplots(1, 4, figsize=(19, 5.2))
    panels = (
        ("semantic", "macro_auroc", "CelebA attributes", "Macro AUROC"),
        ("geometry", "macro_r2", "Decoder-output geometry", "Macro $R^2$"),
        ("source_geometry", "macro_r2", "Source-image geometry", "Macro $R^2$"),
        ("controlled", "balanced_accuracy", "Controlled transformations",
         "Balanced accuracy"),
    )
    colors = ["#4c78a8", "#7f7f7f", "#59a14f", "#f28e2b", "#e15759",
              "#b07aa1", "#76b7b2", "#9c755f", "#edc949"]
    for axis, (family, metric, title, ylabel) in zip(axes, panels):
        selected = {
            row["representation"]: row for row in summaries
            if row["family"] == family and row["metric"] == metric
        }
        rows = [selected[name] for name in primary]
        values = np.array([float(row["score"]) for row in rows])
        low = np.maximum(
            values - np.array([float(row["ci_low"]) for row in rows]), 0
        )
        high = np.maximum(
            np.array([float(row["ci_high"]) for row in rows]) - values, 0
        )
        axis.bar(range(len(rows)), values, color=colors[:len(rows)], width=0.76)
        axis.errorbar(range(len(rows)), values, yerr=np.vstack((low, high)),
                      fmt="none", ecolor="#202020", capsize=3, linewidth=1.2)
        axis.set_xticks(range(len(rows)), [LABELS.get(name, name) for name in primary],
                        rotation=42, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Linear accessibility of semantic and output-geometry information")
    figure.tight_layout()
    figure.savefig(output / "linear_probe_summary.png", dpi=220)
    figure.savefig(output / "linear_probe_summary.svg")
    plt.close(figure)

    if efficiency_summary:
        efficiency_lookup = {}
        for row in efficiency_summary:
            efficiency_lookup.setdefault(row["representation"], []).append(row)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        panels = (
            ("MSE + one objective", (
                "all_mse_2branch", "plus_laplacian", "plus_fourier_low",
                "plus_fourier_mid", "plus_fourier_high",
            )),
            ("MSE + all four objectives", ("all_mse_5branch", "tandem_all4")),
        )
        for axis, (title, names) in zip(axes, panels):
            for name in names:
                rows = sorted(
                    efficiency_lookup.get(name, []),
                    key=lambda row: int(row["feature_budget"]),
                )
                if not rows:
                    continue
                axis.plot(
                    [int(row["feature_budget"]) for row in rows],
                    [float(row["macro_auroc"]) for row in rows],
                    marker="o", markersize=3, linewidth=1.5,
                    label=LABELS.get(name, name),
                )
            axis.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
            axis.set_xscale("log", base=2)
            axis.set_xlabel("Top probe coefficients retained")
            axis.set_title(title)
            axis.grid(alpha=0.22)
            axis.legend(fontsize=8)
        axes[0].set_ylabel("Macro CelebA AUROC")
        figure.suptitle("Semantic probe feature efficiency")
        figure.tight_layout()
        figure.savefig(output / "semantic_feature_efficiency.png", dpi=220)
        figure.savefig(output / "semantic_feature_efficiency.svg")
        plt.close(figure)

    if aligned_geometry:
        figure, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
        for axis, family, title in zip(
            axes,
            ("decoder_geometry", "source_geometry"),
            ("Decoder-output targets", "Source-image targets"),
        ):
            rows = [row for row in aligned_geometry if row["family"] == family]
            labels = [
                f"{LABELS.get(row['changed'], row['changed'])}: "
                f"{row['task'].replace('_log_power', '').replace('_', ' ')}"
                for row in rows
            ]
            values = np.array([float(row["r2_delta"]) for row in rows])
            low = np.maximum(
                values - np.array([float(row["ci_low"]) for row in rows]), 0
            )
            high = np.maximum(
                np.array([float(row["ci_high"]) for row in rows]) - values, 0
            )
            positions = np.arange(len(rows))
            axis.errorbar(
                values, positions, xerr=np.vstack((low, high)), fmt="o",
                color="#3b6ea8", ecolor="#303030", capsize=3,
            )
            axis.axvline(0, color="#777777", linestyle="--", linewidth=1)
            axis.set_yticks(positions, labels)
            axis.invert_yaxis()
            axis.set_xlabel("R2 delta vs matched MSE control")
            axis.set_title(title)
            axis.grid(axis="x", alpha=0.22)
        figure.suptitle("Objective-aligned linear-probe effects")
        figure.tight_layout()
        figure.savefig(output / "aligned_geometry_effects.png", dpi=220)
        figure.savefig(output / "aligned_geometry_effects.svg")
        plt.close(figure)

    report = [
        "# SAE Linear-Probe Benchmark", "",
        "All probes use identity-disjoint train/validation/test splits. Hyperparameters are ",
        "selected on validation data, and the test split is evaluated once. Semantic and ",
        "natural-geometry intervals bootstrap tasks; controlled-transform intervals bootstrap ",
        "held-out identities. Changed-objective deltas use matched architecture controls.", "",
        "## Primary summary", "",
        "| Representation | Semantic AUROC | Decoder geometry R2 | Source geometry R2 | Controlled balanced accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    lookup = {(row["representation"], row["family"]): row for row in summaries}
    for name in primary:
        sem = lookup[(name, "semantic")]
        geo = lookup[(name, "geometry")]
        source_geo = lookup[(name, "source_geometry")]
        ctl = lookup[(name, "controlled")]
        report.append(
            f"| {LABELS.get(name, name)} | {float(sem['score']):.4f} | "
            f"{float(geo['score']):.4f} | {float(source_geo['score']):.4f} | "
            f"{float(ctl['score']):.4f} |"
        )
    report.extend(("", "## Matched changed-objective effects", "",
                   "| Changed model | Control | Family | Delta | 95% CI |",
                   "|---|---|---|---:|---:|"))
    for row in comparisons:
        report.append(
            f"| {LABELS.get(row['changed'], row['changed'])} | "
            f"{LABELS.get(row['matched_control'], row['matched_control'])} | "
            f"{row['family']} | {float(row['mean_delta']):+.4f} | "
            f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}] |"
        )
    if aligned_geometry:
        report.extend((
            "", "## Objective-aligned geometry effects", "",
            "Positive values favor the changed-objective SAE over its "
            "architecture-matched all-MSE control.", "",
            "| Changed model | Target domain | Target | R2 delta | 95% CI |",
            "|---|---|---|---:|---:|",
        ))
        for row in aligned_geometry:
            report.append(
                f"| {LABELS.get(row['changed'], row['changed'])} | "
                f"{row['family']} | {row['task']} | "
                f"{float(row['r2_delta']):+.4f} | "
                f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}] |"
            )
    if concentration_summary:
        report.extend((
            "", "## Semantic feature concentration", "",
            "Top-m curves mask all but the largest standardized probe coefficients; "
            "they do not refit the probe. m90/m95 are computed only for attributes "
            "whose full-probe test AUROC is at least 0.55.", "",
            "| Representation | Effective support | Median m90 | m90 reached | Median m95 | m95 reached |",
            "|---|---:|---:|---:|---:|---:|",
        ))
        for row in concentration_summary:
            report.append(
                f"| {LABELS.get(row['representation'], row['representation'])} | "
                f"{float(row['median_effective_support']):.1f} | "
                f"{float(row['median_m90']):.0f} | "
                f"{float(row['m90_reached_fraction']):.1%} | "
                f"{float(row['median_m95']):.0f} | "
                f"{float(row['m95_reached_fraction']):.1%} |"
            )
    (output / "README.md").write_text("\n".join(report) + "\n")
    (output / "manifest.json").write_text(json.dumps({
        "runs_dir": args.runs_dir, "targets": args.targets,
        "representations": sorted(runs), "seed": args.seed,
    }, indent=2) + "\n")
    print(json.dumps({"representations": sorted(runs),
                      "output_dir": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
