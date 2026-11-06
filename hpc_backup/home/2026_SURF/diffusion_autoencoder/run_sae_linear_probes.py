#!/usr/bin/env python3
"""Fit held-out linear probes to one cached ConvAE/SAE representation."""

import argparse
import csv
import json
import os
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def sparse_matrix(payload, domain):
    values = payload[f"{domain}_values"].float().numpy()
    indices = payload[f"{domain}_indices"].long().numpy()
    rows = np.repeat(np.arange(len(values), dtype=np.int32), values.shape[1])
    matrix = sp.csr_matrix(
        (values.reshape(-1), (rows, indices.reshape(-1))),
        shape=(len(values), int(payload["dimension"])),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    return matrix


def load_matrices(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["format"] == "dense":
        clean = payload["clean"].float().numpy()
        controlled = payload["controlled"].float().numpy()
    else:
        clean = sparse_matrix(payload, "clean")
        controlled = sparse_matrix(payload, "controlled")
    return payload, clean, controlled


def scale_matrices(clean, controlled, train_mask):
    scaler = StandardScaler(with_mean=not sp.issparse(clean), copy=False)
    scaler.fit(clean[train_mask])
    return scaler.transform(clean), scaler.transform(controlled)


def binary_probe(x, y, train, valid, test, c_values, feature_budgets, seed):
    solver = "liblinear"
    best = None
    for c_value in c_values:
        model = LogisticRegression(
            C=c_value, class_weight="balanced", solver=solver,
            max_iter=500, random_state=seed,
        )
        model.fit(x[train], y[train])
        probability = model.predict_proba(x[valid])[:, 1]
        score = roc_auc_score(y[valid], probability)
        if best is None or score > best[0]:
            best = (score, c_value)
    fit = train | valid
    model = LogisticRegression(
        C=best[1], class_weight="balanced", solver=solver,
        max_iter=800, random_state=seed,
    )
    model.fit(x[fit], y[fit])
    probability = model.predict_proba(x[test])[:, 1]
    prediction = probability >= 0.5
    test_auroc = roc_auc_score(y[test], probability)
    coefficient = model.coef_.reshape(-1)
    ranking = np.argsort(-np.abs(coefficient))
    squared = coefficient ** 2
    squared_total = float(squared.sum())
    absolute_total = float(np.abs(coefficient).sum())
    effective_support = absolute_total ** 2 / max(squared_total, 1e-20)
    feature_efficiency = []
    for requested_budget in feature_budgets:
        budget = min(requested_budget, len(coefficient))
        selected = ranking[:budget]
        score = np.asarray(x[test][:, selected] @ coefficient[selected]).reshape(-1)
        top_auroc = roc_auc_score(y[test], score)
        retention = (
            (top_auroc - 0.5) / (test_auroc - 0.5)
            if test_auroc > 0.5 else float("nan")
        )
        feature_efficiency.append({
            "requested_budget": requested_budget,
            "feature_budget": budget,
            "test_auroc": top_auroc,
            "above_chance_retention": retention,
            "coefficient_l2_mass": float(squared[selected].sum())
            / max(squared_total, 1e-20),
        })
    return {
        "best_c": best[1],
        "valid_auroc": best[0],
        "test_auroc": test_auroc,
        "test_balanced_accuracy": balanced_accuracy_score(y[test], prediction),
        "probability": probability.astype(np.float32),
        "coefficient": coefficient.astype(np.float32),
        "effective_support": effective_support,
        "feature_efficiency": feature_efficiency,
    }


def regression_probe(x, y, train, valid, test, alphas):
    train_mean = float(y[train].mean())
    train_std = float(y[train].std())
    standardized = (y - train_mean) / max(train_std, 1e-8)
    best = None
    for alpha in alphas:
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(x[train], standardized[train])
        prediction = model.predict(x[valid])
        score = r2_score(standardized[valid], prediction)
        if best is None or score > best[0]:
            best = (score, alpha)
    fit = train | valid
    model = Ridge(alpha=best[1], solver="lsqr")
    model.fit(x[fit], standardized[fit])
    prediction = model.predict(x[test]) * max(train_std, 1e-8) + train_mean
    return {
        "best_alpha": best[1],
        "valid_r2": best[0],
        "test_r2": r2_score(y[test], prediction),
        "prediction": prediction.astype(np.float32),
    }


def control_probe(x, y, train, valid, test, c_values, seed, jobs):
    solver = "saga" if sp.issparse(x) else "lbfgs"
    best = None
    for c_value in c_values:
        model = LogisticRegression(
            C=c_value, class_weight="balanced", solver=solver, max_iter=600,
            random_state=seed, n_jobs=jobs,
        )
        model.fit(x[train], y[train])
        score = balanced_accuracy_score(y[valid], model.predict(x[valid]))
        if best is None or score > best[0]:
            best = (score, c_value)
    fit = train | valid
    model = LogisticRegression(
        C=best[1], class_weight="balanced", solver=solver, max_iter=900,
        random_state=seed, n_jobs=jobs,
    )
    model.fit(x[fit], y[fit])
    prediction = model.predict(x[test])
    probability = model.predict_proba(x[test])
    return {
        "best_c": best[1], "valid_balanced_accuracy": best[0],
        "test_accuracy": accuracy_score(y[test], prediction),
        "test_balanced_accuracy": balanced_accuracy_score(y[test], prediction),
        "prediction": prediction.astype(np.int16),
        "probability": probability.astype(np.float32),
    }


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--c-values", default="0.03,0.3,3")
    parser.add_argument("--alphas", default="0.1,1,10,100")
    parser.add_argument(
        "--feature-budgets", default="1,2,4,8,16,32,64,128,256,512,1024,2048"
    )
    parser.add_argument("--seed", type=int, default=260818)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    c_values = tuple(float(value) for value in args.c_values.split(","))
    alphas = tuple(float(value) for value in args.alphas.split(","))
    feature_budgets = tuple(
        int(value) for value in args.feature_budgets.split(",")
    )
    targets = torch.load(args.targets, map_location="cpu", weights_only=False)
    representation, clean, controlled = load_matrices(args.representation)
    split = targets["split"].numpy()
    train, valid, test = split == 0, split == 1, split == 2
    clean, controlled = scale_matrices(clean, controlled, train)

    attribute_labels = targets["attributes"].numpy().astype(np.int8)
    attribute_names = list(targets["attribute_names"])
    semantic = Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(binary_probe)(
            clean, attribute_labels[:, index], train, valid, test,
            c_values, feature_budgets, args.seed + index,
        )
        for index in range(len(attribute_names))
    )
    geometry_labels = targets["geometry_targets"].numpy().astype(np.float32)
    geometry_names = list(targets["geometry_names"])
    geometry = Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(regression_probe)(
            clean, geometry_labels[:, index], train, valid, test, alphas,
        )
        for index in range(len(geometry_names))
    )
    source_geometry_labels = targets["source_geometry_targets"].numpy().astype(np.float32)
    source_geometry = Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(regression_probe)(
            clean, source_geometry_labels[:, index], train, valid, test, alphas,
        )
        for index in range(len(geometry_names))
    )
    control_labels = targets["control_labels"].numpy().astype(np.int16)
    controlled_result = control_probe(
        controlled, control_labels, train, valid, test, c_values,
        args.seed, args.jobs,
    )

    rows = []
    for name, result in zip(attribute_names, semantic):
        rows.extend((
            {"representation": representation["name"], "family": "semantic",
             "task": name, "metric": "auroc", "score": result["test_auroc"],
             "validation_score": result["valid_auroc"], "regularization": result["best_c"]},
            {"representation": representation["name"], "family": "semantic",
             "task": name, "metric": "balanced_accuracy",
             "score": result["test_balanced_accuracy"],
             "validation_score": result["valid_auroc"], "regularization": result["best_c"]},
        ))
    for name, result in zip(geometry_names, geometry):
        rows.append({
            "representation": representation["name"], "family": "geometry",
            "task": name, "metric": "r2", "score": result["test_r2"],
            "validation_score": result["valid_r2"],
            "regularization": result["best_alpha"],
        })
    for name, result in zip(geometry_names, source_geometry):
        rows.append({
            "representation": representation["name"],
            "family": "source_geometry", "task": name, "metric": "r2",
            "score": result["test_r2"],
            "validation_score": result["valid_r2"],
            "regularization": result["best_alpha"],
        })
    rows.extend((
        {"representation": representation["name"], "family": "controlled",
         "task": "transform_class", "metric": "accuracy",
         "score": controlled_result["test_accuracy"],
         "validation_score": controlled_result["valid_balanced_accuracy"],
         "regularization": controlled_result["best_c"]},
        {"representation": representation["name"], "family": "controlled",
         "task": "transform_class", "metric": "balanced_accuracy",
         "score": controlled_result["test_balanced_accuracy"],
         "validation_score": controlled_result["valid_balanced_accuracy"],
         "regularization": controlled_result["best_c"]},
    ))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "results.csv", rows)
    efficiency_rows = []
    for name, result in zip(attribute_names, semantic):
        for record in result["feature_efficiency"]:
            efficiency_rows.append({
                "representation": representation["name"],
                "task": name,
                "feature_budget": record["feature_budget"],
                "requested_budget": record["requested_budget"],
                "test_auroc": record["test_auroc"],
                "full_test_auroc": result["test_auroc"],
                "above_chance_retention": record["above_chance_retention"],
                "coefficient_l2_mass": record["coefficient_l2_mass"],
                "effective_support": result["effective_support"],
            })
    write_csv(output / "semantic_feature_efficiency.csv", efficiency_rows)
    prediction_payload = {
        "representation": representation["name"],
        "test_indices": torch.from_numpy(np.where(test)[0]),
        "test_identities": targets["identities"][torch.from_numpy(test)],
        "semantic_probability": torch.from_numpy(np.stack([
            result["probability"] for result in semantic
        ], axis=1)),
        "semantic_coefficients": torch.from_numpy(np.stack([
            result["coefficient"] for result in semantic
        ], axis=0)),
        "geometry_prediction": torch.from_numpy(np.stack([
            result["prediction"] for result in geometry
        ], axis=1)),
        "source_geometry_prediction": torch.from_numpy(np.stack([
            result["prediction"] for result in source_geometry
        ], axis=1)),
        "control_prediction": torch.from_numpy(controlled_result["prediction"]),
        "control_probability": torch.from_numpy(controlled_result["probability"]),
        "attribute_names": attribute_names,
        "geometry_names": geometry_names,
        "control_names": targets["control_names"],
    }
    torch.save(prediction_payload, output / "predictions.pt")
    manifest = {
        "representation": representation["name"],
        "representation_path": args.representation,
        "dimension": int(representation["dimension"]),
        "k": int(representation.get("k", representation["dimension"])),
        "format": representation["format"],
        "metadata": representation["metadata"],
        "split_counts": {"train": int(train.sum()), "validation": int(valid.sum()),
                         "test": int(test.sum())},
        "c_values": c_values, "alphas": alphas,
        "feature_budgets": feature_budgets, "seed": args.seed,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "representation": representation["name"],
        "semantic_macro_auroc": float(np.mean([x["test_auroc"] for x in semantic])),
        "geometry_macro_r2": float(np.mean([x["test_r2"] for x in geometry])),
        "source_geometry_macro_r2": float(np.mean([
            x["test_r2"] for x in source_geometry
        ])),
        "controlled_balanced_accuracy": controlled_result["test_balanced_accuracy"],
        "output_dir": str(output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
