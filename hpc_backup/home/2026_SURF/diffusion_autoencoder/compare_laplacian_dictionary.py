#!/usr/bin/env python3
"""Directly compare active Laplacian SAE atoms with normal SAE dictionaries."""

import argparse
import csv
import json
import os

import torch
import torch.nn.functional as F


SEEDS = (2030, 2031, 2032)
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def load_decoder(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    return F.normalize(state["model"]["decoder"].float(), dim=1).to(device)


def active_laplacian_features(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [int(row["feature"]) for row in rows if row["valid_active"] == "True"]


@torch.no_grad()
def nearest(query, dictionaries, block_size):
    best_values = torch.full((len(query),), -1.0, device=query.device)
    best_features = torch.full(
        (len(query),), -1, dtype=torch.long, device=query.device
    )
    best_seeds = torch.full(
        (len(query),), -1, dtype=torch.long, device=query.device
    )
    for seed, dictionary in dictionaries:
        for start in range(0, len(dictionary), block_size):
            similarities = query @ dictionary[start:start + block_size].T
            values, indices = similarities.max(1)
            update = values > best_values
            best_values[update] = values[update]
            best_features[update] = indices[update] + start
            best_seeds[update] = seed
    return best_values.cpu(), best_features.cpu(), best_seeds.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--block-size", type=int, default=4096)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    laplacian_path = os.path.join(
        args.root, "results", "zsem_transformed_pure_k32", "laplacian", "last.pt"
    )
    feature_csv = os.path.join(
        args.root, "results", "zsem_objective_novelty", "laplacian_features.csv"
    )
    active_features = active_laplacian_features(feature_csv)
    laplacian = load_decoder(laplacian_path, device)
    query = laplacian[active_features]

    normal = []
    for seed in SEEDS:
        path = os.path.join(
            args.root, "results", "zsem_objective_seed_replicates",
            "latent_mse", f"seed_{seed}", "last.pt",
        )
        normal.append((seed, load_decoder(path, device)))

    same_objective = []
    for seed in SEEDS[1:]:
        path = os.path.join(
            args.root, "results", "zsem_objective_seed_replicates",
            "laplacian", f"seed_{seed}", "last.pt",
        )
        same_objective.append((seed, load_decoder(path, device)))

    baseline_cosine, baseline_feature, baseline_seed = nearest(
        query, normal, args.block_size
    )
    laplacian_cosine, laplacian_feature, laplacian_seed = nearest(
        query, same_objective, args.block_size
    )

    rows = []
    for index, feature in enumerate(active_features):
        rows.append({
            "laplacian_feature": feature,
            "nearest_normal_cosine": float(baseline_cosine[index]),
            "nearest_normal_seed": int(baseline_seed[index]),
            "nearest_normal_feature": int(baseline_feature[index]),
            "nearest_other_laplacian_cosine": float(laplacian_cosine[index]),
            "nearest_other_laplacian_seed": int(laplacian_seed[index]),
            "nearest_other_laplacian_feature": int(laplacian_feature[index]),
        })
    rows.sort(key=lambda row: row["nearest_normal_cosine"])
    csv_path = os.path.join(args.out_dir, "laplacian_to_normal_nearest.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "active_laplacian_features": len(rows),
        "normal_dictionary_atoms": sum(len(dictionary) for _, dictionary in normal),
        "mean_nearest_normal_cosine": float(baseline_cosine.mean()),
        "median_nearest_normal_cosine": float(baseline_cosine.median()),
        "minimum_nearest_normal_cosine": float(baseline_cosine.min()),
        "maximum_nearest_normal_cosine": float(baseline_cosine.max()),
        "thresholds": {
            str(threshold): {
                "matched": int((baseline_cosine >= threshold).sum()),
                "unmatched": int((baseline_cosine < threshold).sum()),
                "unmatched_fraction": float((baseline_cosine < threshold).float().mean()),
                "reproduced_laplacian_and_unmatched_normal": int((
                    (laplacian_cosine >= threshold)
                    & (baseline_cosine < threshold)
                ).sum()),
            }
            for threshold in THRESHOLDS
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
