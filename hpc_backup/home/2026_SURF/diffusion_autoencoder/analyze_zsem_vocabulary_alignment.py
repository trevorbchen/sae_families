#!/usr/bin/env python3
"""Published-style vocabulary alignment across transformed-objective z_sem SAEs."""

import argparse
import csv
import json
import math
import os
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


SEEDS = (2030, 2031, 2032)
OBJECTIVES = (
    "latent_mse",
    "laplacian",
    "fourier_bands",
    "fourier_low",
    "fourier_mid",
    "fourier_high",
)
WEIGHTED_OBJECTIVES = (
    "latent_mse",
    "weighted_laplacian",
    "weighted_fourier_low",
    "weighted_fourier_mid",
    "weighted_fourier_high",
)
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def checkpoint_path(root, objective, seed):
    if objective.startswith("weighted_"):
        return os.path.join(
            root, "results", "zsem_weighted_objective_seed_replicates",
            objective, f"seed_{seed}", "last.pt",
        )
    if objective != "latent_mse":
        corrected = os.path.join(
            root, "results", "zsem_objective_seed_replicates_v2", objective,
            f"seed_{seed}", "last.pt",
        )
        if os.path.isfile(corrected):
            return corrected
        if objective == "laplacian" and seed == 2030:
            return os.path.join(
                root, "results", "zsem_transformed_pure_k32", "laplacian",
                "last.pt",
            )
    return os.path.join(
        root, "results", "zsem_objective_seed_replicates", objective,
        f"seed_{seed}", "last.pt",
    )


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def load_vocabulary(path, codes, min_active, batch_size, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = state["model"]
    decoder = F.normalize(model["decoder"].float(), dim=1)
    encoder = F.normalize(model["encoder.weight"].float(), dim=1)
    encoder_weight = model["encoder.weight"].float().to(device)
    encoder_bias = model["encoder.bias"].float().to(device)
    decoder_bias = model["dec_bias"].float().to(device)
    mean = state["input_mean"].float().to(device)
    std = state["input_std"].float().to(device)
    k = int(state["config"]["k"])
    feature_count = len(decoder)
    dense_batches = []
    active_count = torch.zeros(feature_count, dtype=torch.long)
    activation_mass = torch.zeros(feature_count)
    for start in range(0, len(codes), batch_size):
        raw = codes[start:start + batch_size].to(device)
        standardized = (raw - mean) / std
        activations = F.relu(F.linear(
            standardized - decoder_bias, encoder_weight, encoder_bias
        ))
        values, indices = torch.topk(activations, k, dim=1)
        dense = torch.zeros(
            len(raw), feature_count, dtype=torch.float16, device=device
        )
        dense.scatter_(1, indices, values.half())
        dense_batches.append(dense.cpu())
        positive = values > 0
        cpu_indices = indices[positive].cpu()
        cpu_values = values[positive].float().cpu()
        active_count.scatter_add_(
            0, cpu_indices, torch.ones_like(cpu_indices, dtype=torch.long)
        )
        activation_mass.scatter_add_(0, cpu_indices, cpu_values)
    dense = torch.cat(dense_batches)
    alive = torch.where(active_count >= min_active)[0]
    activations = dense[:, alive].float()
    centered = activations - activations.mean(0, keepdim=True)
    normalized_activations = F.normalize(centered, dim=0)
    return {
        "decoder": decoder[alive].to(device),
        "encoder": encoder[alive].to(device),
        "activations": activations.to(device),
        "normalized_activations": normalized_activations.to(device),
        "alive": alive,
        "active_count": active_count[alive],
        "activation_mass": activation_mass[alive],
        "path": path,
        "k": k,
    }


def hungarian_mapping(similarity):
    rows, columns = linear_sum_assignment(
        -similarity.detach().float().cpu().numpy()
    )
    mapping = torch.full(
        (similarity.shape[0],), -1, dtype=torch.long, device=similarity.device
    )
    mapping[torch.from_numpy(rows).to(similarity.device)] = torch.from_numpy(
        columns
    ).to(similarity.device)
    return mapping


def invert_mapping(mapping, output_size):
    inverse = torch.full(
        (output_size,), -1, dtype=torch.long, device=mapping.device
    )
    rows = torch.where(mapping >= 0)[0]
    inverse[mapping[rows]] = rows
    return inverse


@torch.no_grad()
def align_pair(vocab_a, vocab_b):
    decoder_similarity = vocab_a["decoder"] @ vocab_b["decoder"].T
    encoder_similarity = vocab_a["encoder"] @ vocab_b["encoder"].T
    activation_similarity = (
        vocab_a["normalized_activations"].T
        @ vocab_b["normalized_activations"]
    )
    decoder_map_a = hungarian_mapping(decoder_similarity)
    encoder_map_a = hungarian_mapping(encoder_similarity)
    activation_map_a = hungarian_mapping(activation_similarity)
    decoder_map_b = invert_mapping(decoder_map_a, len(vocab_b["alive"]))
    encoder_map_b = invert_mapping(encoder_map_a, len(vocab_b["alive"]))
    activation_map_b = invert_mapping(activation_map_a, len(vocab_b["alive"]))
    max_decoder_a, nearest_decoder_a = decoder_similarity.max(1)
    max_decoder_b, nearest_decoder_b = decoder_similarity.max(0)
    max_activation_a, nearest_activation_a = activation_similarity.max(1)
    max_activation_b, nearest_activation_b = activation_similarity.max(0)
    return {
        "decoder_similarity": decoder_similarity.cpu(),
        "encoder_similarity": encoder_similarity.cpu(),
        "activation_similarity": activation_similarity.cpu(),
        "decoder_map_a": decoder_map_a.cpu(),
        "decoder_map_b": decoder_map_b.cpu(),
        "encoder_map_a": encoder_map_a.cpu(),
        "encoder_map_b": encoder_map_b.cpu(),
        "activation_map_a": activation_map_a.cpu(),
        "activation_map_b": activation_map_b.cpu(),
        "max_decoder_a": max_decoder_a.cpu(),
        "max_decoder_b": max_decoder_b.cpu(),
        "nearest_decoder_a": nearest_decoder_a.cpu(),
        "nearest_decoder_b": nearest_decoder_b.cpu(),
        "max_activation_a": max_activation_a.cpu(),
        "max_activation_b": max_activation_b.cpu(),
        "nearest_activation_a": nearest_activation_a.cpu(),
        "nearest_activation_b": nearest_activation_b.cpu(),
    }


def oriented(pair, direction):
    if direction == "a":
        return {
            "decoder_map": pair["decoder_map_a"],
            "encoder_map": pair["encoder_map_a"],
            "activation_map": pair["activation_map_a"],
            "max_decoder": pair["max_decoder_a"],
            "nearest_decoder": pair["nearest_decoder_a"],
            "max_activation": pair["max_activation_a"],
            "nearest_activation": pair["nearest_activation_a"],
            "decoder_similarity": pair["decoder_similarity"],
            "encoder_similarity": pair["encoder_similarity"],
            "activation_similarity": pair["activation_similarity"],
        }
    return {
        "decoder_map": pair["decoder_map_b"],
        "encoder_map": pair["encoder_map_b"],
        "activation_map": pair["activation_map_b"],
        "max_decoder": pair["max_decoder_b"],
        "nearest_decoder": pair["nearest_decoder_b"],
        "max_activation": pair["max_activation_b"],
        "nearest_activation": pair["nearest_activation_b"],
        "decoder_similarity": pair["decoder_similarity"].T,
        "encoder_similarity": pair["encoder_similarity"].T,
        "activation_similarity": pair["activation_similarity"].T,
    }


def shared_mask(view, threshold):
    rows = torch.arange(len(view["decoder_map"]))
    decoder_map = view["decoder_map"]
    encoder_map = view["encoder_map"]
    assigned = decoder_map >= 0
    agreed = assigned & (decoder_map == encoder_map)
    safe_columns = decoder_map.clamp_min(0)
    decoder_cosine = view["decoder_similarity"][rows, safe_columns]
    encoder_cosine = view["encoder_similarity"][rows, safe_columns]
    return agreed & (decoder_cosine >= threshold) & (encoder_cosine >= threshold)


def assigned_mean(similarity, mapping):
    rows = torch.where(mapping >= 0)[0]
    if not len(rows):
        return float("nan")
    return float(similarity[rows, mapping[rows]].mean())


@torch.no_grad()
def vocabulary_geometry(vocabulary):
    decoder = vocabulary["decoder"].float()
    _, singular_values, vh = torch.linalg.svd(decoder, full_matrices=False)
    energy = singular_values.square()
    probabilities = energy / energy.sum().clamp_min(1e-12)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    )
    stable_rank = energy.sum() / energy.max().clamp_min(1e-12)
    cumulative = probabilities.cumsum(0)
    rank90 = int(torch.searchsorted(
        cumulative, torch.tensor(0.9, device=cumulative.device)
    )) + 1
    return {
        "effective_rank": float(effective_rank),
        "stable_rank": float(stable_rank),
        "rank90": rank90,
        "basis90": vh[:rank90].T.contiguous(),
        "singular_values": singular_values.cpu(),
    }


@torch.no_grad()
def subspace_similarity(geometry_a, geometry_b):
    overlap = geometry_a["basis90"].T @ geometry_b["basis90"]
    principal_cosines = torch.linalg.svdvals(overlap).clamp(0, 1)
    denominator = min(
        geometry_a["rank90"], geometry_b["rank90"]
    )
    return {
        "projection_similarity": float(
            principal_cosines.square().sum() / max(denominator, 1)
        ),
        "mean_principal_cosine": float(principal_cosines.mean()),
        "minimum_principal_cosine": float(principal_cosines.min()),
    }


@torch.no_grad()
def linear_cka(activations_a, activations_b):
    x = activations_a - activations_a.mean(0, keepdim=True)
    y = activations_b - activations_b.mean(0, keepdim=True)
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    return float(
        cross.square().sum()
        / (xx.square().sum().sqrt() * yy.square().sum().sqrt()).clamp_min(1e-12)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-active", type=int, default=10)
    parser.add_argument("--split-seed", type=int, default=2030)
    parser.add_argument("--weighted", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    all_codes = data["codes"].float()
    permutation = torch.randperm(
        len(all_codes), generator=torch.Generator().manual_seed(args.split_seed)
    )
    codes = all_codes[permutation[int(0.9 * len(all_codes)):]]
    objectives = WEIGHTED_OBJECTIVES if args.weighted else OBJECTIVES
    keys = [(objective, seed) for objective in objectives for seed in SEEDS]

    vocabularies = {}
    for key in keys:
        path = checkpoint_path(args.root, *key)
        print(f"loading {key} from {path}", flush=True)
        vocabularies[key] = load_vocabulary(
            path, codes, args.min_active, args.batch_size, device
        )
        print(
            f"{key}: {len(vocabularies[key]['alive'])} alive features", flush=True
        )

    pair_data = {}
    pair_rows = []
    for key_a, key_b in combinations(keys, 2):
        print(f"aligning {key_a} and {key_b}", flush=True)
        pair = align_pair(vocabularies[key_a], vocabularies[key_b])
        pair_data[(key_a, key_b)] = pair
        view_a = oriented(pair, "a")
        view_b = oriented(pair, "b")
        row = {
            "objective_a": key_a[0],
            "seed_a": key_a[1],
            "objective_b": key_b[0],
            "seed_b": key_b[1],
            "alive_a": len(vocabularies[key_a]["alive"]),
            "alive_b": len(vocabularies[key_b]["alive"]),
            "mean_max_decoder_cosine_a_to_b": float(view_a["max_decoder"].mean()),
            "mean_max_decoder_cosine_b_to_a": float(view_b["max_decoder"].mean()),
            "hungarian_decoder_cosine": assigned_mean(
                view_a["decoder_similarity"], view_a["decoder_map"]
            ),
            "hungarian_encoder_cosine": assigned_mean(
                view_a["encoder_similarity"], view_a["encoder_map"]
            ),
            "hungarian_activation_correlation": assigned_mean(
                view_a["activation_similarity"], view_a["activation_map"]
            ),
            "mean_max_activation_correlation_a_to_b": float(
                view_a["max_activation"].mean()
            ),
            "mean_max_activation_correlation_b_to_a": float(
                view_b["max_activation"].mean()
            ),
            "linear_cka": linear_cka(
                vocabularies[key_a]["activations"],
                vocabularies[key_b]["activations"],
            ),
        }
        for threshold in THRESHOLDS:
            shared_a = shared_mask(view_a, threshold)
            shared_b = shared_mask(view_b, threshold)
            label = str(threshold).replace(".", "p")
            row[f"shared_count_{label}"] = int(shared_a.sum())
            row[f"shared_fraction_a_{label}"] = float(shared_a.float().mean())
            row[f"shared_fraction_b_{label}"] = float(shared_b.float().mean())
        pair_rows.append(row)
    write_csv(os.path.join(args.out_dir, "pairwise_alignment.csv"), pair_rows)

    geometries = {
        key: vocabulary_geometry(vocabulary)
        for key, vocabulary in vocabularies.items()
    }
    geometry_rows = []
    for key in keys:
        geometry = geometries[key]
        geometry_rows.append({
            "objective": key[0],
            "seed": key[1],
            "alive_features": len(vocabularies[key]["alive"]),
            "effective_rank": geometry["effective_rank"],
            "stable_rank": geometry["stable_rank"],
            "rank90": geometry["rank90"],
        })
    write_csv(os.path.join(args.out_dir, "vocabulary_geometry.csv"), geometry_rows)

    subspace_rows = []
    for key_a, key_b in combinations(keys, 2):
        metrics = subspace_similarity(geometries[key_a], geometries[key_b])
        subspace_rows.append({
            "objective_a": key_a[0],
            "seed_a": key_a[1],
            "objective_b": key_b[0],
            "seed_b": key_b[1],
            **metrics,
        })
    write_csv(os.path.join(args.out_dir, "subspace_alignment.csv"), subspace_rows)

    def get_view(source, target):
        if keys.index(source) < keys.index(target):
            return oriented(pair_data[(source, target)], "a")
        return oriented(pair_data[(target, source)], "b")

    curve_rows = []
    feature_rows = []
    for source in keys:
        source_vocab = vocabularies[source]
        same_targets = [
            key for key in keys if key != source and key[0] == source[0]
        ]
        cross_targets = [key for key in keys if key[0] != source[0]]
        views_same = [get_view(source, target) for target in same_targets]
        views_cross = [get_view(source, target) for target in cross_targets]
        threshold_masks = {}
        for threshold in THRESHOLDS:
            same_max = torch.stack([
                view["max_decoder"] >= threshold for view in views_same
            ])
            cross_max = torch.stack([
                view["max_decoder"] >= threshold for view in views_cross
            ])
            same_hungarian = torch.stack([
                shared_mask(view, threshold) for view in views_same
            ])
            cross_hungarian = torch.stack([
                shared_mask(view, threshold) for view in views_cross
            ])
            threshold_masks[threshold] = {
                "same_max": same_max,
                "cross_max": cross_max,
                "same_hungarian": same_hungarian,
                "cross_hungarian": cross_hungarian,
            }
            curve_rows.append({
                "objective": source[0],
                "seed": source[1],
                "threshold": threshold,
                "alive_features": len(source_vocab["alive"]),
                "within_mean_maxcos_reappearance_probability": float(
                    same_max.float().mean()
                ),
                "within_fraction_reappears_all_maxcos": float(
                    same_max.all(0).float().mean()
                ),
                "within_mean_hungarian_reappearance_probability": float(
                    same_hungarian.float().mean()
                ),
                "within_fraction_reappears_all_hungarian": float(
                    same_hungarian.all(0).float().mean()
                ),
                "cross_mean_maxcos_match_probability": float(
                    cross_max.float().mean()
                ),
                "cross_mean_hungarian_match_probability": float(
                    cross_hungarian.float().mean()
                ),
                "within_all_cross_none_maxcos": int((
                    same_max.all(0) & ~cross_max.any(0)
                ).sum()),
                "within_all_cross_none_hungarian": int((
                    same_hungarian.all(0) & ~cross_hungarian.any(0)
                ).sum()),
            })

        primary = threshold_masks[0.7]
        for local_index, feature in enumerate(source_vocab["alive"].tolist()):
            same_decoder = [float(view["max_decoder"][local_index]) for view in views_same]
            cross_decoder = [float(view["max_decoder"][local_index]) for view in views_cross]
            same_activation = [
                float(view["max_activation"][local_index]) for view in views_same
            ]
            cross_activation = [
                float(view["max_activation"][local_index]) for view in views_cross
            ]
            feature_rows.append({
                "objective": source[0],
                "seed": source[1],
                "feature": feature,
                "active_count": int(source_vocab["active_count"][local_index]),
                "activation_mass": float(source_vocab["activation_mass"][local_index]),
                "within_min_max_decoder_cosine": min(same_decoder),
                "within_mean_max_decoder_cosine": sum(same_decoder) / len(same_decoder),
                "cross_max_decoder_cosine": max(cross_decoder),
                "cross_mean_max_decoder_cosine": sum(cross_decoder) / len(cross_decoder),
                "within_min_max_activation_correlation": min(same_activation),
                "cross_max_activation_correlation": max(cross_activation),
                "within_maxcos_reappearance_probability_at_0p7": float(
                    primary["same_max"][:, local_index].float().mean()
                ),
                "cross_maxcos_match_probability_at_0p7": float(
                    primary["cross_max"][:, local_index].float().mean()
                ),
                "within_hungarian_reappearance_probability_at_0p7": float(
                    primary["same_hungarian"][:, local_index].float().mean()
                ),
                "cross_hungarian_match_probability_at_0p7": float(
                    primary["cross_hungarian"][:, local_index].float().mean()
                ),
                "within_all_cross_none_maxcos_at_0p7": bool(
                    primary["same_max"][:, local_index].all()
                    and not primary["cross_max"][:, local_index].any()
                ),
                "within_all_cross_none_hungarian_at_0p7": bool(
                    primary["same_hungarian"][:, local_index].all()
                    and not primary["cross_hungarian"][:, local_index].any()
                ),
            })
    write_csv(os.path.join(args.out_dir, "threshold_curves.csv"), curve_rows)
    write_csv(os.path.join(args.out_dir, "feature_recurrence.csv"), feature_rows)

    summary = {
        "method": {
            "space": "standardized 512-dimensional DiffAE z_sem",
            "validation_examples": len(codes),
            "min_active_examples": args.min_active,
            "objectives": list(objectives),
            "seeds": list(SEEDS),
            "primary_threshold": 0.7,
            "hungarian_shared_definition": (
                "encoder and decoder Hungarian assignments agree and both "
                "cosines meet the threshold"
            ),
        },
        "alive_features": {
            f"{key[0]}_seed_{key[1]}": len(vocabularies[key]["alive"])
            for key in keys
        },
        "primary_threshold_rows": [
            row for row in curve_rows if row["threshold"] == 0.7
        ],
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
