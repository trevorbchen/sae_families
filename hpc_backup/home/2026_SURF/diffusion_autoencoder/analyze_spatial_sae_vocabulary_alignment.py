#!/usr/bin/env python3
"""Cross-seed vocabulary alignment for flat or convolutional spatial SAEs."""

import argparse
import csv
import json
import os
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


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
WEIGHTED_OBJECTIVES_WITH_BANDS = (
    "latent_mse",
    "weighted_laplacian",
    "weighted_fourier_bands",
    "weighted_fourier_low",
    "weighted_fourier_mid",
    "weighted_fourier_high",
)
SEEDS = (2030, 2031, 2032)
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def checkpoint_path(root, architecture, objective, seed, checkpoint_root=None):
    if checkpoint_root:
        directory = objective.removeprefix("weighted_")
        return os.path.join(
            checkpoint_root, directory, f"seed_{seed}", "last.pt"
        )
    if architecture == "vector":
        return os.path.join(
            root, "results", "vector_latent_sae_replicates", objective,
            f"seed_{seed}", "last.pt",
        )
    if objective.startswith("weighted_"):
        return os.path.join(
            root, "results", "spatial_map_sae_weighted_seed_replicates",
            architecture, objective, f"seed_{seed}", "last.pt",
        )
    if seed == 2030:
        return os.path.join(
            root, "results", "spatial_map_saes", architecture, objective,
            "last.pt",
        )
    return os.path.join(
        root, "results", "spatial_map_sae_seed_replicates", architecture,
        objective, f"seed_{seed}", "last.pt",
    )


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hungarian_mapping(similarity):
    rows, columns = linear_sum_assignment(
        -similarity.detach().float().cpu().numpy()
    )
    mapping = torch.full(
        (similarity.shape[0],), -1, dtype=torch.long,
        device=similarity.device,
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


def assigned_mean(similarity, mapping):
    rows = torch.where(mapping >= 0)[0]
    if not len(rows):
        return float("nan")
    return float(similarity[rows, mapping[rows]].mean())


def shared_mask(decoder_similarity, encoder_similarity, decoder_map, encoder_map, threshold):
    rows = torch.arange(len(decoder_map), device=decoder_map.device)
    assigned = decoder_map >= 0
    agreed = assigned & (decoder_map == encoder_map)
    columns = decoder_map.clamp_min(0)
    return (
        agreed
        & (decoder_similarity[rows, columns] >= threshold)
        & (encoder_similarity[rows, columns] >= threshold)
    )


@torch.no_grad()
def extract_activations(state, raw_latents, valid_indices, batch_size, device):
    model = state["model"]
    config = state["config"]
    architecture = config["architecture"]
    encoder = model["encoder"].float().to(device)
    encoder_bias = model["encoder_bias"].float().to(device)
    decoder_bias = model["dec_bias"].float().to(device)
    channel_mean = state["channel_mean"].float().to(device)
    channel_std = state["channel_std"].float().to(device)
    if raw_latents.ndim == 2:
        channel_mean = channel_mean[None, :]
        channel_std = channel_std[None, :]
    else:
        channel_mean = channel_mean[None, :, None, None]
        channel_std = channel_std[None, :, None, None]
    feature_count = len(encoder)
    k = int(config["k"])
    batches = []

    for start in range(0, len(valid_indices), batch_size):
        indices = valid_indices[start:start + batch_size]
        raw = raw_latents[indices].to(device).float()
        standardized = (raw - channel_mean) / channel_std
        if architecture == "flat":
            flat = standardized.flatten(1)
            activations = F.relu(F.linear(
                flat - decoder_bias, encoder, encoder_bias
            ))
            values, selected = torch.topk(activations, k, dim=1)
            dense = torch.zeros(
                len(raw), feature_count, dtype=torch.float16, device=device
            )
            dense.scatter_(1, selected, values.half())
        else:
            centered = standardized.flatten(2) - decoder_bias[None, :, None]
            activations = F.relu(
                torch.einsum("bcp,fc->bfp", centered, encoder)
                + encoder_bias[None, :, None]
            )
            values, selected = torch.topk(activations, k, dim=1)
            dense_positions = torch.zeros(
                len(raw), feature_count, activations.shape[2],
                dtype=torch.float16, device=device,
            )
            dense_positions.scatter_(1, selected, values.half())
            # A convolutional atom is position-shared. Max pooling asks whether
            # it is present anywhere in an image without conflating location.
            dense = dense_positions.amax(dim=2)
        batches.append(dense.cpu())
    return torch.cat(batches).float()


@torch.no_grad()
def load_vocabulary(path, raw_latents, valid_indices, top_features, min_active, batch_size, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    activations = extract_activations(
        state, raw_latents, valid_indices, batch_size, device
    )
    split = len(activations) // 2
    match_activations = activations[:split]
    eval_activations = activations[split:]
    active_count = (match_activations > 0).sum(0)
    activation_mass = match_activations.sum(0)
    eligible = torch.where(active_count >= min_active)[0]
    if len(eligible) < top_features:
        raise RuntimeError(
            f"only {len(eligible)} features meet min_active={min_active}; "
            f"need {top_features} for width-controlled matching"
        )
    if len(eligible) > top_features:
        order = torch.argsort(activation_mass[eligible], descending=True)
        selected = eligible[order[:top_features]]
    else:
        selected = eligible

    model = state["model"]
    decoder = F.normalize(model["decoder"][selected].float(), dim=1)
    encoder = F.normalize(model["encoder"][selected].float(), dim=1)
    match_activations = match_activations[:, selected]
    eval_activations = eval_activations[:, selected]

    def normalized_columns(values):
        centered = values - values.mean(0, keepdim=True)
        return F.normalize(centered, dim=0)

    return {
        "decoder": decoder.to(device),
        "encoder": encoder.to(device),
        "match_activations": match_activations.to(device),
        "eval_activations": eval_activations.to(device),
        "normalized_match": normalized_columns(match_activations).to(device),
        "normalized_eval": normalized_columns(eval_activations).to(device),
        "selected": selected,
        "eligible_count": len(eligible),
        "path": path,
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


@torch.no_grad()
def align_pair(vocabulary_a, vocabulary_b):
    decoder_similarity = vocabulary_a["decoder"] @ vocabulary_b["decoder"].T
    encoder_similarity = vocabulary_a["encoder"] @ vocabulary_b["encoder"].T
    activation_match = (
        vocabulary_a["normalized_match"].T
        @ vocabulary_b["normalized_match"]
    )
    activation_eval = (
        vocabulary_a["normalized_eval"].T
        @ vocabulary_b["normalized_eval"]
    )
    decoder_map_a = hungarian_mapping(decoder_similarity)
    encoder_map_a = hungarian_mapping(encoder_similarity)
    activation_map_a = hungarian_mapping(activation_match)
    decoder_map_b = invert_mapping(decoder_map_a, len(vocabulary_b["decoder"]))
    encoder_map_b = invert_mapping(encoder_map_a, len(vocabulary_b["encoder"]))
    max_decoder_a, nearest_decoder_a = decoder_similarity.max(1)
    max_decoder_b, nearest_decoder_b = decoder_similarity.max(0)
    max_activation_a, nearest_activation_a = activation_eval.max(1)
    max_activation_b, nearest_activation_b = activation_eval.max(0)
    return {
        "decoder_similarity": decoder_similarity,
        "encoder_similarity": encoder_similarity,
        "activation_eval": activation_eval,
        "decoder_map_a": decoder_map_a,
        "decoder_map_b": decoder_map_b,
        "encoder_map_a": encoder_map_a,
        "encoder_map_b": encoder_map_b,
        "activation_map_a": activation_map_a,
        "max_decoder_a": max_decoder_a,
        "max_decoder_b": max_decoder_b,
        "nearest_decoder_a": nearest_decoder_a,
        "nearest_decoder_b": nearest_decoder_b,
        "max_activation_a": max_activation_a,
        "max_activation_b": max_activation_b,
        "nearest_activation_a": nearest_activation_a,
        "nearest_activation_b": nearest_activation_b,
    }


def oriented(pair, direction):
    if direction == "a":
        return {
            "decoder_map": pair["decoder_map_a"],
            "encoder_map": pair["encoder_map_a"],
            "max_decoder": pair["max_decoder_a"],
            "nearest_decoder": pair["nearest_decoder_a"],
            "max_activation": pair["max_activation_a"],
            "nearest_activation": pair["nearest_activation_a"],
            "decoder_similarity": pair["decoder_similarity"],
            "encoder_similarity": pair["encoder_similarity"],
        }
    return {
        "decoder_map": pair["decoder_map_b"],
        "encoder_map": pair["encoder_map_b"],
        "max_decoder": pair["max_decoder_b"],
        "nearest_decoder": pair["nearest_decoder_b"],
        "max_activation": pair["max_activation_b"],
        "nearest_activation": pair["nearest_activation_b"],
        "decoder_similarity": pair["decoder_similarity"].T,
        "encoder_similarity": pair["encoder_similarity"].T,
    }


def view_shared_mask(view, threshold):
    rows = torch.arange(len(view["decoder_map"]), device=view["decoder_map"].device)
    assigned = view["decoder_map"] >= 0
    agreed = assigned & (view["decoder_map"] == view["encoder_map"])
    columns = view["decoder_map"].clamp_min(0)
    return (
        agreed
        & (view["decoder_similarity"][rows, columns] >= threshold)
        & (view["encoder_similarity"][rows, columns] >= threshold)
    )


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
    }


@torch.no_grad()
def subspace_similarity(geometry_a, geometry_b):
    overlap = geometry_a["basis90"].T @ geometry_b["basis90"]
    principal_cosines = torch.linalg.svdvals(overlap).clamp(0, 1)
    denominator = min(geometry_a["rank90"], geometry_b["rank90"])
    return {
        "projection_similarity": float(
            principal_cosines.square().sum() / max(denominator, 1)
        ),
        "mean_principal_cosine": float(principal_cosines.mean()),
        "minimum_principal_cosine": float(principal_cosines.min()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--latent-cache", required=True)
    parser.add_argument("--architecture", choices=("flat", "conv", "vector"), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-features", type=int, default=256)
    parser.add_argument("--min-active", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument(
        "--include-fourier-bands", action="store_true",
        help="include weighted_fourier_bands in a weighted objective grid",
    )
    parser.add_argument(
        "--checkpoint-root",
        help="optional objective/seed checkpoint tree overriding legacy paths",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
    raw_latents = cache["latents"].float()
    valid_indices = cache["valid_indices"]
    if args.weighted and args.include_fourier_bands:
        objectives = WEIGHTED_OBJECTIVES_WITH_BANDS
    else:
        objectives = WEIGHTED_OBJECTIVES if args.weighted else OBJECTIVES
    keys = [(objective, seed) for objective in objectives for seed in SEEDS]
    vocabularies = {}
    for key in keys:
        path = checkpoint_path(
            args.root, args.architecture, *key,
            checkpoint_root=args.checkpoint_root,
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        print(f"loading {key} from {path}", flush=True)
        vocabularies[key] = load_vocabulary(
            path, raw_latents, valid_indices, args.top_features,
            args.min_active, args.batch_size, device,
        )
        print(
            f"{key}: selected {len(vocabularies[key]['selected'])} of "
            f"{vocabularies[key]['eligible_count']} eligible features",
            flush=True,
        )

    pair_data = {}
    pair_rows = []
    for key_a, key_b in combinations(keys, 2):
        print(f"aligning {key_a} and {key_b}", flush=True)
        vocabulary_a = vocabularies[key_a]
        vocabulary_b = vocabularies[key_b]
        pair = align_pair(vocabulary_a, vocabulary_b)
        pair_data[(key_a, key_b)] = pair
        view_a = oriented(pair, "a")
        view_b = oriented(pair, "b")
        decoder_map_a = pair["decoder_map_a"]
        decoder_map_b = pair["decoder_map_b"]
        encoder_map_a = pair["encoder_map_a"]
        encoder_map_b = pair["encoder_map_b"]
        row = {
            "objective_a": key_a[0],
            "seed_a": key_a[1],
            "objective_b": key_b[0],
            "seed_b": key_b[1],
            "alive_a": len(vocabulary_a["selected"]),
            "alive_b": len(vocabulary_b["selected"]),
            "eligible_a": vocabulary_a["eligible_count"],
            "eligible_b": vocabulary_b["eligible_count"],
            "mean_max_decoder_cosine_a_to_b": float(view_a["max_decoder"].mean()),
            "mean_max_decoder_cosine_b_to_a": float(view_b["max_decoder"].mean()),
            "hungarian_decoder_cosine": assigned_mean(
                pair["decoder_similarity"], decoder_map_a
            ),
            "hungarian_encoder_cosine": assigned_mean(
                pair["encoder_similarity"], encoder_map_a
            ),
            # Matching and evaluation use disjoint validation-image halves.
            "hungarian_activation_correlation": assigned_mean(
                pair["activation_eval"], pair["activation_map_a"]
            ),
            "mean_max_activation_correlation_a_to_b": float(
                view_a["max_activation"].mean()
            ),
            "mean_max_activation_correlation_b_to_a": float(
                view_b["max_activation"].mean()
            ),
            "linear_cka": linear_cka(
                vocabulary_a["eval_activations"],
                vocabulary_b["eval_activations"],
            ),
        }
        for threshold in THRESHOLDS:
            shared_a = shared_mask(
                pair["decoder_similarity"], pair["encoder_similarity"],
                decoder_map_a, encoder_map_a, threshold,
            )
            shared_b = shared_mask(
                pair["decoder_similarity"].T,
                pair["encoder_similarity"].T,
                decoder_map_b, encoder_map_b, threshold,
            )
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
            "alive_features": len(vocabularies[key]["selected"]),
            "eligible_features": vocabularies[key]["eligible_count"],
            "effective_rank": geometry["effective_rank"],
            "stable_rank": geometry["stable_rank"],
            "rank90": geometry["rank90"],
        })
    write_csv(os.path.join(args.out_dir, "vocabulary_geometry.csv"), geometry_rows)

    subspace_rows = []
    for key_a, key_b in combinations(keys, 2):
        subspace_rows.append({
            "objective_a": key_a[0],
            "seed_a": key_a[1],
            "objective_b": key_b[0],
            "seed_b": key_b[1],
            **subspace_similarity(geometries[key_a], geometries[key_b]),
        })
    write_csv(os.path.join(args.out_dir, "subspace_alignment.csv"), subspace_rows)

    def get_view(source, target):
        if keys.index(source) < keys.index(target):
            return oriented(pair_data[(source, target)], "a")
        return oriented(pair_data[(target, source)], "b")

    curve_rows = []
    feature_rows = []
    for source in keys:
        vocabulary = vocabularies[source]
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
                view_shared_mask(view, threshold) for view in views_same
            ])
            cross_hungarian = torch.stack([
                view_shared_mask(view, threshold) for view in views_cross
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
                "alive_features": len(vocabulary["selected"]),
                "eligible_features": vocabulary["eligible_count"],
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
        for local_index, feature in enumerate(vocabulary["selected"].tolist()):
            same_decoder = [
                float(view["max_decoder"][local_index]) for view in views_same
            ]
            cross_decoder = [
                float(view["max_decoder"][local_index]) for view in views_cross
            ]
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
            "space": {
                "flat": "flattened spatial-AE latent map",
                "conv": "translation-shared spatial-AE channel atoms",
                "vector": "512-dimensional conventional vector-AE latent",
            }[args.architecture],
            "architecture": args.architecture,
            "objectives": list(objectives),
            "seeds": list(SEEDS),
            "validation_examples": len(valid_indices),
            "matching_examples": len(valid_indices) // 2,
            "evaluation_examples": len(valid_indices) - len(valid_indices) // 2,
            "top_features": args.top_features,
            "min_active_examples": args.min_active,
            "primary_threshold": 0.7,
            "conv_activation_pooling": "max_over_spatial_positions",
            "activation_matching": "first half; correlation evaluated on second half",
        },
        "vocabularies": {
            f"{objective}_seed_{seed}": {
                "selected_features": len(vocabularies[(objective, seed)]["selected"]),
                "eligible_features": vocabularies[(objective, seed)]["eligible_count"],
                "checkpoint": vocabularies[(objective, seed)]["path"],
            }
            for objective, seed in keys
        },
        "primary_threshold_rows": [
            row for row in curve_rows if row["threshold"] == 0.7
        ],
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
