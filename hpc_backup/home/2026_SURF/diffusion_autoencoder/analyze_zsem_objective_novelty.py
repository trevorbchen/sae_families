#!/usr/bin/env python3
"""Measure cross-seed stability and objective-specific features in z_sem SAEs."""

import argparse
import csv
import json
import os

import torch
import torch.nn.functional as F


OBJECTIVES = (
    "latent_mse",
    "laplacian",
    "fourier_bands",
    "fourier_low",
    "fourier_mid",
    "fourier_high",
)
SEEDS = (2030, 2031, 2032)


def checkpoint_paths(root):
    replicated = os.path.join(root, "results", "zsem_objective_seed_replicates")
    paths = {
        (objective, seed): os.path.join(
            replicated, objective, f"seed_{seed}", "last.pt"
        )
        for objective in OBJECTIVES
        for seed in SEEDS
        if objective == "latent_mse" or seed != 2030
    }
    paths.update({
        ("laplacian", 2030): os.path.join(
            root, "results", "zsem_transformed_pure_k32", "laplacian", "last.pt"
        ),
        ("fourier_bands", 2030): os.path.join(
            root, "results", "zsem_transformed_pure_k32", "fourier_bands", "last.pt"
        ),
        ("fourier_low", 2030): os.path.join(
            root, "results", "zsem_fourier_specific_pure_k32", "fourier_low", "last.pt"
        ),
        ("fourier_mid", 2030): os.path.join(
            root, "results", "zsem_fourier_specific_pure_k32", "fourier_mid", "last.pt"
        ),
        ("fourier_high", 2030): os.path.join(
            root, "results", "zsem_fourier_specific_pure_k32", "fourier_high", "last.pt"
        ),
    })
    return paths


@torch.no_grad()
def load_representation(path, codes, device, batch_size):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = state["model"]
    decoder = F.normalize(model["decoder"].float(), dim=1).half().cpu()
    encoder_weight = model["encoder.weight"].float().to(device)
    encoder_bias = model["encoder.bias"].float().to(device)
    decoder_bias = model["dec_bias"].float().to(device)
    mean = state["input_mean"].float().to(device)
    std = state["input_std"].float().to(device)
    k = int(state["config"]["k"])
    activation_batches = []
    feature_count = len(decoder)
    active_count = torch.zeros(feature_count, dtype=torch.long)
    activation_mass = torch.zeros(feature_count)
    activation_square_mass = torch.zeros(feature_count)
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
        activation_batches.append(dense.cpu())
        cpu_values = values.float().cpu()
        cpu_indices = indices.cpu()
        active_count.scatter_add_(
            0, cpu_indices.flatten(), torch.ones_like(cpu_indices.flatten())
        )
        activation_mass.scatter_add_(0, cpu_indices.flatten(), cpu_values.flatten())
        activation_square_mass.scatter_add_(
            0, cpu_indices.flatten(), cpu_values.square().flatten()
        )
    return {
        "decoder": decoder,
        "activations": torch.cat(activation_batches),
        "active_count": active_count,
        "activation_mass": activation_mass,
        "activation_square_mass": activation_square_mass,
        "path": path,
        "k": k,
    }


@torch.no_grad()
def decoder_candidates(rep_a, rep_b, candidate_count, device, block_size):
    decoder_b = rep_b["decoder"].to(device=device, dtype=torch.float32)
    values = []
    indices = []
    for start in range(0, len(rep_a["decoder"]), block_size):
        decoder_a = rep_a["decoder"][start:start + block_size].to(
            device=device, dtype=torch.float32
        )
        batch_values, batch_indices = torch.topk(
            decoder_a @ decoder_b.T, candidate_count, dim=1
        )
        values.append(batch_values.cpu())
        indices.append(batch_indices.int().cpu())
    return torch.cat(values), torch.cat(indices)


@torch.no_grad()
def activation_candidate_metrics(rep_a, rep_b, indices, device, block_size):
    acts_a = rep_a["activations"].to(device=device, dtype=torch.float32)
    acts_b = rep_b["activations"].to(device=device, dtype=torch.float32)
    sample_count = acts_a.shape[0]
    correlations = []
    jaccards = []
    for start in range(0, len(indices), block_size):
        selected = indices[start:start + block_size].long().to(device)
        x = acts_a[:, start:start + len(selected)].T
        y = acts_b[:, selected].permute(1, 2, 0)
        sum_x = x.sum(1)
        sum_x2 = x.square().sum(1)
        sum_y = y.sum(2)
        sum_y2 = y.square().sum(2)
        sum_xy = (x[:, None, :] * y).sum(2)
        covariance = sum_xy - sum_x[:, None] * sum_y / sample_count
        variance_x = (sum_x2 - sum_x.square() / sample_count).clamp_min(0)
        variance_y = (sum_y2 - sum_y.square() / sample_count).clamp_min(0)
        denominator = (variance_x[:, None] * variance_y).sqrt()
        correlation = torch.where(
            denominator > 1e-8, covariance / denominator.clamp_min(1e-8),
            torch.zeros_like(covariance),
        )
        active_x = x > 0
        active_y = y > 0
        intersection = (active_x[:, None, :] & active_y).sum(2)
        union = (active_x[:, None, :] | active_y).sum(2)
        jaccard = intersection.float() / union.clamp_min(1)
        correlations.append(correlation.cpu())
        jaccards.append(jaccard.cpu())
    return torch.cat(correlations), torch.cat(jaccards)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def joint_match(match, cosine_threshold, correlation_threshold):
    return (
        (match["cosine"] >= cosine_threshold)
        & (match["correlation"] >= correlation_threshold)
    ).any(1)


def best_candidate(match, correlation_threshold):
    eligible = match["correlation"] >= correlation_threshold
    masked = match["cosine"].masked_fill(~eligible, float("-inf"))
    position = masked.argmax(1)
    no_eligible = ~eligible.any(1)
    position[no_eligible] = 0
    row = torch.arange(len(position))
    return {
        "feature": match["indices"][row, position].long(),
        "cosine": match["cosine"][row, position],
        "correlation": match["correlation"][row, position],
        "jaccard": match["jaccard"][row, position],
        "correlation_eligible": ~no_eligible,
    }


def span_cosines(vector, atoms):
    results = {}
    for count in (1, 2, 4, 8):
        selected = atoms[:count]
        gram = selected @ selected.T
        rhs = selected @ vector
        coefficients = torch.linalg.solve(
            gram + 1e-4 * torch.eye(count), rhs
        )
        reconstruction = coefficients @ selected
        results[count] = float(F.cosine_similarity(
            vector[None], reconstruction[None]
        ))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--match-block-size", type=int, default=512)
    parser.add_argument("--activation-block-size", type=int, default=256)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--min-active", type=int, default=10)
    parser.add_argument("--cosine-threshold", type=float, default=0.8)
    parser.add_argument("--correlation-threshold", type=float, default=0.5)
    parser.add_argument("--composition-limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2030)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    codes = data["codes"].float()
    split_generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(codes), generator=split_generator)
    valid_indices = permutation[int(0.9 * len(codes)):]
    valid_codes = codes[valid_indices]
    paths = checkpoint_paths(args.root)
    missing = [path for path in paths.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")

    representations = {}
    for objective in OBJECTIVES:
        for seed in SEEDS:
            key = (objective, seed)
            print(f"loading activations for {objective} seed {seed}", flush=True)
            representations[key] = load_representation(
                paths[key], valid_codes, device, args.batch_size
            )

    matches = {}

    def get_match(key_a, key_b):
        pair = (key_a, key_b)
        if pair in matches:
            return matches[pair]
        print(f"matching {key_a} -> {key_b}", flush=True)
        cosine, indices = decoder_candidates(
            representations[key_a], representations[key_b],
            args.candidate_count, device, args.match_block_size,
        )
        correlation, jaccard = activation_candidate_metrics(
            representations[key_a], representations[key_b], indices,
            device, args.activation_block_size,
        )
        result = {
            "cosine": cosine,
            "indices": indices,
            "correlation": correlation,
            "jaccard": jaccard,
        }
        matches[pair] = result
        return result

    pair_rows = []
    for objective in OBJECTIVES:
        for left_index, seed_a in enumerate(SEEDS):
            for seed_b in SEEDS[left_index + 1:]:
                key_a = (objective, seed_a)
                key_b = (objective, seed_b)
                forward = get_match(key_a, key_b)
                reverse = get_match(key_b, key_a)
                valid_a = representations[key_a]["active_count"] >= args.min_active
                valid_b = representations[key_b]["active_count"] >= args.min_active
                joint_a = joint_match(
                    forward, args.cosine_threshold, args.correlation_threshold
                ) & valid_a
                joint_b = joint_match(
                    reverse, args.cosine_threshold, args.correlation_threshold
                ) & valid_b
                nearest_a = forward["indices"][:, 0].long()
                nearest_b = reverse["indices"][:, 0].long()
                mutual = nearest_b[nearest_a] == torch.arange(len(nearest_a))
                pair_rows.append({
                    "comparison": "within_objective",
                    "objective_a": objective,
                    "seed_a": seed_a,
                    "objective_b": objective,
                    "seed_b": seed_b,
                    "symmetric_mean_max_cosine": float((
                        forward["cosine"][:, 0].mean()
                        + reverse["cosine"][:, 0].mean()
                    ) / 2),
                    "shared_fraction_a": float(joint_a.sum() / valid_a.sum().clamp_min(1)),
                    "shared_fraction_b": float(joint_b.sum() / valid_b.sum().clamp_min(1)),
                    "mutual_nearest_count": int(mutual.sum()),
                    "active_features_a": int(valid_a.sum()),
                    "active_features_b": int(valid_b.sum()),
                })

    for objective in OBJECTIVES[1:]:
        modified_key = (objective, 2030)
        for baseline_seed in SEEDS:
            baseline_key = ("latent_mse", baseline_seed)
            forward = get_match(modified_key, baseline_key)
            reverse = get_match(baseline_key, modified_key)
            valid_a = representations[modified_key]["active_count"] >= args.min_active
            valid_b = representations[baseline_key]["active_count"] >= args.min_active
            joint_a = joint_match(
                forward, args.cosine_threshold, args.correlation_threshold
            ) & valid_a
            joint_b = joint_match(
                reverse, args.cosine_threshold, args.correlation_threshold
            ) & valid_b
            nearest_a = forward["indices"][:, 0].long()
            nearest_b = reverse["indices"][:, 0].long()
            mutual = nearest_b[nearest_a] == torch.arange(len(nearest_a))
            pair_rows.append({
                "comparison": "modified_to_baseline",
                "objective_a": objective,
                "seed_a": 2030,
                "objective_b": "latent_mse",
                "seed_b": baseline_seed,
                "symmetric_mean_max_cosine": float((
                    forward["cosine"][:, 0].mean()
                    + reverse["cosine"][:, 0].mean()
                ) / 2),
                "shared_fraction_a": float(joint_a.sum() / valid_a.sum().clamp_min(1)),
                "shared_fraction_b": float(joint_b.sum() / valid_b.sum().clamp_min(1)),
                "mutual_nearest_count": int(mutual.sum()),
                "active_features_a": int(valid_a.sum()),
                "active_features_b": int(valid_b.sum()),
            })
    write_csv(os.path.join(args.out_dir, "pair_summary.csv"), pair_rows)

    objective_rows = []
    coverage_rows = []
    feature_rows_by_objective = {}
    for objective in OBJECTIVES:
        reference_key = (objective, 2030)
        reference = representations[reference_key]
        valid = reference["active_count"] >= args.min_active
        within_1 = get_match(reference_key, (objective, 2031))
        within_2 = get_match(reference_key, (objective, 2032))
        stable = (
            valid
            & joint_match(within_1, args.cosine_threshold, args.correlation_threshold)
            & joint_match(within_2, args.cosine_threshold, args.correlation_threshold)
        )
        within_best_1 = best_candidate(within_1, args.correlation_threshold)
        within_best_2 = best_candidate(within_2, args.correlation_threshold)
        baseline_matches = []
        if objective == "latent_mse":
            baseline_matched = stable.clone()
            baseline_bests = []
        else:
            for baseline_seed in SEEDS:
                baseline_matches.append(get_match(
                    reference_key, ("latent_mse", baseline_seed)
                ))
            baseline_bests = [
                best_candidate(match, args.correlation_threshold)
                for match in baseline_matches
            ]
            baseline_matched = torch.stack([
                joint_match(match, args.cosine_threshold, args.correlation_threshold)
                for match in baseline_matches
            ]).any(0)
        new_reproducible = stable & ~baseline_matched
        mass = reference["activation_mass"]
        stable_mass = mass[stable].sum()
        new_mass = mass[new_reproducible].sum()
        rows = []
        for feature in range(len(valid)):
            row = {
                "objective": objective,
                "feature": feature,
                "active_count": int(reference["active_count"][feature]),
                "activation_mass": float(mass[feature]),
                "valid_active": bool(valid[feature]),
                "stable_across_seeds": bool(stable[feature]),
                "baseline_single_match": bool(baseline_matched[feature]),
                "new_reproducible_single_unmatched": bool(new_reproducible[feature]),
                "seed2031_feature": int(within_best_1["feature"][feature]),
                "seed2031_cosine": float(within_best_1["cosine"][feature]),
                "seed2031_activation_corr": float(within_best_1["correlation"][feature]),
                "seed2032_feature": int(within_best_2["feature"][feature]),
                "seed2032_cosine": float(within_best_2["cosine"][feature]),
                "seed2032_activation_corr": float(within_best_2["correlation"][feature]),
                "baseline_seed": "",
                "baseline_feature": "",
                "baseline_cosine": "",
                "baseline_activation_corr": "",
                "baseline_activation_jaccard": "",
                "baseline_span_cosine_1": "",
                "baseline_span_cosine_2": "",
                "baseline_span_cosine_4": "",
                "baseline_span_cosine_8": "",
                "classification": (
                    "new_reproducible_unchecked_composition"
                    if new_reproducible[feature]
                    else "shared" if stable[feature] else "unstable_or_inactive"
                ),
            }
            if baseline_matches:
                choices = []
                for baseline_seed, best in zip(SEEDS, baseline_bests):
                    choices.append((
                        float(best["cosine"][feature]), baseline_seed,
                        int(best["feature"][feature]),
                        float(best["correlation"][feature]),
                        float(best["jaccard"][feature]),
                    ))
                choice = max(choices)
                row.update({
                    "baseline_seed": choice[1],
                    "baseline_feature": choice[2],
                    "baseline_cosine": choice[0],
                    "baseline_activation_corr": choice[3],
                    "baseline_activation_jaccard": choice[4],
                })
            rows.append(row)

        if objective != "latent_mse":
            candidates = torch.where(new_reproducible)[0]
            order = torch.argsort(mass[candidates], descending=True)
            candidates = candidates[order[:args.composition_limit]]
            for feature_tensor in candidates:
                feature = int(feature_tensor)
                row = rows[feature]
                baseline_seed = int(row["baseline_seed"])
                match = get_match(reference_key, ("latent_mse", baseline_seed))
                atom_indices = match["indices"][feature, :8].long()
                atoms = representations[("latent_mse", baseline_seed)]["decoder"][
                    atom_indices
                ].float()
                vector = reference["decoder"][feature].float()
                spans = span_cosines(vector, atoms)
                for count, value in spans.items():
                    row[f"baseline_span_cosine_{count}"] = value
                row["classification"] = (
                    "split_or_merged" if spans[8] >= 0.9 else "new_reproducible"
                )

        feature_rows_by_objective[objective] = rows
        write_csv(os.path.join(args.out_dir, f"{objective}_features.csv"), rows)
        evaluated_new = [row for row in rows if row["classification"] in (
            "new_reproducible", "split_or_merged"
        )]
        objective_rows.append({
            "objective": objective,
            "valid_active_features": int(valid.sum()),
            "stable_features": int(stable.sum()),
            "stable_fraction": float(stable.sum() / valid.sum().clamp_min(1)),
            "baseline_matched_stable_features": int((stable & baseline_matched).sum()),
            "new_reproducible_single_unmatched": int(new_reproducible.sum()),
            "new_fraction_among_stable": float(
                new_reproducible.sum() / stable.sum().clamp_min(1)
            ),
            "new_activation_mass_fraction_among_stable": float(
                new_mass / stable_mass.clamp_min(1e-8)
            ),
            "composition_evaluated": len(evaluated_new),
            "split_or_merged_among_evaluated": sum(
                row["classification"] == "split_or_merged" for row in evaluated_new
            ),
            "new_noncompositional_among_evaluated": sum(
                row["classification"] == "new_reproducible" for row in evaluated_new
            ),
        })

        for correlation_threshold in (-1.0, 0.0, 0.1, 0.25, 0.5):
            for cosine_threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                stable_at = (
                    valid
                    & joint_match(within_1, cosine_threshold, correlation_threshold)
                    & joint_match(within_2, cosine_threshold, correlation_threshold)
                )
                if objective == "latent_mse":
                    baseline_at = stable_at.clone()
                else:
                    baseline_at = torch.stack([
                        joint_match(match, cosine_threshold, correlation_threshold)
                        for match in baseline_matches
                    ]).any(0)
                new_at = stable_at & ~baseline_at
                stable_mass_at = mass[stable_at].sum()
                new_mass_at = mass[new_at].sum()
                coverage_rows.append({
                    "objective": objective,
                    "cosine_threshold": cosine_threshold,
                    "activation_correlation_threshold": correlation_threshold,
                    "stable_features": int(stable_at.sum()),
                    "stable_fraction": float(
                        stable_at.sum() / valid.sum().clamp_min(1)
                    ),
                    "baseline_matched_stable": int((stable_at & baseline_at).sum()),
                    "new_reproducible_single_unmatched": int(new_at.sum()),
                    "new_fraction_among_stable": float(
                        new_at.sum() / stable_at.sum().clamp_min(1)
                    ),
                    "new_activation_mass_fraction_among_stable": float(
                        new_mass_at / stable_mass_at.clamp_min(1e-8)
                    ),
                })

    write_csv(os.path.join(args.out_dir, "objective_summary.csv"), objective_rows)
    write_csv(os.path.join(args.out_dir, "coverage_curves.csv"), coverage_rows)
    summary = {
        "method": {
            "reference_seed": 2030,
            "validation_examples": len(valid_codes),
            "candidate_count": args.candidate_count,
            "min_active": args.min_active,
            "cosine_threshold": args.cosine_threshold,
            "activation_correlation_threshold": args.correlation_threshold,
            "composition_limit_per_objective": args.composition_limit,
            "new_definition": (
                "active reference feature matching both same-objective seeds but no "
                "single feature in any of three latent-MSE baseline seeds"
            ),
        },
        "objectives": objective_rows,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
