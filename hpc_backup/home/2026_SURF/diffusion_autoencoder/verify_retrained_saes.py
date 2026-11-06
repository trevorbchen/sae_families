#!/usr/bin/env python3
"""Sanity-check the freshly retrained vector_16x_k32 SAEs against the paper's
reported numbers: held-out EV per objective (compare to Figure 4a: MSE~55.3%,
Laplacian~49.0%, Fourier low~54.0%, Fourier mid~52.1%, Fourier high~50.6%),
and whether the same-objective > different-objective decoder-similarity
pattern from Figure 1 still holds. Not expecting an exact match (fresh
training run, different autoencoder weights) -- just checking this is in the
same ballpark and not some degenerate/broken result.
"""

import itertools
from pathlib import Path

import torch

from train_spatial_map_sae import FlatTopKSAE

BASE = Path("/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1")
ARM = "vector_16x_k32"
OBJECTIVES = ("latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high")
SEEDS = (2030, 2031, 2032)
LAMBDA_TAG = "10p0"


def checkpoint_path(objective, seed):
    subdir = "mse_control" if objective == "latent_mse" else f"lambda_{LAMBDA_TAG}"
    return BASE / "experiments" / ARM / "checkpoints" / subdir / objective / f"seed_{seed}" / "last.pt"


def load_sae(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    input_shape = tuple(state["input_shape"])
    config = state["config"]
    model = FlatTopKSAE(input_shape, config["feature_count"], config["k"]).to(device)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    spatial_axes = max(0, len(input_shape) - 1)
    stats_shape = (1, -1) + (1,) * spatial_axes
    mean = state["channel_mean"].float().to(device).view(stats_shape)
    std = state["channel_std"].float().to(device).view(stats_shape)
    return model, mean, std


@torch.no_grad()
def held_out_ev(model, mean, std, latents, indices, device, batch_size=1024):
    # Matches evaluate_lambda_sweep_model.py's reconstruction_metrics exactly:
    # pooled E[X^2]-E[X]^2 (i.e. VARIANCE of the residual, not raw MSE around
    # zero) over every element (all examples x all 512 dims flattened
    # together), same denominator convention for the target.
    target_sum = target_square_sum = 0.0
    residual_sum = residual_square_sum = 0.0
    count = 0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        raw = latents[batch_idx].to(device).float()
        target = (raw - mean) / std
        reconstruction, _, _ = model(target)
        residual = target - reconstruction
        target_sum += float(target.double().sum())
        target_square_sum += float(target.double().square().sum())
        residual_sum += float(residual.double().sum())
        residual_square_sum += float(residual.double().square().sum())
        count += target.numel()
    target_variance = target_square_sum / count - (target_sum / count) ** 2
    residual_variance = residual_square_sum / count - (residual_sum / count) ** 2
    residual_mse = residual_square_sum / count
    return {
        "ev_variance_based": 1.0 - residual_variance / target_variance,
        "ev_raw_mse_based": 1.0 - residual_mse / target_variance,
        "residual_mean": residual_sum / count,
    }


def best_match_cosine(a, b):
    a_n = a / a.norm(dim=1, keepdim=True).clamp_min(1e-8)
    b_n = b / b.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cos = a_n @ b_n.T
    return cos.max(dim=1).values.mean().item()


def decoder_cosine_matrix(decoders_a, decoders_b, keys):
    # decoders_a/decoders_b: two DIFFERENT seeds' decoders, keyed by objective.
    # Matches the paper's methodology: every cell (diagonal included) compares
    # across seeds, so the diagonal is same-objective/different-seed, not a
    # trivial self-match.
    n = len(keys)
    sim = torch.zeros(n, n)
    for i, j in itertools.product(range(n), range(n)):
        sim[i, j] = best_match_cosine(decoders_a[keys[i]], decoders_b[keys[j]])
    return sim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(BASE / "latents" / "vector_ffhq_32768.pt", map_location="cpu", weights_only=False)
    latents = cache["latents"].float()
    valid_indices = cache["valid_indices"].long()
    print(f"held-out set: {len(valid_indices)} examples\n")

    print("=== Held-out EV per (objective, seed), lambda=10 ===")
    print(f"{'objective':<14} " + " ".join(f"seed{s}" for s in SEEDS) + "   mean")
    ev_by_objective = {}
    decoders_2030 = {}
    decoders_2031 = {}
    ev_details_by_objective = {}
    for objective in OBJECTIVES:
        evs = []
        details = []
        for seed in SEEDS:
            path = checkpoint_path(objective, seed)
            if not path.is_file():
                print(f"  MISSING: {path}")
                continue
            model, mean, std = load_sae(path, device)
            result = held_out_ev(model, mean, std, latents, valid_indices, device)
            evs.append(result["ev_variance_based"])
            details.append(result)
            if seed == 2030:
                decoders_2030[objective] = model.decoder.detach().cpu()
            elif seed == 2031:
                decoders_2031[objective] = model.decoder.detach().cpu()
            del model
        ev_by_objective[objective] = evs
        ev_details_by_objective[objective] = details
        row = " ".join(f"{e*100:5.1f}%" for e in evs)
        mean_ev = sum(evs) / len(evs) if evs else float("nan")
        print(f"{objective:<14} {row}   {mean_ev*100:5.1f}%")

    print("\n=== Diagnostic: variance-based EV (correct formula) vs raw-MSE-based EV (my earlier bug), and residual mean ===")
    print(f"{'objective':<14} {'ev_variance':>12} {'ev_raw_mse':>12} {'residual_mean':>14}")
    for objective in OBJECTIVES:
        for d in ev_details_by_objective[objective]:
            print(f"{objective:<14} {d['ev_variance_based']*100:11.1f}% {d['ev_raw_mse_based']*100:11.1f}% {d['residual_mean']:14.5f}")

    paper_ev = {
        "latent_mse": 55.3, "laplacian": 49.0, "fourier_low": 54.0,
        "fourier_mid": 52.1, "fourier_high": 50.6,
    }
    print("\n=== Compare to paper's Figure 4a (lambda=10, anchored), using correct variance-based EV ===")
    print(f"{'objective':<14} {'paper':>8} {'retrained':>10} {'diff':>8}")
    for objective in OBJECTIVES:
        mean_ev = sum(ev_by_objective[objective]) / len(ev_by_objective[objective]) * 100
        diff = mean_ev - paper_ev[objective]
        print(f"{objective:<14} {paper_ev[objective]:7.1f}% {mean_ev:9.1f}% {diff:+7.1f}pp")

    print("\n=== Decoder-vector cosine similarity (seed 2030 vs seed 2031, all 5 objectives) ===")
    keys = list(OBJECTIVES)
    sim = decoder_cosine_matrix(decoders_2030, decoders_2031, keys)
    header = "            " + " ".join(f"{k[:10]:>10}" for k in keys)
    print(header)
    for i, k in enumerate(keys):
        row = " ".join(f"{sim[i, j].item():10.3f}" for j in range(len(keys)))
        print(f"{k[:10]:<12}{row}")

    diag = sim.diagonal()
    off_diag_mask = ~torch.eye(len(keys), dtype=torch.bool)
    off_diag_mean_per_row = (sim * off_diag_mask).sum(dim=1) / off_diag_mask.sum(dim=1)
    print("\nDiagonal (same-objective) vs row-mean-off-diagonal (different-objective):")
    all_diag_higher = True
    for i, k in enumerate(keys):
        higher = diag[i].item() > off_diag_mean_per_row[i].item()
        all_diag_higher &= higher
        print(f"  {k:<14} diag={diag[i].item():.3f}  off-diag-mean={off_diag_mean_per_row[i].item():.3f}  {'OK' if higher else 'FAIL'}")
    print(f"\nSame-objective > different-objective for every objective: {all_diag_higher}")


if __name__ == "__main__":
    main()
