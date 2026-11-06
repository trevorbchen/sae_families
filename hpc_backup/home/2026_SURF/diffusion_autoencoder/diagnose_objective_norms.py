#!/usr/bin/env python3
"""Does the by_objective_absolute t-SNE separation reduce to a trivial
per-objective norm difference? Compute mean/std norm of the h-projected
residual per objective (seed 2030, ffhq_heldout) and the across-objective
norm spread relative to within-objective spread."""
import torch
from pathlib import Path

base = Path("/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1")
data = torch.load(base / "tsne" / "residuals_vector_16x_k32_lambda10.pt", map_location="cpu", weights_only=False)
arm, lambda_value, records = data["arm"], data["lambda_value"], data["records"]

cache = torch.load(base / "latents" / "vector_ffhq_32768.pt", map_location="cpu", weights_only=False)
raw_latents = cache["latents"].float()

ckpt = torch.load(base / "experiments" / arm / "checkpoints" / "mse_control" / "latent_mse" / "seed_2030" / "last.pt",
                   map_location="cpu", weights_only=False)
mean = ckpt["channel_mean"].float().view(1, -1)
std = ckpt["channel_std"].float().view(1, -1)
h_full = (raw_latents - mean) / std


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def get(records, source, objective, seed):
    for r in records:
        if r["source"] == source and r["objective"] == objective and r["seed"] == seed:
            return r
    return None


print(f"{'objective':14s} {'n':>5s} {'mean|r|':>10s} {'std|r|':>10s} {'mean|r_perp|':>13s} {'std|r_perp|':>12s}")
norms_by_obj = {}
for obj in ["latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high"]:
    rec = get(records, "ffhq_heldout", obj, 2030)
    r = rec["residual"].float()
    idx = rec["latent_index"]
    h = h_full[idx]
    h_u = unit(h)
    coeff = (r * h_u).sum(dim=1, keepdim=True)
    r_perp = r - coeff * h_u
    raw_norm = r.norm(dim=1)
    perp_norm = r_perp.norm(dim=1)
    norms_by_obj[obj] = perp_norm
    print(f"{obj:14s} {r.shape[0]:5d} {raw_norm.mean():10.4f} {raw_norm.std():10.4f} "
          f"{perp_norm.mean():13.4f} {perp_norm.std():12.4f}")

means = torch.tensor([norms_by_obj[o].mean() for o in norms_by_obj])
pooled_std = torch.cat(list(norms_by_obj.values())).std()
print(f"\nacross-objective spread of mean|r_perp|: std={means.std():.4f}, range={means.max()-means.min():.4f}")
print(f"pooled within-objective std|r_perp|: {pooled_std:.4f}")
print(f"ratio (across-objective mean spread / within-objective std): {means.std()/pooled_std:.4f}")
