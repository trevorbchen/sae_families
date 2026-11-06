#!/usr/bin/env python3
"""Direct test: in the exact space t-SNE embeds for by_objective_absolute
(h-projected, NOT further normalized, PCA-100), are a point's k=15 nearest
neighbors more often the same image (different objective) or the same
objective (different image)? This is the thing that actually determines
what the t-SNE layout looks like -- settles whether the visible clustering
is objective-driven or a residual same-image artifact."""
import torch
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

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


vectors, objectives, image_ids = [], [], []
for obj in ["latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high"]:
    rec = get(records, "ffhq_heldout", obj, 2030)
    r = rec["residual"].float()
    idx = rec["latent_index"]
    h = h_full[idx]
    h_u = unit(h)
    coeff = (r * h_u).sum(dim=1, keepdim=True)
    r_perp = r - coeff * h_u  # "absolute": no further normalization, same as by_objective_absolute
    vectors.append(r_perp.numpy())
    objectives.extend([obj] * r_perp.shape[0])
    image_ids.extend(idx.tolist())

matrix = np.concatenate(vectors, axis=0)
objectives = np.array(objectives)
image_ids = np.array(image_ids)
print(f"matrix shape: {matrix.shape}")

matrix_pca = PCA(n_components=100, random_state=0).fit_transform(matrix)

nn = NearestNeighbors(n_neighbors=16, n_jobs=-1).fit(matrix_pca)  # 16 so we can drop self
_, neighbor_idx = nn.kneighbors(matrix_pca)
neighbor_idx = neighbor_idx[:, 1:]  # drop self (first neighbor is always the point itself)

same_obj_frac = []
same_img_frac = []
for i in range(matrix.shape[0]):
    nbrs = neighbor_idx[i]
    same_obj_frac.append((objectives[nbrs] == objectives[i]).mean())
    same_img_frac.append((image_ids[nbrs] == image_ids[i]).mean())

same_obj_frac = np.array(same_obj_frac)
same_img_frac = np.array(same_img_frac)

n_total = matrix.shape[0]
n_per_obj = n_total // 5
base_rate_same_obj = (n_per_obj - 1) / (n_total - 1)
base_rate_same_img = 4 / (n_total - 1)  # 4 other objectives share this image

print(f"\nmean fraction of 15 nearest neighbors that share OBJECTIVE: {same_obj_frac.mean():.4f}  (random baseline {base_rate_same_obj:.4f})")
print(f"mean fraction of 15 nearest neighbors that share IMAGE:     {same_img_frac.mean():.4f}  (random baseline {base_rate_same_img:.4f})")
print(f"\nenrichment over random: same-objective x{same_obj_frac.mean()/base_rate_same_obj:.2f}, same-image x{same_img_frac.mean()/base_rate_same_img:.2f}")

# per-objective breakdown
print("\nper-objective same-objective-neighbor enrichment:")
for obj in ["latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high"]:
    mask = objectives == obj
    print(f"  {obj:14s} mean same-obj frac={same_obj_frac[mask].mean():.4f}  (x{same_obj_frac[mask].mean()/base_rate_same_obj:.2f} over random)")
