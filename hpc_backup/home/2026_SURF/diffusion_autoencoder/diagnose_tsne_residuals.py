import torch
import numpy as np

d = torch.load("/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/residuals_vector_16x_k32_lambda10.pt",
                map_location="cpu", weights_only=False)
recs = d["records"]

def get(source, objective, seed):
    for r in recs:
        if r["source"] == source and r["objective"] == objective and r["seed"] == seed:
            return r
    return None

source = "ffhq_heldout"
objectives = ["latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high"]
seed0 = 2030

# 1. norms per objective
print("=== norms per objective (seed=2030) ===")
for obj in objectives:
    r = get(source, obj, seed0)
    norms = r["residual"].norm(dim=1)
    print(f"{obj:14s} mean={norms.mean():.4f} std={norms.std():.4f} median={norms.median():.4f}")

# build index->residual map per objective for seed0, matched by latent_index
cells = {obj: get(source, obj, seed0) for obj in objectives}
idx_maps = {}
for obj in objectives:
    idx_maps[obj] = {int(li): i for i, li in enumerate(cells[obj]["latent_index"].tolist())}

common_idx = set(idx_maps["latent_mse"].keys())
for obj in objectives[1:]:
    common_idx &= set(idx_maps[obj].keys())
common_idx = sorted(common_idx)
print(f"\ncommon images across all 5 objectives (seed=2030): {len(common_idx)}")

def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

# 2. same-image, different-objective cosine similarity distribution (pairwise across all 5 objectives, averaged per pair-type)
import itertools
pairs = list(itertools.combinations(objectives, 2))
print("\n=== same-image cosine sim, across objective pairs (seed=2030) ===")
all_same_image_diff_obj = []
for a, b in pairs:
    va = cells[a]["residual"][[idx_maps[a][i] for i in common_idx]].float()
    vb = cells[b]["residual"][[idx_maps[b][i] for i in common_idx]].float()
    cs = (unit(va) * unit(vb)).sum(dim=1)
    all_same_image_diff_obj.append(cs)
    print(f"{a:12s} vs {b:14s} mean={cs.mean():.4f} std={cs.std():.4f} p10={cs.quantile(0.1):.4f} p50={cs.quantile(0.5):.4f} p90={cs.quantile(0.9):.4f}")
all_same_image_diff_obj = torch.cat(all_same_image_diff_obj)
print(f"POOLED same-image-diff-objective: mean={all_same_image_diff_obj.mean():.4f} std={all_same_image_diff_obj.std():.4f}")

# 3. different-image, same-objective cosine similarity (random pairs, baseline for "how similar do unrelated images look")
print("\n=== different-image cosine sim, SAME objective (random pairs, seed=2030) ===")
g = torch.Generator().manual_seed(0)
for obj in objectives:
    v = unit(cells[obj]["residual"].float())
    n = v.shape[0]
    perm = torch.randperm(n, generator=g)
    cs = (v * v[perm]).sum(dim=1)
    # remove accidental self-pairs
    mask = perm != torch.arange(n)
    cs = cs[mask]
    print(f"{obj:14s} mean={cs.mean():.4f} std={cs.std():.4f} p10={cs.quantile(0.1):.4f} p90={cs.quantile(0.9):.4f}")

# 4. fully random baseline: shuffle both image and objective
print("\n=== fully random pairs (diff image AND diff objective, seed=2030) ===")
all_vecs = torch.cat([unit(cells[obj]["residual"].float()) for obj in objectives], dim=0)
n = all_vecs.shape[0]
perm = torch.randperm(n, generator=g)
cs = (all_vecs * all_vecs[perm]).sum(dim=1)
print(f"mean={cs.mean():.4f} std={cs.std():.4f} p10={cs.quantile(0.1):.4f} p90={cs.quantile(0.9):.4f}")

# theoretical floor for random unit vectors in 512 dims: mean~0, std ~ 1/sqrt(512)
print(f"\ntheoretical random-direction std in 512 dims: {1/np.sqrt(512):.4f}")

# 5. same-image, same-objective, different seed (previously reported ~0.61) -- recompute with distribution
print("\n=== same-image cosine sim, same objective DIFFERENT seed (latent_mse) ===")
seeds = [2030, 2031, 2032]
cellsS = {s: get(source, "latent_mse", s) for s in seeds}
idxS = {s: {int(li): i for i, li in enumerate(cellsS[s]["latent_index"].tolist())} for s in seeds}
common_s = set(idxS[2030].keys()) & set(idxS[2031].keys()) & set(idxS[2032].keys())
common_s = sorted(common_s)
print(f"common images across 3 seeds: {len(common_s)}")
for a, b in itertools.combinations(seeds, 2):
    va = cellsS[a]["residual"][[idxS[a][i] for i in common_s]].float()
    vb = cellsS[b]["residual"][[idxS[b][i] for i in common_s]].float()
    cs = (unit(va) * unit(vb)).sum(dim=1)
    print(f"seed {a} vs {b}: mean={cs.mean():.4f} std={cs.std():.4f} p10={cs.quantile(0.1):.4f} p90={cs.quantile(0.9):.4f}")
