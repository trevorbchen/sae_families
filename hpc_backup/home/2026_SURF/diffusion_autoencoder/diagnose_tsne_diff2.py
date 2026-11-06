import torch
import itertools

d = torch.load("/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/residuals_vector_16x_k32_lambda10.pt",
                map_location="cpu", weights_only=False)
recs = d["records"]

def get(source, objective, seed):
    for r in recs:
        if r["source"] == source and r["objective"] == objective and r["seed"] == seed:
            return r
    return None

source = "ffhq_heldout"
seed0 = 2030
objectives = ["latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high"]
cells = {obj: get(source, obj, seed0) for obj in objectives}
idx_maps = {obj: {int(li): i for i, li in enumerate(cells[obj]["latent_index"].tolist())} for obj in objectives}
common_idx = set(idx_maps[objectives[0]].keys())
for obj in objectives[1:]:
    common_idx &= set(idx_maps[obj].keys())
common_idx = sorted(common_idx)

def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

raw = {obj: cells[obj]["residual"][[idx_maps[obj][i] for i in common_idx]].float() for obj in objectives}
stacked = torch.stack([raw[obj] for obj in objectives], dim=0)  # [5, N, 512]
per_image_mean = stacked.mean(dim=0)  # [N, 512]
diffs = {obj: raw[obj] - per_image_mean for obj in objectives}

print("=== diff-from-per-image-mean norms ===")
for obj in objectives:
    n = diffs[obj].norm(dim=1)
    print(f"{obj:14s} mean={n.mean():.4f} std={n.std():.4f}")

print("\n=== same-image cosine sim of DIFF vectors, across objective pairs ===")
all_same_image = []
for a, b in itertools.combinations(objectives, 2):
    cs = (unit(diffs[a]) * unit(diffs[b])).sum(dim=1)
    all_same_image.append(cs)
    print(f"{a:12s} vs {b:14s} mean={cs.mean():.4f} std={cs.std():.4f}")
print(f"POOLED same-image diff-objective (of DIFF vectors): mean={torch.cat(all_same_image).mean():.4f}")

print("\n=== different-image cosine sim of DIFF vectors, SAME objective (random pairs) ===")
g = torch.Generator().manual_seed(0)
for obj in objectives:
    v = unit(diffs[obj])
    n = v.shape[0]
    perm = torch.randperm(n, generator=g)
    mask = perm != torch.arange(n)
    cs = (v * v[perm]).sum(dim=1)[mask]
    print(f"{obj:14s} mean={cs.mean():.4f} std={cs.std():.4f}")

print("\n=== fully random pairs of DIFF vectors ===")
all_vecs = torch.cat([unit(diffs[obj]) for obj in objectives], dim=0)
n = all_vecs.shape[0]
perm = torch.randperm(n, generator=g)
cs = (all_vecs * all_vecs[perm]).sum(dim=1)
print(f"mean={cs.mean():.4f} std={cs.std():.4f}")

# bonus: how much of total variance does the per-image mean (shared component) account for vs residual-of-residual?
total_var = stacked.var(dim=0, unbiased=False).sum()
shared_var = per_image_mean.var(dim=0, unbiased=False).sum()
print(f"\nNote: per-image-mean vector norm vs raw residual norm (seed=2030, latent_mse): mean_norm={per_image_mean.norm(dim=1).mean():.4f} vs raw={raw['latent_mse'].norm(dim=1).mean():.4f}")
