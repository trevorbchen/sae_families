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
objectives = ["laplacian", "fourier_low", "fourier_mid", "fourier_high"]
cells = {obj: get(source, obj, seed0) for obj in objectives + ["latent_mse"]}
idx_maps = {obj: {int(li): i for i, li in enumerate(cells[obj]["latent_index"].tolist())} for obj in objectives + ["latent_mse"]}
common_idx = set(idx_maps["latent_mse"].keys())
for obj in objectives:
    common_idx &= set(idx_maps[obj].keys())
common_idx = sorted(common_idx)
print(f"common images: {len(common_idx)}")

def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

mse_r = cells["latent_mse"]["residual"][[idx_maps["latent_mse"][i] for i in common_idx]].float()
diffs = {}
for obj in objectives:
    r = cells[obj]["residual"][[idx_maps[obj][i] for i in common_idx]].float()
    diffs[obj] = r - mse_r

print("\n=== diff-vector norms per objective ===")
for obj in objectives:
    n = diffs[obj].norm(dim=1)
    print(f"{obj:14s} mean={n.mean():.4f} std={n.std():.4f}")

print("\n=== same-image cosine sim of DIFF vectors, across objective pairs ===")
all_same_image = []
for a, b in itertools.combinations(objectives, 2):
    cs = (unit(diffs[a]) * unit(diffs[b])).sum(dim=1)
    all_same_image.append(cs)
    print(f"{a:12s} vs {b:12s} mean={cs.mean():.4f} std={cs.std():.4f}")
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

print("\n=== fully random pairs of DIFF vectors (diff image AND diff objective) ===")
all_vecs = torch.cat([unit(diffs[obj]) for obj in objectives], dim=0)
n = all_vecs.shape[0]
perm = torch.randperm(n, generator=g)
cs = (all_vecs * all_vecs[perm]).sum(dim=1)
print(f"mean={cs.mean():.4f} std={cs.std():.4f}")
