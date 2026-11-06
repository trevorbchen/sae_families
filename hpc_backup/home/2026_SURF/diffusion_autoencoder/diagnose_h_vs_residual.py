import torch
import itertools
import sys
sys.path.insert(0, "/home/tbchen/2026 SURF/diffusion_autoencoder")
from train_spatial_map_sae import FlatTopKSAE

base = "/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d = torch.load(f"{base}/tsne/residuals_vector_16x_k32_lambda10.pt", map_location="cpu", weights_only=False)
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
print(f"common images: {len(common_idx)}")

# load raw ffhq latents + the checkpoint's standardization stats
cache = torch.load(f"{base}/latents/vector_ffhq_32768.pt", map_location="cpu", weights_only=False)
raw_latents = cache["latents"].float()

ckpt_path = f"{base}/experiments/vector_16x_k32/checkpoints/mse_control/latent_mse/seed_2030/last.pt"
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
mean = state["channel_mean"].float().view(1, -1)
std = state["channel_std"].float().view(1, -1)

h_raw = raw_latents[common_idx]
h_std = (h_raw - mean) / std  # standardized activation, same space as "residual"

def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

print("\n=== cos(residual, standardized original activation h) per objective, same image ===")
for obj in objectives:
    r = cells[obj]["residual"][[idx_maps[obj][i] for i in common_idx]].float()
    cs = (unit(r) * unit(h_std)).sum(dim=1)
    print(f"{obj:14s} mean={cs.mean():.4f} std={cs.std():.4f} p10={cs.quantile(0.1):.4f} p90={cs.quantile(0.9):.4f}")

print(f"\nh_std norm: mean={h_std.norm(dim=1).mean():.4f}")
for obj in objectives:
    r = cells[obj]["residual"][[idx_maps[obj][i] for i in common_idx]].float()
    print(f"{obj:14s} residual norm mean={r.norm(dim=1).mean():.4f}")

# Now: project out h's direction from each residual, then redo the same-image vs diff-image comparison
print("\n=== after projecting out h's direction from residual: same-image cross-objective cosine ===")
projected = {}
for obj in objectives:
    r = cells[obj]["residual"][[idx_maps[obj][i] for i in common_idx]].float()
    h_u = unit(h_std)
    coeff = (r * h_u).sum(dim=1, keepdim=True)
    r_perp = r - coeff * h_u
    projected[obj] = r_perp

all_same_image = []
for a, b in itertools.combinations(objectives, 2):
    cs = (unit(projected[a]) * unit(projected[b])).sum(dim=1)
    all_same_image.append(cs)
print(f"POOLED same-image diff-objective (h-projected-out): mean={torch.cat(all_same_image).mean():.4f}")

g = torch.Generator().manual_seed(0)
print("\n=== after projecting out h: different-image, SAME objective (random pairs) ===")
for obj in objectives:
    v = unit(projected[obj])
    n = v.shape[0]
    perm = torch.randperm(n, generator=g)
    mask = perm != torch.arange(n)
    cs = (v * v[perm]).sum(dim=1)[mask]
    print(f"{obj:14s} mean={cs.mean():.4f} std={cs.std():.4f}")
