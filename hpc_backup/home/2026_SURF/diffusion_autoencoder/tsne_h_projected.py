import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE

base = "/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
d = torch.load(f"{base}/tsne/residuals_vector_16x_k32_lambda10.pt", map_location="cpu", weights_only=False)
recs = d["records"]

def get(source, objective, seed):
    for r in recs:
        if r["source"] == source and r["objective"] == objective and r["seed"] == seed:
            return r

source = "ffhq_heldout"
seed0 = 2030
objectives = ["latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high"]
cells = {obj: get(source, obj, seed0) for obj in objectives}
idx_maps = {obj: {int(li): i for i, li in enumerate(cells[obj]["latent_index"].tolist())} for obj in objectives}
common_idx = set(idx_maps[objectives[0]].keys())
for obj in objectives[1:]:
    common_idx &= set(idx_maps[obj].keys())
common_idx = sorted(common_idx)

cache = torch.load(f"{base}/latents/vector_ffhq_32768.pt", map_location="cpu", weights_only=False)
raw_latents = cache["latents"].float()
ckpt_path = f"{base}/experiments/vector_16x_k32/checkpoints/mse_control/latent_mse/seed_2030/last.pt"
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
mean = state["channel_mean"].float().view(1, -1)
std = state["channel_std"].float().view(1, -1)
h_std = (raw_latents[common_idx] - mean) / std

def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

h_u = unit(h_std)
vectors, meta = [], []
for obj in objectives:
    r = cells[obj]["residual"][[idx_maps[obj][i] for i in common_idx]].float()
    coeff = (r * h_u).sum(dim=1, keepdim=True)
    r_perp = r - coeff * h_u
    vectors.append(unit(r_perp).numpy())
    meta.extend([obj] * r_perp.shape[0])

matrix = np.concatenate(vectors, axis=0)
print(f"matrix shape: {matrix.shape}")
embedding = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(matrix)

OBJECTIVE_COLORS = {"latent_mse": "#999999", "laplacian": "#d6604d", "fourier_low": "#4393c3",
                     "fourier_mid": "#92c5de", "fourier_high": "#2166ac"}
OBJECTIVE_LABELS = {"latent_mse": "MSE (baseline density)", "laplacian": "Laplacian", "fourier_low": "Fourier low",
                     "fourier_mid": "Fourier mid", "fourier_high": "Fourier high"}
OBJECTIVE_MARKERS = {"latent_mse": "o", "laplacian": "s", "fourier_low": "^", "fourier_mid": "D", "fourier_high": "v"}
DRAW_ORDER = ["fourier_mid", "fourier_low", "fourier_high", "laplacian"]
BG_KEY = "latent_mse"

def plot_background_blob(ax, points, color):
    kde = gaussian_kde(points.T)
    pad_x = max((points[:, 0].max() - points[:, 0].min()) * 0.15, 1e-3)
    pad_y = max((points[:, 1].max() - points[:, 1].min()) * 0.15, 1e-3)
    xmin, xmax = points[:, 0].min() - pad_x, points[:, 0].max() + pad_x
    ymin, ymax = points[:, 1].min() - pad_y, points[:, 1].max() + pad_y
    xx, yy = np.mgrid[xmin:xmax:150j, ymin:ymax:150j]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[zz.max() * 0.08, zz.max()], colors=[color], alpha=0.3, zorder=0)

meta_arr = np.array(meta)
fig, ax = plt.subplots(figsize=(7, 6.5))
handles = []
mask_bg = meta_arr == BG_KEY
plot_background_blob(ax, embedding[mask_bg], OBJECTIVE_COLORS[BG_KEY])
handles.append(Patch(facecolor=OBJECTIVE_COLORS[BG_KEY], alpha=0.35, label=OBJECTIVE_LABELS[BG_KEY]))
for key in DRAW_ORDER:
    mask = meta_arr == key
    sc = ax.scatter(embedding[mask, 0], embedding[mask, 1], s=6, alpha=0.6,
                     color=OBJECTIVE_COLORS[key], marker=OBJECTIVE_MARKERS[key],
                     linewidths=0, label=OBJECTIVE_LABELS[key], zorder=2)
    handles.append(sc)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(handles=handles, markerscale=3, fontsize=9, frameon=False, loc="best")
ax.set_title(f"vector_16x_k32, {source}, seed={seed0}, h-projected residual direction, perp=30 ({len(meta)} pts)", fontsize=10)
out = f"{base}/tsne/figures/ffhq_heldout_diff_objective_hprojected_perp30.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {out}")
