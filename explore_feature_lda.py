#!/usr/bin/env python3
"""Linear-discriminant view of the same per-(image, active feature) ablation
deltas used by explore_feature_tsne.py, in raw 512-dim ConvAE latent space.

The t-SNE-style local-neighborhood embedding of this data comes out looking
like a diffuse gaussian blob with no stable per-objective pockets (checked
directly: local neighborhood purity around the single best density anchor
never clears what pure chance produces when you search thousands of
candidates, except for Laplacian). But a plain linear classifier on the same
unit-normalized deltas gets ~64-68% held-out accuracy against a 20% chance
floor -- so there IS real separating structure, it's just a *global linear*
direction, not a local density cluster. t-SNE is a local-neighborhood method
and is structurally blind to exactly this kind of structure; LDA is built to
find it directly (it looks for the projection that maximizes between-class
separation relative to within-class scatter).

Same output set as explore_feature_tsne.py, just on the LDA projection
instead of t-SNE, and using density contours throughout instead of raw
scatter (matches the contour style later adopted for the t-SNE plots too --
at 100k+ points scatter is unreadable, and LDA has no t-SNE-style repulsion
to spread same-class points apart):
  _lda_by_objective.png   -- 2-panel: 2D contours (LD1 vs LD2) + 1D ridgeline
                              along LD1 (the dominant axis, usually ~60-70%
                              of the separation -- check explained_variance_
                              ratio_ in the console output)
  _lda_by_seed.png         -- same LD1/LD2 embedding (still the
                              objective-trained projection), recolored by
                              seed -- a leakage sanity check: seed shouldn't
                              line up with the objective-discriminating axes
  _lda_by_fidelity.png     -- continuous reconstruction-fidelity coloring
  _lda_highlight_<obj>.png -- one objective's contour over gray "other" dots,
                              x5
  _lda_embedding.npz       -- embedding + objective/seed/latent_index/value/
                              fidelity, for reuse (e.g. visualize_pure_cluster.py)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tsne_viz_utils import kde_contours_multi, kde_contour_single

OBJECTIVES = ("latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high")
SEEDS = (2030, 2031, 2032)
OBJECTIVE_COLORS = {
    "latent_mse": "#984ea3", "laplacian": "#e41a1c", "fourier_low": "#4daf4a",
    "fourier_mid": "#ff7f00", "fourier_high": "#377eb8",
}
OBJECTIVE_LABELS = {
    "latent_mse": "MSE", "laplacian": "Laplacian", "fourier_low": "Fourier low",
    "fourier_mid": "Fourier mid", "fourier_high": "Fourier high",
}
SEED_COLORS = {2030: "#1b9e77", 2031: "#d95f02", 2032: "#7570b3"}
SEED_LABELS = {s: f"seed {s}" for s in SEED_COLORS}

CACHE_BY_SOURCE = {
    "ffhq_heldout": "latents/vector_ffhq_32768.pt",
    "celeba": "linear_probe_vector16x_lambda_sweep/celeba_vector_32768.pt",
    "fairface": "latents/fairface_vector_8192.pt",
}


def load_sae_parts(checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    decoder = state["model"]["decoder"].float()
    dec_bias = state["model"]["dec_bias"].float()
    mean = state["channel_mean"].float().view(-1)
    std = state["channel_std"].float().view(-1)
    return decoder, dec_bias, mean, std


def subsample(n, k, generator):
    if n <= k:
        return torch.arange(n)
    return torch.randperm(n, generator=generator)[:k]


def build_points(records, source, base, arm, lambda_tag, seeds, per_cell_images,
                  features_per_image, generator, latents):
    """One unit-normalized ablation-delta point per (image, active feature).
    Each point also carries that image's own overall reconstruction fidelity
    (using ALL its active features, not just the ones sampled into points)."""
    vectors, meta = [], []
    for objective in OBJECTIVES:
        subdir = "mse_control" if objective == "latent_mse" else f"lambda_{lambda_tag}"
        for seed in seeds:
            ckpt = Path(base) / "experiments" / arm / "checkpoints" / subdir / objective / f"seed_{seed}" / "last.pt"
            decoder, dec_bias, mean, std = load_sae_parts(ckpt)
            rec = next(r for r in records if r["source"] == source and r["objective"] == objective and r["seed"] == seed)

            n_images = rec["feature_indices"].shape[0]
            img_idx = subsample(n_images, per_cell_images, generator)
            values_full = rec["feature_values"][img_idx]
            indices_full = rec["feature_indices"][img_idx]
            latent_idx = rec["latent_index"][img_idx]

            raw = latents[latent_idx].float()
            h_std = (raw - mean) / std
            recon_full = (values_full.unsqueeze(-1) * decoder[indices_full]).sum(1) + dec_bias
            fidelity_full = -(h_std - recon_full).norm(dim=1)

            k = values_full.shape[1]
            if features_per_image and features_per_image < k:
                feat_idx = subsample(k, features_per_image, generator)
                values = values_full[:, feat_idx]
                indices = indices_full[:, feat_idx]
            else:
                values, indices = values_full, indices_full

            atoms = decoder[indices]
            deltas = values.unsqueeze(-1) * atoms * std.view(1, 1, -1)
            n, f, d = deltas.shape
            deltas = deltas.reshape(n * f, d)
            unit = deltas / deltas.norm(dim=1, keepdim=True).clamp_min(1e-8)
            vectors.append(unit)
            for i in range(n):
                for j in range(f):
                    meta.append({
                        "objective": objective, "seed": seed,
                        "latent_index": int(latent_idx[i]), "value": float(values[i, j]),
                        "fidelity": float(fidelity_full[i]),
                    })
    return torch.cat(vectors, dim=0).numpy(), meta


def write_details(out_path, lines):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}", flush=True)


def plot_categorical_contours(emb2d, keys_arr, color_map, label_map, draw_order, title, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    present = {k for k in keys_arr}
    keys = [k for k in draw_order if k in present]
    xy_by_class = {k: emb2d[keys_arr == k] for k in keys}
    kde_contours_multi(ax, xy_by_class, color_map)
    handles = [plt.Line2D([0], [0], color=color_map.get(k, "#999999"), lw=2, label=label_map.get(k, str(k)))
               for k in keys]
    ax.set_xlabel("LD1", fontsize=9)
    ax.set_ylabel("LD2", fontsize=9)
    ax.legend(handles=handles, fontsize=9, frameon=False, loc="best")
    ax.set_title(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def plot_highlight(emb2d, objective_arr, highlight, title, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    mask_other = objective_arr != highlight
    mask_high = ~mask_other
    ax.scatter(emb2d[mask_other, 0], emb2d[mask_other, 1], s=4, alpha=0.15,
               color="#bbbbbb", marker="o", linewidths=0, zorder=1)
    kde_contour_single(ax, emb2d[mask_high], OBJECTIVE_COLORS[highlight])
    handles = [plt.Line2D([0], [0], marker="o", color="#bbbbbb", lw=0, markersize=5, label="other"),
               plt.Line2D([0], [0], color=OBJECTIVE_COLORS[highlight], lw=2, label=OBJECTIVE_LABELS[highlight])]
    ax.set_xlabel("LD1", fontsize=9)
    ax.set_ylabel("LD2", fontsize=9)
    ax.legend(handles=handles, fontsize=9, frameon=False, loc="best")
    ax.set_title(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def plot_fidelity(emb2d, fidelity, title, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    norm = Normalize(vmin=np.percentile(fidelity, 2), vmax=np.percentile(fidelity, 98))
    ax.scatter(emb2d[:, 0], emb2d[:, 1], s=4, alpha=0.5, c=fidelity, cmap="coolwarm",
               norm=norm, linewidths=0, zorder=2)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="coolwarm"), ax=ax, shrink=0.8)
    cbar.set_label("reconstruction fidelity (hot=good, cool=bad)", fontsize=8)
    ax.set_xlabel("LD1", fontsize=9)
    ax.set_ylabel("LD2", fontsize=9)
    ax.set_title(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usage", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--arm", default="vector_16x_k32")
    parser.add_argument("--lambda-tag", default="10p0")
    parser.add_argument("--source", choices=("ffhq_heldout", "celeba", "fairface"), default="ffhq_heldout")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--per-cell-images", type=int, default=800)
    parser.add_argument("--features-per-image", type=int, default=0, help="0 = use all active features")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    data = torch.load(args.usage, map_location="cpu", weights_only=False)
    records = data["records"]
    cache = torch.load(Path(args.base) / CACHE_BY_SOURCE[args.source], map_location="cpu", weights_only=False)
    latent_key = "latents" if "latents" in cache else "clean_latents"
    latents = cache[latent_key].float()

    generator = torch.Generator().manual_seed(0)
    matrix, meta = build_points(records, args.source, args.base, args.arm, args.lambda_tag,
                                 args.seeds, args.per_cell_images, args.features_per_image, generator, latents)
    objective_arr = np.array([m["objective"] for m in meta])
    seed_arr = np.array([m["seed"] for m in meta])
    fidelity_arr = np.array([m["fidelity"] for m in meta])
    print(f"{args.source}: {matrix.shape[0]} unit-normalized ablation-delta points, {matrix.shape[1]}-dim, "
          f"seeds={args.seeds}", flush=True)

    Xtr, Xte, ytr, yte = train_test_split(matrix, objective_arr, test_size=args.test_size,
                                           random_state=0, stratify=objective_arr)

    lda = LinearDiscriminantAnalysis(n_components=min(4, len(OBJECTIVES) - 1))
    lda.fit(Xtr, ytr)
    acc = lda.score(Xte, yte)
    pred = lda.predict(Xte)
    cm = confusion_matrix(yte, pred, labels=OBJECTIVES, normalize="true")
    print(f"held-out LDA accuracy: {acc:.1%} (chance = {100/len(OBJECTIVES):.1f}%)", flush=True)
    print("explained_variance_ratio_:", np.round(lda.explained_variance_ratio_, 3), flush=True)
    print("confusion matrix (rows=true, cols=pred):", flush=True)
    print("            " + " ".join(f"{OBJECTIVE_LABELS[o][:8]:>9s}" for o in OBJECTIVES))
    for i, o in enumerate(OBJECTIVES):
        print(f"{OBJECTIVE_LABELS[o]:12s}" + " ".join(f"{cm[i, j]:9.0%}" for j in range(len(OBJECTIVES))))

    emb_all = lda.transform(matrix)  # project everything (train+test) for the figures
    emb2d = emb_all[:, :2]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix

    # by_objective
    plot_categorical_contours(emb2d, objective_arr, OBJECTIVE_COLORS, OBJECTIVE_LABELS, OBJECTIVES,
                               f"{args.source}, LDA by objective",
                               out_dir / f"{prefix}_lda_by_objective.png")
    ld1_by_class = {o: emb_all[objective_arr == o, 0] for o in OBJECTIVES}
    order = sorted(OBJECTIVES, key=lambda o: ld1_by_class[o].mean())
    cm_lines = ["            " + " ".join(f"{OBJECTIVE_LABELS[o][:8]:>9s}" for o in OBJECTIVES)]
    for i, o in enumerate(OBJECTIVES):
        cm_lines.append(f"{OBJECTIVE_LABELS[o]:12s}" + " ".join(f"{cm[i, j]:9.0%}" for j in range(len(OBJECTIVES))))
    write_details(out_dir / f"{prefix}_lda_by_objective.txt", [
        f"{args.source}, LDA by objective",
        f"points: {matrix.shape[0]}  (per-(image, active feature) ablation deltas, unit-normalized, 512-dim)",
        f"seeds pooled: {args.seeds}",
        f"held-out linear classifier accuracy: {acc:.1%}  (chance = {100/len(OBJECTIVES):.1f}%)",
        f"LDA explained_variance_ratio_ (LD1..LD{len(lda.explained_variance_ratio_)}): "
        + ", ".join(f"{v:.1%}" for v in lda.explained_variance_ratio_),
        f"objective order along LD1 (low to high): {', '.join(OBJECTIVE_LABELS[o] for o in order)}",
        "",
        "confusion matrix (rows=true, cols=predicted), held-out set:",
        *cm_lines,
    ])

    # by_seed: same (objective-trained) embedding, recolored by seed -- leakage sanity check
    plot_categorical_contours(emb2d, seed_arr, SEED_COLORS, SEED_LABELS, list(SEED_COLORS),
                               f"{args.source}, LDA by seed",
                               out_dir / f"{prefix}_lda_by_seed.png")
    write_details(out_dir / f"{prefix}_lda_by_seed.txt", [
        f"{args.source}, LDA by seed",
        f"points: {matrix.shape[0]}",
        "same LD1/LD2 axes as the by_objective plot (LDA was trained on objective labels only), "
        "just recolored by seed -- a leakage check: if the 3 seeds' contours don't line up with "
        "each other here, something about seed identity is leaking into the objective-discriminating "
        "axes, which would undermine the by_objective result. They should (and do) overlap closely.",
    ])

    # by_fidelity
    plot_fidelity(emb2d, fidelity_arr,
                  f"{args.source}, LDA by fidelity",
                  out_dir / f"{prefix}_lda_by_fidelity.png")
    write_details(out_dir / f"{prefix}_lda_by_fidelity.txt", [
        f"{args.source}, LDA by fidelity",
        f"points: {matrix.shape[0]}",
        "fidelity = -||residual|| of that point's own image's full reconstruction (all active "
        "features, not just the ones sampled into this point) -- hot/red = good reconstruction, "
        "cool/blue = poor reconstruction.",
        f"fidelity range: [{fidelity_arr.min():.3f}, {fidelity_arr.max():.3f}], "
        f"mean={fidelity_arr.mean():.3f}",
    ])

    # highlight x5
    for obj in OBJECTIVES:
        plot_highlight(emb2d, objective_arr, obj,
                        f"{args.source}, LDA, {OBJECTIVE_LABELS[obj]} highlighted",
                        out_dir / f"{prefix}_lda_highlight_{obj}.png")
        n_high = int((objective_arr == obj).sum())
        write_details(out_dir / f"{prefix}_lda_highlight_{obj}.txt", [
            f"{args.source}, LDA, {OBJECTIVE_LABELS[obj]} highlighted",
            f"points: {matrix.shape[0]} total, {n_high} belong to {OBJECTIVE_LABELS[obj]} "
            f"({n_high / matrix.shape[0]:.1%})",
            f"held-out recall for {OBJECTIVE_LABELS[obj]}: {cm[OBJECTIVES.index(obj), OBJECTIVES.index(obj)]:.0%} "
            f"(chance = {100/len(OBJECTIVES):.1f}%)",
        ])

    out_npz = out_dir / f"{prefix}_lda_embedding.npz"
    np.savez(out_npz, embedding=emb_all, objective=objective_arr, seed=seed_arr,
             latent_index=np.array([m["latent_index"] for m in meta]),
             value=np.array([m["value"] for m in meta]),
             fidelity=fidelity_arr, held_out_accuracy=acc)
    print(f"wrote {out_npz}", flush=True)


if __name__ == "__main__":
    main()
