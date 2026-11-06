#!/usr/bin/env python3
"""Extract per-image top-k active SAE feature indices/values for the same
hard-image set used by extract_residual_vectors_for_tsne.py (same sampling
seed and per-cell count, so image i here is the same image i there).

Where the residual extraction asks "in which direction did the SAE fail,"
this asks "which dictionary atoms did the SAE actually use" -- the literal
vocabulary, not just the reconstruction geometry. Each objective trains its
own SAE with its own randomly-initialized dictionary, so raw feature index j
means a different thing per cell -- comparing those directly ("absolute")
just measures "which independently-initialized dictionary produced this."
To get a cross-cell-comparable ("difference") representation, every cell's
dictionary atoms are also remapped into one fixed reference cell's atom
space by nearest decoder-direction cosine similarity (same method as
analyze_sae_feature_similarity.py): since every cell's decoder reconstructs
directions in the same underlying 512-dim ConvAE latent space, this
alignment is a meaningful correspondence, not just picking the best of
random noise. Both the raw ("feature_indices") and aligned
("aligned_feature_indices") indices are saved per record, along with each
cell's mean top-1 alignment cosine (a sanity check on how trustworthy the
"difference" representation is -- low cosine means the mapping is weak).
"""

import argparse
from pathlib import Path

import torch

from train_spatial_map_sae import FlatTopKSAE

OBJECTIVES = ("latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high")
SEEDS = (2030, 2031, 2032)
REF_OBJECTIVE = "latent_mse"
REF_SEED = 2030


def nearest(a, b, topk, block):
    best_values, best_indices = [], []
    for start in range(0, len(a), block):
        sim = a[start:start + block] @ b.T
        values, indices = torch.topk(sim, min(topk, len(b)), dim=1)
        best_values.append(values.cpu())
        best_indices.append(indices.cpu())
    return torch.cat(best_values), torch.cat(best_indices)


def normalized_decoder(model):
    decoder = model.decoder.detach().float()
    return decoder / decoder.norm(dim=1, keepdim=True).clamp_min(1e-8)

CELEBA_CACHE = {
    "vector": "linear_probe_vector16x_lambda_sweep/celeba_vector_32768.pt",
    "spatial": "linear_probe_spatial16x_lambda_sweep/celeba_spatial_32768.pt",
}
FAIRFACE_CACHE = {
    "vector": "latents/fairface_vector_8192.pt",
    "spatial": "latents/fairface_spatial_8192.pt",
}


def load_sae(checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_shape = tuple(state["input_shape"])
    config = state["config"]
    model = FlatTopKSAE(input_shape, config["feature_count"], config["k"]).to(device)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    spatial_axes = max(0, len(input_shape) - 1)
    stats_shape = (1, -1) + (1,) * spatial_axes
    mean = state["channel_mean"].float().to(device).view(stats_shape)
    std = state["channel_std"].float().to(device).view(stats_shape)
    return model, mean, std, config["feature_count"], config["k"]


@torch.no_grad()
def per_example_codes(model, mean, std, latents, indices, batch_size, device):
    all_indices, all_values = [], []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        raw = latents[batch_idx].to(device).float()
        standardized = (raw - mean) / std
        _, values, idx = model(standardized)
        all_values.append(values.cpu())
        all_indices.append(idx.cpu())
    return torch.cat(all_values, dim=0), torch.cat(all_indices, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1")
    parser.add_argument("--arm", default="vector_16x_k32")
    parser.add_argument("--lambda-tag", default="10p0")
    parser.add_argument("--lambda-value", type=float, default=10.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--objectives", nargs="+", default=list(OBJECTIVES))
    parser.add_argument("--sources", nargs="+", default=["ffhq_heldout", "celeba", "fairface"])
    parser.add_argument("--per-cell", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sample-seed", type=int, default=0,
                         help="must match extract_residual_vectors_for_tsne.py's --sample-seed (default 0) "
                              "to land on the identical image set")
    parser.add_argument("--out", required=True, help="output .pt path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(args.base)
    ae_type = "vector" if "vector" in args.arm else "spatial"

    sources = {}
    if "ffhq_heldout" in args.sources:
        cache = torch.load(base / "latents" / f"{ae_type}_ffhq_32768.pt", map_location="cpu", weights_only=False)
        sources["ffhq_heldout"] = (cache["latents"].float(), cache["valid_indices"].long())
    if "celeba" in args.sources:
        cache = torch.load(base / CELEBA_CACHE[ae_type], map_location="cpu", weights_only=False)
        latents = cache["clean_latents"].float()
        sources["celeba"] = (latents, torch.arange(len(latents)))
    if "fairface" in args.sources:
        cache = torch.load(base / FAIRFACE_CACHE[ae_type], map_location="cpu", weights_only=False)
        latents = cache["latents"].float()
        sources["fairface"] = (latents, torch.arange(len(latents)))

    generator = torch.Generator().manual_seed(args.sample_seed)
    sampled_indices = {}
    for source_name, (_, valid_indices) in sources.items():
        n = min(args.per_cell, len(valid_indices))
        perm = torch.randperm(len(valid_indices), generator=generator)[:n]
        sampled_indices[source_name] = valid_indices[perm]
        print(f"{source_name}: sampling {n}/{len(valid_indices)} images", flush=True)

    ref_subdir = "mse_control" if REF_OBJECTIVE == "latent_mse" else f"lambda_{args.lambda_tag}"
    ref_checkpoint = base / "experiments" / args.arm / "checkpoints" / ref_subdir / REF_OBJECTIVE / f"seed_{REF_SEED}" / "last.pt"
    ref_model, _, _, _, _ = load_sae(ref_checkpoint, device)
    ref_decoder = normalized_decoder(ref_model).to(device)
    del ref_model
    print(f"reference cell for alignment: {REF_OBJECTIVE}/seed_{REF_SEED}", flush=True)

    records = []
    feature_count = None
    k = None
    align_quality = {}
    for objective in args.objectives:
        for seed in args.seeds:
            subdir = "mse_control" if objective == "latent_mse" else f"lambda_{args.lambda_tag}"
            checkpoint_path = base / "experiments" / args.arm / "checkpoints" / subdir / objective / f"seed_{seed}" / "last.pt"
            if not checkpoint_path.is_file():
                print(f"skipping {objective}/seed_{seed}: no checkpoint at {checkpoint_path}", flush=True)
                continue
            model, mean, std, feature_count, k = load_sae(checkpoint_path, device)

            if objective == REF_OBJECTIVE and seed == REF_SEED:
                index_map = torch.arange(feature_count)
                mean_cosine = 1.0
            else:
                decoder = normalized_decoder(model).to(device)
                vals, inds = nearest(decoder, ref_decoder, topk=1, block=1024)
                index_map = inds[:, 0]
                mean_cosine = float(vals[:, 0].mean())
            align_quality[f"{objective}/seed_{seed}"] = mean_cosine
            print(f"  align {objective}/seed_{seed} -> ref: mean_top1_cosine={mean_cosine:.3f}", flush=True)

            for source_name, (latents, _) in sources.items():
                idx = sampled_indices[source_name]
                values, feat_idx = per_example_codes(model, mean, std, latents, idx, args.batch_size, device)
                aligned_feat_idx = index_map[feat_idx]
                records.append({
                    "objective": objective, "seed": seed, "source": source_name,
                    "latent_index": idx.clone(), "feature_values": values,
                    "feature_indices": feat_idx, "aligned_feature_indices": aligned_feat_idx,
                })
            print(f"{objective}/seed_{seed}: done", flush=True)
            del model

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arm": args.arm, "lambda_value": args.lambda_value,
        "feature_count": feature_count, "k": k, "records": records,
        "ref_cell": f"{REF_OBJECTIVE}/seed_{REF_SEED}", "align_quality": align_quality,
    }, out_path)
    total_points = sum(r["feature_indices"].shape[0] for r in records)
    print(f"wrote {len(records)} cells, {total_points} total codes (feature_count={feature_count}, k={k}), to {out_path}", flush=True)


if __name__ == "__main__":
    main()
