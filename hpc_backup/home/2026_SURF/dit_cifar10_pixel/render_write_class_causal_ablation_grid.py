#!/usr/bin/env python3
import argparse
import json
import os

import torch
from torchvision import utils

from write_class_causal_ablation import (
    CLASS_NAMES,
    Diffusion,
    baseline_means_for_t,
    forward_with_write_ablation,
    load_all_saes,
    load_cifar,
    load_model,
    load_ranked_features,
    make_balanced_batch,
    make_class_batch,
    model_eps,
    selected_sites,
)


def save_grid(path, x, nrow):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    utils.save_image(x.clamp(-1, 1), path, nrow=nrow, normalize=True, value_range=(-1, 1), padding=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/dit_s2_cifar10/last.pt")
    p.add_argument("--block0-sae-ckpt", default="runs/sae_block0_writes_16x_k32/last.pt")
    p.add_argument("--remaining-sae-ckpt", default="runs/sae_remaining_block_writes_16x_k32/last.pt")
    p.add_argument("--feature-csv", default="runs/write_class_direction_attribution_16x_k32/feature_class_specificity_zscore_filled_filtered.csv")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs/write_class_causal_ablation_grids")
    p.add_argument("--class-id", type=int, default=5)
    p.add_argument("--t", type=int, default=500)
    p.add_argument("--set-size", type=int, default=100)
    p.add_argument("--num-images", type=int, default=100)
    p.add_argument("--nrow", type=int, default=10)
    p.add_argument("--baseline-per-class", type=int, default=4)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--timesteps", type=int, default=1000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset = load_cifar(args.data_dir)
    model = load_model(args.ckpt, device)
    diffusion = Diffusion(args.timesteps, device=device)
    saes = load_all_saes(args, device)
    ranked = load_ranked_features(args.feature_csv, min_active=0.005, min_abs_attr=1e-5, min_hits=2)
    features = ranked[args.class_id]
    chosen = selected_sites(features, min(args.set_size, len(features)))

    x_base, y_base, _ = make_balanced_batch(dataset, args.baseline_per_class, args.seed + 77, device)
    t_base = torch.full((x_base.shape[0],), args.t, device=device, dtype=torch.long)
    gen_base = torch.Generator(device=device).manual_seed(args.seed + 50000 + args.t)
    noise_base = torch.randn(x_base.shape, device=device, generator=gen_base)
    baseline_means = baseline_means_for_t(model, diffusion, saes, x_base, y_base, t_base, args.precision, noise_base)

    x0, labels, indices = make_class_batch(dataset, args.class_id, args.num_images, args.seed + 123, device)
    t = torch.full((x0.shape[0],), args.t, device=device, dtype=torch.long)
    gen = torch.Generator(device=device).manual_seed(args.seed + args.class_id * 1000 + args.t)
    noise = torch.randn(x0.shape, device=device, generator=gen)
    x_t = diffusion.q_sample(x0, t, noise)

    eps_on = model_eps(model, x_t, t, labels, args.precision)
    eps_off = forward_with_write_ablation(model, saes, x_t, t, labels, chosen, baseline_means, args.precision)
    sqrt_ac = diffusion.extract(diffusion.sqrt_ac, t, x_t.shape)
    sqrt_om = diffusion.extract(diffusion.sqrt_one_minus_ac, t, x_t.shape)
    pred_x0_on = (x_t - sqrt_om * eps_on) / sqrt_ac
    pred_x0_off = (x_t - sqrt_om * eps_off) / sqrt_ac

    name = CLASS_NAMES[args.class_id]
    stem = f"class{args.class_id}_{name}_t{args.t:03d}_top{args.set_size}"
    save_grid(os.path.join(args.out_dir, f"{stem}_x0.png"), x0, args.nrow)
    save_grid(os.path.join(args.out_dir, f"{stem}_x_t.png"), x_t, args.nrow)
    save_grid(os.path.join(args.out_dir, f"{stem}_pred_x0_features_on.png"), pred_x0_on, args.nrow)
    save_grid(os.path.join(args.out_dir, f"{stem}_pred_x0_features_off.png"), pred_x0_off, args.nrow)
    save_grid(os.path.join(args.out_dir, f"{stem}_absdiff_x8.png"), (pred_x0_on - pred_x0_off).abs() * 8 - 1, args.nrow)

    meta = {
        "class_id": args.class_id,
        "class_name": name,
        "t": args.t,
        "set_size": args.set_size,
        "actual_set_size": min(args.set_size, len(features)),
        "num_images": args.num_images,
        "cifar_train_indices": indices,
        "selected_sites": {f"block{b:02d}_{stream}": fs for (b, stream), fs in chosen.items()},
        "baseline": "same-t mixed-class/global SAE feature mean",
    }
    with open(os.path.join(args.out_dir, f"{stem}_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
