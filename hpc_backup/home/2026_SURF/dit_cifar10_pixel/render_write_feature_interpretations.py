#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import torch
from torchvision import utils

from train_dit_cifar import Diffusion
from write_onoff_ablation import (
    encode_dense,
    finish_after_block,
    load_model,
    load_write_sae,
    make_batch,
    mlp_write_from_after_attn,
    run_prefix_to_write,
)


CLASS_NAMES = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def read_top_features(metrics_path, top_n):
    rows = []
    with open(metrics_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["simple_category"] != "consistent_cyclic":
                continue
            rows.append({
                "block": row["block"],
                "block_index": int(row["block"][5:7]),
                "stream": "attn" if row["block"].endswith("_attn") else "mlp",
                "feature": int(row["feature"]),
                "agg_rank_t000": int(row["agg_rank_t000"]),
                "peak_mean_color": float(row["peak_mean_color"]),
                "median_color_amplitude": float(row["median_color_amplitude"]),
                "modal_peak_t_pm1": int(row["modal_peak_t_pm1"]),
            })
    rows.sort(key=lambda r: (r["peak_mean_color"], r["median_color_amplitude"]), reverse=True)
    return rows[:top_n]


def read_selected_csv(path, class_id, stream, top_n):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["class_id"]) != class_id or row["stream"] != stream:
                continue
            rows.append({
                "block": row["block"],
                "block_index": int(row["block_index"]),
                "stream": stream,
                "feature": int(row["feature"]),
                "agg_rank_t000": int(row["agg_rank_t000"]),
                "peak_mean_color": float(row["target_peak_mean_color"]),
                "median_color_amplitude": float(row["target_median_color_amplitude"]),
                "modal_peak_t_pm1": int(row["modal_peak_t_pm1"]),
                "specificity_score": float(row.get("specificity_score", 0.0)),
            })
    rows.sort(key=lambda r: r.get("specificity_score", 0.0), reverse=True)
    return rows[:top_n]


def to_img(x):
    return ((x.detach().float().cpu().clamp(-1, 1) + 1) * 0.5).clamp(0, 1)


def diff_img(a, b):
    d = (a.detach().float().cpu() - b.detach().float().cpu()).abs()
    scale = torch.quantile(d.flatten(1), 0.99, dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
    return (d / scale).clamp(0, 1)


@torch.no_grad()
def render_one(args, model, diffusion, feature_row, x0_all, labels_all, class_indices, out_dir, device):
    block_index = feature_row["block_index"]
    stream = feature_row["stream"]
    feature = feature_row["feature"]
    t_int = feature_row["modal_peak_t_pm1"]

    sae = load_write_sae(
        block_index,
        stream,
        args.block0_sae_ckpt,
        args.remaining_sae_ckpt,
        device,
    )
    decoder = sae.decoder.float()

    t = torch.full((x0_all.shape[0],), t_int, device=device, dtype=torch.long)
    gen = torch.Generator(device=device).manual_seed(args.seed + 1000 + t_int)
    noise = torch.randn(x0_all.shape, device=device, generator=gen)
    parts = run_prefix_to_write(model, diffusion, x0_all, labels_all, t, block_index, stream, args.precision, noise)

    base_state = parts["x_after_block"]
    c = parts["c"]
    base_out = finish_after_block(model, base_state, c, block_index, args.precision)
    base_eps = torch.chunk(base_out, 2, dim=1)[0]
    sqrt_ac = diffusion.extract(diffusion.sqrt_ac, t, parts["x_t"].shape)
    sqrt_om = diffusion.extract(diffusion.sqrt_one_minus_ac, t, parts["x_t"].shape)
    base_pred_x0 = (parts["x_t"] - sqrt_om * base_eps) / sqrt_ac

    dense = encode_dense(sae, parts["write"].flatten(0, 1))
    coeff = dense[:, :, feature].float()
    image_scores = coeff.abs().mean(dim=1)
    top_images = torch.argsort(image_scores, descending=True)[: args.images_per_feature].tolist()

    states = []
    selected = []
    for ii in top_images:
        delta = -coeff[ii].unsqueeze(1) * decoder[feature].unsqueeze(0)
        if stream == "attn":
            states.append(parts["x_after_attn"][ii] + delta)
        else:
            states.append(parts["x_after_block"][ii] + delta)
        selected.append(ii)

    states = torch.stack(states, dim=0)
    ids = torch.tensor(selected, device=device, dtype=torch.long)
    c_batch = c[ids]
    if stream == "attn":
        states = states + mlp_write_from_after_attn(model, states, c_batch, block_index, args.precision)

    off_out = finish_after_block(model, states, c_batch, block_index, args.precision)
    off_eps = torch.chunk(off_out, 2, dim=1)[0]
    off_pred_x0 = (parts["x_t"][ids] - sqrt_om[ids] * off_eps) / sqrt_ac[ids]

    x0 = to_img(x0_all[ids])
    xt = to_img(parts["x_t"][ids])
    base = to_img(base_pred_x0[ids])
    off = to_img(off_pred_x0)
    diff = diff_img(base_pred_x0[ids], off_pred_x0)

    rows = []
    for j in range(len(selected)):
        rows.extend([x0[j], xt[j], base[j], off[j], diff[j]])
    grid = utils.make_grid(rows, nrow=5, padding=2, pad_value=1.0)

    stem = f"{feature_row['block']}_f{feature:04d}_t{t_int:03d}"
    png_path = out_dir / f"{stem}.png"
    utils.save_image(grid, png_path)

    metadata = {
        **feature_row,
        "timestep": t_int,
        "selected_local_image_ids": selected,
        "selected_cifar_train_indices": [int(class_indices[i]) for i in selected],
        "selected_coeff_abs_mean": [float(image_scores[i].item()) for i in selected],
        "columns": ["x0", "x_t", "pred_x0_original", "pred_x0_feature_off", "abs_difference_scaled_per_image"],
        "image_path": str(png_path),
    }
    with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classes", default="0,1,2")
    p.add_argument("--streams", default="attn,mlp")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--images-per-feature", type=int, default=4)
    p.add_argument("--metrics-root", default="runs")
    p.add_argument("--out-root", default="runs/write_feature_interpretation_grids_top20")
    p.add_argument("--selection-csv", default=None)
    p.add_argument("--ckpt", default="runs/dit_s2_cifar10/last.pt")
    p.add_argument("--block0-sae-ckpt", default="runs/sae_block0_writes_16x_k32/last.pt")
    p.add_argument("--remaining-sae-ckpt", default="runs/sae_remaining_block_writes_16x_k32/last.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--num-images", type=int, default=12)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    args = p.parse_args()

    classes = [int(x) for x in args.classes.split(",") if x.strip()]
    streams = [x.strip() for x in args.streams.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = load_model(args.ckpt, device)
    diffusion = Diffusion(args.timesteps, device=device)
    all_metadata = []

    for class_id in classes:
        class_name = CLASS_NAMES[class_id]
        x0, labels, class_indices = make_batch(args.data_dir, args.num_images, args.seed, device, class_id)
        for stream in streams:
            folder_stream = "attn_writes" if stream == "attn" else "mlp_writes"
            folder = Path(args.metrics_root) / f"block_{folder_stream}_onoff_ablation_16x_k32_class_{class_id}_{class_name}"
            metrics_path = folder / "consistent_feature_trajectory_metrics_all_blocks.csv"
            if args.selection_csv:
                selected = read_selected_csv(args.selection_csv, class_id, stream, args.top_n)
            else:
                selected = read_top_features(metrics_path, args.top_n)
            out_dir = Path(args.out_root) / f"class_{class_id}_{class_name}" / folder_stream
            out_dir.mkdir(parents=True, exist_ok=True)
            for rank, row in enumerate(selected, 1):
                print(f"class={class_name} stream={stream} rank={rank}/{len(selected)} block={row['block']} feature={row['feature']}", flush=True)
                meta = render_one(args, model, diffusion, row, x0, labels, class_indices, out_dir, device)
                meta["selection_rank"] = rank
                meta["class_id"] = class_id
                meta["class_name"] = class_name
                meta["stream"] = stream
                all_metadata.append(meta)

    fields = [
        "class_id", "class_name", "stream", "selection_rank", "block", "block_index", "feature",
        "agg_rank_t000", "peak_mean_color", "median_color_amplitude", "modal_peak_t_pm1",
        "timestep", "image_path",
    ]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "selected_features.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_metadata:
            writer.writerow({k: row.get(k) for k in fields})
    with open(out_root / "README.md", "w", encoding="utf-8") as f:
        f.write("# Write Feature Interpretation Grids\n\n")
        f.write("Each PNG row is one image. Columns are: x0, x_t, original pred_x0, feature-off pred_x0, scaled absolute difference.\n\n")
        f.write("Features are selected from `consistent_cyclic` write features, sorted by peak heatmap color then trajectory amplitude.\n")


if __name__ == "__main__":
    main()
