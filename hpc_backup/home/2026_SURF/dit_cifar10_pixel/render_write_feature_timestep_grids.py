#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import utils
from torchvision.transforms.functional import to_pil_image

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
T_VALUES = (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999)


def parse_ints(s):
    return tuple(int(x) for x in s.split(",") if x.strip())


def read_selected(path, classes, streams, top_n):
    rows = []
    seen = defaultdict(int)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            class_id = int(row["class_id"])
            stream = row["stream"]
            if class_id not in classes or stream not in streams:
                continue
            key = (class_id, stream)
            if seen[key] >= top_n:
                continue
            seen[key] += 1
            rows.append({
                "class_id": class_id,
                "class_name": row.get("class_name", CLASS_NAMES[class_id]),
                "stream": stream,
                "selection_rank": int(row.get("selection_rank", seen[key])),
                "group": row.get("group", ""),
                "block": row["block"],
                "block_index": int(row["block_index"]),
                "feature": int(row["feature"]),
                "peak_timestep": int(row.get("timestep", row.get("modal_peak_t_pm1", 0))),
            })
    return rows


def to_img(x):
    return ((x.detach().float().cpu().clamp(-1, 1) + 1) * 0.5).clamp(0, 1)


def diff_img(a, b):
    d = (a.detach().float().cpu() - b.detach().float().cpu()).abs()
    scale = torch.quantile(d.flatten(1), 0.99, dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
    return (d / scale).clamp(0, 1)


def upsample(images, size):
    return F.interpolate(images, size=(size, size), mode="nearest")


def save_labeled_grid(path, cells, t_values, image_size):
    grid = utils.make_grid(cells, nrow=5, padding=2, pad_value=1.0)
    img = to_pil_image(grid)
    from PIL import Image
    top = 24
    left = 58
    canvas = Image.new("RGB", (img.width + left, img.height + top), "white")
    canvas.paste(img.convert("RGB"), (left, top))
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        col_labels = ["x0", "x_t", "pred", "off", "diff"]
        step = image_size + 2
        for ci, label in enumerate(col_labels):
            draw.text((left + 2 + ci * step + image_size // 2 - 12, 6), label, fill=(20, 20, 20))
        for ri, t in enumerate(t_values):
            draw.text((8, top + 2 + ri * step + image_size // 2 - 6), f"t{t:03d}", fill=(20, 20, 20))
    except Exception:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_image_block_grid(path, cells, t_values, example_ids, image_size):
    from PIL import Image, ImageDraw
    ncols = len(example_ids) * 5
    grid = utils.make_grid(cells, nrow=ncols, padding=2, pad_value=1.0)
    img = to_pil_image(grid)
    top = 42
    left = 58
    canvas = Image.new("RGB", (img.width + left, img.height + top), "white")
    canvas.paste(img.convert("RGB"), (left, top))
    draw = ImageDraw.Draw(canvas)
    step = image_size + 2
    col_labels = ["x0", "x_t", "pred", "off", "diff"]
    for ei, example_id in enumerate(example_ids):
        block_x = left + ei * 5 * step
        draw.text((block_x + 2 * step, 6), f"example {example_id}", fill=(20, 20, 20))
        for ci, label in enumerate(col_labels):
            draw.text((block_x + 2 + ci * step + image_size // 2 - 12, 24), label, fill=(20, 20, 20))
    for ri, t in enumerate(t_values):
        draw.text((8, top + 2 + ri * step + image_size // 2 - 6), f"t{t:03d}", fill=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def compute_group(args, model, diffusion, device, x0, labels, feature_rows):
    block_index = feature_rows[0]["block_index"]
    stream = feature_rows[0]["stream"]
    features = [r["feature"] for r in feature_rows]
    sae = load_write_sae(block_index, stream, args.block0_sae_ckpt, args.remaining_sae_ckpt, device)
    decoder = sae.decoder.float()
    data = {feature: {"timesteps": [], "coeff_scores": []} for feature in features}

    for t_int in args.t_values:
        print(f"class={feature_rows[0]['class_name']} stream={stream} block={block_index} t={t_int} features={len(features)}", flush=True)
        t = torch.full((x0.shape[0],), t_int, device=device, dtype=torch.long)
        gen = torch.Generator(device=device).manual_seed(args.seed + 1000 + t_int)
        noise = torch.randn(x0.shape, device=device, generator=gen)
        parts = run_prefix_to_write(model, diffusion, x0, labels, t, block_index, stream, args.precision, noise)
        base_out = finish_after_block(model, parts["x_after_block"], parts["c"], block_index, args.precision)
        base_eps = torch.chunk(base_out, 2, dim=1)[0]
        sqrt_ac = diffusion.extract(diffusion.sqrt_ac, t, parts["x_t"].shape)
        sqrt_om = diffusion.extract(diffusion.sqrt_one_minus_ac, t, parts["x_t"].shape)
        base_pred_x0 = (parts["x_t"] - sqrt_om * base_eps) / sqrt_ac
        dense = encode_dense(sae, parts["write"].flatten(0, 1))

        for feature in features:
            coeff = dense[:, :, feature].float()
            states = []
            for ii in range(x0.shape[0]):
                delta = -coeff[ii].unsqueeze(1) * decoder[feature].unsqueeze(0)
                if stream == "attn":
                    states.append(parts["x_after_attn"][ii] + delta)
                else:
                    states.append(parts["x_after_block"][ii] + delta)
            states = torch.stack(states, dim=0)
            if stream == "attn":
                states = states + mlp_write_from_after_attn(model, states, parts["c"], block_index, args.precision)
            off_out = finish_after_block(model, states, parts["c"], block_index, args.precision)
            off_eps = torch.chunk(off_out, 2, dim=1)[0]
            off_pred_x0 = (parts["x_t"] - sqrt_om * off_eps) / sqrt_ac
            data[feature]["timesteps"].append({
                "timestep": t_int,
                "x_t": parts["x_t"].detach().cpu(),
                "base_pred_x0": base_pred_x0.detach().cpu(),
                "off_pred_x0": off_pred_x0.detach().cpu(),
                "pred_delta_mse": (base_pred_x0 - off_pred_x0).pow(2).flatten(1).mean(dim=1).detach().cpu(),
            })
            data[feature]["coeff_scores"].append(coeff.abs().mean(dim=1).detach().cpu())
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection-csv", required=True)
    p.add_argument("--classes", default="0,1,2")
    p.add_argument("--streams", default="attn,mlp")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--t-values", type=parse_ints, default=T_VALUES)
    p.add_argument("--out-root", default="runs/write_feature_timestep_grids_class_specific_top20_first3")
    p.add_argument("--ckpt", default="runs/dit_s2_cifar10/last.pt")
    p.add_argument("--block0-sae-ckpt", default="runs/sae_block0_writes_16x_k32/last.pt")
    p.add_argument("--remaining-sae-ckpt", default="runs/sae_remaining_block_writes_16x_k32/last.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--num-images", type=int, default=12)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--examples-per-feature", type=int, default=1)
    p.add_argument("--layout", choices=["single_image", "image_blocks"], default="single_image")
    args = p.parse_args()

    classes = [int(x) for x in args.classes.split(",") if x.strip()]
    streams = [x.strip() for x in args.streams.split(",") if x.strip()]
    selected = read_selected(args.selection_csv, set(classes), set(streams), args.top_n)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = load_model(args.ckpt, device)
    diffusion = Diffusion(args.timesteps, device=device)

    grouped = defaultdict(list)
    for row in selected:
        grouped[(row["class_id"], row["stream"], row["block_index"])].append(row)

    metadata_rows = []
    for (class_id, stream, block_index), rows in sorted(grouped.items()):
        class_name = CLASS_NAMES[class_id]
        x0, labels, class_indices = make_batch(args.data_dir, args.num_images, args.seed, device, class_id)
        data = compute_group(args, model, diffusion, device, x0, labels, rows)
        x0_cpu = x0.detach().cpu()
        folder_stream = "attn_writes" if stream == "attn" else "mlp_writes"
        out_dir = out_root / f"class_{class_id}_{class_name}" / folder_stream
        for row in rows:
            feature = row["feature"]
            coeff_by_t = torch.stack(data[feature]["coeff_scores"])
            representative_images = torch.argsort(coeff_by_t.mean(dim=0), descending=True)[:args.examples_per_feature].tolist()
            representative_image = int(representative_images[0])
            cells = []
            pred_delta_by_t = []
            if args.layout == "image_blocks":
                for rec in data[feature]["timesteps"]:
                    per_t_delta = []
                    for ii in representative_images:
                        base = rec["base_pred_x0"][ii:ii + 1]
                        off = rec["off_pred_x0"][ii:ii + 1]
                        cells.extend([
                            to_img(x0_cpu[ii:ii + 1])[0],
                            to_img(rec["x_t"][ii:ii + 1])[0],
                            to_img(base)[0],
                            to_img(off)[0],
                            diff_img(base, off)[0],
                        ])
                        per_t_delta.append(float(rec["pred_delta_mse"][ii].item()))
                    pred_delta_by_t.append(per_t_delta)
            else:
                for rec in data[feature]["timesteps"]:
                    ii = representative_image
                    base = rec["base_pred_x0"][ii:ii + 1]
                    off = rec["off_pred_x0"][ii:ii + 1]
                    cells.extend([
                        to_img(x0_cpu[ii:ii + 1])[0],
                        to_img(rec["x_t"][ii:ii + 1])[0],
                        to_img(base)[0],
                        to_img(off)[0],
                        diff_img(base, off)[0],
                    ])
                    pred_delta_by_t.append(float(rec["pred_delta_mse"][ii].item()))
            cells = upsample(torch.stack(cells), args.image_size)
            stem = f"{row['block']}_f{feature:04d}_all_t"
            png_path = out_dir / f"{stem}.png"
            if args.layout == "image_blocks":
                save_image_block_grid(png_path, cells, args.t_values, representative_images, args.image_size)
            else:
                save_labeled_grid(png_path, cells, args.t_values, args.image_size)
            meta = {
                **row,
                "representative_local_image_id": representative_image,
                "representative_local_image_ids": representative_images,
                "representative_cifar_train_index": int(class_indices[representative_image]),
                "representative_cifar_train_indices": [int(class_indices[ii]) for ii in representative_images],
                "columns": ["x0", "x_t", "pred_x0_original", "pred_x0_feature_off", "abs_difference_p99_scaled_per_timestep"],
                "rows": [f"t{t:03d}" for t in args.t_values],
                "pred_x0_delta_mse_for_representative_image_by_t": pred_delta_by_t,
                "image_path": str(png_path),
            }
            with (out_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            metadata_rows.append(meta)

    fields = [
        "class_id", "class_name", "stream", "group", "selection_rank", "block", "block_index", "feature",
        "peak_timestep", "representative_local_image_id", "representative_cifar_train_index", "image_path",
    ]
    with (out_root / "selected_features.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in metadata_rows:
            writer.writerow({k: row.get(k) for k in fields})
    with (out_root / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Write Feature All-Timestep Interpretation Grids\n\n")
        f.write("Each PNG is one SAE write feature over all 11 timesteps. Rows are timesteps; columns are `x0`, `x_t`, original `pred_x0`, feature-off `pred_x0`, and a per-timestep p99-scaled absolute difference image.\n\n")
        f.write("The representative image is the class image with the largest mean absolute SAE coefficient for that feature across the 11 timesteps.\n")


if __name__ == "__main__":
    main()
