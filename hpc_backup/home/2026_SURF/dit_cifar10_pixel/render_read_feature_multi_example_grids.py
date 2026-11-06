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

from render_read_feature_timestep_grids import (
    CLASS_NAMES,
    T_VALUES,
    compute_input_group,
    compute_middle_group,
    diff_img,
    load_model,
    make_batch,
    parse_ints,
    read_selected,
    to_img,
    Diffusion,
)


def upsample(images, size):
    return F.interpolate(images, size=(size, size), mode="nearest")


def save_grid(path, cells, row_labels, image_size):
    from PIL import Image, ImageDraw
    grid = utils.make_grid(cells, nrow=3, padding=2, pad_value=1.0)
    img = to_pil_image(grid)
    top = 24
    left = 72
    canvas = Image.new("RGB", (img.width + left, img.height + top), "white")
    canvas.paste(img.convert("RGB"), (left, top))
    draw = ImageDraw.Draw(canvas)
    step = image_size + 2
    for ci, label in enumerate(["x0", "x_t", "diff"]):
        draw.text((left + 2 + ci * step + image_size // 2 - 10, 6), label, fill=(20, 20, 20))
    for ri, label in enumerate(row_labels):
        draw.text((8, top + 2 + ri * step + image_size // 2 - 6), label, fill=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection-csv", required=True)
    p.add_argument("--sites", default="input,middle")
    p.add_argument("--top-n", type=int, default=999)
    p.add_argument("--examples-per-feature", type=int, default=5)
    p.add_argument("--t-values", type=parse_ints, default=T_VALUES)
    p.add_argument("--out-root", default="runs/read_feature_multi_example_grids_review_50_each")
    p.add_argument("--ckpt", default="runs/dit_s2_cifar10/last.pt")
    p.add_argument("--input-sae-ckpt", default="runs/sae_block_inputs_16x_k32_long/last.pt")
    p.add_argument("--middle-sae-ckpt", default="runs/sae_middle_reads_16x_k32_long/last.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--num-images", type=int, default=12)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--image-size", type=int, default=64)
    args = p.parse_args()

    sites = [x.strip() for x in args.sites.split(",") if x.strip()]
    selected = read_selected(args.selection_csv, set(sites), args.top_n)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = load_model(args.ckpt, device)
    diffusion = Diffusion(args.timesteps, device=device)

    grouped = defaultdict(list)
    for row in selected:
        grouped[(row["class_id"], row["site"], row["block_index"])].append(row)

    metadata_rows = []
    batch_cache = {}
    for (class_id, site, block_index), rows in sorted(grouped.items()):
        if class_id not in batch_cache:
            batch_cache[class_id] = make_batch(args.data_dir, args.num_images, args.seed, device, class_id)
        x0, labels, class_indices = batch_cache[class_id]
        data = compute_input_group(args, model, diffusion, device, x0, labels, rows) if site == "input" else compute_middle_group(args, model, diffusion, device, x0, labels, rows)
        x0_cpu = x0.detach().cpu()
        out_dir = out_root / f"class_{class_id}_{CLASS_NAMES[class_id]}" / f"{site}_reads"
        for row in rows:
            feature = row["feature"]
            coeff_by_t = torch.stack(data[feature]["coeff_scores"])
            image_order = torch.argsort(coeff_by_t.mean(dim=0), descending=True)[: args.examples_per_feature].tolist()
            cells = []
            row_labels = []
            pred_delta_by_t_image = {}
            for rec in data[feature]["timesteps"]:
                t = rec["timestep"]
                for local_j, ii in enumerate(image_order):
                    base = rec["base_pred_x0"][ii:ii + 1]
                    off = rec["off_pred_x0"][ii:ii + 1]
                    cells.extend([
                        to_img(x0_cpu[ii:ii + 1])[0],
                        to_img(rec["x_t"][ii:ii + 1])[0],
                        diff_img(base, off)[0],
                    ])
                    row_labels.append(f"t{t:03d} e{local_j}")
                    pred_delta_by_t_image[f"t{t:03d}_e{local_j}"] = float(rec["pred_delta_mse"][ii].item())
            cells = upsample(torch.stack(cells), args.image_size)
            stem = f"{row['block']}_f{feature:04d}_multi_examples"
            png_path = out_dir / f"{stem}.png"
            save_grid(png_path, cells, row_labels, args.image_size)
            meta = {
                **row,
                "example_local_image_ids": image_order,
                "example_cifar_train_indices": [int(class_indices[i]) for i in image_order],
                "columns": ["x0", "x_t", "abs_difference_p99_scaled_per_timestep_example"],
                "pred_x0_delta_mse_by_timestep_example": pred_delta_by_t_image,
                "image_path": str(png_path),
            }
            with (out_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            metadata_rows.append(meta)

    fields = [
        "class_id", "class_name", "site", "group", "selection_rank", "block", "block_index", "feature",
        "peak_timestep", "image_path",
    ]
    with (out_root / "selected_features.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in metadata_rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    with (out_root / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Read Feature Multi-Example Review Grids\n\n")
        f.write("Each PNG is one SAE read feature. Rows are timestep/example pairs; columns are `x0`, `x_t`, and p99-scaled `abs(pred_x0_original - pred_x0_feature_off)`.\n")


if __name__ == "__main__":
    main()
