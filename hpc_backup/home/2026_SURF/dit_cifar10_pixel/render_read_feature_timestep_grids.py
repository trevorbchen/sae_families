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

from block_input_onoff_ablation_dog import (
    Diffusion,
    encode_dense as encode_dense_input,
    finish_from_block,
    get_block_input_and_c,
    load_model,
    load_sae as load_input_sae,
    make_batch,
)
from middle_read_onoff_ablation import (
    encode_dense as encode_dense_middle,
    finish_from_middle,
    load_middle_sae,
    run_prefix_to_middle,
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


def read_selected(path, sites, top_n):
    rows = []
    seen = defaultdict(int)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            site = row["site"]
            if site not in sites:
                continue
            key = site
            if seen[key] >= top_n:
                continue
            seen[key] += 1
            class_id = int(row["class_id"])
            rows.append({
                "class_id": class_id,
                "class_name": row.get("class_name", CLASS_NAMES[class_id]),
                "site": site,
                "group": row.get("group", ""),
                "selection_rank": int(row.get("selection_rank", seen[key])),
                "block": row["block"],
                "block_index": int(row["block_index"]),
                "feature": int(row["feature"]),
                "peak_timestep": int(row.get("timestep", 0)),
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
    from PIL import Image, ImageDraw
    grid = utils.make_grid(cells, nrow=5, padding=2, pad_value=1.0)
    img = to_pil_image(grid)
    top = 24
    left = 58
    canvas = Image.new("RGB", (img.width + left, img.height + top), "white")
    canvas.paste(img.convert("RGB"), (left, top))
    draw = ImageDraw.Draw(canvas)
    col_labels = ["x0", "x_t", "pred", "off", "diff"]
    step = image_size + 2
    for ci, label in enumerate(col_labels):
        draw.text((left + 2 + ci * step + image_size // 2 - 12, 6), label, fill=(20, 20, 20))
    for ri, t in enumerate(t_values):
        draw.text((8, top + 2 + ri * step + image_size // 2 - 6), f"t{t:03d}", fill=(20, 20, 20))
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
def compute_input_group(args, model, diffusion, device, x0, labels, feature_rows):
    block_index = feature_rows[0]["block_index"]
    features = [r["feature"] for r in feature_rows]
    sae = load_input_sae(args.input_sae_ckpt, block_index, device)
    decoder = sae.decoder.float()
    data = {feature: {"timesteps": [], "coeff_scores": []} for feature in features}
    for t_int in args.t_values:
        print(f"class={feature_rows[0]['class_name']} site=input block={block_index} t={t_int} features={len(features)}", flush=True)
        t = torch.full((x0.shape[0],), t_int, device=device, dtype=torch.long)
        gen = torch.Generator(device=device).manual_seed(args.seed + 1000 + t_int)
        noise = torch.randn(x0.shape, device=device, generator=gen)
        x_in, c, x_t, _ = get_block_input_and_c(model, diffusion, x0, labels, t, block_index, args.precision, noise)
        base_out = finish_from_block(model, x_in, c, block_index, args.precision)
        base_eps = torch.chunk(base_out, 2, dim=1)[0]
        sqrt_ac = diffusion.extract(diffusion.sqrt_ac, t, x_t.shape)
        sqrt_om = diffusion.extract(diffusion.sqrt_one_minus_ac, t, x_t.shape)
        base_pred_x0 = (x_t - sqrt_om * base_eps) / sqrt_ac
        dense = encode_dense_input(sae, x_in.flatten(0, 1))
        for feature in features:
            coeff = dense[:, :, feature].float()
            states = []
            for ii in range(x0.shape[0]):
                delta = -coeff[ii].unsqueeze(1) * decoder[feature].unsqueeze(0)
                states.append(x_in[ii] + delta)
            states = torch.stack(states, dim=0)
            off_out = finish_from_block(model, states, c, block_index, args.precision)
            off_eps = torch.chunk(off_out, 2, dim=1)[0]
            off_pred_x0 = (x_t - sqrt_om * off_eps) / sqrt_ac
            data[feature]["timesteps"].append({
                "timestep": t_int,
                "x_t": x_t.detach().cpu(),
                "base_pred_x0": base_pred_x0.detach().cpu(),
                "off_pred_x0": off_pred_x0.detach().cpu(),
                "pred_delta_mse": (base_pred_x0 - off_pred_x0).pow(2).flatten(1).mean(dim=1).detach().cpu(),
            })
            data[feature]["coeff_scores"].append(coeff.abs().mean(dim=1).detach().cpu())
    return data


@torch.no_grad()
def compute_middle_group(args, model, diffusion, device, x0, labels, feature_rows):
    block_index = feature_rows[0]["block_index"]
    features = [r["feature"] for r in feature_rows]
    sae = load_middle_sae(args.middle_sae_ckpt, block_index, device)
    decoder = sae.decoder.float()
    data = {feature: {"timesteps": [], "coeff_scores": []} for feature in features}
    for t_int in args.t_values:
        print(f"class={feature_rows[0]['class_name']} site=middle block={block_index} t={t_int} features={len(features)}", flush=True)
        t = torch.full((x0.shape[0],), t_int, device=device, dtype=torch.long)
        gen = torch.Generator(device=device).manual_seed(args.seed + 1000 + t_int)
        noise = torch.randn(x0.shape, device=device, generator=gen)
        x_middle, c, x_t = run_prefix_to_middle(model, diffusion, x0, labels, t, block_index, args.precision, noise)
        base_out = finish_from_middle(model, x_middle, c, block_index, args.precision)
        base_eps = torch.chunk(base_out, 2, dim=1)[0]
        sqrt_ac = diffusion.extract(diffusion.sqrt_ac, t, x_t.shape)
        sqrt_om = diffusion.extract(diffusion.sqrt_one_minus_ac, t, x_t.shape)
        base_pred_x0 = (x_t - sqrt_om * base_eps) / sqrt_ac
        dense = encode_dense_middle(sae, x_middle.flatten(0, 1))
        for feature in features:
            coeff = dense[:, :, feature].float()
            states = []
            for ii in range(x0.shape[0]):
                delta = -coeff[ii].unsqueeze(1) * decoder[feature].unsqueeze(0)
                states.append(x_middle[ii] + delta)
            states = torch.stack(states, dim=0)
            off_out = finish_from_middle(model, states, c, block_index, args.precision)
            off_eps = torch.chunk(off_out, 2, dim=1)[0]
            off_pred_x0 = (x_t - sqrt_om * off_eps) / sqrt_ac
            data[feature]["timesteps"].append({
                "timestep": t_int,
                "x_t": x_t.detach().cpu(),
                "base_pred_x0": base_pred_x0.detach().cpu(),
                "off_pred_x0": off_pred_x0.detach().cpu(),
                "pred_delta_mse": (base_pred_x0 - off_pred_x0).pow(2).flatten(1).mean(dim=1).detach().cpu(),
            })
            data[feature]["coeff_scores"].append(coeff.abs().mean(dim=1).detach().cpu())
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection-csv", required=True)
    p.add_argument("--sites", default="input,middle")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--t-values", type=parse_ints, default=T_VALUES)
    p.add_argument("--out-root", default="runs/read_feature_timestep_grids_review_50_each")
    p.add_argument("--ckpt", default="runs/dit_s2_cifar10/last.pt")
    p.add_argument("--input-sae-ckpt", default="runs/sae_block_inputs_16x_k32_long/last.pt")
    p.add_argument("--middle-sae-ckpt", default="runs/sae_middle_reads_16x_k32_long/last.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--num-images", type=int, default=12)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--examples-per-feature", type=int, default=1)
    p.add_argument("--layout", choices=["single_image", "image_blocks"], default="single_image")
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
        "class_id", "class_name", "site", "group", "selection_rank", "block", "block_index", "feature",
        "peak_timestep", "representative_local_image_id", "representative_cifar_train_index", "image_path",
    ]
    with (out_root / "selected_features.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in metadata_rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    with (out_root / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Read Feature All-Timestep Interpretation Grids\n\n")
        f.write("Each PNG is one SAE read feature over all 11 timesteps. Rows are timesteps; columns are `x0`, `x_t`, original `pred_x0`, feature-off `pred_x0`, and a per-timestep p99-scaled absolute difference image.\n")


if __name__ == "__main__":
    main()
