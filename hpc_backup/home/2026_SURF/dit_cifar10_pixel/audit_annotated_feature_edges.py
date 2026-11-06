#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import utils
from torchvision.transforms.functional import to_pil_image

from block_input_onoff_ablation_dog import (
    encode_dense as encode_dense_input,
    finish_from_block,
    get_block_input_and_c,
    load_sae as load_input_sae,
)
from middle_read_onoff_ablation import (
    encode_dense as encode_dense_middle,
    finish_from_middle,
    load_middle_sae,
    run_prefix_to_middle,
)
from write_onoff_ablation import (
    Diffusion,
    encode_dense as encode_dense_write,
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


CATEGORY_KEYWORDS = {
    "outline": ("outline", "edge", "boundary", "silhouette", "shape"),
    "color": ("color", "green", "pink", "purple", "yellow", "orange", "blue", "red", "channel"),
    "noise": ("noise", "noisy", "frequency", "high freq", "high-frequency", "cleanup", "cleaning"),
    "background": ("background", "bg", "cloud", "sky"),
    "subject": ("subject", "plane", "dog", "car", "truck", "ship", "bird", "cat", "deer", "frog", "horse"),
    "brightness": ("light", "dark", "bright", "shadow"),
}


def parse_ints(s):
    return tuple(int(x) for x in s.split(",") if x.strip())


def to_img(x):
    return ((x.detach().float().cpu().clamp(-1, 1) + 1) * 0.5).clamp(0, 1)


def normalize01(x, eps=1e-8):
    x = x.detach().float()
    dims = tuple(range(1, x.ndim))
    lo = x.amin(dim=dims, keepdim=True)
    hi = x.amax(dim=dims, keepdim=True)
    return ((x - lo) / (hi - lo).clamp_min(eps)).clamp(0, 1)


def gray(x01):
    return (0.2989 * x01[:, 0] + 0.5870 * x01[:, 1] + 0.1140 * x01[:, 2]).unsqueeze(1)


def sobel_edges(x01):
    g = gray(x01)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x01.device, dtype=x01.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x01.device, dtype=x01.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    return normalize01((gx.square() + gy.square()).sqrt())


def soft_enrichment(signal, mask, eps=1e-8):
    signal = signal.float()
    mask = mask.float().clamp(0, 1)
    inside = (signal * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(eps)
    outside_mask = 1 - mask
    outside = (signal * outside_mask).sum(dim=(1, 2, 3)) / outside_mask.sum(dim=(1, 2, 3)).clamp_min(eps)
    return inside / outside.clamp_min(eps)


def soft_corr(a, b, eps=1e-8):
    a = a.flatten(1).float()
    b = b.flatten(1).float()
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return (a * b).mean(dim=1) / (a.square().mean(dim=1).sqrt() * b.square().mean(dim=1).sqrt()).clamp_min(eps)


def high_freq_frac(diff01, cutoff=0.35, eps=1e-8):
    g = gray(diff01).squeeze(1)
    fft = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1)).abs().square()
    h, w = g.shape[-2:]
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=g.device), torch.linspace(-1, 1, w, device=g.device), indexing="ij")
    rr = (xx.square() + yy.square()).sqrt()
    high = fft[:, rr >= cutoff].sum(dim=1)
    total = fft.flatten(1).sum(dim=1)
    return high / total.clamp_min(eps)


def dominant_channel(diff):
    vals = diff.abs().flatten(2).mean(dim=2)
    idx = vals.argmax(dim=1)
    names = ["R", "G", "B"]
    return [names[int(i)] for i in idx], vals


def center_subject_mask(n, device):
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 32, device=device), torch.linspace(-1, 1, 32, device=device), indexing="ij")
    mask = ((xx / 0.72).square() + (yy / 0.72).square() <= 1).float().view(1, 1, 32, 32)
    return mask.repeat(n, 1, 1, 1)


class DinoSubject:
    def __init__(self, device, model_name="dinov2_vits14"):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, x01):
        x = F.interpolate(x01, size=(224, 224), mode="bicubic", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        out = self.model.forward_features(x)
        cls = out["x_norm_clstoken"]
        patches = out["x_norm_patchtokens"]
        sim = F.cosine_similarity(patches, cls.unsqueeze(1), dim=-1).view(-1, 1, 16, 16)
        sim = F.interpolate(normalize01(sim), size=(32, 32), mode="bilinear", align_corners=False)
        return sim


def load_subject_model(args, device):
    if args.subject_method == "center":
        return None
    try:
        return DinoSubject(device, args.dino_model)
    except Exception as e:
        if args.subject_method == "dino":
            raise
        print(f"DINO unavailable; falling back to center subject mask: {e}", flush=True)
        return None


def categories_for(note):
    low = note.lower()
    cats = []
    for name, kws in CATEGORY_KEYWORDS.items():
        if any(kw in low for kw in kws):
            cats.append(name)
    return cats or ["uncategorized"]


def parse_notes(path, include_categories, include_uncategorized):
    include_categories = set(include_categories)
    rows = []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pat = re.compile(r"^(write|read):(\d+):(attn|mlp|input|middle):block(\d{2})_(attn|mlp|input|middle):(\d+)$")
    for key, val in data.items():
        note = (val.get("note") or "").strip()
        if val.get("not_interesting") and not note:
            continue
        m = pat.match(key)
        if not m:
            continue
        kind, class_id, site, block_index, _, feature = m.groups()
        cats = categories_for(note)
        if not include_uncategorized and cats == ["uncategorized"]:
            continue
        if include_categories and not (set(cats) & include_categories):
            continue
        class_id = int(class_id)
        rows.append({
            "kind": kind,
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "site": site,
            "block": f"block{int(block_index):02d}_{site}",
            "block_index": int(block_index),
            "feature": int(feature),
            "note": note,
            "categories": cats,
            "not_interesting": bool(val.get("not_interesting", False)),
        })
    rows.sort(key=lambda r: (r["kind"], r["class_id"], r["site"], r["block_index"], r["feature"]))
    return rows


def pred_x0_from_out(diffusion, x_t, out, t):
    eps = torch.chunk(out, 2, dim=1)[0]
    sqrt_ac = diffusion.extract(diffusion.sqrt_ac, t, x_t.shape)
    sqrt_om = diffusion.extract(diffusion.sqrt_one_minus_ac, t, x_t.shape)
    return (x_t - sqrt_om * eps) / sqrt_ac


def overlay(mask, signal):
    mask = mask.float().clamp(0, 1)
    signal = signal.float().clamp(0, 1)
    return torch.cat([signal, mask, torch.zeros_like(mask)], dim=1).clamp(0, 1)


def save_grid(path, cells, labels, image_size):
    from PIL import Image, ImageDraw
    cells = F.interpolate(torch.stack(cells), size=(image_size, image_size), mode="nearest")
    grid = utils.make_grid(cells, nrow=7, padding=2, pad_value=1.0)
    img = to_pil_image(grid)
    top, left = 24, 70
    canvas = Image.new("RGB", (img.width + left, img.height + top), "white")
    canvas.paste(img.convert("RGB"), (left, top))
    draw = ImageDraw.Draw(canvas)
    col_labels = ["x0", "edge", "subject", "act", "diff", "diff_edge", "diff_subject"]
    step = image_size + 2
    for ci, label in enumerate(col_labels):
        draw.text((left + 2 + ci * step, 6), label, fill=(20, 20, 20))
    for ri, label in enumerate(labels):
        draw.text((8, top + 2 + ri * step + image_size // 2 - 6), label, fill=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def metrics_for_batch(base_pred, off_pred, coeff, x0, subject):
    x001 = to_img(x0).to(base_pred.device)
    edge = sobel_edges(x001)
    subject_edge = normalize01(edge * subject)
    diff = (base_pred - off_pred).abs()
    diff01 = normalize01(diff.mean(dim=1, keepdim=True))
    act = normalize01(F.interpolate(coeff.abs().view(-1, 1, 16, 16), size=(32, 32), mode="nearest"))
    diff_rgb01 = normalize01(diff)
    channels, channel_vals = dominant_channel(diff)
    return {
        "edge": edge,
        "subject": subject,
        "subject_edge": subject_edge,
        "diff": diff01,
        "act": act,
        "diff_edge_enrichment": soft_enrichment(diff01, edge),
        "diff_subject_enrichment": soft_enrichment(diff01, subject),
        "diff_subject_edge_enrichment": soft_enrichment(diff01, subject_edge),
        "act_edge_enrichment": soft_enrichment(act, edge),
        "act_subject_enrichment": soft_enrichment(act, subject),
        "act_subject_edge_enrichment": soft_enrichment(act, subject_edge),
        "diff_edge_corr": soft_corr(diff01, edge),
        "act_edge_corr": soft_corr(act, edge),
        "high_freq_frac": high_freq_frac(diff_rgb01),
        "pred_x0_delta_mse": diff.square().flatten(1).mean(dim=1),
        "dominant_channel": channels,
        "channel_vals": channel_vals,
    }


@torch.no_grad()
def run_write_group(args, model, diffusion, subject_fn, device, x0, labels, rows, csv_rows, overlay_rows):
    block_index, stream = rows[0]["block_index"], rows[0]["site"]
    features = [r["feature"] for r in rows]
    by_feature = {r["feature"]: r for r in rows}
    sae = load_write_sae(block_index, stream, args.block0_sae_ckpt, args.remaining_sae_ckpt, device)
    decoder = sae.decoder.float()
    subject = subject_fn(to_img(x0).to(device)) if subject_fn else center_subject_mask(x0.shape[0], device)
    for t_int in args.t_values:
        print(f"write class={rows[0]['class_name']} {stream} block={block_index} t={t_int} n_features={len(features)}", flush=True)
        t = torch.full((x0.shape[0],), t_int, device=device, dtype=torch.long)
        gen = torch.Generator(device=device).manual_seed(args.seed + 1000 + t_int)
        noise = torch.randn(x0.shape, device=device, generator=gen)
        parts = run_prefix_to_write(model, diffusion, x0, labels, t, block_index, stream, args.precision, noise)
        base_pred = pred_x0_from_out(diffusion, parts["x_t"], finish_after_block(model, parts["x_after_block"], parts["c"], block_index, args.precision), t)
        dense = encode_dense_write(sae, parts["write"].flatten(0, 1))
        for feature in features:
            coeff = dense[:, :, feature].float()
            states = []
            for ii in range(x0.shape[0]):
                delta = -coeff[ii].unsqueeze(1) * decoder[feature].unsqueeze(0)
                states.append((parts["x_after_attn"] if stream == "attn" else parts["x_after_block"])[ii] + delta)
            states = torch.stack(states, dim=0)
            if stream == "attn":
                states = states + mlp_write_from_after_attn(model, states, parts["c"], block_index, args.precision)
            off_pred = pred_x0_from_out(diffusion, parts["x_t"], finish_after_block(model, states, parts["c"], block_index, args.precision), t)
            append_metrics(args, by_feature[feature], t_int, x0, base_pred, off_pred, coeff, subject, csv_rows, overlay_rows)


@torch.no_grad()
def run_read_group(args, model, diffusion, subject_fn, device, x0, labels, rows, csv_rows, overlay_rows):
    block_index, site = rows[0]["block_index"], rows[0]["site"]
    features = [r["feature"] for r in rows]
    by_feature = {r["feature"]: r for r in rows}
    if site == "input":
        sae = load_input_sae(args.input_sae_ckpt, block_index, device)
    else:
        sae = load_middle_sae(args.middle_sae_ckpt, block_index, device)
    decoder = sae.decoder.float()
    subject = subject_fn(to_img(x0).to(device)) if subject_fn else center_subject_mask(x0.shape[0], device)
    for t_int in args.t_values:
        print(f"read class={rows[0]['class_name']} {site} block={block_index} t={t_int} n_features={len(features)}", flush=True)
        t = torch.full((x0.shape[0],), t_int, device=device, dtype=torch.long)
        gen = torch.Generator(device=device).manual_seed(args.seed + 1000 + t_int)
        noise = torch.randn(x0.shape, device=device, generator=gen)
        if site == "input":
            x_read, c, x_t, _ = get_block_input_and_c(model, diffusion, x0, labels, t, block_index, args.precision, noise)
            base_pred = pred_x0_from_out(diffusion, x_t, finish_from_block(model, x_read, c, block_index, args.precision), t)
            dense = encode_dense_input(sae, x_read.flatten(0, 1))
        else:
            x_read, c, x_t = run_prefix_to_middle(model, diffusion, x0, labels, t, block_index, args.precision, noise)
            base_pred = pred_x0_from_out(diffusion, x_t, finish_from_middle(model, x_read, c, block_index, args.precision), t)
            dense = encode_dense_middle(sae, x_read.flatten(0, 1))
        for feature in features:
            coeff = dense[:, :, feature].float()
            states = []
            for ii in range(x0.shape[0]):
                delta = -coeff[ii].unsqueeze(1) * decoder[feature].unsqueeze(0)
                states.append(x_read[ii] + delta)
            states = torch.stack(states, dim=0)
            out = finish_from_block(model, states, c, block_index, args.precision) if site == "input" else finish_from_middle(model, states, c, block_index, args.precision)
            off_pred = pred_x0_from_out(diffusion, x_t, out, t)
            append_metrics(args, by_feature[feature], t_int, x0, base_pred, off_pred, coeff, subject, csv_rows, overlay_rows)


def append_metrics(args, row, t_int, x0, base_pred, off_pred, coeff, subject, csv_rows, overlay_rows):
    m = metrics_for_batch(base_pred, off_pred, coeff, x0, subject)
    for ii in range(x0.shape[0]):
        csv_rows.append({
            "kind": row["kind"],
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "site": row["site"],
            "block": row["block"],
            "block_index": row["block_index"],
            "feature": row["feature"],
            "timestep": t_int,
            "image_id": ii,
            "categories": "|".join(row["categories"]),
            "note": row["note"],
            "pred_x0_delta_mse": float(m["pred_x0_delta_mse"][ii]),
            "diff_edge_enrichment": float(m["diff_edge_enrichment"][ii]),
            "diff_subject_enrichment": float(m["diff_subject_enrichment"][ii]),
            "diff_subject_edge_enrichment": float(m["diff_subject_edge_enrichment"][ii]),
            "act_edge_enrichment": float(m["act_edge_enrichment"][ii]),
            "act_subject_enrichment": float(m["act_subject_enrichment"][ii]),
            "act_subject_edge_enrichment": float(m["act_subject_edge_enrichment"][ii]),
            "diff_edge_corr": float(m["diff_edge_corr"][ii]),
            "act_edge_corr": float(m["act_edge_corr"][ii]),
            "high_freq_frac": float(m["high_freq_frac"][ii]),
            "dominant_channel": m["dominant_channel"][ii],
            "diff_R": float(m["channel_vals"][ii, 0]),
            "diff_G": float(m["channel_vals"][ii, 1]),
            "diff_B": float(m["channel_vals"][ii, 2]),
        })
    if len(overlay_rows[row_key(row)]) < args.max_overlay_rows:
        best = int(m["pred_x0_delta_mse"].argmax().item())
        overlay_rows[row_key(row)].append({
            "label": f"t{t_int:03d} e{best}",
            "cells": [
                to_img(x0[best:best + 1])[0].cpu(),
                m["edge"][best].repeat(3, 1, 1).cpu(),
                m["subject"][best].repeat(3, 1, 1).cpu(),
                m["act"][best].repeat(3, 1, 1).cpu(),
                m["diff"][best].repeat(3, 1, 1).cpu(),
                overlay(m["edge"][best:best + 1], m["diff"][best:best + 1])[0].cpu(),
                overlay(m["subject"][best:best + 1], m["diff"][best:best + 1])[0].cpu(),
            ],
        })


def row_key(row):
    return (row["kind"], row["class_id"], row["site"], row["block_index"], row["feature"])


def write_outputs(args, selected, csv_rows, overlay_rows):
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "kind", "class_id", "class_name", "site", "block", "block_index", "feature", "timestep", "image_id",
        "categories", "note", "pred_x0_delta_mse", "diff_edge_enrichment", "diff_subject_enrichment",
        "diff_subject_edge_enrichment", "act_edge_enrichment", "act_subject_enrichment",
        "act_subject_edge_enrichment", "diff_edge_corr", "act_edge_corr", "high_freq_frac",
        "dominant_channel", "diff_R", "diff_G", "diff_B",
    ]
    with (out_root / "feature_edge_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)

    summary = {}
    for r in csv_rows:
        key = (r["kind"], r["class_id"], r["site"], r["block_index"], r["feature"])
        acc = summary.setdefault(key, {k: [] for k in fields if k not in ("note", "categories", "dominant_channel")})
        for metric in ("pred_x0_delta_mse", "diff_edge_enrichment", "diff_subject_enrichment",
                       "diff_subject_edge_enrichment", "act_edge_enrichment", "act_subject_enrichment",
                       "act_subject_edge_enrichment", "diff_edge_corr", "act_edge_corr", "high_freq_frac"):
            acc[metric].append(float(r[metric]))
    selected_by_key = {row_key(r): r for r in selected}
    summary_rows = []
    for key, vals in summary.items():
        src = selected_by_key[key]
        rec = {k: src[k] for k in ("kind", "class_id", "class_name", "site", "block", "block_index", "feature", "note")}
        rec["categories"] = "|".join(src["categories"])
        for metric, xs in vals.items():
            if not xs or metric not in ("pred_x0_delta_mse", "diff_edge_enrichment", "diff_subject_enrichment",
                                        "diff_subject_edge_enrichment", "act_edge_enrichment", "act_subject_enrichment",
                                        "act_subject_edge_enrichment", "diff_edge_corr", "act_edge_corr", "high_freq_frac"):
                continue
            rec[f"mean_{metric}"] = sum(xs) / len(xs)
            rec[f"max_{metric}"] = max(xs)
        summary_rows.append(rec)
    summary_rows.sort(key=lambda r: r.get("mean_diff_subject_edge_enrichment", 0), reverse=True)
    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    if summary_rows:
        with (out_root / "feature_edge_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            w.writeheader()
            w.writerows(summary_rows)

    overlay_dir = out_root / "overlays"
    for key, rows in overlay_rows.items():
        if not rows:
            continue
        src = selected_by_key[key]
        cells, labels = [], []
        for r in rows:
            labels.append(r["label"])
            cells.extend(r["cells"])
        path = overlay_dir / f"class_{src['class_id']}_{src['class_name']}" / src["kind"] / src["site"] / f"{src['block']}_f{src['feature']:04d}.png"
        save_grid(path, cells, labels, args.image_size)

    with (out_root / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Annotated Feature Edge/Subject Audit\n\n")
        f.write("This pass re-runs annotated SAE features and compares feature activation maps and feature-off causal differences against image edges and a subject/objectness mask.\n\n")
        f.write("- `feature_edge_metrics.csv`: one row per feature/timestep/image.\n")
        f.write("- `feature_edge_summary.csv`: averaged per feature, sorted by subject-edge causal enrichment.\n")
        f.write("- `overlays/`: grids with `x0 | edge | subject | act | diff | diff_edge | diff_subject`.\n\n")
        f.write("DINO subject masks use cls-token to patch-token cosine similarity. If DINO is unavailable and `--subject-method auto` is used, the script falls back to a centered object mask.\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--notes-json", default="sae_feature_review_notes (1).json")
    p.add_argument("--out-root", default="runs/annotated_feature_edge_audit")
    p.add_argument("--include-categories", default="outline")
    p.add_argument("--include-uncategorized", action="store_true")
    p.add_argument("--max-features", type=int, default=80)
    p.add_argument("--t-values", type=parse_ints, default=T_VALUES)
    p.add_argument("--num-images", type=int, default=5)
    p.add_argument("--max-overlay-rows", type=int, default=8)
    p.add_argument("--subject-method", choices=["auto", "dino", "center"], default="auto")
    p.add_argument("--dino-model", default="dinov2_vits14")
    p.add_argument("--ckpt", default="runs/dit_s2_cifar10/last.pt")
    p.add_argument("--block0-sae-ckpt", default="runs/sae_block0_writes_16x_k32/last.pt")
    p.add_argument("--remaining-sae-ckpt", default="runs/sae_remaining_block_writes_16x_k32/last.pt")
    p.add_argument("--input-sae-ckpt", default="runs/sae_block_inputs_16x_k32_long/last.pt")
    p.add_argument("--middle-sae-ckpt", default="runs/sae_middle_reads_16x_k32_long/last.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--image-size", type=int, default=96)
    args = p.parse_args()
    args.t_values = tuple(args.t_values)

    include_categories = [x.strip() for x in args.include_categories.split(",") if x.strip()]
    selected = parse_notes(args.notes_json, include_categories, args.include_uncategorized)
    if args.max_features:
        selected = selected[:args.max_features]
    if not selected:
        raise SystemExit("No annotated features matched the requested categories.")
    print(f"selected annotated features: {len(selected)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = load_model(args.ckpt, device)
    diffusion = Diffusion(args.timesteps, device=device)
    subject_model = load_subject_model(args, device)

    grouped = defaultdict(list)
    for row in selected:
        grouped[(row["kind"], row["class_id"], row["site"], row["block_index"])].append(row)

    batch_cache = {}
    csv_rows = []
    overlay_rows = defaultdict(list)
    for (kind, class_id, site, block_index), rows in sorted(grouped.items()):
        if class_id not in batch_cache:
            batch_cache[class_id] = make_batch(args.data_dir, args.num_images, args.seed, device, class_id)
        x0, labels, _ = batch_cache[class_id]
        if kind == "write":
            run_write_group(args, model, diffusion, subject_model, device, x0, labels, rows, csv_rows, overlay_rows)
        else:
            run_read_group(args, model, diffusion, subject_model, device, x0, labels, rows, csv_rows, overlay_rows)
        write_outputs(args, selected, csv_rows, overlay_rows)
    write_outputs(args, selected, csv_rows, overlay_rows)
    print(f"wrote {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
