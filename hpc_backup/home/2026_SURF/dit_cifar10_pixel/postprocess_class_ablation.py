#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from pathlib import Path

import torch


TICK_LABELS = None
PALETTE = [
    (49, 54, 149),
    (69, 117, 180),
    (116, 173, 209),
    (224, 243, 248),
    (255, 255, 191),
    (254, 224, 144),
    (253, 174, 97),
    (244, 109, 67),
    (215, 48, 39),
    (165, 0, 38),
]


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def color_at(x):
    x = float(max(0.0, min(1.0, x)))
    pos = x * (len(PALETTE) - 1)
    i = int(math.floor(pos))
    j = min(i + 1, len(PALETTE) - 1)
    a = pos - i
    rgb = tuple(round(PALETTE[i][k] * (1 - a) + PALETTE[j][k] * a) for k in range(3))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def write_svg_heatmap(path, matrix, row_labels, col_labels, title, cell_w=42, cell_h=7):
    rows, cols = matrix.shape
    left = 118
    top = 46
    label_w = left - 8
    width = left + cols * cell_w + 20
    height = top + rows * cell_h + 42
    y_font = 7 if rows > 120 else 8
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="12" y="22" font-family="Arial" font-size="14" font-weight="700" fill="#202124">{title}</text>',
    ]
    for ci, label in enumerate(col_labels):
        x = left + ci * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.1f}" y="39" text-anchor="middle" font-family="Arial" font-size="9" fill="#333">{label}</text>')
    for ri, label in enumerate(row_labels):
        y = top + ri * cell_h
        parts.append(f'<text x="{label_w}" y="{y + cell_h - 1:.1f}" text-anchor="end" font-family="Arial" font-size="{y_font}" fill="#333">{label}</text>')
        for ci in range(cols):
            x = left + ci * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color_at(matrix[ri, ci])}"/>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def active_features_by_cutoff(order_t000, max_rel):
    ordered_vals = max_rel[order_t000].float()
    # Keep the visible head above the flat near-zero tail. The floor keeps weak rows out;
    # the knee catches blocks where the head is much larger than the rest.
    floor = 1.25
    logs = torch.log10(ordered_vals.clamp_min(1e-12))
    if len(logs) > 5:
        drops = logs[:-1] - logs[1:]
        knee_idx = int(torch.argmax(drops).item()) + 1
        knee_val = float(ordered_vals[knee_idx].item())
        cutoff = max(floor, min(knee_val, 3.0))
    else:
        cutoff = floor
    keep = [int(f) for f in order_t000.tolist() if float(max_rel[f]) >= cutoff]
    if len(keep) < 25:
        keep = [int(f) for f in order_t000[:25].tolist()]
    return keep[:300]


def rank_lookup(values):
    order = torch.argsort(values, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, len(order) + 1)
    return ranks


def result_candidates(block_dir, block_name, site):
    candidates = [block_dir / f"{block_name}_onoff_ablation.pt"]
    if site in {"attn", "mlp"}:
        candidates.append(block_dir / f"{block_name}_write_onoff_ablation.pt")
    return candidates


def process_block(block_dir, block_name, site, all_rows, count_rows):
    result_path = next((p for p in result_candidates(block_dir, block_name, site) if p.exists()), None)
    if result_path is None:
        return False
    obj = load_pt(result_path)
    pred = obj["metrics"]["pred_x0_delta_mse"].float()
    t_values = [int(x) for x in obj["t_values"]]
    col_labels = [f"t{x:03d}" for x in t_values]
    agg = pred.mean(dim=1)
    agg_rel = agg / agg.mean(dim=1, keepdim=True).clamp_min(1e-12)
    rel = pred / pred.mean(dim=2, keepdim=True).clamp_min(1e-12)

    order_t000 = torch.argsort(agg[0], descending=True)
    ranks_by_t = torch.stack([rank_lookup(agg[ti]) for ti in range(len(t_values))])
    max_rel = agg_rel.max(dim=0).values
    active = active_features_by_cutoff(order_t000, max_rel)

    time_dir = block_dir / "time_evolution"
    rows = []
    for fid in active:
        rows.append({
            "feature": fid,
            "rank_t000": int(ranks_by_t[0, fid]),
            "max_relative_importance": float(max_rel[fid]),
            "best_rank": int(ranks_by_t[:, fid].min()),
        })
    write_csv(time_dir / "active_cutoff_features.csv", rows, ["feature", "rank_t000", "max_relative_importance", "best_rank"])

    rank_rows = []
    fields_rank = ["feature", "rank_t000"] + [f"rank_t{t:03d}" for t in t_values]
    for fid in active:
        rec = {"feature": fid, "rank_t000": int(ranks_by_t[0, fid])}
        for ti, t in enumerate(t_values):
            rec[f"rank_t{t:03d}"] = int(ranks_by_t[ti, fid])
        rank_rows.append(rec)
    write_csv(time_dir / "feature_time_evolution_summary_ordered_by_t000.csv", rank_rows, fields_rank)

    vals = rel[:, :, active].reshape(-1)
    low = float(torch.quantile(vals, 0.02))
    high = float(torch.quantile(vals, 0.98))
    if high <= low:
        high = low + 1e-6
    colors = ((rel[:, :, active] - low) / (high - low)).clamp(0, 1)
    agg_colors = colors.mean(dim=1).transpose(0, 1).numpy()
    row_labels = [f"f{fid}" for fid in active]
    write_svg_heatmap(
        time_dir / "heatmap_active_cutoff_global_relative_importance_multicolor.svg",
        agg_colors,
        row_labels,
        col_labels,
        f"{block_name} aggregate relative importance",
        cell_h=8 if len(active) <= 90 else 6,
    )

    per_image_dir = time_dir / "per_image_aligned_by_aggregate_rank"
    for image_idx in range(colors.shape[1]):
        image_matrix = colors[:, image_idx, :].transpose(0, 1).numpy()
        write_svg_heatmap(
            per_image_dir / f"image{image_idx:02d}_aligned_by_aggregate_rank.svg",
            image_matrix,
            row_labels,
            col_labels,
            f"{block_name} image {image_idx:02d}",
            cell_h=8 if len(active) <= 90 else 6,
        )

    metric_fields = [
        "block", "feature", "agg_rank_t000", "simple_category", "is_active", "is_consistent", "is_cyclic",
        "peak_mean_color", "mean_color", "median_color_amplitude",
        "mean_abs_deviation_from_mean_traj", "max_abs_deviation_from_mean_traj",
        "peak_t_pm1_agreement", "modal_peak_t_pm1", "active_image_frac_peak_color_ge_0p20",
        "heatmap_color_low_rel", "heatmap_color_high_rel",
    ] + [f"mean_color_t{t:03d}" for t in t_values]

    category_counts = {}
    metric_rows = []
    for local_idx, fid in enumerate(active):
        traj = colors[:, :, local_idx].transpose(0, 1)
        mean_traj = traj.mean(dim=0)
        peak_idx = int(torch.argmax(mean_traj))
        image_peak = torch.argmax(traj, dim=1)
        modal_idx = int(torch.mode(image_peak).values.item())
        amp = traj.max(dim=1).values - traj.min(dim=1).values
        mad = (traj - mean_traj).abs()
        peak_mean = float(mean_traj[peak_idx])
        active_frac = float((traj[:, peak_idx] >= 0.20).float().mean())
        mean_abs_dev = float(mad.mean())
        max_abs_dev = float(mad.max())
        is_active = peak_mean >= 0.20 and active_frac >= 0.75
        is_consistent = mean_abs_dev <= 0.15 and max_abs_dev <= 0.35
        is_cyclic = float(torch.median(amp)) >= 0.30
        if not is_active:
            category = "inactive"
        elif not is_consistent:
            category = "inconsistent"
        elif is_cyclic:
            category = "consistent_cyclic"
        else:
            category = "consistent_flat"
        category_counts[category] = category_counts.get(category, 0) + 1
        rec = {
            "block": block_name,
            "feature": int(fid),
            "agg_rank_t000": int(ranks_by_t[0, fid]),
            "simple_category": category,
            "is_active": int(is_active),
            "is_consistent": int(is_consistent),
            "is_cyclic": int(is_cyclic and is_active and is_consistent),
            "peak_mean_color": peak_mean,
            "mean_color": float(mean_traj.mean()),
            "median_color_amplitude": float(torch.median(amp)),
            "mean_abs_deviation_from_mean_traj": mean_abs_dev,
            "max_abs_deviation_from_mean_traj": max_abs_dev,
            "peak_t_pm1_agreement": float(((image_peak - modal_idx).abs() <= 1).float().mean()),
            "modal_peak_t_pm1": f"{t_values[modal_idx]:03d}",
            "active_image_frac_peak_color_ge_0p20": active_frac,
            "heatmap_color_low_rel": low,
            "heatmap_color_high_rel": high,
        }
        for ti, t in enumerate(t_values):
            rec[f"mean_color_t{t:03d}"] = float(mean_traj[ti])
        metric_rows.append(rec)
        all_rows.append(rec)

    metric_rows.sort(key=lambda r: (r["simple_category"] != "consistent_cyclic", r["simple_category"] != "consistent_flat", r["agg_rank_t000"]))
    write_csv(time_dir / "consistent_feature_trajectory_metrics_sorted.csv", metric_rows, metric_fields)
    for category, count in sorted(category_counts.items()):
        count_rows.append({"block": block_name, "simple_category": category, "count": count})
    return True


def write_docs(root, site):
    blocks = [f"block{i:02d}_{site}" for i in range(12)]
    (root / "STATIC_HEATMAPS.md").write_text(
        "# Static Heatmaps\n\n"
        f"Class-specific folder: `{root.name}`. Preferred aggregate view per block.\n\n"
        + "\n".join(
            f"- [{b} active-cutoff multicolor]({b}/time_evolution/heatmap_active_cutoff_global_relative_importance_multicolor.svg)"
            for b in blocks
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Aligned Per-Image Heatmaps",
        "",
        f"Class-specific folder: `{root.name}`. Rows are fixed within each block by aggregate active features ordered by aggregate `t000` rank.",
    ]
    for b in blocks:
        lines += ["", f"## {b}"]
        for i in range(12):
            lines.append(f"- [image {i:02d}]({b}/time_evolution/per_image_aligned_by_aggregate_rank/image{i:02d}_aligned_by_aggregate_rank.svg)")
    (root / "PER_IMAGE_ALIGNED_HEATMAPS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "CONSISTENT_FEATURES.md").write_text(
        "# Consistent Feature Trajectories\n\n"
        "Uses the same hierarchy as totalclass/dog: inactive -> inconsistent -> consistent_cyclic/consistent_flat.\n\n"
        "- [all blocks metrics](consistent_feature_trajectory_metrics_all_blocks.csv)\n"
        "- [category counts](consistent_feature_category_counts.csv)\n",
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=["input", "middle", "attn", "mlp"], default="input")
    ap.add_argument("roots", nargs="+")
    args = ap.parse_args()
    for root_s in args.roots:
        root = Path(root_s)
        all_rows = []
        count_rows = []
        processed = 0
        for i in range(12):
            b = f"block{i:02d}_{args.site}"
            if process_block(root / b, b, args.site, all_rows, count_rows):
                processed += 1
        if processed:
            fields = list(all_rows[0].keys())
            write_csv(root / "consistent_feature_trajectory_metrics_all_blocks.csv", all_rows, fields)
            write_csv(root / "consistent_feature_category_counts.csv", count_rows, ["block", "simple_category", "count"])
            write_docs(root, args.site)
        print(f"{root}: site={args.site} processed {processed} blocks")


if __name__ == "__main__":
    main()
