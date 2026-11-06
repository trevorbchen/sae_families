#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

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


def metrics_folder(metrics_root, site, class_id):
    class_name = CLASS_NAMES[class_id]
    if site == "input":
        if class_id == 5:
            return Path(metrics_root) / "block_inputs_onoff_ablation_16x_k32_dog"
        return Path(metrics_root) / f"block_inputs_onoff_ablation_16x_k32_class_{class_id}_{class_name}"
    if site == "middle":
        return Path(metrics_root) / f"block_middle_reads_onoff_ablation_16x_k32_class_{class_id}_{class_name}"
    raise ValueError(site)


def load_candidates(metrics_root, site, class_id, clean_max, noisy_min, top_n):
    path = metrics_folder(metrics_root, site, class_id) / "consistent_feature_trajectory_metrics_all_blocks.csv"
    groups = {"clean": [], "noisy": []}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["simple_category"] != "consistent_cyclic":
                continue
            t_peak = int(float(row["modal_peak_t_pm1"]))
            if t_peak <= clean_max:
                group = "clean"
            elif t_peak >= noisy_min:
                group = "noisy"
            else:
                continue
            block = row["block"]
            rec = {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "site": site,
                "group": group,
                "block": block,
                "block_index": int(block[5:7]),
                "feature": int(row["feature"]),
                "timestep": t_peak,
                "peak_mean_color": float(row["peak_mean_color"]),
                "median_color_amplitude": float(row["median_color_amplitude"]),
                "agg_rank_t000": int(float(row["agg_rank_t000"])),
            }
            groups[group].append(rec)
    selected = []
    for group, rows in groups.items():
        rows.sort(key=lambda r: (r["peak_mean_color"], r["median_color_amplitude"]), reverse=True)
        selected.extend(rows[:top_n])
    return selected


def radial_setup(size, device):
    y, x = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing="ij")
    center = size // 2
    radius = torch.sqrt((x.float() - center) ** 2 + (y.float() - center) ** 2).round().long()
    return [(radius == r) for r in range(int(radius.max().item()) + 1)]


def radial_power(signal, masks):
    gray = signal.float().mean(dim=1)
    fft = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
    power = fft.abs().pow(2)
    radial = torch.stack([power[:, mask].mean(dim=1) for mask in masks], dim=1)
    total = radial.sum(dim=1).clamp_min(1e-12)
    norm = radial / total.unsqueeze(1)
    radii = torch.arange(radial.shape[1], device=signal.device, dtype=torch.float32)
    return radial, norm, {
        "total_power": total,
        "spectral_centroid": (norm * radii.unsqueeze(0)).sum(dim=1),
        "low_power_frac": norm[:, :5].sum(dim=1),
        "mid_power_frac": norm[:, 5:11].sum(dim=1),
        "high_power_frac": norm[:, 11:].sum(dim=1),
        "high_low_ratio": norm[:, 11:].sum(dim=1) / norm[:, :5].sum(dim=1).clamp_min(1e-12),
    }


@torch.no_grad()
def evaluate_input(args, model, diffusion, device, x0, labels, features, block_index, timestep):
    sae = load_input_sae(args.input_sae_ckpt, block_index, device)
    decoder = sae.decoder.float()
    t = torch.full((x0.shape[0],), timestep, device=device, dtype=torch.long)
    gen = torch.Generator(device=device).manual_seed(args.seed + 1000 + timestep)
    noise = torch.randn(x0.shape, device=device, generator=gen)
    x_in, c, _x_t, _ = get_block_input_and_c(model, diffusion, x0, labels, t, block_index, args.precision, noise)
    base_out = finish_from_block(model, x_in, c, block_index, args.precision)
    base_eps = torch.chunk(base_out, 2, dim=1)[0]
    dense = encode_dense_input(sae, x_in.flatten(0, 1))
    out = {}
    for feature in features:
        coeff = dense[:, :, feature].float()
        states = []
        for ii in range(x0.shape[0]):
            delta = -coeff[ii].unsqueeze(1) * decoder[feature].unsqueeze(0)
            states.append(x_in[ii] + delta)
        states = torch.stack(states, dim=0)
        off_out = finish_from_block(model, states, c, block_index, args.precision)
        off_eps = torch.chunk(off_out, 2, dim=1)[0]
        out[feature] = {
            "eps_delta": base_eps - off_eps,
            "coeff_abs_mean": coeff.abs().mean(dim=1),
            "active_frac": (coeff > 0).float().mean(dim=1),
        }
    return out


@torch.no_grad()
def evaluate_middle(args, model, diffusion, device, x0, labels, features, block_index, timestep):
    sae = load_middle_sae(args.middle_sae_ckpt, block_index, device)
    decoder = sae.decoder.float()
    t = torch.full((x0.shape[0],), timestep, device=device, dtype=torch.long)
    gen = torch.Generator(device=device).manual_seed(args.seed + 1000 + timestep)
    noise = torch.randn(x0.shape, device=device, generator=gen)
    x_middle, c, _x_t = run_prefix_to_middle(model, diffusion, x0, labels, t, block_index, args.precision, noise)
    base_out = finish_from_middle(model, x_middle, c, block_index, args.precision)
    base_eps = torch.chunk(base_out, 2, dim=1)[0]
    dense = encode_dense_middle(sae, x_middle.flatten(0, 1))
    out = {}
    for feature in features:
        coeff = dense[:, :, feature].float()
        states = []
        for ii in range(x0.shape[0]):
            delta = -coeff[ii].unsqueeze(1) * decoder[feature].unsqueeze(0)
            states.append(x_middle[ii] + delta)
        states = torch.stack(states, dim=0)
        off_out = finish_from_middle(model, states, c, block_index, args.precision)
        off_eps = torch.chunk(off_out, 2, dim=1)[0]
        out[feature] = {
            "eps_delta": base_eps - off_eps,
            "coeff_abs_mean": coeff.abs().mean(dim=1),
            "active_frac": (coeff > 0).float().mean(dim=1),
        }
    return out


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def color(i):
    return ["#2166ac", "#b2182b", "#1b7837", "#762a83"][i % 4]


def write_spectrum_svg(path, series, title, width=760, height=430):
    left, right, top, bottom = 62, 24, 40, 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_x = max(max(xs) for xs, _ in series.values())
    max_y = max(max(ys) for _, ys in series.values()) * 1.08
    max_y = max(max_y, 1e-9)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="20" y="24" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
    ]
    for tick in range(0, int(max_x) + 1, 4):
        x = left + plot_w * tick / max_x
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 5}" stroke="#222"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 34}" text-anchor="middle" font-family="Arial" font-size="11">{tick}</text>')
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        y = top + plot_h * (1 - frac)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{max_y * frac:.3g}</text>')
    for idx, (name, (xs, ys)) in enumerate(series.items()):
        pts = []
        for x0, y0 in zip(xs, ys):
            x = left + plot_w * x0 / max_x
            y = top + plot_h * (1 - y0 / max_y)
            pts.append(f"{x:.1f},{y:.1f}")
        c = color(idx)
        parts.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{" ".join(pts)}"/>')
        lx = left + 18 + idx * 170
        parts.append(f'<line x1="{lx}" y1="{height - 18}" x2="{lx + 22}" y2="{height - 18}" stroke="{c}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 28}" y="{height - 14}" font-family="Arial" font-size="12">{name}</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 8}" text-anchor="middle" font-family="Arial" font-size="12">radial spatial frequency bin</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classes", default="0,1,2")
    p.add_argument("--sites", default="input,middle")
    p.add_argument("--features-per-group", type=int, default=20)
    p.add_argument("--clean-max", type=int, default=200)
    p.add_argument("--noisy-min", type=int, default=800)
    p.add_argument("--metrics-root", default="runs")
    p.add_argument("--out-root", default="runs/read_feature_eps_frequency_clean_vs_noisy_first3")
    p.add_argument("--ckpt", default="runs/dit_s2_cifar10/last.pt")
    p.add_argument("--input-sae-ckpt", default="runs/sae_block_inputs_16x_k32_long/last.pt")
    p.add_argument("--middle-sae-ckpt", default="runs/sae_middle_reads_16x_k32_long/last.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--num-images", type=int, default=12)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    args = p.parse_args()

    classes = [int(x) for x in args.classes.split(",") if x.strip()]
    sites = [x.strip() for x in args.sites.split(",") if x.strip()]
    selected = []
    for cid in classes:
        for site in sites:
            selected.extend(load_candidates(args.metrics_root, site, cid, args.clean_max, args.noisy_min, args.features_per_group))

    out_root = Path(args.out_root)
    write_csv(out_root / "selected_features.csv", selected, [
        "class_id", "class_name", "site", "group", "block", "block_index", "feature", "timestep",
        "peak_mean_color", "median_color_amplitude", "agg_rank_t000",
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = load_model(args.ckpt, device)
    diffusion = Diffusion(args.timesteps, device=device)
    masks = radial_setup(32, device)
    class_batches = {cid: make_batch(args.data_dir, args.num_images, args.seed, device, cid) for cid in classes}

    work = defaultdict(list)
    for rec in selected:
        work[(rec["class_id"], rec["site"], rec["block_index"], rec["timestep"])].append(rec)

    rows = []
    spectra_accum = defaultdict(list)
    for (cid, site, block_index, timestep), recs in sorted(work.items()):
        print(f"class={CLASS_NAMES[cid]} site={site} block={block_index} t={timestep} features={len(recs)}", flush=True)
        x0, labels, _indices = class_batches[cid]
        feature_ids = [r["feature"] for r in recs]
        if site == "input":
            result = evaluate_input(args, model, diffusion, device, x0, labels, feature_ids, block_index, timestep)
        else:
            result = evaluate_middle(args, model, diffusion, device, x0, labels, feature_ids, block_index, timestep)
        for rec in recs:
            feature = rec["feature"]
            eps_delta = result[feature]["eps_delta"]
            _radial, norm, stats = radial_power(eps_delta, masks)
            spectra_accum[(rec["site"], rec["group"])].append(norm.mean(dim=0).detach().cpu())
            rows.append({
                **rec,
                "n_images": int(eps_delta.shape[0]),
                "total_power": float(stats["total_power"].mean().item()),
                "spectral_centroid": float(stats["spectral_centroid"].mean().item()),
                "low_power_frac": float(stats["low_power_frac"].mean().item()),
                "mid_power_frac": float(stats["mid_power_frac"].mean().item()),
                "high_power_frac": float(stats["high_power_frac"].mean().item()),
                "high_low_ratio": float(stats["high_low_ratio"].mean().item()),
                "coeff_abs_mean": float(result[feature]["coeff_abs_mean"].mean().item()),
                "active_frac": float(result[feature]["active_frac"].mean().item()),
            })

    metric_fields = [
        "class_id", "class_name", "site", "group", "block", "block_index", "feature", "timestep",
        "peak_mean_color", "median_color_amplitude", "agg_rank_t000", "n_images", "total_power",
        "spectral_centroid", "low_power_frac", "mid_power_frac", "high_power_frac", "high_low_ratio",
        "coeff_abs_mean", "active_frac",
    ]
    write_csv(out_root / "feature_eps_frequency_metrics.csv", rows, metric_fields)

    summary_rows = []
    for site in sites:
        series = {}
        for group in ["clean", "noisy"]:
            vals = spectra_accum.get((site, group), [])
            if not vals:
                continue
            mean = torch.stack(vals).mean(dim=0)
            series[group] = (list(range(len(mean))), [float(x) for x in mean.tolist()])
            group_rows = [r for r in rows if r["site"] == site and r["group"] == group]
            summary_rows.append({
                "site": site,
                "group": group,
                "n_features": len(group_rows),
                "mean_total_power": sum(r["total_power"] for r in group_rows) / max(1, len(group_rows)),
                "mean_spectral_centroid": sum(r["spectral_centroid"] for r in group_rows) / max(1, len(group_rows)),
                "mean_low_power_frac": sum(r["low_power_frac"] for r in group_rows) / max(1, len(group_rows)),
                "mean_mid_power_frac": sum(r["mid_power_frac"] for r in group_rows) / max(1, len(group_rows)),
                "mean_high_power_frac": sum(r["high_power_frac"] for r in group_rows) / max(1, len(group_rows)),
                "mean_high_low_ratio": sum(r["high_low_ratio"] for r in group_rows) / max(1, len(group_rows)),
            })
        if series:
            write_spectrum_svg(out_root / f"{site}_clean_vs_noisy_radial_spectrum.svg", series, f"{site} read eps-delta frequency: clean vs noisy")

    write_csv(out_root / "summary_by_site_group.csv", summary_rows, [
        "site", "group", "n_features", "mean_total_power", "mean_spectral_centroid",
        "mean_low_power_frac", "mean_mid_power_frac", "mean_high_power_frac", "mean_high_low_ratio",
    ])
    with (out_root / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Read Feature Epsilon-Frequency Analysis\n\n")
        f.write("Features are consistent-cyclic read SAE features split by peak timestep: clean `t <= clean_max`, noisy `t >= noisy_min`.\n\n")
        f.write("For each selected feature, the script zeroes that read-site SAE feature, recomputes the model output, and analyzes the raw epsilon delta:\n\n")
        f.write("`eps_delta = eps_original - eps_feature_off`\n\n")
        f.write("Sites: `input` is the beginning read into a block; `middle` is the read after the attention write and before the MLP.\n")
    with (out_root / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
