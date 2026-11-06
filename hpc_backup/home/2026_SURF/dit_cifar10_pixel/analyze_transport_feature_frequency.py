#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from train_dit_cifar import modulate
from train_dit_cifar_consistency_transport import TransportDiT, q_sample_continuous
from train_transport_saes import TopKSAE
from write_onoff_ablation import make_batch


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
SITE_KINDS = ("input", "middle", "attn", "mlp")


def load_pt(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(path, device):
    model = TransportDiT(learn_sigma=True).to(device)
    ckpt = load_pt(path, device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_sae_bank(path, device):
    ckpt = load_pt(path, device)
    cfg = ckpt.get("config", {})
    input_dim = int(cfg.get("input_dim", 384))
    dict_mult = int(cfg.get("dict_mult", 16))
    dict_size = input_dim * dict_mult
    k = int(cfg.get("k", 32))
    bank = {}
    for name, state in ckpt["saes"].items():
        sae = TopKSAE(input_dim=input_dim, dict_size=dict_size, k=k).to(device)
        sae.load_state_dict(state)
        sae.eval()
        for p in sae.parameters():
            p.requires_grad_(False)
        bank[name] = sae
    return bank


def encode_dense(sae, x_flat):
    pre = sae.encoder(x_flat - sae.dec_bias)
    acts = F.relu(pre)
    vals, idx = torch.topk(acts, sae.k, dim=-1)
    dense = torch.zeros(x_flat.shape[0], sae.dict_size, device=x_flat.device, dtype=x_flat.dtype)
    dense.scatter_(1, idx, vals)
    return dense.reshape(-1, 256, sae.dict_size)


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


def color(i):
    return ["#2166ac", "#b2182b", "#1b7837", "#762a83", "#d6604d", "#4393c3"][i % 6]


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
        ly = height - 30 + (idx // 4) * 16
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 22}" y2="{ly}" stroke="{c}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 28}" y="{ly + 4}" font-family="Arial" font-size="12">{name}</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 8}" text-anchor="middle" font-family="Arial" font-size="12">radial spatial frequency bin</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


@torch.no_grad()
def run_transport_prefix(model, x_t, t, s, labels, precision):
    enabled = x_t.device.type == "cuda" and precision == "bf16"
    records = {}
    with torch.autocast(device_type=x_t.device.type, dtype=torch.bfloat16, enabled=enabled):
        x = model.x_embedder(x_t).flatten(2).transpose(1, 2) + model.pos_embed
        c = model.t_embedder(t) + model.s_embedder(s) + model.y_embedder(labels)
        for i, block in enumerate(model.blocks):
            records[f"block{i}_input"] = {"x": x.float(), "c": c.float(), "block_index": i}
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaLN(c).chunk(6, dim=1)
            attn_in = modulate(block.norm1(x), shift_msa, scale_msa)
            attn_out = block.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
            attn_write = gate_msa.unsqueeze(1) * attn_out
            x_after_attn = x + attn_write
            records[f"b{i:02d}_attn"] = {
                "write": attn_write.float(),
                "x_after_attn": x_after_attn.float(),
                "c": c.float(),
                "block_index": i,
            }
            records[f"block{i:02d}_middle_read"] = {"x": x_after_attn.float(), "c": c.float(), "block_index": i}
            mlp_in = modulate(block.norm2(x_after_attn), shift_mlp, scale_mlp)
            mlp_out = block.mlp(mlp_in)
            mlp_write = gate_mlp.unsqueeze(1) * mlp_out
            x_after_block = x_after_attn + mlp_write
            records[f"b{i:02d}_mlp"] = {
                "write": mlp_write.float(),
                "x_after_block": x_after_block.float(),
                "c": c.float(),
                "block_index": i,
            }
            x = x_after_block
        out = model.unpatchify(model.final_layer(x, c))
    return out.float(), records


@torch.no_grad()
def mlp_write_from_after_attn(model, x_after_attn, c, block_index, precision):
    enabled = x_after_attn.device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type=x_after_attn.device.type, dtype=torch.bfloat16, enabled=enabled):
        block = model.blocks[block_index]
        _, _, _, shift_mlp, scale_mlp, gate_mlp = block.adaLN(c).chunk(6, dim=1)
        mlp_in = modulate(block.norm2(x_after_attn), shift_mlp, scale_mlp)
        mlp_out = block.mlp(mlp_in)
        return (gate_mlp.unsqueeze(1) * mlp_out).float()


@torch.no_grad()
def finish_from_block_input(model, x, c, block_index, precision):
    enabled = x.device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=enabled):
        for i in range(block_index, len(model.blocks)):
            x = model.blocks[i](x, c)
        out = model.unpatchify(model.final_layer(x, c))
    return out.float()


@torch.no_grad()
def finish_from_middle(model, x_middle, c, block_index, precision):
    enabled = x_middle.device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type=x_middle.device.type, dtype=torch.bfloat16, enabled=enabled):
        block = model.blocks[block_index]
        _, _, _, shift_mlp, scale_mlp, gate_mlp = block.adaLN(c).chunk(6, dim=1)
        mlp_in = modulate(block.norm2(x_middle), shift_mlp, scale_mlp)
        mlp_out = block.mlp(mlp_in)
        x = x_middle + gate_mlp.unsqueeze(1) * mlp_out
        for i in range(block_index + 1, len(model.blocks)):
            x = model.blocks[i](x, c)
        out = model.unpatchify(model.final_layer(x, c))
    return out.float()


@torch.no_grad()
def finish_after_block(model, x, c, block_index, precision):
    enabled = x.device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=enabled):
        for i in range(block_index + 1, len(model.blocks)):
            x = model.blocks[i](x, c)
        out = model.unpatchify(model.final_layer(x, c))
    return out.float()


def sae_name(site_kind, block_index):
    if site_kind == "input":
        return f"block{block_index}_input"
    if site_kind == "middle":
        return f"block{block_index:02d}_middle_read"
    if site_kind == "attn":
        return f"b{block_index:02d}_attn"
    if site_kind == "mlp":
        return f"b{block_index:02d}_mlp"
    raise ValueError(site_kind)


def block_from_name(name):
    if name.startswith("block") and name.endswith("_input"):
        return int(name[5:name.index("_")])
    if name.startswith("block") and name.endswith("_middle_read"):
        return int(name[5:7])
    return int(name[1:3])


def site_kind_from_name(name):
    if name.endswith("_input"):
        return "input"
    if name.endswith("_middle_read"):
        return "middle"
    if name.endswith("_attn"):
        return "attn"
    if name.endswith("_mlp"):
        return "mlp"
    raise ValueError(name)


@torch.no_grad()
def select_features(args, model, sae_banks, device, x0, labels, class_id, group, t_value, s_value):
    t = torch.full((x0.shape[0],), float(t_value), device=device)
    s = torch.full((x0.shape[0],), float(s_value), device=device)
    gen = torch.Generator(device=device).manual_seed(args.seed + 1000 + int(t_value) + 17 * int(s_value) + class_id)
    noise = torch.randn(x0.shape, device=device, generator=gen)
    x_t = q_sample_continuous(x0, t, noise, args.timesteps)
    _base_out, records = run_transport_prefix(model, x_t, t, s, labels, args.precision)
    selected = []
    for site_kind in args.site_kinds:
        candidates = []
        bank = sae_banks["writes"] if site_kind in ("attn", "mlp") else sae_banks[site_kind]
        for name, sae in bank.items():
            if site_kind_from_name(name) != site_kind:
                continue
            rec = records[name]
            x = rec["write"] if site_kind in ("attn", "mlp") else rec["x"]
            dense = encode_dense(sae, x.flatten(0, 1))
            coeff_mean = dense.abs().mean(dim=(0, 1))
            vals, idx = torch.topk(coeff_mean, min(args.features_per_group, coeff_mean.numel()))
            block_index = block_from_name(name)
            for rank, (val, fid) in enumerate(zip(vals.tolist(), idx.tolist()), 1):
                candidates.append({
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "group": group,
                    "source_t": int(t_value),
                    "target_s": int(s_value),
                    "site": site_kind,
                    "block": name,
                    "block_index": block_index,
                    "feature": int(fid),
                    "selection_rank": rank,
                    "coeff_abs_mean_for_selection": float(val),
                })
        candidates.sort(key=lambda r: r["coeff_abs_mean_for_selection"], reverse=True)
        selected.extend(candidates[:args.features_per_group])
    return selected


@torch.no_grad()
def evaluate_feature_group(args, model, sae_banks, masks, device, x0, labels, class_id, group, t_value, s_value, rows):
    t = torch.full((x0.shape[0],), float(t_value), device=device)
    s = torch.full((x0.shape[0],), float(s_value), device=device)
    gen = torch.Generator(device=device).manual_seed(args.seed + 1000 + int(t_value) + 17 * int(s_value) + class_id)
    noise = torch.randn(x0.shape, device=device, generator=gen)
    x_t = q_sample_continuous(x0, t, noise, args.timesteps)
    base_out, records = run_transport_prefix(model, x_t, t, s, labels, args.precision)
    out_rows = []
    spectra = []
    for row in rows:
        site_kind = row["site"]
        block_index = int(row["block_index"])
        feature = int(row["feature"])
        bank = sae_banks["writes"] if site_kind in ("attn", "mlp") else sae_banks[site_kind]
        sae = bank[sae_name(site_kind, block_index)]
        rec = records[sae_name(site_kind, block_index)]
        x = rec["write"] if site_kind in ("attn", "mlp") else rec["x"]
        dense = encode_dense(sae, x.flatten(0, 1))
        coeff = dense[:, :, feature].float()
        decoder = sae.decoder.float()
        states = []
        for ii in range(x0.shape[0]):
            delta = -coeff[ii].unsqueeze(1) * decoder[feature].unsqueeze(0)
            if site_kind == "input":
                states.append(rec["x"][ii] + delta)
            elif site_kind == "middle":
                states.append(rec["x"][ii] + delta)
            elif site_kind == "attn":
                states.append(rec["x_after_attn"][ii] + delta)
            else:
                states.append(rec["x_after_block"][ii] + delta)
        states = torch.stack(states, dim=0)
        c = rec["c"]
        if site_kind == "input":
            off_out = finish_from_block_input(model, states, c, block_index, args.precision)
        elif site_kind == "middle":
            off_out = finish_from_middle(model, states, c, block_index, args.precision)
        elif site_kind == "attn":
            states = states + mlp_write_from_after_attn(model, states, c, block_index, args.precision)
            off_out = finish_after_block(model, states, c, block_index, args.precision)
        else:
            off_out = finish_after_block(model, states, c, block_index, args.precision)
        delta = base_out[:, :3] - off_out[:, :3]
        _radial, norm, stats = radial_power(delta, masks)
        spectra.append(norm.mean(dim=0).detach().cpu())
        out_rows.append({
            **row,
            "n_images": int(delta.shape[0]),
            "total_power": float(stats["total_power"].mean().item()),
            "spectral_centroid": float(stats["spectral_centroid"].mean().item()),
            "low_power_frac": float(stats["low_power_frac"].mean().item()),
            "mid_power_frac": float(stats["mid_power_frac"].mean().item()),
            "high_power_frac": float(stats["high_power_frac"].mean().item()),
            "high_low_ratio": float(stats["high_low_ratio"].mean().item()),
            "coeff_abs_mean": float(coeff.abs().mean().item()),
            "active_frac": float((coeff > 0).float().mean().item()),
        })
    return out_rows, spectra


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--site-kinds", default="input,middle,attn,mlp")
    p.add_argument("--features-per-group", type=int, default=20)
    p.add_argument("--clean-source-t", type=int, default=100)
    p.add_argument("--noisy-source-t", type=int, default=900)
    p.add_argument("--target-s", type=int, default=0)
    p.add_argument("--out-root", default="runs/transport_feature_frequency_clean_vs_noisy")
    p.add_argument("--ckpt", default="runs/dit_s2_cifar10_consistency_transport/last.pt")
    p.add_argument("--input-sae-ckpt", default="runs/sae_transport_block_inputs_16x_k32/last.pt")
    p.add_argument("--middle-sae-ckpt", default="runs/sae_transport_middle_reads_16x_k32/last.pt")
    p.add_argument("--writes-sae-ckpt", default="runs/sae_transport_writes_16x_k32/last.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--num-images", type=int, default=12)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    args = p.parse_args()
    args.site_kinds = [x.strip() for x in args.site_kinds.split(",") if x.strip()]
    classes = [int(x) for x in args.classes.split(",") if x.strip()]

    out_root = Path(args.out_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = load_model(args.ckpt, device)
    sae_banks = {
        "input": load_sae_bank(args.input_sae_ckpt, device),
        "middle": load_sae_bank(args.middle_sae_ckpt, device),
        "writes": load_sae_bank(args.writes_sae_ckpt, device),
    }
    masks = radial_setup(32, device)
    class_batches = {cid: make_batch(args.data_dir, args.num_images, args.seed, device, cid) for cid in classes}

    selected = []
    for cid in classes:
        x0, labels, _indices = class_batches[cid]
        selected.extend(select_features(args, model, sae_banks, device, x0, labels, cid, "clean", args.clean_source_t, args.target_s))
        selected.extend(select_features(args, model, sae_banks, device, x0, labels, cid, "noisy", args.noisy_source_t, args.target_s))
    selected_fields = [
        "class_id", "class_name", "group", "source_t", "target_s", "site", "block", "block_index",
        "feature", "selection_rank", "coeff_abs_mean_for_selection",
    ]
    write_csv(out_root / "selected_features.csv", selected, selected_fields)

    work = defaultdict(list)
    for row in selected:
        work[(row["class_id"], row["group"], row["source_t"], row["target_s"], row["site"], row["block_index"])].append(row)

    metric_rows = []
    spectra_accum = defaultdict(list)
    for (cid, group, t_value, s_value, site, block_index), rows in sorted(work.items()):
        print(f"class={CLASS_NAMES[cid]} group={group} source_t={t_value} target_s={s_value} site={site} block={block_index} features={len(rows)}", flush=True)
        x0, labels, _indices = class_batches[cid]
        out_rows, spectra = evaluate_feature_group(args, model, sae_banks, masks, device, x0, labels, cid, group, t_value, s_value, rows)
        metric_rows.extend(out_rows)
        spectra_accum[(site, group)].extend(spectra)

    metric_fields = selected_fields + [
        "n_images", "total_power", "spectral_centroid", "low_power_frac", "mid_power_frac",
        "high_power_frac", "high_low_ratio", "coeff_abs_mean", "active_frac",
    ]
    write_csv(out_root / "feature_xs_frequency_metrics.csv", metric_rows, metric_fields)

    summary_rows = []
    for site in args.site_kinds:
        series = {}
        for group in ("clean", "noisy"):
            vals = spectra_accum.get((site, group), [])
            if not vals:
                continue
            mean = torch.stack(vals).mean(dim=0)
            series[group] = (list(range(len(mean))), [float(x) for x in mean.tolist()])
            rows = [r for r in metric_rows if r["site"] == site and r["group"] == group]
            summary_rows.append({
                "site": site,
                "group": group,
                "n_features": len(rows),
                "mean_total_power": sum(r["total_power"] for r in rows) / max(1, len(rows)),
                "mean_spectral_centroid": sum(r["spectral_centroid"] for r in rows) / max(1, len(rows)),
                "mean_low_power_frac": sum(r["low_power_frac"] for r in rows) / max(1, len(rows)),
                "mean_mid_power_frac": sum(r["mid_power_frac"] for r in rows) / max(1, len(rows)),
                "mean_high_power_frac": sum(r["high_power_frac"] for r in rows) / max(1, len(rows)),
                "mean_high_low_ratio": sum(r["high_low_ratio"] for r in rows) / max(1, len(rows)),
            })
        if series:
            write_spectrum_svg(out_root / f"{site}_clean_vs_noisy_radial_spectrum.svg", series, f"transport {site} x_s-delta frequency: clean vs noisy source t")
    write_csv(out_root / "summary_by_site_group.csv", summary_rows, [
        "site", "group", "n_features", "mean_total_power", "mean_spectral_centroid",
        "mean_low_power_frac", "mean_mid_power_frac", "mean_high_power_frac", "mean_high_low_ratio",
    ])
    with (out_root / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Transport Feature Frequency Analysis\n\n")
        f.write("This analyzes the trained consistency-transport DiT (`x_t, t, s, y -> x_s`).\n\n")
        f.write("Features are selected by highest mean absolute SAE coefficient at clean source `t` and noisy source `t`, with target `s=0` by default. For each selected feature, the feature is turned off and the raw output delta is analyzed:\n\n")
        f.write("`x_s_delta = model_original(x_t, t, s, y) - model_feature_off(x_t, t, s, y)`\n\n")
        f.write("The spectra are normalized radial FFT power over the 32x32 output delta. This is not epsilon-space; it is direct transport output space.\n")
    with (out_root / "config.json").open("w", encoding="utf-8") as f:
        json.dump({**vars(args), "site_kinds": args.site_kinds}, f, indent=2)


if __name__ == "__main__":
    main()
