#!/usr/bin/env python3
import argparse
import json
import os

import torch


def svg_panel(out_path, values, candidate_t, true_t, pred_t, rows, title, ylabel):
    cols = 4
    rows_n = (len(rows) + cols - 1) // cols
    cell_w, cell_h = 330, 240
    pad_l, pad_r, pad_t, pad_b = 42, 14, 42, 34
    width = cols * cell_w
    height = rows_n * cell_h + 42
    min_t = float(candidate_t.min())
    max_t = float(candidate_t.max())

    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="22" text-anchor="middle" font-family="Arial" font-size="16" font-weight="600">{esc(title)}</text>',
        '<text x="18" y="22" font-family="Arial" font-size="11" fill="#2ca02c">green=true t</text>',
        '<text x="18" y="36" font-family="Arial" font-size="11" fill="#d62728">red=argmin t</text>',
    ]
    for panel_i, idx in enumerate(rows):
        col = panel_i % cols
        row = panel_i // cols
        x0 = col * cell_w
        y0 = row * cell_h + 42
        px0 = x0 + pad_l
        py0 = y0 + pad_t
        pw = cell_w - pad_l - pad_r
        ph = cell_h - pad_t - pad_b
        y = values[idx].float()
        y_norm = (y - y.min()) / (y.max() - y.min()).clamp_min(1e-12)

        def x_map(t):
            return px0 + (float(t) - min_t) / (max_t - min_t) * pw

        def y_map(v):
            return py0 + (1.0 - float(v)) * ph

        points = " ".join(f"{x_map(t):.2f},{y_map(v):.2f}" for t, v in zip(candidate_t.tolist(), y_norm.tolist()))
        true_x = x_map(int(true_t[idx]))
        pred_x = x_map(int(pred_t[idx]))
        pred_y = y_map(float(y_norm[torch.argmin(y)]))
        err = abs(int(pred_t[idx]) - int(true_t[idx]))
        parts.extend([
            f'<text x="{x0 + 8}" y="{y0 + 18}" font-family="Arial" font-size="12" font-weight="600">row {idx}: true {int(true_t[idx])}, pred {int(pred_t[idx])}, err {err}</text>',
            f'<rect x="{px0}" y="{py0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#d0d0d0"/>',
            f'<line x1="{px0}" y1="{py0 + ph}" x2="{px0 + pw}" y2="{py0 + ph}" stroke="#555"/>',
            f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py0 + ph}" stroke="#555"/>',
        ])
        for frac in (0.25, 0.5, 0.75):
            gy = py0 + frac * ph
            parts.append(f'<line x1="{px0}" y1="{gy}" x2="{px0 + pw}" y2="{gy}" stroke="#e4e4e4"/>')
        for t_tick in (0, 250, 500, 750, 999):
            tx = x_map(t_tick)
            parts.append(f'<line x1="{tx}" y1="{py0 + ph}" x2="{tx}" y2="{py0 + ph + 4}" stroke="#555"/>')
            parts.append(f'<text x="{tx}" y="{py0 + ph + 17}" text-anchor="middle" font-family="Arial" font-size="9">{t_tick}</text>')
        parts.extend([
            f'<polyline points="{points}" fill="none" stroke="#1f77b4" stroke-width="1.7"/>',
            f'<line x1="{true_x}" y1="{py0}" x2="{true_x}" y2="{py0 + ph}" stroke="#2ca02c" stroke-width="1.5" stroke-dasharray="5,4"/>',
            f'<line x1="{pred_x}" y1="{py0}" x2="{pred_x}" y2="{py0 + ph}" stroke="#d62728" stroke-width="1.7" stroke-dasharray="2,3"/>',
            f'<circle cx="{pred_x}" cy="{pred_y}" r="3.2" fill="#d62728"/>',
            f'<text x="{px0 + pw/2}" y="{y0 + cell_h - 5}" text-anchor="middle" font-family="Arial" font-size="10">candidate timestep</text>',
            f'<text x="{x0 + 11}" y="{py0 + ph/2}" transform="rotate(-90 {x0 + 11} {py0 + ph/2})" text-anchor="middle" font-family="Arial" font-size="10">{esc(ylabel)}</text>',
        ])
    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", default="runs/timestep_multi_metric_eval")
    p.add_argument("--metric", default="score_norm")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.result_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    obj = torch.load(os.path.join(args.result_dir, "multi_metric_curves.pt"), map_location="cpu", weights_only=False)
    values = obj["metrics"][args.metric]
    candidate_t = obj["candidate_t_values"]
    true_t = obj["true_t"]
    pred_t = candidate_t[torch.argmin(values, dim=1)]
    err = (pred_t - true_t).abs()
    first16 = list(range(min(16, values.shape[0])))
    best8 = torch.argsort(err)[:8].tolist()
    worst8 = torch.argsort(err, descending=True)[:8].tolist()
    mixed = best8 + worst8
    svg_panel(
        os.path.join(out_dir, f"{args.metric}_curves_first16.svg"),
        values,
        candidate_t,
        true_t,
        pred_t,
        first16,
        f"{args.metric}: first 16 samples",
        f"min-max {args.metric}",
    )
    svg_panel(
        os.path.join(out_dir, f"{args.metric}_curves_best8_worst8.svg"),
        values,
        candidate_t,
        true_t,
        pred_t,
        mixed,
        f"{args.metric}: best 8 and worst 8 predictions",
        f"min-max {args.metric}",
    )
    with open(os.path.join(out_dir, f"{args.metric}_plot_selection.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": args.metric, "first16": first16, "best8": best8, "worst8": worst8}, f, indent=2)
    print(json.dumps({"out_dir": out_dir, "metric": args.metric, "best8": best8, "worst8": worst8}, indent=2), flush=True)


if __name__ == "__main__":
    main()
