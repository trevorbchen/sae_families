#!/usr/bin/env python3
import argparse
import csv
import json
import os

import torch


def plot_panel(out_path, curves, candidate_t, true_t, pred_t, rows, title):
    n = len(rows)
    cols = 4
    rows_n = (n + cols - 1) // cols
    cell_w = 330
    cell_h = 240
    pad_l = 42
    pad_r = 14
    pad_t = 42
    pad_b = 34
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
        plot_x0 = x0 + pad_l
        plot_y0 = y0 + pad_t
        plot_w = cell_w - pad_l - pad_r
        plot_h = cell_h - pad_t - pad_b
        y = curves[idx].float()
        y_norm = (y - y.min()) / (y.max() - y.min()).clamp_min(1e-12)

        def x_map(t):
            return plot_x0 + (float(t) - min_t) / (max_t - min_t) * plot_w

        def y_map(v):
            return plot_y0 + (1.0 - float(v)) * plot_h

        points = " ".join(f"{x_map(t):.2f},{y_map(v):.2f}" for t, v in zip(candidate_t.tolist(), y_norm.tolist()))
        true_x = x_map(int(true_t[idx]))
        pred_x = x_map(int(pred_t[idx]))
        pred_y = y_map(float(y_norm[torch.argmin(y)]))
        err = abs(int(pred_t[idx]) - int(true_t[idx]))
        parts.extend([
            f'<text x="{x0 + 8}" y="{y0 + 18}" font-family="Arial" font-size="12" font-weight="600">row {idx}: true {int(true_t[idx])}, pred {int(pred_t[idx])}, err {err}</text>',
            f'<rect x="{plot_x0}" y="{plot_y0}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#d0d0d0"/>',
            f'<line x1="{plot_x0}" y1="{plot_y0 + plot_h}" x2="{plot_x0 + plot_w}" y2="{plot_y0 + plot_h}" stroke="#555"/>',
            f'<line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y0 + plot_h}" stroke="#555"/>',
        ])
        for frac in (0.25, 0.5, 0.75):
            gy = plot_y0 + frac * plot_h
            parts.append(f'<line x1="{plot_x0}" y1="{gy}" x2="{plot_x0 + plot_w}" y2="{gy}" stroke="#e4e4e4"/>')
        for t_tick in (0, 250, 500, 750, 999):
            tx = x_map(t_tick)
            parts.append(f'<line x1="{tx}" y1="{plot_y0 + plot_h}" x2="{tx}" y2="{plot_y0 + plot_h + 4}" stroke="#555"/>')
            parts.append(f'<text x="{tx}" y="{plot_y0 + plot_h + 17}" text-anchor="middle" font-family="Arial" font-size="9">{t_tick}</text>')
        parts.extend([
            f'<polyline points="{points}" fill="none" stroke="#1f77b4" stroke-width="1.7"/>',
            f'<line x1="{true_x}" y1="{plot_y0}" x2="{true_x}" y2="{plot_y0 + plot_h}" stroke="#2ca02c" stroke-width="1.5" stroke-dasharray="5,4"/>',
            f'<line x1="{pred_x}" y1="{plot_y0}" x2="{pred_x}" y2="{plot_y0 + plot_h}" stroke="#d62728" stroke-width="1.7" stroke-dasharray="2,3"/>',
            f'<circle cx="{pred_x}" cy="{pred_y}" r="3.2" fill="#d62728"/>',
            f'<text x="{plot_x0 + plot_w/2}" y="{y0 + cell_h - 5}" text-anchor="middle" font-family="Arial" font-size="10">candidate timestep</text>',
            f'<text x="{x0 + 11}" y="{plot_y0 + plot_h/2}" transform="rotate(-90 {x0 + 11} {plot_y0 + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="10">min-max ||eps||</text>',
        ])

    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def write_selected_csv(out_path, curves, candidate_t, true_t, pred_t, rows):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fields = ["row", "true_t", "pred_t", "candidate_t", "eps_norm", "eps_norm_minmax"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for idx in rows:
            y = curves[idx].float()
            y_norm = (y - y.min()) / (y.max() - y.min()).clamp_min(1e-12)
            for j, t in enumerate(candidate_t.tolist()):
                w.writerow({
                    "row": int(idx),
                    "true_t": int(true_t[idx]),
                    "pred_t": int(pred_t[idx]),
                    "candidate_t": int(t),
                    "eps_norm": float(y[j]),
                    "eps_norm_minmax": float(y_norm[j]),
                })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", default="runs/timestep_eps_norm_curve_eval")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    result_dir = args.result_dir
    out_dir = args.out_dir or os.path.join(result_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    obj = torch.load(os.path.join(result_dir, "eps_norm_curves.pt"), map_location="cpu", weights_only=False)
    curves = obj["curves"]
    candidate_t = obj["candidate_t_values"]
    true_t = obj["true_t"]
    pred_t = obj["pred_t"]
    summary = obj["summary"]
    errors = (pred_t - true_t).abs()

    first16 = list(range(min(16, curves.shape[0])))
    best8 = torch.argsort(errors)[:8].tolist()
    worst8 = torch.argsort(errors, descending=True)[:8].tolist()
    mixed = best8 + worst8

    plot_panel(
        os.path.join(out_dir, "eps_norm_curves_first16.svg"),
        curves,
        candidate_t,
        true_t,
        pred_t,
        first16,
        "Raw ||eps_pred|| timestep sweep: first 16 samples",
    )
    plot_panel(
        os.path.join(out_dir, "eps_norm_curves_best8_worst8.svg"),
        curves,
        candidate_t,
        true_t,
        pred_t,
        mixed,
        "Raw ||eps_pred|| timestep sweep: best 8 and worst 8 predictions",
    )
    write_selected_csv(os.path.join(out_dir, "selected_curve_values.csv"), curves, candidate_t, true_t, pred_t, sorted(set(first16 + mixed)))

    with open(os.path.join(out_dir, "plot_selection_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "source_summary": summary,
            "first16": first16,
            "best8": best8,
            "worst8": worst8,
        }, f, indent=2)
    print(json.dumps({"out_dir": out_dir, "first16": first16, "best8": best8, "worst8": worst8}, indent=2), flush=True)


if __name__ == "__main__":
    main()
