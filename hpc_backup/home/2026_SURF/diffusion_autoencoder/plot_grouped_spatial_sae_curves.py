#!/usr/bin/env python3
"""Create dependency-free SVG plots for grouped Conv-AE SAE trajectories."""

import argparse
import html
import json
import math
from pathlib import Path


RUNS = (
    ("Original", "original_16k_k64_seed2026", "#2563eb"),
    ("Proper loss", "proper_loss_16k_k64_seed2026", "#ea580c"),
    ("Transformed loss", "transformed_loss_16k_k64_seed2026", "#059669"),
)
GROUP_COLORS = {
    "mse": "#2563eb",
    "laplacian": "#dc2626",
    "fourier_low": "#059669",
    "fourier_mid": "#d97706",
    "fourier_high": "#7c3aed",
}


def load_json(path):
    return json.loads(path.read_text())


def transform(value, mode):
    if mode == "log":
        return math.log10(max(value, 1e-12))
    if mode == "symlog":
        return math.copysign(math.log10(1 + abs(value) / 10), value)
    return value


def inverse(value, mode):
    if mode == "log":
        return 10 ** value
    if mode == "symlog":
        return math.copysign(10 * (10 ** abs(value) - 1), value)
    return value


def format_number(value):
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 10:
        return f"{value:.0f}"
    if magnitude >= 1:
        return f"{value:.1f}"
    if magnitude >= 0.01:
        return f"{value:.2f}"
    return f"{value:.1e}"


class Svg:
    def __init__(self, width, height, title):
        self.width = width
        self.height = height
        self.items = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#111827;letter-spacing:0} .small{font-size:11px}.label{font-size:12px}.title{font-size:15px;font-weight:600}.main{font-size:22px;font-weight:700}</style>',
            f'<text class="main" x="{width / 2}" y="30" text-anchor="middle">{html.escape(title)}</text>',
        ]

    def add(self, item):
        self.items.append(item)

    def save(self, path):
        self.items.append("</svg>")
        path.write_text("\n".join(self.items))


def panel(svg, x, y, width, height, title, series, mode="linear", reference=None):
    left, right, top, bottom = 62, 16, 34, 42
    px, py = x + left, y + top
    pw, ph = width - left - right, height - top - bottom
    all_points = [point for _, _, points in series for point in points]
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    if reference is not None:
        y_values.append(reference)
    x_min, x_max = min(x_values), max(x_values)
    transformed = [transform(value, mode) for value in y_values]
    y_min, y_max = min(transformed), max(transformed)
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    padding = 0.06 * (y_max - y_min)
    y_min -= padding
    y_max += padding

    def map_x(value):
        return px + pw * (value - x_min) / max(x_max - x_min, 1)

    def map_y(value):
        scaled = transform(value, mode)
        return py + ph * (y_max - scaled) / (y_max - y_min)

    svg.add(f'<text class="title" x="{x + width / 2}" y="{y + 19}" text-anchor="middle">{html.escape(title)}</text>')
    svg.add(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" fill="#f9fafb" stroke="#9ca3af"/>')
    for index in range(6):
        fraction = index / 5
        grid_y = py + ph * fraction
        raw_value = inverse(y_max - fraction * (y_max - y_min), mode)
        svg.add(f'<line x1="{px}" y1="{grid_y:.2f}" x2="{px + pw}" y2="{grid_y:.2f}" stroke="#e5e7eb"/>')
        svg.add(f'<text class="small" x="{px - 7}" y="{grid_y + 4:.2f}" text-anchor="end">{html.escape(format_number(raw_value))}</text>')
    for index in range(5):
        fraction = index / 4
        grid_x = px + pw * fraction
        raw_value = x_min + fraction * (x_max - x_min)
        svg.add(f'<line x1="{grid_x:.2f}" y1="{py}" x2="{grid_x:.2f}" y2="{py + ph}" stroke="#eef2f7"/>')
        svg.add(f'<text class="small" x="{grid_x:.2f}" y="{py + ph + 17}" text-anchor="middle">{raw_value:,.0f}</text>')
    if reference is not None and y_min <= transform(reference, mode) <= y_max:
        reference_y = map_y(reference)
        svg.add(f'<line x1="{px}" y1="{reference_y:.2f}" x2="{px + pw}" y2="{reference_y:.2f}" stroke="#dc2626" stroke-width="1.2" stroke-dasharray="5 4"/>')

    legend_x = px + 8
    for label, color, points in series:
        coordinates = " ".join(
            f"{map_x(step):.2f},{map_y(value):.2f}" for step, value in points
        )
        svg.add(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        svg.add(f'<line x1="{legend_x}" y1="{py + 14}" x2="{legend_x + 18}" y2="{py + 14}" stroke="{color}" stroke-width="3"/>')
        svg.add(f'<text class="small" x="{legend_x + 23}" y="{py + 18}">{html.escape(label)}</text>')
        legend_x += 28 + 7 * len(label)


def read_group_losses(path, stride=20):
    losses = {}
    with path.open() as handle:
        for index, line in enumerate(handle):
            if index % stride:
                continue
            record = json.loads(line)
            for name, value in record["loss_by_group_mean"].items():
                losses.setdefault(name, []).append((record["step"], value))
    return losses


def overview(root, output):
    svg = Svg(1440, 1100, "Full Conv-AE latent SAE trajectories")
    panel_width, panel_height = 470, 340
    for row, (label, directory, color) in enumerate(RUNS):
        metrics = load_json(root / directory / "metrics.json")
        history = metrics["history"]
        validation = metrics["validation_history"]
        train_mse = [(record["step"], record["latent_mse"]) for record in history]
        valid_mse = [(record["step"], record["valid_mse"]) for record in validation]
        valid_ev = [(record["step"], 100 * record["valid_ev"]) for record in validation]
        gradients = [
            (record["step"], record["gradient_norm_before_clip"])
            for record in history
        ]
        top = 45 + row * panel_height
        panel(svg, 0, top, panel_width, panel_height, f"{label}: latent MSE", (("training", color, train_mse), ("held-out", "#111827", valid_mse)), mode="log")
        best_ev, best_step = max((value, step) for step, value in valid_ev)
        panel(svg, 480, top, panel_width, panel_height, f"{label}: validation EV (best {best_ev:.1f}% at {best_step:,})", (("held-out EV", color, valid_ev),), mode="symlog", reference=0)
        panel(svg, 960, top, panel_width, panel_height, f"{label}: gradient norm", (("pre-clip", color, gradients),), mode="log", reference=1)
    svg.save(output / "training_validation_curves.svg")


def specialist_losses(root, output):
    svg = Svg(1400, 790, "Specialist group losses over training")
    for row, (label, directory, _) in enumerate(RUNS[1:]):
        losses = read_group_losses(root / directory / "step_trace.jsonl")
        series = tuple(
            (name.replace("_", " "), GROUP_COLORS[name], points)
            for name, points in losses.items()
        )
        panel(svg, 20, 45 + row * 360, 1360, 350, label, series, mode="log")
    svg.save(output / "specialist_group_losses.svg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    overview(args.root, args.output)
    specialist_losses(args.root, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
