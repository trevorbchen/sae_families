#!/usr/bin/env python3
"""Export compact, presentation-ready panels from the running review API."""

import argparse
import base64
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FAMILIES = {
    "mse": ("direct_mse", [(42, "teeth"), (45, "sunglasses"), (71, "gray hair")]),
    "laplacian": (
        "grouped_objective_laplacian",
        [(4, "eyelashes"), (5, "eye opening"), (34, "pupil / eye color")],
    ),
    "fourier_low": (
        "grouped_objective_fourier_low",
        [(0, "global brightness"), (24, "ear size"), (26, "hair density")],
    ),
    "fourier_mid": (
        "grouped_objective_fourier_mid",
        [(23, "teeth"), (35, "eyebrows"), (54, "fine shadows")],
    ),
    "fourier_high": (
        "grouped_objective_fourier_high",
        [(25, "eyebrow hair"), (62, "nostril size"), (83, "ear lighting")],
    ),
}


def fetch(base_url, source, rank):
    query = urllib.parse.urlencode({"source": source, "rank": rank})
    with urllib.request.urlopen(f"{base_url}/api/feature?{query}", timeout=300) as response:
        return json.load(response)


def decode(uri):
    return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGB")


def build_panel(records, labels, output):
    cell = 156
    gap = 8
    label_h = 34
    header_h = 27
    rows = 3
    cols_per_feature = 3
    feature_w = cols_per_feature * cell + 2 * gap
    width = 3 * feature_w + 4 * gap
    height = header_h + label_h + rows * cell + (rows + 2) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    teal = (22, 124, 128)
    ink = (32, 38, 46)
    gray = (102, 113, 126)

    for feature_index, (record, label) in enumerate(zip(records, labels)):
        x0 = gap + feature_index * (feature_w + gap)
        draw.rectangle((x0, 0, x0 + feature_w, 4), fill=teal)
        draw.text((x0 + 4, 10), label, fill=ink, font=font)
        feature_id = record.get("feature")
        draw.text((x0 + feature_w - 92, 10), f"feature {feature_id}", fill=gray, font=font)
        for row in range(rows):
            variants = record["images"][row]["variants"]
            selected = (variants[1], variants[3], variants[5])  # -4, 0, +4
            for col, variant in enumerate(selected):
                image = decode(variant["image"]).resize((cell, cell), Image.Resampling.LANCZOS)
                x = x0 + col * (cell + gap)
                y = header_h + label_h + row * (cell + gap)
                canvas.paste(image, (x, y))
                if row == 0:
                    draw.text((x + 4, header_h + 5), variant["label"], fill=gray, font=font)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://hpc-21-18:8813")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for family, (source, selections) in FAMILIES.items():
        records = [fetch(args.base_url, source, rank) for rank, _ in selections]
        labels = [label for _, label in selections]
        build_panel(records, labels, Path(args.output_dir) / f"{family}_features.png")


if __name__ == "__main__":
    main()
