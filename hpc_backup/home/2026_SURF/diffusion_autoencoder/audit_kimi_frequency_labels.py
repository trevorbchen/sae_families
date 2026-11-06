#!/usr/bin/env python3
"""Create deterministic visual spot-check sheets for Kimi feature labels."""

import argparse
import csv
import json
import random
import textwrap
import urllib.parse
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from label_residual_gui_kimi import request_json, subject_sheet


def family_name(record):
    source = record.get("source", "")
    if source != "tandem_canonical":
        return source
    component = record.get("component_label", "").removesuffix(" branch")
    return f"tandem_{component.lower().replace(' ', '_')}"


def read_labels(root):
    records = []
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            label = record.get("label")
            if isinstance(label, dict) and not record.get("parse_error"):
                records.append(record)
    return records


def choose_samples(records, samples_per_family, seed):
    grouped = defaultdict(list)
    for record in records:
        grouped[family_name(record)].append(record)
    rng = random.Random(seed)
    selected = []
    for family, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["rank"]))
        selected.extend(rng.sample(rows, min(samples_per_family, len(rows))))
    return selected


def wrapped_lines(text, width=76):
    return textwrap.wrap(str(text), width=width) or [""]


def render_record(endpoint, record, examples):
    query = {"source": record["source"], "rank": int(record["rank"])}
    component = record.get("requested_component", "")
    if component:
        query["component"] = component
    feature = request_json(
        f"{endpoint.rstrip('/')}/api/feature?{urllib.parse.urlencode(query)}"
    )
    label = record["label"]
    header_lines = [
        f"{family_name(record)} | rank {record['rank']} | feature {record['feature']}",
        f"Kimi: {label.get('primary_effect', 'unclear')}",
        (
            f"type={label.get('candidate_type', 'unclear')} | "
            f"band={label.get('frequency_band', 'unclear')} | "
            f"scale={label.get('spatial_scale', 'unclear')} | "
            f"confidence={label.get('confidence', 0)}"
        ),
    ]
    header_lines.extend(wrapped_lines(label.get("evidence", "")))
    header_height = 24 * len(header_lines) + 18
    sheets = [subject_sheet(item["variants"]) for item in feature["images"][:examples]]
    width = max(sheet.width for sheet in sheets)
    height = header_height + sum(sheet.height for sheet in sheets)
    output = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(output)
    y = 8
    for index, line in enumerate(header_lines):
        draw.text((10, y), line, fill="black")
        y += 24
    y = header_height
    for sheet in sheets:
        output.paste(sheet, (0, y))
        y += sheet.height
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples-per-family", type=int, default=2)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = read_labels(args.labels_dir)
    selected = choose_samples(records, args.samples_per_family, args.seed)
    rows = []
    for index, record in enumerate(selected):
        name = (
            f"{index:02d}_{family_name(record)}_rank{int(record['rank']):03d}_"
            f"feature{int(record['feature']):05d}.png"
        )
        render_record(args.endpoint, record, args.examples).save(args.out_dir / name)
        label = record["label"]
        rows.append({
            "image": name,
            "family": family_name(record),
            "source": record["source"],
            "component": record.get("requested_component", ""),
            "rank": record["rank"],
            "feature": record["feature"],
            "primary_effect": label.get("primary_effect", ""),
            "frequency_band": label.get("frequency_band", ""),
            "confidence": label.get("confidence", ""),
        })
        print(name, flush=True)

    with (args.out_dir / "audit_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
