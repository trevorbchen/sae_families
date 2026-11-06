#!/usr/bin/env python3
"""Consolidate frequency-family Kimi JSONL labels into analysis-ready CSVs."""

import argparse
import collections
import csv
import json
from pathlib import Path


CATEGORICAL_FIELDS = (
    "candidate_type",
    "effect_category",
    "frequency_band",
    "spatial_scale",
    "spatial_distribution",
)


def scalar(value):
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return ""
    return str(value)


def family_name(record):
    source = record.get("source", "")
    if source == "direct_mse":
        return "direct_mse"
    if source.startswith("grouped_objective_"):
        return source
    if source == "tandem_canonical":
        component = record.get("component_label", "").removesuffix(" branch")
        return f"tandem_{component.lower().replace(' ', '_')}" if component else source
    return source


def read_records(inputs):
    records = {}
    for pattern in inputs:
        paths = sorted(Path().glob(pattern)) if not Path(pattern).is_absolute() else []
        if Path(pattern).is_file():
            paths = [Path(pattern)]
        elif Path(pattern).is_dir():
            paths = sorted(Path(pattern).glob("*.jsonl"))
        for path in paths:
            for line in path.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record.get("label"), dict):
                    continue
                key = (
                    record.get("source", ""),
                    record.get("requested_component", ""),
                    int(record.get("rank", -1)),
                )
                records[key] = record
    return sorted(records.values(), key=lambda row: (family_name(row), int(row["rank"])))


def write_rows(records, path):
    label_fields = (
        "primary_effect", *CATEGORICAL_FIELDS, "affected_regions", "direction",
        "secondary_effects", "semantic_confounds", "frequency_selective_candidate",
        "texture_only", "generic_blur_only", "subject_consistency",
        "identity_and_geometry_preserved", "artifact_onset", "confidence", "evidence",
    )
    fields = (
        "family", "source", "component_label", "rank", "feature", "interesting",
        "dataset_active_fraction", "active_rms", "prompt_version", *label_fields,
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            label = record["label"]
            row = {
                "family": family_name(record),
                "source": record.get("source", ""),
                "component_label": record.get("component_label", ""),
                "rank": record.get("rank", ""),
                "feature": record.get("feature", ""),
                "interesting": scalar(record.get("interesting", False)),
                "dataset_active_fraction": record.get("dataset_active_fraction", ""),
                "active_rms": record.get("active_rms", ""),
                "prompt_version": record.get("prompt_version", ""),
            }
            row.update({field: scalar(label.get(field, "")) for field in label_fields})
            writer.writerow(row)


def write_summary(records, path):
    grouped = collections.defaultdict(list)
    for record in records:
        grouped[family_name(record)].append(record)
    fields = ("family", "n", "field", "value", "count", "fraction")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for family, rows in sorted(grouped.items()):
            for field in CATEGORICAL_FIELDS + (
                "frequency_selective_candidate", "generic_blur_only",
            ):
                counts = collections.Counter(
                    scalar(row["label"].get(field, "unclear")) for row in rows
                )
                for value, count in counts.most_common():
                    writer.writerow({
                        "family": family,
                        "n": len(rows),
                        "field": field,
                        "value": value,
                        "count": count,
                        "fraction": count / len(rows),
                    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = read_records(args.inputs)
    write_rows(records, out / "feature_labels.csv")
    write_summary(records, out / "family_label_distributions.csv")
    (out / "summary.json").write_text(json.dumps({
        "features": len(records),
        "families": dict(collections.Counter(family_name(row) for row in records)),
    }, indent=2) + "\n")
    print(f"wrote {len(records)} labels to {out}")


if __name__ == "__main__":
    main()
