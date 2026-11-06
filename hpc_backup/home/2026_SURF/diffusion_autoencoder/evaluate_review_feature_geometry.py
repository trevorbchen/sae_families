#!/usr/bin/env python3
"""Cache feature-level image geometry for every feature in the review GUI."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluate_feature_output_geometry import METRICS as GEOMETRY_METRICS, image_geometry
from residual_sae_feature_review_gui import ResidualFeatureServer, STAGES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--direct-mse", required=True)
    parser.add_argument("--grouped-objective-root", required=True)
    parser.add_argument("--grouped-objective-seed", default="seed_2030")
    parser.add_argument("--tandem-checkpoint", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--latent-cache", required=True)
    parser.add_argument("--rank-examples", type=int, default=4096)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--decode-batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    return parser.parse_args()


def internal_source(source, component_label):
    if source != "tandem_canonical":
        return source
    stage = component_label.removesuffix(" branch").lower().replace(" ", "_")
    if stage not in STAGES:
        raise ValueError(f"unknown routed component: {component_label}")
    return f"tandem_{stage}"


def aligned_metric(source, component_label):
    name = component_label.removesuffix(" branch").lower().replace(" ", "_")
    if source.startswith("grouped_objective_"):
        name = source.removeprefix("grouped_objective_")
    return {
        "laplacian": "laplacian_norm",
        "fourier_low": "fourier_low_fraction",
        "fourier_mid": "fourier_mid_fraction",
        "fourier_high": "fourier_high_fraction",
    }.get(name, "")


def load_manifest(path):
    records = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("interesting", "false")).strip().lower() == "true":
                continue
            source = row["source"]
            component = row.get("component_label", "")
            records.append(
                {
                    "source": source,
                    "internal_source": internal_source(source, component),
                    "component_label": component,
                    "rank": int(row["rank"]),
                    "feature": int(row["feature"]),
                    "aligned_metric": aligned_metric(source, component),
                }
            )
    return records


def load_completed(path):
    completed = {}
    path = Path(path)
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            completed[(row["source"], row["component_label"], int(row["feature"]))] = row
    return completed


@torch.no_grad()
def collect_examples(server, source, features):
    model = server.sources[source]["model"]
    feature_to_slot = torch.full(
        (model.dict_size,), -1, dtype=torch.long, device=server.device
    )
    feature_to_slot[torch.tensor(features, device=server.device)] = torch.arange(
        len(features), device=server.device
    )
    candidates = [[] for _ in features]
    for start in range(0, len(server.latents), server.args.batch_size):
        raw = server.latents[start : start + server.args.batch_size]
        standardized = server.standardize(raw, source)
        values, indices = server.selected_codes(model, server.incoming(source, standardized))
        slots = feature_to_slot[indices]
        rows = torch.arange(len(raw), device=server.device)[:, None].expand_as(indices)
        selected = (slots >= 0) & (values > 0)
        triples = torch.stack(
            (slots[selected], values[selected], rows[selected] + start), dim=1
        ).cpu().tolist()
        for slot, value, index in triples:
            candidates[int(slot)].append((float(value), int(index)))
    output = {}
    for feature, feature_candidates in zip(features, candidates):
        feature_candidates.sort(reverse=True)
        output[feature] = feature_candidates[: server.args.examples]
    return output


@torch.no_grad()
def decode_chunks(server, latents, batch_size):
    images = []
    for start in range(0, len(latents), batch_size):
        decoded = (
            server.autoencoder.decode(latents[start : start + batch_size])
            .add(1)
            .div(2)
            .clamp(0, 1)
        )
        images.append(decoded)
    return torch.cat(images)


@torch.no_grad()
def evaluate_batch(server, source, records, examples, decode_batch_size):
    model = server.sources[source]["model"]
    pending_latents = []
    layouts = []
    for record in records:
        selected = examples[record["feature"]]
        if not selected:
            continue
        coefficients = torch.tensor(
            [value for value, _ in selected], device=server.device
        )
        indices = torch.tensor([index for _, index in selected], dtype=torch.long)
        raw = server.latents[indices].to(server.device).float()
        standardized = server.standardize(raw, source)
        scale = server.stats[source]["active_rms"][record["feature"]].clamp_min(1e-6)
        edited = []
        for multiplier in (-2.0, 2.0):
            target = torch.clamp(coefficients + multiplier * scale, min=0)
            delta = (target - coefficients)[:, None] * model.decoder[record["feature"]]
            edited.append(server.unstandardize(standardized + delta, source))
        start = sum(len(group) for group in pending_latents)
        pending_latents.extend((raw, edited[0], edited[1]))
        layouts.append((record, start, len(raw)))
    if not layouts:
        return []
    decoded = decode_chunks(server, torch.cat(pending_latents), decode_batch_size)
    output = []
    for record, start, count in layouts:
        clean = decoded[start : start + count]
        minus = decoded[start + count : start + 2 * count]
        plus = decoded[start + 2 * count : start + 3 * count]
        measurements = {
            "edited_output": image_geometry(torch.cat((minus, plus))),
            "causal_delta": image_geometry(
                torch.cat((minus - clean, plus - clean))
            ),
        }
        result = dict(record)
        result["examples"] = count
        result["multipliers"] = [-2.0, 2.0]
        for measurement, values in measurements.items():
            for metric in GEOMETRY_METRICS:
                result[f"{measurement}_{metric}"] = float(values[metric].mean())
        output.append(result)
    return output


def write_csv(path, records):
    fields = [
        "source",
        "internal_source",
        "component_label",
        "rank",
        "feature",
        "aligned_metric",
        "examples",
        "edited_output_laplacian_norm",
        "edited_output_fourier_low_fraction",
        "edited_output_fourier_mid_fraction",
        "edited_output_fourier_high_fraction",
        "causal_delta_laplacian_norm",
        "causal_delta_fourier_low_fraction",
        "causal_delta_fourier_mid_fraction",
        "causal_delta_fourier_high_fraction",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main():
    args = parse_args()
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest_csv)
    completed = load_completed(output_jsonl)
    server_args = SimpleNamespace(
        direct_mse=args.direct_mse,
        grouped_objective_root=args.grouped_objective_root,
        grouped_objective_seed=args.grouped_objective_seed,
        tandem_checkpoint=args.tandem_checkpoint,
        autoencoder=args.autoencoder,
        latent_cache=args.latent_cache,
        annotations=str(output_jsonl.parent / "geometry_annotations_unused.json"),
        port=0,
        examples=args.examples,
        rank_examples=args.rank_examples,
        precompute_ranks=0,
        batch_size=args.batch_size,
    )
    server = ResidualFeatureServer(server_args)
    sources = []
    for record in manifest:
        if record["internal_source"] not in sources:
            sources.append(record["internal_source"])
    with output_jsonl.open("a", encoding="utf-8") as output:
        for source in sources:
            records = [
                record
                for record in manifest
                if record["internal_source"] == source
                and (record["source"], record["component_label"], record["feature"])
                not in completed
            ]
            if not records:
                print(f"{source}: already complete", flush=True)
                continue
            features = [record["feature"] for record in records]
            print(f"{source}: collecting examples for {len(features)} features", flush=True)
            examples = collect_examples(server, source, features)
            for start in range(0, len(records), args.feature_batch_size):
                batch = records[start : start + args.feature_batch_size]
                results = evaluate_batch(
                    server, source, batch, examples, args.decode_batch_size
                )
                for result in results:
                    output.write(json.dumps(result, sort_keys=True) + "\n")
                    output.flush()
                    key = (
                        result["source"],
                        result["component_label"],
                        result["feature"],
                    )
                    completed[key] = result
                print(
                    f"{source}: {min(start + len(batch), len(records))}/{len(records)}",
                    flush=True,
                )
    ordered = [
        completed[(record["source"], record["component_label"], record["feature"])]
        for record in manifest
        if (record["source"], record["component_label"], record["feature"])
        in completed
    ]
    write_csv(args.output_csv, ordered)
    print(f"wrote {len(ordered)} cached features to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
