#!/usr/bin/env python3
"""Cache dense ConvAE latents and sparse SAE codes for linear probing."""

import argparse
import json
import os
from pathlib import Path

import torch

from train_grouped_objective_sae import GroupSpec, PartitionedTopKSAE
from train_independent_routed_union_sae import load_independent_routed_union
from train_spatial_map_sae import FlatTopKSAE


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def parse_named_paths(records):
    parsed = []
    for record in records:
        name, separator, path = record.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"expected NAME=PATH, received {record!r}")
        parsed.append((name, path))
    return parsed


def load_direct(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    specs = tuple(GroupSpec(**record) for record in payload["groups"])
    model = PartitionedTopKSAE(payload["input_shape"].numel() if hasattr(
        payload["input_shape"], "numel"
    ) else int(torch.tensor(payload["input_shape"]).prod()), specs)
    model.load_state_dict(payload["model"])
    model.eval().requires_grad_(False).to(device)
    mean = payload["input_mean"].float().to(device)
    std = payload["input_std"].float().to(device)

    def encode(raw):
        standardized = ((raw.float() - mean) / std).flatten(1)
        _, groups = model(standardized)
        values, indices = [], []
        for spec in specs:
            values.append(groups[spec.name]["values"])
            indices.append(groups[spec.name]["indices"])
        return torch.cat(values, 1), torch.cat(indices, 1)

    return encode, sum(spec.size for spec in specs), sum(spec.k for spec in specs), {
        "kind": "partitioned", "path": path,
        "branches": [{"name": spec.name, "size": spec.size, "k": spec.k}
                     for spec in specs],
    }


def load_union(path, device):
    model, payload = load_independent_routed_union(path, device)
    mean = payload["channel_mean"].float().to(device)[None, :, None, None]
    std = payload["channel_std"].float().to(device)[None, :, None, None]
    offsets = {}
    offset = 0
    for spec in model.specs:
        offsets[spec.name] = offset
        offset += spec.feature_count

    def encode(raw):
        standardized = (raw.float() - mean) / std
        _, outputs = model(standardized)
        values, indices = [], []
        for spec in model.specs:
            values.append(outputs[spec.name]["values"])
            indices.append(outputs[spec.name]["indices"] + offsets[spec.name])
        return torch.cat(values, 1), torch.cat(indices, 1)

    return encode, offset, sum(spec.k for spec in model.specs), {
        "kind": "independent_routed_union", "path": path,
        "branches": [
            {"name": spec.name, "objective": spec.objective,
             "size": spec.feature_count, "k": spec.k,
             "offset": offsets[spec.name]}
            for spec in model.specs
        ],
    }


def load_flat(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = FlatTopKSAE(payload["input_shape"], config["feature_count"], config["k"])
    model.load_state_dict(payload["model"])
    model.eval().requires_grad_(False).to(device)
    mean = payload["channel_mean"].float().to(device)
    std = payload["channel_std"].float().to(device)
    if len(payload["input_shape"]) == 1:
        mean, std = mean[None, :], std[None, :]
    else:
        mean, std = mean[None, :, None, None], std[None, :, None, None]

    def encode(raw):
        _, values, indices = model((raw.float() - mean) / std)
        return values, indices

    return encode, model.h, model.k, {
        "kind": "standalone_flat", "path": path,
        "objective": config["objective"], "branches": [
            {"name": "features", "objective": config["objective"],
             "size": model.h, "k": model.k, "offset": 0}
        ],
    }


@torch.no_grad()
def cache_sparse(name, loader, path, dataset, output_dir, batch_size, device):
    encode, dimension, k, metadata = loader(path, device)
    cached = {}
    for domain in ("clean", "controlled"):
        latents = dataset[f"{domain}_latents"]
        values, indices = [], []
        for start in range(0, len(latents), batch_size):
            value, index = encode(latents[start:start + batch_size].to(device))
            values.append(value.half().cpu())
            indices.append(index.int().cpu())
        cached[f"{domain}_values"] = torch.cat(values)
        cached[f"{domain}_indices"] = torch.cat(indices)
    train = dataset["split"] == 0
    train_values = cached["clean_values"][train].float()
    train_indices = cached["clean_indices"][train].long()
    sum_squares = torch.zeros(dimension, dtype=torch.float64)
    counts = torch.zeros(dimension, dtype=torch.long)
    positive = train_values > 0
    sum_squares.scatter_add_(0, train_indices[positive], train_values[positive].double().square())
    counts.scatter_add_(0, train_indices[positive], torch.ones_like(train_indices[positive]))
    cached.update({
        "name": name, "format": "topk", "dimension": dimension, "k": k,
        "feature_rms": (sum_squares / counts.clamp_min(1)).sqrt().float(),
        "feature_train_count": counts, "metadata": metadata,
    })
    output = Path(output_dir) / f"{name}.pt"
    atomic_save(cached, output)
    print(json.dumps({"cached": name, "dimension": dimension, "k": k,
                      "output": str(output)}), flush=True)
    del cached, encode
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cache_raw(dataset, output_dir):
    clean = dataset["clean_latents"].float().flatten(1)
    controlled = dataset["controlled_latents"].float().flatten(1)
    train = dataset["split"] == 0
    mean = clean[train].mean(0)
    std = clean[train].std(0).clamp_min(1e-6)
    payload = {
        "name": "raw_latent", "format": "dense", "dimension": clean.shape[1],
        "clean": ((clean - mean) / std).half(),
        "controlled": ((controlled - mean) / std).half(),
        "train_mean": mean, "train_std": std,
        "metadata": {"kind": "raw_spatial_convae_latent"},
    }
    output = Path(output_dir) / "raw_latent.pt"
    atomic_save(payload, output)
    print(json.dumps({"cached": "raw_latent", "dimension": clean.shape[1],
                      "output": str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--direct", help="NAME=PATH")
    parser.add_argument("--union", action="append", default=[], help="NAME=PATH")
    parser.add_argument("--flat", action="append", default=[], help="NAME=PATH")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--skip-targets", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load(args.data, map_location="cpu", weights_only=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        key: value for key, value in dataset.items()
        if key not in ("clean_latents", "controlled_latents")
    }
    targets["source_cache"] = args.data
    if not args.skip_targets:
        atomic_save(targets, output_dir / "targets.pt")
    if not args.skip_raw:
        cache_raw(dataset, output_dir)
    if args.direct:
        direct_name, direct_path = parse_named_paths([args.direct])[0]
        cache_sparse(direct_name, load_direct, direct_path, dataset, output_dir,
                     args.batch_size, device)
    for name, path in parse_named_paths(args.union):
        cache_sparse(name, load_union, path, dataset, output_dir,
                     args.batch_size, device)
    for name, path in parse_named_paths(args.flat):
        cache_sparse(name, load_flat, path, dataset, output_dir,
                     args.batch_size, device)


if __name__ == "__main__":
    main()
