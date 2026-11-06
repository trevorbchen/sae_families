#!/usr/bin/env python3
"""PCA control: optimal dense linear autoencoder at selected ranks."""

import argparse
import json
import os

import torch


def load_codes(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    codes = state["conds"].float() if isinstance(state, dict) else state.float()
    return codes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--codes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    args = p.parse_args()

    codes = load_codes(args.codes)
    mean = codes.mean(0)
    std = codes.std(0).clamp_min(1e-6)
    x = (codes - mean) / std
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(x), generator=g)
    split = int(len(x) * 0.9)
    train, valid = x[perm[:split]], x[perm[split:]]
    train_center = train.mean(0)
    train_centered = train - train_center
    # PCA is the globally optimal dense linear autoencoder for MSE.
    _, _, vh = torch.linalg.svd(train_centered, full_matrices=False)
    total_var = valid.var().item()
    results = []
    for rank in args.ranks:
        basis = vh[:rank]
        valid_recon = (valid - train_center) @ basis.T @ basis + train_center
        mse = torch.mean((valid_recon - valid) ** 2).item()
        results.append({"rank": rank, "valid_mse": mse,
                        "valid_ev": 1.0 - mse / max(total_var, 1e-8)})
    result = {"n_codes": len(x), "input_dim": x.shape[1],
              "split": split, "valid_variance": total_var,
              "results": results,
              "note": "PCA is the optimal dense linear autoencoder under MSE"}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
