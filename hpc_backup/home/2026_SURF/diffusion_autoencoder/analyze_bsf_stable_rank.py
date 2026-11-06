#!/usr/bin/env python3
"""Measure the effective dimensionality of each learned BSF block."""

import argparse
import csv
import json
import os

import torch


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bsf", required=True)
    p.add_argument("--codes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=1024)
    args = p.parse_args()

    state = torch.load(args.bsf, map_location="cpu", weights_only=False)
    cfg = state["config"]
    from train_zsem_bsf import BlockSparseFeaturizer

    model = BlockSparseFeaturizer(512, int(cfg["dict_mult"]), int(cfg["block_dim"]), int(cfg["top_blocks"]))
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    mean = state["input_mean"].float()
    std = state["input_std"].float()
    codes_state = torch.load(args.codes, map_location="cpu", weights_only=False)
    codes = codes_state["conds"].float()

    n_blocks = model.n_blocks
    block_dim = model.block_dim
    count = torch.zeros(n_blocks)
    sums = torch.zeros(n_blocks, block_dim)
    second = torch.zeros(n_blocks, block_dim, block_dim)

    for start in range(0, len(codes), args.batch_size):
        x = ((codes[start:start + args.batch_size] - mean) / std)
        coeff = model.encoder(x - model.dec_bias).view(-1, n_blocks, block_dim)
        norms = coeff.norm(dim=-1)
        _, selected = torch.topk(norms, model.top_blocks, dim=-1)
        mask = torch.zeros_like(norms, dtype=torch.bool)
        mask.scatter_(1, selected, True)
        active = coeff * mask.unsqueeze(-1)
        count += mask.sum(0).float()
        sums += active.sum(0)
        second += torch.einsum("bni,bnj->nij", active, active)

    rows = []
    for block in range(n_blocks):
        n = count[block].item()
        if n < 2:
            eig = torch.zeros(block_dim)
        else:
            cov = second[block] / n - torch.outer(sums[block] / n, sums[block] / n)
            eig = torch.linalg.eigvalsh(cov.clamp_min(0)).flip(0).clamp_min(0)
        total = eig.sum().item()
        largest = eig[0].item() if len(eig) else 0.0
        stable_rank = total / max(largest, 1e-12)
        participation = total * total / max(float((eig * eig).sum()), 1e-12)
        rows.append({
            "block": block,
            "selected_count": int(n),
            "selected_fraction": n / len(codes),
            "stable_rank": stable_rank,
            "participation_ratio": participation,
            "eig1_fraction": (eig[0].item() / total) if total > 0 else 0.0,
            **{f"eig_{i + 1}": float(v) for i, v in enumerate(eig)},
        })

    ranks = torch.tensor([r["stable_rank"] for r in rows])
    result = {
        "definition": "stable_rank = trace(covariance) / largest_covariance_eigenvalue, conditional on block being selected",
        "n_codes": len(codes), "n_blocks": n_blocks, "block_dim": block_dim,
        "top_blocks": model.top_blocks,
        "mean_stable_rank": float(ranks.mean()),
        "median_stable_rank": float(ranks.median()),
        "p10_stable_rank": float(torch.quantile(ranks, 0.10)),
        "p90_stable_rank": float(torch.quantile(ranks, 0.90)),
        "fraction_rank_le_1_25": float((ranks <= 1.25).float().mean()),
        "fraction_rank_le_2": float((ranks <= 2).float().mean()),
        "fraction_rank_le_3": float((ranks <= 3).float().mean()),
        "fraction_rank_gt_4": float((ranks > 4).float().mean()),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    with open(args.out, "w") as f:
        json.dump({"summary": result, "blocks": rows}, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
