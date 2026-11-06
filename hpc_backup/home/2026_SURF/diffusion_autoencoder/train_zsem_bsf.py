#!/usr/bin/env python3
"""Train a block-sparse featurizer on the same Diff-AE z_sem codes as the SAE.

The representation is dense within a selected block, but sparse over blocks:
each block contributes ``block_dim`` coordinates and block TopK selects the
blocks with the largest coefficient L2 norms. This keeps the scalar dictionary
size and active scalar budget comparable to the 32x / k32 SAE baseline.
"""
import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class BlockSparseFeaturizer(nn.Module):
    def __init__(self, d, dict_mult=32, block_dim=8, top_blocks=4):
        super().__init__()
        if dict_mult * d % block_dim:
            raise ValueError("dict_mult * input_dim must be divisible by block_dim")
        self.d = d
        self.block_dim = block_dim
        self.n_blocks = dict_mult * d // block_dim
        self.top_blocks = top_blocks
        self.encoder = nn.Linear(d, self.n_blocks * block_dim)
        self.decoder = nn.Parameter(torch.empty(self.n_blocks, block_dim, d))
        self.dec_bias = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.decoder, std=d ** -0.5)
        self.normalize_decoder()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder.flatten(0, 1))
            self.encoder.bias.zero_()

    @torch.no_grad()
    def normalize_decoder(self):
        # Normalize each block's rows so block scale is controlled by its code.
        self.decoder.div_(self.decoder.norm(dim=-1, keepdim=True).clamp_min(1e-6))

    def encode(self, x):
        coeff = self.encoder(x - self.dec_bias).view(-1, self.n_blocks, self.block_dim)
        norms = coeff.norm(dim=-1)
        _, idx = torch.topk(norms, self.top_blocks, dim=-1)
        mask = torch.zeros_like(norms, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        selected = coeff * mask.unsqueeze(-1)
        recon = torch.einsum("bnk,nkd->bd", selected, self.decoder) + self.dec_bias
        return recon, coeff, norms, idx

    def forward(self, x):
        return self.encode(x)


def load_codes(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    codes = state["conds"].float() if isinstance(state, dict) else state.float()
    return codes, codes.mean(0), codes.std(0).clamp_min(1e-6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--codes", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dict-mult", type=int, default=32)
    p.add_argument("--block-dim", type=int, default=8)
    p.add_argument("--top-blocks", type=int, default=4)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    codes, mean, std = load_codes(args.codes)
    x = ((codes - mean) / std).float()
    perm = torch.randperm(len(x))
    split = int(len(x) * 0.9)
    train, valid = x[perm[:split]], x[perm[split:]]
    loader = DataLoader(TensorDataset(train), batch_size=args.batch_size,
                        shuffle=True, drop_last=False)
    iterator = iter(loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlockSparseFeaturizer(x.shape[1], args.dict_mult, args.block_dim,
                                  args.top_blocks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log = []

    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)[0]
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)[0]
        batch = batch.to(device, non_blocking=True)
        recon, coeff, norms, idx = model(batch)
        loss = F.mse_loss(recon, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.warmup and step <= args.warmup:
            scale = step / args.warmup
            for group in opt.param_groups:
                group["lr"] = args.lr * scale
        opt.step()
        model.normalize_decoder()
        if step == 1 or step % 1000 == 0:
            log.append({
                "step": step,
                "loss": float(loss),
                "active_blocks": args.top_blocks,
                "block_dim": args.block_dim,
            })
            print(log[-1], flush=True)

    with torch.no_grad():
        val = valid.to(device)
        recon, coeff, norms, idx = model(val)
        mse = float(F.mse_loss(recon, val))
        variance = float(val.var().clamp_min(1e-8))
        usage = torch.bincount(idx.flatten(), minlength=model.n_blocks)
        result = {
            "n_codes": len(x), "split": split, "input_dim": x.shape[1],
            "dict_size_scalar": model.n_blocks * model.block_dim,
            "n_blocks": model.n_blocks, "block_dim": model.block_dim,
            "top_blocks": model.top_blocks,
            "active_scalar_dims": model.top_blocks * model.block_dim,
            "valid_mse": mse, "valid_ev": 1 - mse / variance,
            "valid_variance": variance,
            "used_blocks": int((usage > 0).sum()),
            "config": vars(args), "log": log,
        }
    torch.save({"model": model.state_dict(), "input_mean": mean,
                "input_std": std, "config": vars(args), "metrics": result},
               os.path.join(args.out_dir, "last.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
