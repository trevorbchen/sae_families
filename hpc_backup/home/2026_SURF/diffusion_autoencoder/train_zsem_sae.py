#!/usr/bin/env python3
"""Train a first TopK SAE on the official Diff-AE semantic codes."""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class TopKSAE(nn.Module):
    def __init__(self, input_dim, dict_size, k):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.k = k
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Parameter(torch.empty(dict_size, input_dim))
        self.dec_bias = nn.Parameter(torch.zeros(input_dim))
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder, std=input_dim ** -0.5)
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.div_(self.decoder.norm(dim=1, keepdim=True).clamp_min(1e-6))

    def forward(self, x):
        pre = self.encoder(x - self.dec_bias)
        acts = F.relu(pre)
        vals, idx = torch.topk(acts, self.k, dim=-1)
        recon = (vals.unsqueeze(-1) * self.decoder[idx]).sum(dim=1) + self.dec_bias
        return recon, vals, idx


def load_codes(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "conds" in state:
        codes = state["conds"].float()
        mean = state.get("conds_mean")
        std = state.get("conds_std")
    else:
        codes = state.float() if torch.is_tensor(state) else state[0].float()
        mean = std = None
    if mean is None or std is None:
        mean = codes.mean(0)
        std = codes.std(0).clamp_min(1e-6)
    return codes, mean.float().flatten(), std.float().flatten()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--codes", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dict-mult", type=int, default=8)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    codes, raw_mean, raw_std = load_codes(args.codes)
    mean, std = codes.mean(0), codes.std(0).clamp_min(1e-6)
    x = ((codes - mean) / std).float()
    n, d = x.shape
    perm = torch.randperm(n)
    split = max(1, int(n * 0.9))
    train, valid = x[perm[:split]], x[perm[split:]]
    loader = DataLoader(TensorDataset(train), batch_size=args.batch_size,
                        shuffle=True, drop_last=False)
    it = iter(loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = TopKSAE(d, args.dict_mult * d, args.k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    log = []
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)[0]
        except StopIteration:
            it = iter(loader)
            batch = next(it)[0]
        batch = batch.to(device, non_blocking=True)
        recon, vals, _ = sae(batch)
        mse = F.mse_loss(recon, batch)
        opt.zero_grad(set_to_none=True)
        mse.backward()
        opt.step()
        sae.normalize_decoder()
        if step == 1 or step % 100 == 0:
            with torch.no_grad():
                log.append({"step": step, "loss": float(mse),
                            "l0": float((vals > 0).sum(1).float().mean())})
    with torch.no_grad():
        val = valid.to(device)
        recon, vals, idx = sae(val)
        mse = float(F.mse_loss(recon, val))
        var = float(val.var().clamp_min(1e-8))
        ev = 1.0 - mse / var
        used = int(torch.unique(idx).numel())
    torch.save({"model": sae.state_dict(), "input_mean": mean,
                "input_std": std, "raw_mean": raw_mean,
                "raw_std": raw_std, "config": vars(args),
                "valid_mse": mse, "valid_ev": ev,
                "valid_used_features": used},
               os.path.join(args.out_dir, "last.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"n_codes": n, "input_dim": d, "dict_size": args.dict_mult * d,
                   "k": args.k, "valid_mse": mse, "valid_ev": ev,
                   "valid_used_features": used, "log": log}, f, indent=2)
    print(json.dumps({"n_codes": n, "input_dim": d, "valid_ev": ev,
                      "valid_used_features": used}, indent=2))


if __name__ == "__main__":
    main()
