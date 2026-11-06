#!/usr/bin/env python3
"""Controlled z_sem SAE variants: TopK, AuxK, L1-ReLU, and JumpReLU."""

import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def load_codes(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    return (state["conds"] if isinstance(state, dict) else state).float()


class VariantSAE(nn.Module):
    def __init__(self, d, h, k, mode, aux_k=16, jump_threshold=0.1):
        super().__init__()
        self.d, self.h, self.k, self.mode = d, h, k, mode
        self.aux_k = aux_k
        self.encoder = nn.Linear(d, h)
        self.decoder = nn.Parameter(torch.empty(h, d))
        self.dec_bias = nn.Parameter(torch.zeros(d))
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder, std=d ** -0.5)
        if mode == "jump":
            self.log_threshold = nn.Parameter(
                torch.tensor(math.log(math.expm1(jump_threshold))))
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.div_(self.decoder.norm(dim=1, keepdim=True).clamp_min(1e-6))

    def forward(self, x):
        pre = self.encoder(x - self.dec_bias)
        acts = F.relu(pre)
        aux = None
        if self.mode in ("topk", "auxk"):
            vals, idx = torch.topk(acts, self.k, dim=-1)
            recon_delta = (vals.unsqueeze(-1) * self.decoder[idx]).sum(1)
            if self.mode == "auxk":
                mask = torch.zeros_like(acts, dtype=torch.bool)
                mask.scatter_(1, idx, True)
                residual_acts = acts.masked_fill(mask, 0)
                av, ai = torch.topk(residual_acts, self.aux_k, dim=-1)
                aux = (av.unsqueeze(-1) * self.decoder[ai]).sum(1)
            z = vals
            active_idx = idx
        elif self.mode == "l1":
            recon_delta = acts @ self.decoder
            z = acts
            active_idx = None
        elif self.mode == "jump":
            threshold = F.softplus(self.log_threshold)
            z = acts * (acts > threshold)
            recon_delta = z @ self.decoder
            active_idx = None
        else:
            raise ValueError(self.mode)
        return recon_delta + self.dec_bias, z, active_idx, aux


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--codes", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", choices=["topk", "auxk", "l1", "jump"], default="topk")
    p.add_argument("--norm", choices=["raw", "center", "standard"], default="standard")
    p.add_argument("--dict-mult", type=int, default=16)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--aux-k", type=int, default=16)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--l1-coeff", type=float, default=1e-4)
    p.add_argument("--aux-coeff", type=float, default=0.1)
    p.add_argument("--l0-coeff", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    codes = load_codes(args.codes)
    mean = codes.mean(0)
    std = codes.std(0).clamp_min(1e-6)
    if args.norm == "raw":
        x = codes
    elif args.norm == "center":
        x = codes - mean
    else:
        x = (codes - mean) / std
    x = x.float()
    perm = torch.randperm(len(x))
    split = int(len(x) * 0.9)
    train, valid = x[perm[:split]], x[perm[split:]]
    loader = DataLoader(TensorDataset(train), batch_size=args.batch_size,
                        shuffle=True, drop_last=False)
    it = iter(loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = VariantSAE(x.shape[1], args.dict_mult * x.shape[1], args.k,
                     args.mode, args.aux_k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    log = []
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)[0]
        except StopIteration:
            it = iter(loader)
            batch = next(it)[0]
        batch = batch.to(device, non_blocking=True)
        recon, z, active_idx, aux = sae(batch)
        main_mse = F.mse_loss(recon, batch)
        loss = main_mse
        if args.mode == "auxk":
            residual = (batch - recon).detach()
            loss = loss + args.aux_coeff * F.mse_loss(aux, residual)
        elif args.mode == "l1":
            loss = loss + args.l1_coeff * z.mean()
        elif args.mode == "jump":
            loss = loss + args.l0_coeff * (z > 0).float().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.normalize_decoder()
        if step == 1 or step % 1000 == 0:
            log.append({"step": step, "loss": float(loss),
                        "main_mse": float(main_mse),
                        "active_mean": float((z > 0).sum(1).float().mean())})

    with torch.no_grad():
        val = valid.to(device)
        recon, z, active_idx, _ = sae(val)
        mse = float(F.mse_loss(recon, val))
        var = float(val.var().clamp_min(1e-8))
        if active_idx is not None:
            used = int(torch.unique(active_idx).numel())
        else:
            used = int((z > 0).any(0).sum())
        result = {"n_codes": len(x), "split": split, "input_dim": x.shape[1],
                  "dict_size": args.dict_mult * x.shape[1], "k": args.k,
                  "mode": args.mode, "norm": args.norm,
                  "valid_mse": mse, "valid_ev": 1.0 - mse / var,
                  "valid_variance": var, "valid_used_features": used,
                  "valid_active_mean": float((z > 0).sum(1).float().mean()),
                  "config": vars(args), "log": log}
    torch.save({"model": sae.state_dict(), "input_mean": mean,
                "input_std": std, "config": vars(args),
                "metrics": result}, os.path.join(args.out_dir, "last.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
