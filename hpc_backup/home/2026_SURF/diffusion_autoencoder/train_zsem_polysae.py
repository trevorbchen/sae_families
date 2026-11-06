#!/usr/bin/env python3
"""Train a TopK SAE with a polynomial decoder: z, z^2, and z^3."""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class PolyTopKSAE(nn.Module):
    def __init__(self, d, h, k, aux_k, degree=3):
        super().__init__()
        if degree != 3:
            raise ValueError("this experiment uses degree=3")
        self.d, self.h, self.k, self.aux_k, self.degree = d, h, k, aux_k, degree
        self.encoder = nn.Linear(d, h)
        self.decoder = nn.Parameter(torch.empty(3, h, d))
        self.dec_bias = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.decoder[0], std=d ** -0.5)
        nn.init.zeros_(self.decoder[1])
        nn.init.zeros_(self.decoder[2])
        self.normalize_linear_decoder()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder[0])
            self.encoder.bias.zero_()

    @torch.no_grad()
    def normalize_linear_decoder(self):
        self.decoder[0].div_(self.decoder[0].norm(dim=1, keepdim=True).clamp_min(1e-6))

    def decode_sparse(self, vals, idx):
        powers = torch.stack((vals, vals.square(), vals.pow(3)), dim=-1)
        dec = self.decoder[:, idx].permute(1, 2, 0, 3)
        return torch.einsum("bkp,bkpd->bd", powers, dec) + self.dec_bias

    def encode(self, x):
        acts = F.relu(self.encoder(x - self.dec_bias))
        vals, idx = torch.topk(acts, self.k, dim=-1)
        recon = self.decode_sparse(vals, idx)
        return recon, vals, idx, acts

    def forward(self, x):
        recon, vals, idx, acts = self.encode(x)
        mask = torch.zeros_like(acts, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        residual_acts = acts.masked_fill(mask, 0)
        aux_vals, aux_idx = torch.topk(residual_acts, self.aux_k, dim=-1)
        aux_recon = self.decode_sparse(aux_vals, aux_idx) - self.dec_bias
        return recon, vals, idx, aux_recon, acts


@torch.no_grad()
def rescue_dead(sae, batch, recon, last_fire, step, dead_after, n_rescue):
    dead = torch.where(last_fire < step - dead_after)[0]
    if not len(dead):
        return 0
    errors = (batch - recon).pow(2).mean(1)
    chosen = torch.argsort(errors, descending=True)[:min(len(errors), len(dead), n_rescue)]
    if not len(chosen):
        return 0
    residual = batch[chosen] - recon[chosen]
    directions = F.normalize(residual, dim=1)
    features = dead[:len(chosen)]
    sae.decoder[0, features].copy_(directions)
    sae.decoder[1:, features].zero_()
    sae.encoder.weight[features].copy_(directions)
    centered = batch[chosen] - sae.dec_bias
    response = (directions * centered).sum(1)
    sae.encoder.bias[features].copy_(-response + 0.1)
    last_fire[features] = step
    return len(features)


def load_codes(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    codes = state["conds"].float() if isinstance(state, dict) else state.float()
    return codes, codes.mean(0), codes.std(0).clamp_min(1e-6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--codes", required=True); p.add_argument("--out-dir", required=True)
    p.add_argument("--dict-mult", type=int, default=32); p.add_argument("--k", type=int, default=64)
    p.add_argument("--aux-k", type=int, default=64); p.add_argument("--aux-coeff", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=100000); p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--reinit-every", type=int, default=1000); p.add_argument("--dead-after", type=int, default=3000)
    p.add_argument("--max-rescue", type=int, default=256); p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args(); torch.manual_seed(args.seed); os.makedirs(args.out_dir, exist_ok=True)
    codes, mean, std = load_codes(args.codes); x = ((codes - mean) / std).float()
    perm = torch.randperm(len(x)); split = int(len(x) * 0.9)
    train, valid = x[perm[:split]], x[perm[split:]]
    loader = DataLoader(TensorDataset(train), batch_size=args.batch_size, shuffle=True)
    it = iter(loader); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = PolyTopKSAE(x.shape[1], args.dict_mult * x.shape[1], args.k, args.aux_k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    last_fire = torch.zeros(sae.h, dtype=torch.long, device=device)
    log = []; rescued_total = 0
    for step in range(1, args.steps + 1):
        try: batch = next(it)[0]
        except StopIteration: it = iter(loader); batch = next(it)[0]
        batch = batch.to(device, non_blocking=True)
        recon, vals, idx, aux_recon, acts = sae(batch)
        main_mse = F.mse_loss(recon, batch)
        residual = (batch - recon).detach()
        loss = main_mse + args.aux_coeff * F.mse_loss(aux_recon, residual)
        opt.zero_grad(set_to_none=True); loss.backward()
        if args.warmup and step <= args.warmup:
            for group in opt.param_groups: group["lr"] = args.lr * step / args.warmup
        opt.step(); sae.normalize_linear_decoder()
        fired = idx[vals > 0].unique(); last_fire[fired] = step
        rescued = 0
        if step % args.reinit_every == 0:
            rescued = rescue_dead(sae, batch, recon.detach(), last_fire, step, args.dead_after, args.max_rescue)
            rescued_total += rescued
        if step == 1 or step % 1000 == 0:
            entry = {"step": step, "loss": float(loss), "main_mse": float(main_mse),
                     "aux_mse": float(F.mse_loss(aux_recon, residual)),
                     "active_mean": float((vals > 0).sum(1).float().mean()),
                     "dead": int((last_fire < step - args.dead_after).sum()), "rescued": rescued}
            log.append(entry); print(entry, flush=True)
    with torch.no_grad():
        val = valid.to(device); recon, vals, idx, _, _ = sae(val)
        mse = float(F.mse_loss(recon, val)); variance = float(val.var().clamp_min(1e-8))
        result = {"model": "polynomial_decoder_z_z2_z3", "n_codes": len(x), "split": split,
                  "input_dim": x.shape[1], "dict_size": sae.h, "k": args.k, "aux_k": args.aux_k,
                  "valid_mse": mse, "valid_ev": 1 - mse / variance, "valid_variance": variance,
                  "valid_used_features": int(torch.unique(idx).numel()),
                  "valid_active_mean": float((vals > 0).sum(1).float().mean()),
                  "rescued_total": rescued_total, "config": vars(args), "log": log}
    torch.save({"model": sae.state_dict(), "input_mean": mean, "input_std": std,
                "config": vars(args), "metrics": result}, os.path.join(args.out_dir, "last.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f: json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
