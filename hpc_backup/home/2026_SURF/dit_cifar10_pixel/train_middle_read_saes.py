#!/usr/bin/env python3
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_dit_cifar import DiT, Diffusion, modulate


class TopKSAE(nn.Module):
    def __init__(self, input_dim=384, dict_size=6144, k=32):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.k = k
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Parameter(torch.empty(dict_size, input_dim))
        self.dec_bias = nn.Parameter(torch.zeros(input_dim))
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder, std=1.0 / input_dim**0.5)
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.div_(self.decoder.norm(dim=1, keepdim=True).clamp_min(1e-6))

    def forward(self, x):
        pre = self.encoder(x - self.dec_bias)
        acts = F.relu(pre)
        vals, idx = torch.topk(acts, self.k, dim=-1)
        recon = (vals.unsqueeze(-1) * self.decoder[idx]).sum(dim=1) + self.dec_bias
        return recon, vals


@torch.no_grad()
def get_middle_reads(model, diffusion, x0, labels, timesteps, blocks, precision):
    noise = torch.randn_like(x0)
    x_t = diffusion.q_sample(x0, timesteps, noise)
    enabled = x_t.device.type == "cuda" and precision == "bf16"
    reads = {}

    with torch.autocast(device_type=x_t.device.type, dtype=torch.bfloat16, enabled=enabled):
        x = model.x_embedder(x_t).flatten(2).transpose(1, 2) + model.pos_embed
        c = model.t_embedder(timesteps) + model.y_embedder(labels)

        for i, block in enumerate(model.blocks):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaLN(c).chunk(6, dim=1)

            attn_in = modulate(block.norm1(x), shift_msa, scale_msa)
            attn_out = block.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
            x = x + gate_msa.unsqueeze(1) * attn_out

            if i in blocks:
                reads[f"block{i:02d}_middle_read"] = x.float().flatten(0, 1)

            mlp_in = modulate(block.norm2(x), shift_mlp, scale_mlp)
            mlp_out = block.mlp(mlp_in)
            x = x + gate_mlp.unsqueeze(1) * mlp_out

    return reads


def train_one(sae, opt, x):
    recon, vals = sae(x)
    mse = F.mse_loss(recon, x)
    denom = x.pow(2).mean().clamp_min(1e-8)
    loss = mse / denom
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sae.normalize_decoder()
    with torch.no_grad():
        return {
            "loss": float(loss.item()),
            "mse": float(mse.item()),
            "var_explained": float(1.0 - mse.item() / denom.item()),
            "l0": float((vals > 0).sum(dim=1).float().mean().item()),
            "target_rms": float(x.pow(2).mean().sqrt().item()),
        }


def save_saes(path, step, saes, args):
    torch.save(
        {
            "step": step,
            "saes": {name: sae.state_dict() for name, sae in saes.items()},
            "config": vars(args),
            "activation_site": "middle_read_after_attention_write_before_norm2",
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/dit_s2_cifar10/last.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="runs/sae_middle_reads_16x_k32_long")
    parser.add_argument("--block-start", type=int, default=0)
    parser.add_argument("--block-end", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dict-mult", type=int, default=16)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    blocks = list(range(args.block_start, args.block_end + 1))
    if not blocks or min(blocks) < 0 or max(blocks) > 11:
        raise ValueError(f"Invalid block range: {blocks}")

    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(args.data_dir, train=True, transform=transform, download=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = DiT(learn_sigma=True).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    diffusion = Diffusion(args.timesteps, device=device)
    input_dim = 384
    dict_size = input_dim * args.dict_mult

    names = [f"block{i:02d}_middle_read" for i in blocks]
    saes = nn.ModuleDict({name: TopKSAE(input_dim, dict_size, args.k) for name in names}).to(device)
    opts = {name: torch.optim.AdamW(saes[name].parameters(), lr=args.lr, weight_decay=0.0) for name in names}

    step = 0
    start = time.time()
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    while step < args.max_steps:
        for x0, labels in loader:
            x0 = x0.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            t = torch.randint(0, args.timesteps, (x0.shape[0],), device=device)

            reads = get_middle_reads(model, diffusion, x0, labels, t, blocks, args.precision)
            stats = {}
            for name in names:
                stats[name] = train_one(saes[name], opts[name], reads[name])
            step += 1

            if step % args.log_every == 0 or step == 1:
                rec = {
                    "step": step,
                    "tokens_per_sae": step * args.batch_size * 256,
                    "elapsed_sec": round(time.time() - start, 2),
                    "stats": stats,
                }
                print(json.dumps(rec), flush=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

            if step % args.save_every == 0:
                save_saes(os.path.join(args.out_dir, f"saes_step_{step:06d}.pt"), step, saes, args)

            if step >= args.max_steps:
                break

    save_saes(os.path.join(args.out_dir, "last.pt"), step, saes, args)


if __name__ == "__main__":
    main()
