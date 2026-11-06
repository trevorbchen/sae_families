#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from train_dit_cifar import DiT, TimestepEmbedder, update_ema


def cosine_alpha_bar(t, timesteps=1000, s=0.008):
    t01 = (t.float() / float(timesteps)).clamp(0, 1)
    base = math.cos((s / (1 + s)) * math.pi * 0.5) ** 2
    ac = torch.cos(((t01 + s) / (1 + s)) * math.pi * 0.5) ** 2
    return (ac / base).clamp(1e-5, 1.0)


def q_sample_continuous(x0, time, noise, timesteps):
    ac = cosine_alpha_bar(time, timesteps).view(-1, 1, 1, 1)
    return ac.sqrt() * x0 + (1.0 - ac).sqrt() * noise


class TransportDiT(DiT):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=384,
        depth=12,
        num_heads=6,
        num_classes=10,
        learn_sigma=True,
    ):
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
        )
        self.s_embedder = TimestepEmbedder(hidden_size)
        self._init_target_embedder()

    def _init_target_embedder(self):
        for module in self.s_embedder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, t, s, y):
        x = self.x_embedder(x).flatten(2).transpose(1, 2) + self.pos_embed
        c = self.t_embedder(t) + self.s_embedder(s) + self.y_embedder(y)
        for block in self.blocks:
            x = block(x, c)
        return self.unpatchify(self.final_layer(x, c))


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    epochs: int
    max_steps: int
    batch_size: int
    lr: float
    num_workers: int
    timesteps: int
    min_gap: float
    endpoint_prob: float
    log_every: int
    save_every: int
    sample_every: int
    sample_steps: int
    learn_sigma: bool
    precision: str
    resume: str


def sample_time_pairs(batch, timesteps, min_gap, endpoint_prob, device):
    lo = torch.full((batch,), float(min_gap), device=device)
    hi = torch.full((batch,), float(timesteps - 1), device=device)
    t = lo + torch.rand(batch, device=device) * (hi - lo)
    s = torch.rand(batch, device=device) * (t - min_gap).clamp_min(1.0)
    use_endpoint = torch.rand(batch, device=device) < endpoint_prob
    s = torch.where(use_endpoint, torch.zeros_like(s), s)
    return t, s


def model_xs(model, x_t, t, s, y, learn_sigma):
    out = model(x_t, t, s, y)
    return torch.chunk(out, 2, dim=1)[0] if learn_sigma else out


@torch.no_grad()
def transport_sample_loop(model, shape, y, learn_sigma, device, timesteps=1000, steps=18):
    ts = torch.linspace(timesteps - 1, 0, steps, device=device)
    x = torch.randn(shape, device=device)
    for i in tqdm(range(len(ts) - 1), desc="transport sampling", leave=False):
        t = torch.full((shape[0],), float(ts[i].item()), device=device)
        s = torch.full((shape[0],), float(ts[i + 1].item()), device=device)
        x = model_xs(model, x, t, s, y, learn_sigma).clamp(-1, 1)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out-dir", default="./runs/dit_s2_cifar10_consistency_transport")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--min-gap", type=float, default=10.0)
    parser.add_argument("--endpoint-prob", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--sample-every", type=int, default=10000)
    parser.add_argument("--sample-steps", type=int, default=18)
    parser.add_argument("--learn-sigma", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        obj = asdict(cfg)
        obj["prediction_type"] = "xs_transport"
        obj["notes"] = "Model conditions on source timestep t and target timestep s, then maps x_t -> x_s for arbitrary s < t. Same DiT-S/2 residual stream width/depth as the DDPM model."
        json.dump(obj, f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(cfg.data_dir, train=True, transform=transform, download=True)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = TransportDiT(learn_sigma=cfg.learn_sigma).to(device)
    ema = TransportDiT(learn_sigma=cfg.learn_sigma).to(device)
    ema.load_state_dict(model.state_dict())
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    autocast_dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float32

    step = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        step = int(ckpt.get("step", 0))
        print(f"Resumed from {cfg.resume} at step {step}", flush=True)

    start = time.time()
    log_path = os.path.join(cfg.out_dir, "train_log.jsonl")
    model.train()
    start_epoch = step // len(loader)
    for epoch in range(start_epoch, cfg.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for x0, y in pbar:
            x0 = x0.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            t, s = sample_time_pairs(x0.shape[0], cfg.timesteps, cfg.min_gap, cfg.endpoint_prob, device)
            noise = torch.randn_like(x0)
            x_t = q_sample_continuous(x0, t, noise, cfg.timesteps)
            x_s = q_sample_continuous(x0, s, noise, cfg.timesteps)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=(device == "cuda" and cfg.precision == "bf16")):
                pred_s = model_xs(model, x_t, t, s, y, cfg.learn_sigma)
                loss = F.mse_loss(pred_s, x_s)
                endpoint_mask = (s == 0).view(-1, 1, 1, 1)
                endpoint_loss = F.mse_loss(pred_s[endpoint_mask.expand_as(pred_s)], x0[endpoint_mask.expand_as(x0)]) if endpoint_mask.any() else loss.detach() * 0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            update_ema(ema, model, 0.9999)
            step += 1
            mean_gap = (t - s).mean().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", gap=f"{mean_gap:.1f}", endpoint=f"{endpoint_loss.item():.4f}")

            if step % cfg.log_every == 0:
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "endpoint_loss": float(endpoint_loss.item()),
                    "mean_t": float(t.mean().item()),
                    "mean_s": float(s.mean().item()),
                    "mean_gap": float(mean_gap),
                    "elapsed_sec": round(time.time() - start, 2),
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

            if step % cfg.sample_every == 0:
                ema.eval()
                labels = torch.arange(16, device=device) % 10
                samples = transport_sample_loop(
                    ema, (16, 3, 32, 32), labels, cfg.learn_sigma, device, cfg.timesteps, cfg.sample_steps
                )
                samples = (samples.clamp(-1, 1) + 1) / 2
                utils.save_image(samples, os.path.join(cfg.out_dir, f"samples_{step:07d}.png"), nrow=4)
                model.train()

            if step % cfg.save_every == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "step": step,
                        "config": {**asdict(cfg), "prediction_type": "xs_transport"},
                    },
                    os.path.join(cfg.out_dir, f"ckpt_{step:07d}.pt"),
                )

            if cfg.max_steps and step >= cfg.max_steps:
                break
        if cfg.max_steps and step >= cfg.max_steps:
            break

    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "config": {**asdict(cfg), "prediction_type": "xs_transport"},
        },
        os.path.join(cfg.out_dir, "last.pt"),
    )


if __name__ == "__main__":
    main()
