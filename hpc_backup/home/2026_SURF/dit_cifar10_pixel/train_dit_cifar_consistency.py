#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from train_dit_cifar import DiT, Diffusion, update_ema


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
    t_gap: int
    consistency_weight: float
    denoise_weight: float
    log_every: int
    save_every: int
    sample_every: int
    sample_steps: int
    learn_sigma: bool
    precision: str
    resume: str


def model_x0(model, x_t, t, y, learn_sigma):
    out = model(x_t, t, y)
    return torch.chunk(out, 2, dim=1)[0] if learn_sigma else out


@torch.no_grad()
def consistency_sample_loop(model, diffusion, shape, y, learn_sigma, device, steps=18):
    ts = torch.linspace(diffusion.timesteps - 1, 0, steps, device=device).round().long().tolist()
    ts = sorted(set(ts), reverse=True)
    x = torch.randn(shape, device=device)
    for i, t_int in enumerate(tqdm(ts, desc="consistency sampling", leave=False)):
        t = torch.full((shape[0],), int(t_int), device=device, dtype=torch.long)
        pred_x0 = model_x0(model, x, t, y, learn_sigma).clamp(-1, 1)
        if i == len(ts) - 1:
            x = pred_x0
            break
        s_int = int(ts[i + 1])
        s = torch.full((shape[0],), s_int, device=device, dtype=torch.long)
        sqrt_ac_t = diffusion.extract(diffusion.sqrt_ac, t, x.shape)
        sqrt_om_t = diffusion.extract(diffusion.sqrt_one_minus_ac, t, x.shape).clamp_min(1e-6)
        eps = (x - sqrt_ac_t * pred_x0) / sqrt_om_t
        x = diffusion.extract(diffusion.sqrt_ac, s, x.shape) * pred_x0 + diffusion.extract(
            diffusion.sqrt_one_minus_ac, s, x.shape
        ) * eps
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out-dir", default="./runs/dit_s2_cifar10_consistency")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--t-gap", type=int, default=10)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--denoise-weight", type=float, default=1.0)
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
        obj["prediction_type"] = "x0_consistency"
        obj["notes"] = "DiT output first 3 channels are trained as direct x0 predictions. Hidden architecture matches train_dit_cifar.py."
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

    model = DiT(learn_sigma=cfg.learn_sigma).to(device)
    ema = DiT(learn_sigma=cfg.learn_sigma).to(device)
    ema.load_state_dict(model.state_dict())
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    diffusion = Diffusion(cfg.timesteps, device=device)
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
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            t = torch.randint(max(1, cfg.t_gap), cfg.timesteps, (x.shape[0],), device=device)
            s = (t - cfg.t_gap).clamp_min(0)
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise)
            x_s = diffusion.q_sample(x, s, noise)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=(device == "cuda" and cfg.precision == "bf16")):
                pred_t = model_x0(model, x_t, t, y, cfg.learn_sigma)
                denoise = F.mse_loss(pred_t, x)
                with torch.no_grad():
                    target_s = model_x0(ema, x_s, s, y, cfg.learn_sigma).clamp(-1, 1)
                    target_s = torch.where((s == 0).view(-1, 1, 1, 1), x, target_s)
                consistency = F.mse_loss(pred_t, target_s)
                loss = cfg.denoise_weight * denoise + cfg.consistency_weight * consistency

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            update_ema(ema, model, 0.9999)
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", denoise=f"{denoise.item():.4f}", consistency=f"{consistency.item():.4f}")

            if step % cfg.log_every == 0:
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "denoise": float(denoise.item()),
                    "consistency": float(consistency.item()),
                    "elapsed_sec": round(time.time() - start, 2),
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

            if step % cfg.sample_every == 0:
                ema.eval()
                labels = torch.arange(16, device=device) % 10
                samples = consistency_sample_loop(
                    ema, diffusion, (16, 3, 32, 32), labels, cfg.learn_sigma, device, cfg.sample_steps
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
                        "config": {**asdict(cfg), "prediction_type": "x0_consistency"},
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
            "config": {**asdict(cfg), "prediction_type": "x0_consistency"},
        },
        os.path.join(cfg.out_dir, "last.pt"),
    )


if __name__ == "__main__":
    main()
