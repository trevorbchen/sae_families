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


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_size=256):
        super().__init__()
        self.freq_size = freq_size
        self.mlp = nn.Sequential(
            nn.Linear(freq_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        half = self.freq_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_size % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(6, dim=1)
        y = modulate(self.norm1(x), shift_msa, scale_msa)
        y = self.attn(y, y, y, need_weights=False)[0]
        x = x + gate_msa.unsqueeze(1) * y
        y = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x + gate_mlp.unsqueeze(1) * y


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DiT(nn.Module):
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
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_patches_side = input_size // patch_size
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Embedding(num_classes, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_side**2, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init)
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        nn.init.xavier_uniform_(self.x_embedder.weight.view(self.x_embedder.weight.shape[0], -1))
        nn.init.zeros_(self.x_embedder.bias)
        self.pos_embed.data.copy_(get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches_side))
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = self.num_patches_side
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x, t, y):
        x = self.x_embedder(x).flatten(2).transpose(1, 2) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y)
        for block in self.blocks:
            x = block(x, c)
        return self.unpatchify(self.final_layer(x, c))


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="ij")
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1).unsqueeze(0)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))
    out = torch.einsum("m,d->md", pos.reshape(-1), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


class Diffusion:
    def __init__(self, timesteps=1000, device="cpu"):
        self.timesteps = timesteps
        betas = self.cosine_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        ac_prev = torch.cat([torch.ones(1, device=device), ac[:-1]])
        self.betas = betas
        self.alphas_cumprod = ac
        self.alphas_cumprod_prev = ac_prev
        self.sqrt_ac = torch.sqrt(ac)
        self.sqrt_one_minus_ac = torch.sqrt(1.0 - ac)
        self.posterior_variance = betas * (1.0 - ac_prev) / (1.0 - ac)
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = betas * torch.sqrt(ac_prev) / (1.0 - ac)
        self.posterior_mean_coef2 = (1.0 - ac_prev) * torch.sqrt(alphas) / (1.0 - ac)

    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        ac = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        ac = ac / ac[0]
        betas = 1 - (ac[1:] / ac[:-1])
        return torch.clip(betas, 0.0001, 0.999)

    def extract(self, a, t, x_shape):
        return a.gather(0, t).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise):
        return self.extract(self.sqrt_ac, t, x_start.shape) * x_start + self.extract(
            self.sqrt_one_minus_ac, t, x_start.shape
        ) * noise

    def q_posterior(self, x_start, x_t, t):
        mean = self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + self.extract(
            self.posterior_mean_coef2, t, x_t.shape
        ) * x_t
        var = self.extract(self.posterior_variance, t, x_t.shape)
        log_var = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def training_losses(self, model, x_start, t, y, learn_sigma):
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        model_out = model(x_t, t, y)
        if learn_sigma:
            eps, var_values = torch.chunk(model_out, 2, dim=1)
        else:
            eps, var_values = model_out, None
        mse = F.mse_loss(eps, noise)
        if not learn_sigma:
            return mse, {"mse": mse.detach(), "vb": torch.zeros_like(mse.detach())}

        pred_x0 = ((x_t - self.extract(self.sqrt_one_minus_ac, t, x_t.shape) * eps.detach()) / self.extract(
            self.sqrt_ac, t, x_t.shape
        )).clamp(-1, 1)
        true_mean, _, true_log_var = self.q_posterior(x_start, x_t, t)
        model_mean, _, _ = self.q_posterior(pred_x0, x_t, t)
        frac = ((var_values + 1) * 0.5).clamp(0, 1)
        min_log = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        max_log = self.extract(torch.log(self.betas), t, x_t.shape)
        model_log_var = frac * max_log + (1 - frac) * min_log
        kl = normal_kl(true_mean, true_log_var, model_mean, model_log_var)
        vb = kl.mean() / math.log(2.0)
        return mse + 0.001 * vb, {"mse": mse.detach(), "vb": vb.detach()}

    @torch.no_grad()
    def p_sample_loop(self, model, shape, y, learn_sigma, device):
        x = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(self.timesteps)), total=self.timesteps, desc="sampling", leave=False):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            out = model(x, t, y)
            eps = torch.chunk(out, 2, dim=1)[0] if learn_sigma else out
            pred_x0 = ((x - self.extract(self.sqrt_one_minus_ac, t, x.shape) * eps) / self.extract(
                self.sqrt_ac, t, x.shape
            )).clamp(-1, 1)
            mean, _, log_var = self.q_posterior(pred_x0, x, t)
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = mean + torch.exp(0.5 * log_var) * noise
        return x


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))


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
    log_every: int
    save_every: int
    sample_every: int
    learn_sigma: bool
    precision: str
    resume: str


def update_ema(ema, model, decay):
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(msd[k].detach(), alpha=1 - decay)
            else:
                v.copy_(msd[k])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out-dir", default="./runs/dit_s2_cifar10")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--sample-every", type=int, default=5000)
    parser.add_argument("--learn-sigma", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

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
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    diffusion = Diffusion(cfg.timesteps, device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and cfg.precision == "fp32"))
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
            t = torch.randint(0, cfg.timesteps, (x.shape[0],), device=device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=(device == "cuda" and cfg.precision == "bf16")):
                loss, parts = diffusion.training_losses(model, x, t, y, cfg.learn_sigma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            update_ema(ema, model, 0.9999)
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", mse=f"{parts['mse'].item():.4f}", vb=f"{parts['vb'].item():.4f}")

            if step % cfg.log_every == 0:
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "mse": float(parts["mse"].item()),
                    "vb": float(parts["vb"].item()),
                    "elapsed_sec": round(time.time() - start, 2),
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

            if step % cfg.sample_every == 0:
                ema.eval()
                labels = torch.arange(16, device=device) % 10
                samples = diffusion.p_sample_loop(ema, (16, 3, 32, 32), labels, cfg.learn_sigma, device)
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
                        "config": asdict(cfg),
                    },
                    os.path.join(cfg.out_dir, f"ckpt_{step:07d}.pt"),
                )

            if cfg.max_steps and step >= cfg.max_steps:
                break
        if cfg.max_steps and step >= cfg.max_steps:
            break

    torch.save(
        {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step, "config": asdict(cfg)},
        os.path.join(cfg.out_dir, "last.pt"),
    )


if __name__ == "__main__":
    main()
