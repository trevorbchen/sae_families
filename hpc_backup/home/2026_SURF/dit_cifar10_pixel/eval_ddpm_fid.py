#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm

from train_dit_cifar import DiT, Diffusion


def load_pt(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_dit(path, device, learn_sigma):
    model = DiT(learn_sigma=learn_sigma).to(device)
    ckpt = load_pt(path, device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_inception(device):
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, aux_logits=True, transform_input=False)
    model.fc = nn.Identity()
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def inception_features(model, images, device, batch_size):
    feats = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    for x in torch.split(images, batch_size):
        x = x.to(device, non_blocking=True).float().clamp(0, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
        x = (x - mean) / std
        y = model(x)
        if isinstance(y, tuple):
            y = y[0]
        feats.append(y.detach().cpu().double())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def sample_ddpm(model, diffusion, num_samples, sample_batch_size, device, learn_sigma, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    all_samples = []
    made = 0
    pbar = tqdm(total=num_samples, desc="generate", dynamic_ncols=True)
    while made < num_samples:
        n = min(sample_batch_size, num_samples - made)
        labels = (torch.arange(made, made + n, device=device) % 10).long()
        x = torch.randn((n, 3, 32, 32), device=device, generator=gen)
        for i in reversed(range(diffusion.timesteps)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            out = model(x, t, labels)
            eps = torch.chunk(out, 2, dim=1)[0] if learn_sigma else out
            pred_x0 = ((x - diffusion.extract(diffusion.sqrt_one_minus_ac, t, x.shape) * eps) / diffusion.extract(
                diffusion.sqrt_ac, t, x.shape
            )).clamp(-1, 1)
            mean, _, log_var = diffusion.q_posterior(pred_x0, x, t)
            noise = torch.randn(x.shape, device=device, generator=gen) if i > 0 else torch.zeros_like(x)
            x = mean + torch.exp(0.5 * log_var) * noise
        all_samples.append(((x.clamp(-1, 1) + 1) * 0.5).detach().cpu())
        made += n
        pbar.update(n)
    pbar.close()
    return torch.cat(all_samples, dim=0)


def real_images(data_dir, num_real, seed):
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(data_dir, train=True, transform=transform, download=True)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=gen)[:num_real].tolist()
    loader = DataLoader(Subset(dataset, idx), batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    xs = []
    for x, _ in tqdm(loader, desc="real images", dynamic_ncols=True):
        xs.append(x)
    return torch.cat(xs, dim=0)


def stats(feats):
    x = feats.double()
    mu = x.mean(dim=0)
    xc = x - mu
    cov = xc.T.matmul(xc) / (x.shape[0] - 1)
    return mu, cov


def frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    mu1 = mu1.double()
    mu2 = mu2.double()
    cov1 = cov1.double()
    cov2 = cov2.double()
    eye = torch.eye(cov1.shape[0], dtype=torch.double)
    cov1 = (cov1 + cov1.T) * 0.5 + eps * eye
    cov2 = (cov2 + cov2.T) * 0.5 + eps * eye
    vals1, vecs1 = torch.linalg.eigh(cov1)
    vals1 = vals1.clamp_min(0)
    sqrt_cov1 = (vecs1 * torch.sqrt(vals1).unsqueeze(0)).matmul(vecs1.T)
    middle = sqrt_cov1.matmul(cov2).matmul(sqrt_cov1)
    middle = (middle + middle.T) * 0.5
    vals_mid = torch.linalg.eigvalsh(middle).clamp_min(0)
    tr_covmean = torch.sqrt(vals_mid).sum()
    diff = mu1 - mu2
    fid = diff.dot(diff) + torch.trace(cov1) + torch.trace(cov2) - 2 * tr_covmean
    return float(fid.item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/dit_s2_cifar10/last.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs/ddpm_fid")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--num-real", type=int, default=10000)
    p.add_argument("--sample-batch-size", type=int, default=128)
    p.add_argument("--feature-batch-size", type=int, default=128)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--learn-sigma", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-samples", action="store_true")
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"device={device}", flush=True)

    dit = load_dit(args.ckpt, device, args.learn_sigma)
    diffusion = Diffusion(args.timesteps, device=device)
    inception = load_inception(device)

    samples = sample_ddpm(dit, diffusion, args.num_samples, args.sample_batch_size, device, args.learn_sigma, args.seed)
    if args.save_samples:
        utils.save_image(samples[: min(100, len(samples))], Path(args.out_dir) / "sample_grid_first100.png", nrow=10)
    del dit
    torch.cuda.empty_cache()

    real = real_images(args.data_dir, args.num_real, args.seed + 999)
    gen_feats = inception_features(inception, samples, device, args.feature_batch_size)
    real_feats = inception_features(inception, real, device, args.feature_batch_size)
    mu_g, cov_g = stats(gen_feats)
    mu_r, cov_r = stats(real_feats)
    fid = frechet_distance(mu_g, cov_g, mu_r, cov_r)

    result = {
        "fid": fid,
        "num_samples": args.num_samples,
        "num_real": args.num_real,
        "feature_extractor": "torchvision.models.inception_v3 IMAGENET1K_V1, fc identity, 32x32 resized to 299x299, ImageNet normalization",
        "note": "This is a torchvision-Inception FID estimate, not clean-fid/torch-fidelity exact standard FID.",
    }
    with open(Path(args.out_dir) / "fid_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
