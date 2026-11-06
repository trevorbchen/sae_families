#!/usr/bin/env python3
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import datasets, transforms

from train_dit_cifar import DiT, Diffusion


T_VALUES = (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999)


class TopKSAE(nn.Module):
    def __init__(self, input_dim=384, dict_size=6144, k=32):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.k = k
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Parameter(torch.empty(dict_size, input_dim))
        self.dec_bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        pre = self.encoder(x - self.dec_bias)
        acts = F.relu(pre)
        vals, idx = torch.topk(acts, self.k, dim=-1)
        recon = (vals.unsqueeze(-1) * self.decoder[idx]).sum(dim=1) + self.dec_bias
        return recon, vals, idx


def parse_ints(s):
    return tuple(int(x) for x in s.split(",") if x.strip())


def load_model(path, device):
    model = DiT(learn_sigma=True).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_block0_input_sae(path, device, input_dim=384, dict_size=6144, k=32):
    ckpt = torch.load(path, map_location=device)
    sae = TopKSAE(input_dim=input_dim, dict_size=dict_size, k=k).to(device)
    state = ckpt["saes"]["block0_input"] if "saes" in ckpt else ckpt["input_sae"]
    sae.load_state_dict(state)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae


@torch.no_grad()
def make_batch(data_dir, num_images, seed, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(data_dir, train=True, transform=transform, download=True)
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=gen)[:num_images].tolist()
    xs, ys = [], []
    for idx in indices:
        x, y = dataset[idx]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs).to(device), torch.tensor(ys, device=device, dtype=torch.long), indices


@torch.no_grad()
def block0_input_and_c(model, x_t, t, labels, precision):
    enabled = x_t.device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type=x_t.device.type, dtype=torch.bfloat16, enabled=enabled):
        x = model.x_embedder(x_t).flatten(2).transpose(1, 2) + model.pos_embed
        c = model.t_embedder(t) + model.y_embedder(labels)
    return x.float(), c.float()


@torch.no_grad()
def finish_from_block0(model, x, c, precision):
    enabled = x.device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=enabled):
        for block in model.blocks:
            x = block(x, c)
        out = model.unpatchify(model.final_layer(x, c))
    return out.float()


@torch.no_grad()
def encode_dense(sae, x_flat):
    pre = sae.encoder(x_flat - sae.dec_bias)
    acts = F.relu(pre)
    vals, idx = torch.topk(acts, sae.k, dim=-1)
    dense = torch.zeros(x_flat.shape[0], sae.dict_size, device=x_flat.device, dtype=x_flat.dtype)
    dense.scatter_(1, idx, vals)
    return dense.reshape(-1, 256, sae.dict_size), vals, idx


def empty_metrics(num_t, num_images, num_features):
    shape = (num_t, num_images, num_features)
    return {
        "eps_delta_mse": torch.empty(shape, dtype=torch.float32),
        "eps_delta_l2": torch.empty(shape, dtype=torch.float32),
        "pred_x0_delta_mse": torch.empty(shape, dtype=torch.float32),
        "pred_x0_mse_to_clean": torch.empty(shape, dtype=torch.float32),
        "pred_x0_mse_to_clean_delta": torch.empty(shape, dtype=torch.float32),
        "active_frac": torch.empty(shape, dtype=torch.float32),
        "coeff_abs_mean": torch.empty(shape, dtype=torch.float32),
        "coeff_max": torch.empty(shape, dtype=torch.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/dit_s2_cifar10/last.pt")
    parser.add_argument("--sae-ckpt", default="runs/sae_block_inputs_16x_k32_long/last.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="runs/block0_input_onoff_ablation_16x_k32")
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--t-values", type=parse_ints, default=T_VALUES)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--features-per-forward", type=int, default=8)
    parser.add_argument("--forward-batch-size", type=int, default=96)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if max(args.t_values) >= args.timesteps:
        raise ValueError("t-values must be inside [0, timesteps); use 999 for a 1000-step diffusion")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    model = load_model(args.ckpt, device)
    sae = load_block0_input_sae(args.sae_ckpt, device)
    diffusion = Diffusion(args.timesteps, device=device)
    x0, labels, data_indices = make_batch(args.data_dir, args.num_images, args.seed, device)
    metrics = empty_metrics(len(args.t_values), args.num_images, sae.dict_size)
    baselines = {
        "eps_mse_to_noise": torch.empty((len(args.t_values), args.num_images), dtype=torch.float32),
        "pred_x0_mse_to_clean": torch.empty((len(args.t_values), args.num_images), dtype=torch.float32),
    }

    gen = torch.Generator(device=device).manual_seed(args.seed + 1000)
    decoder = sae.decoder.float()
    start = time.time()

    for ti, t_int in enumerate(args.t_values):
        t = torch.full((args.num_images,), t_int, device=device, dtype=torch.long)
        noise = torch.randn(x0.shape, device=device, generator=gen)
        x_t = diffusion.q_sample(x0, t, noise)
        x_in, c = block0_input_and_c(model, x_t, t, labels, args.precision)
        base_out = finish_from_block0(model, x_in, c, args.precision)
        base_eps = torch.chunk(base_out, 2, dim=1)[0]
        sqrt_ac = diffusion.extract(diffusion.sqrt_ac, t, x_t.shape)
        sqrt_om = diffusion.extract(diffusion.sqrt_one_minus_ac, t, x_t.shape)
        base_pred_x0 = (x_t - sqrt_om * base_eps) / sqrt_ac
        base_eps_mse = (base_eps - noise).pow(2).flatten(1).mean(dim=1)
        base_x0_mse = (base_pred_x0 - x0).pow(2).flatten(1).mean(dim=1)
        baselines["eps_mse_to_noise"][ti] = base_eps_mse.cpu()
        baselines["pred_x0_mse_to_clean"][ti] = base_x0_mse.cpu()

        dense, _, _ = encode_dense(sae, x_in.flatten(0, 1))
        for f0 in range(0, sae.dict_size, args.features_per_forward):
            f1 = min(f0 + args.features_per_forward, sae.dict_size)
            states = []
            image_ids = []
            feature_ids = []
            for ii in range(args.num_images):
                x_img = x_in[ii]
                coeff = dense[ii, :, f0:f1].float()
                for local_f, fid in enumerate(range(f0, f1)):
                    delta = -coeff[:, local_f].unsqueeze(1) * decoder[fid].unsqueeze(0)
                    states.append(x_img + delta)
                    image_ids.append(ii)
                    feature_ids.append(fid)
                    metrics["active_frac"][ti, ii, fid] = (coeff[:, local_f] > 0).float().mean().cpu()
                    metrics["coeff_abs_mean"][ti, ii, fid] = coeff[:, local_f].abs().mean().cpu()
                    metrics["coeff_max"][ti, ii, fid] = coeff[:, local_f].max().cpu()

            states = torch.stack(states, dim=0)
            image_ids_t = torch.tensor(image_ids, device=device, dtype=torch.long)
            c_batch = c[image_ids_t]
            for b0 in range(0, states.shape[0], args.forward_batch_size):
                b1 = min(b0 + args.forward_batch_size, states.shape[0])
                out = finish_from_block0(model, states[b0:b1], c_batch[b0:b1], args.precision)
                eps = torch.chunk(out, 2, dim=1)[0]
                ids = image_ids_t[b0:b1]
                local_base_eps = base_eps[ids]
                local_x_t = x_t[ids]
                local_x0 = x0[ids]
                local_t = t[ids]
                local_sqrt_ac = diffusion.extract(diffusion.sqrt_ac, local_t, local_x_t.shape)
                local_sqrt_om = diffusion.extract(diffusion.sqrt_one_minus_ac, local_t, local_x_t.shape)
                pred_x0 = (local_x_t - local_sqrt_om * eps) / local_sqrt_ac
                eps_delta = eps - local_base_eps
                pred_x0_delta = pred_x0 - base_pred_x0[ids]
                vals = {
                    "eps_delta_mse": eps_delta.pow(2).flatten(1).mean(dim=1).cpu(),
                    "eps_delta_l2": eps_delta.flatten(1).norm(dim=1).cpu(),
                    "pred_x0_delta_mse": pred_x0_delta.pow(2).flatten(1).mean(dim=1).cpu(),
                    "pred_x0_mse_to_clean": (pred_x0 - local_x0).pow(2).flatten(1).mean(dim=1).cpu(),
                }
                vals["pred_x0_mse_to_clean_delta"] = vals["pred_x0_mse_to_clean"] - base_x0_mse[ids].cpu()
                for local_i, global_i in enumerate(range(b0, b1)):
                    ii = image_ids[global_i]
                    fid = feature_ids[global_i]
                    for key, tensor in vals.items():
                        metrics[key][ti, ii, fid] = tensor[local_i]

            if f1 % 512 == 0 or f1 == sae.dict_size:
                print(f"t={t_int} features=[0,{f1}) elapsed={time.time() - start:.1f}s", flush=True)

    out_path = os.path.join(args.out_dir, "block0_input_onoff_ablation.pt")
    torch.save({
        "metrics": metrics,
        "baselines": baselines,
        "t_values": tuple(args.t_values),
        "feature_ids": torch.arange(sae.dict_size),
        "labels": labels.cpu(),
        "cifar_train_indices": data_indices,
        "config": vars(args),
        "intervention": "For each block0_input SAE feature, set the actual TopK coefficient to zero at all 256 tokens for one image/timestep, preserving SAE residual via x + decode(z_ablate)-decode(z_orig).",
        "primary_local_metric": "pred_x0_delta_mse and eps_delta_mse versus original unperturbed prediction",
        "clean_image_metric": "pred_x0_mse_to_clean and pred_x0_mse_to_clean_delta versus true CIFAR x0",
    }, out_path)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
