#!/usr/bin/env python3
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from train_block0_write_saes import TopKSAE
from train_dit_cifar import DiT, Diffusion, modulate


SCALES = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
T_VALUES = (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999)


def parse_ints(s):
    return tuple(int(x) for x in s.split(",") if x.strip())


def parse_floats(s):
    return tuple(float(x) for x in s.split(",") if x.strip())


def load_dit(path, device):
    model = DiT(learn_sigma=True).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_saes(path, device, input_dim=384, dict_size=6144, k=32):
    ckpt = torch.load(path, map_location=device)
    attn = TopKSAE(input_dim=input_dim, dict_size=dict_size, k=k).to(device)
    mlp = TopKSAE(input_dim=input_dim, dict_size=dict_size, k=k).to(device)
    attn.load_state_dict(ckpt["attn_sae"])
    mlp.load_state_dict(ckpt["mlp_sae"])
    attn.eval()
    mlp.eval()
    for sae in (attn, mlp):
        for p in sae.parameters():
            p.requires_grad_(False)
    return {"attn": attn, "mlp": mlp}


def encode_topk_dense(sae, x):
    pre = sae.encoder(x - sae.dec_bias)
    acts = F.relu(pre)
    vals, idx = torch.topk(acts, sae.k, dim=-1)
    dense = torch.zeros(x.shape[0], sae.dict_size, device=x.device, dtype=x.dtype)
    dense.scatter_(1, idx, vals)
    return dense


@torch.no_grad()
def block0_parts(model, x_t, t, labels, precision):
    enabled = x_t.device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type=x_t.device.type, dtype=torch.bfloat16, enabled=enabled):
        x = model.x_embedder(x_t).flatten(2).transpose(1, 2) + model.pos_embed
        c = model.t_embedder(t) + model.y_embedder(labels)
        block = model.blocks[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaLN(c).chunk(6, dim=1)
        attn_in = modulate(block.norm1(x), shift_msa, scale_msa)
        attn_out = block.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        attn_write = gate_msa.unsqueeze(1) * attn_out
        x_after_attn = x + attn_write
        mlp_in = modulate(block.norm2(x_after_attn), shift_mlp, scale_mlp)
        mlp_out = block.mlp(mlp_in)
        mlp_write = gate_mlp.unsqueeze(1) * mlp_out
        x_after_block0 = x_after_attn + mlp_write
    return {
        "x_before": x.float(),
        "c": c.float(),
        "attn_write": attn_write.float(),
        "x_after_attn": x_after_attn.float(),
        "mlp_write": mlp_write.float(),
        "x_after_block0": x_after_block0.float(),
    }


@torch.no_grad()
def mlp_from_after_attn(model, x_after_attn, c, precision):
    enabled = x_after_attn.device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type=x_after_attn.device.type, dtype=torch.bfloat16, enabled=enabled):
        block = model.blocks[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaLN(c).chunk(6, dim=1)
        del shift_msa, scale_msa, gate_msa
        mlp_in = modulate(block.norm2(x_after_attn), shift_mlp, scale_mlp)
        mlp_out = block.mlp(mlp_in)
        mlp_write = gate_mlp.unsqueeze(1) * mlp_out
    return mlp_write.float()


@torch.no_grad()
def finish_from_block1(model, x, c, precision):
    enabled = x.device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=enabled):
        for block in model.blocks[1:]:
            x = block(x, c)
        out = model.unpatchify(model.final_layer(x, c))
    return out.float()


@torch.no_grad()
def make_starting_batch(data_dir, num_images, seed, device):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
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
def calibrate_feature_stats(args, model, diffusion, saes, device):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(args.data_dir, train=True, transform=transform, download=True)
    gen = torch.Generator().manual_seed(args.seed + 17)
    indices = torch.randperm(len(dataset), generator=gen)[: args.calib_images].tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.calib_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    sums = {name: torch.zeros(sae.dict_size, device=device) for name, sae in saes.items()}
    sumsqs = {name: torch.zeros(sae.dict_size, device=device) for name, sae in saes.items()}
    total_tokens = 0
    t_values = torch.tensor(args.t_values, device=device, dtype=torch.long)

    for batch_i, (x0, labels) in enumerate(loader):
        x0 = x0.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        t = t_values[torch.arange(x0.shape[0], device=device) % len(t_values)]
        noise = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, noise)
        parts = block0_parts(model, x_t, t, labels, args.precision)
        writes = {
            "attn": parts["attn_write"].flatten(0, 1),
            "mlp": parts["mlp_write"].flatten(0, 1),
        }
        for name, write in writes.items():
            sae = saes[name]
            pre = sae.encoder(write - sae.dec_bias)
            acts = F.relu(pre)
            vals, idx = torch.topk(acts, sae.k, dim=-1)
            sums[name].scatter_add_(0, idx.reshape(-1), vals.reshape(-1))
            sumsqs[name].scatter_add_(0, idx.reshape(-1), vals.reshape(-1).pow(2))
        total_tokens += x0.shape[0] * 256
        if (batch_i + 1) % 10 == 0:
            print(f"calibration batches={batch_i + 1} tokens={total_tokens}", flush=True)

    stats = {}
    for name, sae in saes.items():
        mean = sums[name] / total_tokens
        var = (sumsqs[name] / total_tokens - mean.pow(2)).clamp_min(0)
        stats[name] = {
            "mean": mean.cpu(),
            "std": var.sqrt().clamp_min(1e-8).cpu(),
            "tokens": total_tokens,
        }
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(
        {
            "stats": stats,
            "config": vars(args),
            "note": "Feature std includes inactive-token zeros from the TopK sparse code.",
        },
        args.feature_stats,
    )
    print(f"saved feature stats to {args.feature_stats}", flush=True)


def init_metric_buffers(num_streams, num_t, num_images, num_features, num_scales):
    shape = (num_streams, num_t, num_images, num_features, num_scales)
    return {
        "eps_delta_mse": torch.empty(shape, dtype=torch.float32),
        "eps_delta_l2": torch.empty(shape, dtype=torch.float32),
        "eps_mse_to_noise": torch.empty(shape, dtype=torch.float32),
        "eps_mse_to_noise_delta": torch.empty(shape, dtype=torch.float32),
        "pred_x0_mse": torch.empty(shape, dtype=torch.float32),
        "pred_x0_mse_delta": torch.empty(shape, dtype=torch.float32),
    }


@torch.no_grad()
def run_ablation(args, model, diffusion, saes, device):
    stats_ckpt = torch.load(args.feature_stats, map_location="cpu")
    feature_std = {name: stats_ckpt["stats"][name]["std"].to(device) for name in saes}
    x0, labels, data_indices = make_starting_batch(args.data_dir, args.num_images, args.seed, device)

    streams = tuple(s for s in args.streams.split(",") if s)
    scales = torch.tensor(args.scales, device=device, dtype=torch.float32)
    t_values = tuple(args.t_values)
    feature_ids = torch.arange(args.feature_start, args.feature_end, device=device, dtype=torch.long)
    metrics = init_metric_buffers(len(streams), len(t_values), args.num_images, len(feature_ids), len(scales))
    baselines = {
        "eps_mse_to_noise": torch.empty((len(t_values), args.num_images), dtype=torch.float32),
        "pred_x0_mse": torch.empty((len(t_values), args.num_images), dtype=torch.float32),
    }

    gen = torch.Generator(device=device).manual_seed(args.seed + 1000)
    decoder_cache = {name: saes[name].decoder[feature_ids].float() for name in streams}
    start = time.time()

    for ti, t_int in enumerate(t_values):
        t = torch.full((args.num_images,), t_int, device=device, dtype=torch.long)
        noise = torch.randn(x0.shape, device=device, generator=gen)
        x_t = diffusion.q_sample(x0, t, noise)
        parts = block0_parts(model, x_t, t, labels, args.precision)
        base_out = finish_from_block1(model, parts["x_after_block0"], parts["c"], args.precision)
        base_eps = torch.chunk(base_out, 2, dim=1)[0]
        sqrt_ac = diffusion.extract(diffusion.sqrt_ac, t, x_t.shape)
        sqrt_om = diffusion.extract(diffusion.sqrt_one_minus_ac, t, x_t.shape)
        base_pred_x0 = (x_t - sqrt_om * base_eps) / sqrt_ac
        base_eps_mse = (base_eps - noise).pow(2).flatten(1).mean(dim=1)
        base_x0_mse = (base_pred_x0 - x0).pow(2).flatten(1).mean(dim=1)
        baselines["eps_mse_to_noise"][ti] = base_eps_mse.cpu()
        baselines["pred_x0_mse"][ti] = base_x0_mse.cpu()

        dense_codes = {}
        for stream in streams:
            write = parts[f"{stream}_write"].flatten(0, 1)
            dense_codes[stream] = encode_topk_dense(saes[stream], write).reshape(args.num_images, 256, -1)

        for si, stream in enumerate(streams):
            decoder = decoder_cache[stream]
            std = feature_std[stream][feature_ids]
            cur_codes = dense_codes[stream][..., feature_ids].float()
            for f0 in range(0, len(feature_ids), args.features_per_forward):
                f1 = min(f0 + args.features_per_forward, len(feature_ids))
                chunk_features = f1 - f0
                for sc0 in range(0, len(scales), args.scales_per_forward):
                    sc1 = min(sc0 + args.scales_per_forward, len(scales))
                    chunk_scales = sc1 - sc0
                    states = []
                    image_ids = []
                    feature_offsets = []
                    scale_offsets = []
                    for ii in range(args.num_images):
                        current = cur_codes[ii, :, f0:f1]
                        for local_f in range(chunk_features):
                            feature_decoder = decoder[f0 + local_f]
                            for local_s in range(chunk_scales):
                                new_coeff = scales[sc0 + local_s] * std[f0 + local_f]
                                delta = (new_coeff - current[:, local_f]).unsqueeze(1) * feature_decoder.unsqueeze(0)
                                if stream == "attn":
                                    x_after_attn = parts["x_after_attn"][ii] + delta
                                    states.append(x_after_attn)
                                else:
                                    x_after_block0 = parts["x_after_block0"][ii] + delta
                                    states.append(x_after_block0)
                                image_ids.append(ii)
                                feature_offsets.append(f0 + local_f)
                                scale_offsets.append(sc0 + local_s)

                    states = torch.stack(states, dim=0)
                    image_ids_t = torch.tensor(image_ids, device=device, dtype=torch.long)
                    c = parts["c"][image_ids_t]
                    if stream == "attn":
                        mlp_write = mlp_from_after_attn(model, states, c, args.precision)
                        states = states + mlp_write

                    for b0 in range(0, states.shape[0], args.forward_batch_size):
                        b1 = min(b0 + args.forward_batch_size, states.shape[0])
                        out = finish_from_block1(model, states[b0:b1], c[b0:b1], args.precision)
                        eps = torch.chunk(out, 2, dim=1)[0]
                        ids = image_ids_t[b0:b1]
                        local_base_eps = base_eps[ids]
                        local_noise = noise[ids]
                        local_x_t = x_t[ids]
                        local_x0 = x0[ids]
                        local_t = t[ids]
                        local_sqrt_ac = diffusion.extract(diffusion.sqrt_ac, local_t, local_x_t.shape)
                        local_sqrt_om = diffusion.extract(diffusion.sqrt_one_minus_ac, local_t, local_x_t.shape)
                        pred_x0 = (local_x_t - local_sqrt_om * eps) / local_sqrt_ac
                        eps_delta = eps - local_base_eps
                        vals = {
                            "eps_delta_mse": eps_delta.pow(2).flatten(1).mean(dim=1).cpu(),
                            "eps_delta_l2": eps_delta.flatten(1).norm(dim=1).cpu(),
                            "eps_mse_to_noise": (eps - local_noise).pow(2).flatten(1).mean(dim=1).cpu(),
                            "pred_x0_mse": (pred_x0 - local_x0).pow(2).flatten(1).mean(dim=1).cpu(),
                        }
                        vals["eps_mse_to_noise_delta"] = vals["eps_mse_to_noise"] - base_eps_mse[ids].cpu()
                        vals["pred_x0_mse_delta"] = vals["pred_x0_mse"] - base_x0_mse[ids].cpu()
                        for local_i, global_i in enumerate(range(b0, b1)):
                            img_i = image_ids[global_i]
                            feat_i = feature_offsets[global_i]
                            scale_i = scale_offsets[global_i]
                            for key, tensor in vals.items():
                                metrics[key][si, ti, img_i, feat_i, scale_i] = tensor[local_i]

            print(
                f"t={t_int} stream={stream} done feature_range=[{args.feature_start},{args.feature_end}) "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"block0_ablation_features_{args.feature_start:04d}_{args.feature_end:04d}.pt")
    torch.save(
        {
            "metrics": metrics,
            "baselines": baselines,
            "streams": streams,
            "t_values": t_values,
            "scales": tuple(args.scales),
            "feature_ids": feature_ids.cpu(),
            "cifar_train_indices": data_indices,
            "labels": labels.cpu(),
            "config": vars(args),
            "intervention": "For each feature and image, replace that feature's TopK SAE coefficient at all 256 tokens with scale * feature_std, then continue the model.",
        },
        out_path,
    )
    print(f"saved {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/dit_s2_cifar10/last.pt")
    parser.add_argument("--sae-ckpt", default="runs/sae_block0_writes_16x_k32/last.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="runs/block0_feature_ablation_16x_k32")
    parser.add_argument("--feature-stats", default="runs/block0_feature_ablation_16x_k32/feature_stats.pt")
    parser.add_argument("--feature-start", type=int, default=0)
    parser.add_argument("--feature-end", type=int, default=6144)
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--t-values", type=parse_ints, default=T_VALUES)
    parser.add_argument("--scales", type=parse_floats, default=SCALES)
    parser.add_argument("--streams", default="attn,mlp")
    parser.add_argument("--features-per-forward", type=int, default=4)
    parser.add_argument("--scales-per-forward", type=int, default=3)
    parser.add_argument("--forward-batch-size", type=int, default=96)
    parser.add_argument("--calib-images", type=int, default=4096)
    parser.add_argument("--calib-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--only-stats", action="store_true")
    args = parser.parse_args()

    if args.feature_end <= args.feature_start:
        raise ValueError("--feature-end must be greater than --feature-start")
    if args.feature_start < 0 or args.feature_end > 6144:
        raise ValueError("feature range must be inside [0, 6144]")
    if max(args.t_values) >= args.timesteps:
        raise ValueError(f"t-values must be < timesteps={args.timesteps}; use 999 for the endpoint")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = load_dit(args.ckpt, device)
    saes = load_saes(args.sae_ckpt, device)
    diffusion = Diffusion(args.timesteps, device=device)

    if args.only_stats:
        calibrate_feature_stats(args, model, diffusion, saes, device)
        return
    if not os.path.exists(args.feature_stats):
        raise FileNotFoundError(f"missing feature stats: {args.feature_stats}")
    run_ablation(args, model, diffusion, saes, device)


if __name__ == "__main__":
    main()
