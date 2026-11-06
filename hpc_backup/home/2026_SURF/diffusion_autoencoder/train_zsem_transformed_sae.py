#!/usr/bin/env python3
"""Train a TopK z_sem SAE using only a fixed DiffAE output geometry."""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


OBJECTIVES = (
    "latent_mse",
    "gradient",
    "laplacian",
    "fourier_bands",
    "fourier_low",
    "fourier_mid",
    "fourier_high",
    "steerable",
    "weighted_laplacian",
    "weighted_fourier_bands",
    "weighted_fourier_low",
    "weighted_fourier_mid",
    "weighted_fourier_high",
)


def load_diffae(repo, checkpoint, device):
    sys.path.insert(0, os.path.abspath(repo))
    from experiment import LitModel
    from templates import ffhq128_autoenc_130M

    conf = ffhq128_autoenc_130M()
    conf.pretrain = None
    conf.latent_infer_path = None
    lit = LitModel(conf)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    lit.load_state_dict(state["state_dict"], strict=False)
    lit.eval().to(device)
    for parameter in lit.parameters():
        parameter.requires_grad_(False)
    return lit


class TopKSAE(nn.Module):
    def __init__(self, d, h, k):
        super().__init__()
        self.d, self.h, self.k = d, h, k
        self.encoder = nn.Linear(d, h)
        self.decoder = nn.Parameter(torch.empty(h, d))
        self.dec_bias = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.decoder, std=d ** -0.5)
        self.normalize_decoder()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder)
            self.encoder.bias.zero_()

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.div_(self.decoder.norm(dim=1, keepdim=True).clamp_min(1e-6))

    def forward(self, x):
        activations = F.relu(self.encoder(x - self.dec_bias))
        values, indices = torch.topk(activations, self.k, dim=-1)
        reconstruction = (
            values.unsqueeze(-1) * self.decoder[indices]
        ).sum(1) + self.dec_bias
        return reconstruction, values, indices


@torch.no_grad()
def rescue_dead_features(sae, inputs, reconstruction, objective_loss, last_fire,
                         step, dead_after, max_rescue):
    dead = torch.where(last_fire < step - dead_after)[0]
    if not len(dead):
        return 0
    count = min(len(dead), len(inputs), max_rescue)
    examples = torch.argsort(objective_loss, descending=True)[:count]
    residual = inputs[examples] - reconstruction[examples]
    directions = F.normalize(residual, dim=1)
    features = dead[:count]
    sae.decoder[features].copy_(directions)
    sae.encoder.weight[features].copy_(directions)
    centered = inputs[examples] - sae.dec_bias
    response = (directions * centered).sum(1)
    sae.encoder.bias[features].copy_(-response + 0.1)
    last_fire[features] = step
    return count


def finite_differences(image):
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    return [dx, dy]


def depthwise_filter(image, kernel):
    channels = image.shape[1]
    kernel = kernel.to(device=image.device, dtype=image.dtype)
    kernel = kernel[None, None].expand(channels, 1, -1, -1)
    padded = F.pad(image, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, kernel, groups=channels)


def laplacian_components(image):
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=image.device,
        dtype=image.dtype,
    )
    return [depthwise_filter(image, kernel)]


def fourier_band_components(image):
    height, width = image.shape[-2:]
    spectrum = torch.fft.rfft2(image, norm="ortho")
    fy = torch.fft.fftfreq(height, device=image.device) * height
    fx = torch.fft.rfftfreq(width, device=image.device) * width
    radius = (fy[:, None].square() + fx[None, :].square()).sqrt()
    base = max(1.0, min(height, width) / 64)
    edges = (0.0, base, 2 * base, 4 * base, 8 * base, 16 * base, float("inf"))
    return [
        spectrum[..., (radius >= low) & (radius < high)]
        for low, high in zip(edges[:-1], edges[1:])
    ]


def gaussian_blur(image):
    weights = torch.tensor(
        [1.0, 4.0, 6.0, 4.0, 1.0], device=image.device, dtype=image.dtype
    )
    weights = weights / weights.sum()
    channels = image.shape[1]
    horizontal = weights[None, None, None, :].expand(channels, 1, 1, -1)
    vertical = weights[None, None, :, None].expand(channels, 1, -1, 1)
    image = F.conv2d(F.pad(image, (2, 2, 0, 0), mode="reflect"), horizontal,
                     groups=channels)
    return F.conv2d(F.pad(image, (0, 0, 2, 2), mode="reflect"), vertical,
                    groups=channels)


def steerable_components(image):
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ) / 8.0
    sobel_y = sobel_x.t().contiguous()
    angles = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)
    components = []
    current = image
    for scale in range(4):
        gx = depthwise_filter(current, sobel_x)
        gy = depthwise_filter(current, sobel_y)
        components.extend(math.cos(theta) * gx + math.sin(theta) * gy
                          for theta in angles)
        if scale != 3:
            current = F.avg_pool2d(gaussian_blur(current), 2)
    return components


def transform_components(objective, image):
    if objective == "gradient":
        return finite_differences(image)
    if objective == "laplacian":
        return laplacian_components(image)
    if objective.startswith("fourier_"):
        components = fourier_band_components(image)
        if objective == "fourier_bands":
            return components
        group = {
            "fourier_low": slice(0, 2),
            "fourier_mid": slice(2, 4),
            "fourier_high": slice(4, 6),
        }[objective]
        return components[group]
    if objective == "steerable":
        return steerable_components(image)
    raise ValueError(objective)


def component_power(component):
    return component.abs().square().flatten(1).mean(1)


def transformed_loss(objective, error, bin_ids, powers):
    losses = torch.stack(
        [component_power(component) for component in transform_components(objective, error)],
        dim=1,
    )
    selected_powers = powers[bin_ids]
    return (losses / selected_powers).mean(1)


def transform_objective_name(objective):
    return objective.removeprefix("weighted_")


def timestep_batch(step, batch, seed, image_shape, device):
    generator = torch.Generator().manual_seed(seed + step)
    bin_ids = (step * batch + torch.arange(batch, device=device)) % 10
    offsets = torch.randint(0, 100, (batch,), generator=generator).to(device)
    timesteps = (bin_ids * 100 + offsets).clamp_max(999).long()
    noise = torch.randn(image_shape, generator=generator).to(device)
    return bin_ids, timesteps, noise


@torch.no_grad()
def calibrate(lit, objective, images, codes, batch_size, seed, device):
    sums = None
    counts = torch.zeros(10, dtype=torch.long)
    for start in range(0, len(images), batch_size):
        image = images[start:start + batch_size].to(device).float().div(255)
        code = codes[start:start + batch_size].to(device)
        bin_ids, timesteps, noise = timestep_batch(
            start // batch_size, len(image), seed, image.shape, device
        )
        x_t = lit.sampler.q_sample(image.mul(2).sub(1), timesteps, noise=noise)
        prediction = lit.sampler.p_mean_variance(
            lit.ema_model,
            x_t,
            timesteps,
            clip_denoised=True,
            model_kwargs={"cond": code},
        )["pred_xstart"].add(1).div(2)
        batch_powers = torch.stack(
            [component_power(component)
             for component in transform_components(objective, prediction)],
            dim=1,
        ).double().cpu()
        if sums is None:
            sums = torch.zeros(10, batch_powers.shape[1], dtype=torch.float64)
        cpu_bins = bin_ids.cpu()
        for bin_id in cpu_bins.unique().tolist():
            selected = batch_powers[cpu_bins == bin_id]
            sums[bin_id] += selected.sum(0)
            counts[bin_id] += len(selected)
        if start == 0 or start + len(image) == len(images) or start % (20 * batch_size) == 0:
            print(f"calibration {min(start + len(image), len(images))}/{len(images)}", flush=True)
    powers = sums / counts.clamp_min(1)[:, None]
    floor = torch.quantile(powers[powers > 0], 0.01).clamp_min(1e-10)
    return powers.clamp_min(floor).float().to(device), counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=OBJECTIVES, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dict-mult", type=int, default=32)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--calibration-examples", type=int, default=2048)
    parser.add_argument("--reinit-every", type=int, default=1000)
    parser.add_argument("--dead-after", type=int, default=3000)
    parser.add_argument("--max-rescue", type=int, default=256)
    parser.add_argument("--transform-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--split-seed", type=int, default=2030)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    images = data["images_uint8"]
    codes = data["codes"].float()
    split_generator = torch.Generator().manual_seed(args.split_seed)
    permutation = torch.randperm(len(codes), generator=split_generator)
    split = int(0.9 * len(codes))
    train_indices, valid_indices = permutation[:split], permutation[split:]
    mean = codes[train_indices].mean(0)
    std = codes[train_indices].std(0).clamp_min(1e-6)
    device_mean = mean.to(device)
    device_std = std.to(device)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(images[train_indices], codes[train_indices]),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    iterator = iter(loader)

    lit = None
    powers = None
    power_counts = None
    if args.objective != "latent_mse":
        lit = load_diffae(args.repo, args.checkpoint, device)
        calibration_count = min(len(train_indices), args.calibration_examples)
        calibration_indices = train_indices[:calibration_count]
        powers, power_counts = calibrate(
            lit,
            transform_objective_name(args.objective),
            images[calibration_indices],
            codes[calibration_indices],
            max(args.batch_size, 4),
            args.seed + 100_000,
            device,
        )
        torch.save(
            {"component_powers": powers.cpu(), "counts": power_counts,
             "objective": args.objective},
            os.path.join(args.out_dir, "transform_stats.pt"),
        )

    # DiffAE construction may reset the global RNG. Re-seed immediately before
    # SAE initialization so independent SAE seeds actually start differently.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    sae = TopKSAE(512, args.dict_mult * 512, args.k).to(device)
    with torch.no_grad():
        sae.dec_bias.copy_(((codes[train_indices] - mean) / std).mean(0).to(device))
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    feature_counts = torch.zeros(sae.h, dtype=torch.long, device=device)
    last_fire = torch.zeros(sae.h, dtype=torch.long, device=device)
    rescued_total = 0
    history = []

    for step in range(1, args.steps + 1):
        try:
            image_u8, raw_code = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            image_u8, raw_code = next(iterator)
        image = image_u8.to(device).float().div(255)
        raw_code = raw_code.to(device)
        standardized = (raw_code - device_mean) / device_std
        reconstruction, values, indices = sae(standardized)
        latent_per_example = (reconstruction - standardized).square().mean(1)
        if args.objective == "latent_mse":
            per_example_loss = latent_per_example
        else:
            reconstructed_code = reconstruction * device_std + device_mean
            bin_ids, timesteps, noise = timestep_batch(
                step, len(image), args.seed, image.shape, device
            )
            x_t = lit.sampler.q_sample(image.mul(2).sub(1), timesteps, noise=noise)
            with torch.no_grad():
                clean_prediction = lit.sampler.p_mean_variance(
                    lit.ema_model,
                    x_t,
                    timesteps,
                    clip_denoised=True,
                    model_kwargs={"cond": raw_code},
                )["pred_xstart"].add(1).div(2)
            reconstructed_prediction = lit.sampler.p_mean_variance(
                lit.ema_model,
                x_t,
                timesteps,
                clip_denoised=True,
                model_kwargs={"cond": reconstructed_code},
            )["pred_xstart"].add(1).div(2)
            geometry_loss = transformed_loss(
                transform_objective_name(args.objective),
                clean_prediction - reconstructed_prediction,
                bin_ids,
                powers,
            )
            per_example_loss = (
                latent_per_example + args.transform_weight * geometry_loss
                if args.objective.startswith("weighted_") else geometry_loss
            )
        loss = per_example_loss.mean()
        latent_mse = F.mse_loss(reconstruction, standardized)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip)
        if args.warmup and step <= args.warmup:
            for group in optimizer.param_groups:
                group["lr"] = args.lr * step / args.warmup
        optimizer.step()
        sae.normalize_decoder()
        active = values > 0
        feature_counts.scatter_add_(
            0, indices[active], torch.ones_like(indices[active], dtype=torch.long)
        )
        fired = indices[active].unique()
        last_fire[fired] = step
        rescued = 0
        if args.reinit_every and step % args.reinit_every == 0:
            rescued = rescue_dead_features(
                sae,
                standardized,
                reconstruction.detach(),
                per_example_loss.detach(),
                last_fire,
                step,
                args.dead_after,
                args.max_rescue,
            )
            rescued_total += rescued

        if step == 1 or step % 100 == 0:
            record = {
                "step": step,
                "transformed_loss": float(loss.detach()),
                "latent_mse_monitor_only": float(latent_mse.detach()),
                "active_mean": float(active.sum(1).float().mean()),
                "used_features": int((feature_counts > 0).sum()),
                "dead_features": int((last_fire < step - args.dead_after).sum()),
                "rescued": rescued,
                "gradient_norm_before_clip": float(gradient_norm),
            }
            history.append(record)
            print(record, flush=True)

    sae.eval()
    valid_transformed_sum = 0.0
    valid_pixel_sum = 0.0
    valid_latent_sum = 0.0
    valid_count = 0
    valid_loader = DataLoader(
        TensorDataset(images[valid_indices], codes[valid_indices]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for valid_step, (image_u8, raw_code) in enumerate(valid_loader):
            image = image_u8.to(device).float().div(255)
            raw_code = raw_code.to(device)
            standardized = (raw_code - device_mean) / device_std
            reconstruction, _, indices = sae(standardized)
            latent = (standardized - reconstruction).square().mean(1)
            if args.objective == "latent_mse":
                transformed = latent
                pixel = torch.full_like(transformed, float("nan"))
            else:
                reconstructed_code = reconstruction * device_std + device_mean
                bin_ids, timesteps, noise = timestep_batch(
                    1_000_000 + valid_step,
                    len(image),
                    args.seed,
                    image.shape,
                    device,
                )
                x_t = lit.sampler.q_sample(image.mul(2).sub(1), timesteps, noise=noise)
                clean_prediction = lit.sampler.p_mean_variance(
                    lit.ema_model,
                    x_t,
                    timesteps,
                    clip_denoised=True,
                    model_kwargs={"cond": raw_code},
                )["pred_xstart"].add(1).div(2)
                reconstructed_prediction = lit.sampler.p_mean_variance(
                    lit.ema_model,
                    x_t,
                    timesteps,
                    clip_denoised=True,
                    model_kwargs={"cond": reconstructed_code},
                )["pred_xstart"].add(1).div(2)
                geometry_loss = transformed_loss(
                    transform_objective_name(args.objective),
                    clean_prediction - reconstructed_prediction,
                    bin_ids,
                    powers,
                )
                transformed = (
                    latent + args.transform_weight * geometry_loss
                    if args.objective.startswith("weighted_") else geometry_loss
                )
                pixel = (clean_prediction - reconstructed_prediction).square().flatten(1).mean(1)
            valid_transformed_sum += float(transformed.sum())
            if args.objective != "latent_mse":
                valid_pixel_sum += float(pixel.sum())
            valid_latent_sum += float(latent.sum())
            valid_count += len(image)

    result = {
        "objective": args.objective,
        "pure_transformed_objective": (
            args.objective != "latent_mse"
            and not args.objective.startswith("weighted_")
        ),
        "pixel_or_latent_anchor_coefficient": (
            1.0 if args.objective.startswith("weighted_") else 0.0
        ),
        "examples": len(codes),
        "train_examples": len(train_indices),
        "valid_examples": len(valid_indices),
        "used_features": int((feature_counts > 0).sum()),
        "dead_features_at_end": int((last_fire < args.steps - args.dead_after).sum()),
        "rescued_total": rescued_total,
        "valid_transformed_loss": valid_transformed_sum / valid_count,
        "valid_pixel_mse_monitor_only": (
            valid_pixel_sum / valid_count if args.objective != "latent_mse" else None
        ),
        "valid_latent_mse_monitor_only": valid_latent_sum / valid_count,
        "component_powers": powers.cpu().tolist() if powers is not None else None,
        "history": history,
        "config": vars(args),
    }
    torch.save(
        {
            "model": sae.state_dict(),
            "input_mean": mean,
            "input_std": std,
            "config": vars(args),
            "representation_training": result,
        },
        os.path.join(args.out_dir, "last.pt"),
    )
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
