#!/usr/bin/env python3
"""Train flat or translation-shared TopK SAEs on a frozen spatial AE map."""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from spatial_conv_autoencoder import load_spatial_autoencoder
from vector_conv_autoencoder import load_vector_autoencoder
from train_zsem_transformed_sae import component_power, transform_components


OBJECTIVES = (
    "latent_mse",
    "laplacian",
    "fourier_bands",
    "fourier_low",
    "fourier_mid",
    "fourier_high",
    "weighted_laplacian",
    "weighted_fourier_bands",
    "weighted_fourier_low",
    "weighted_fourier_mid",
    "weighted_fourier_high",
)


class FlatTopKSAE(nn.Module):
    def __init__(self, input_shape, feature_count, k):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.d = int(torch.tensor(input_shape).prod())
        self.h = feature_count
        self.k = k
        self.encoder = nn.Parameter(torch.empty(feature_count, self.d))
        self.encoder_bias = nn.Parameter(torch.zeros(feature_count))
        self.decoder = nn.Parameter(torch.empty(feature_count, self.d))
        self.dec_bias = nn.Parameter(torch.zeros(self.d))
        nn.init.normal_(self.decoder, std=self.d ** -0.5)
        self.normalize_decoder()
        with torch.no_grad():
            self.encoder.copy_(self.decoder)

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.div_(self.decoder.norm(dim=1, keepdim=True).clamp_min(1e-6))

    def forward(self, x):
        flat = x.flatten(1)
        activations = F.relu(F.linear(flat - self.dec_bias, self.encoder, self.encoder_bias))
        values, indices = torch.topk(activations, self.k, dim=1)
        reconstruction = (
            values.unsqueeze(-1) * self.decoder[indices]
        ).sum(1) + self.dec_bias
        return reconstruction.view(-1, *self.input_shape), values, indices


class ConvTopKSAE(nn.Module):
    """Shared 1x1 dictionary with TopK selected independently at each position."""

    def __init__(self, input_shape, feature_count, k):
        super().__init__()
        channels, height, width = input_shape
        self.input_shape = tuple(input_shape)
        self.d = channels
        self.h = feature_count
        self.k = k
        self.positions = height * width
        self.encoder = nn.Parameter(torch.empty(feature_count, channels))
        self.encoder_bias = nn.Parameter(torch.zeros(feature_count))
        self.decoder = nn.Parameter(torch.empty(feature_count, channels))
        self.dec_bias = nn.Parameter(torch.zeros(channels))
        nn.init.normal_(self.decoder, std=channels ** -0.5)
        self.normalize_decoder()
        with torch.no_grad():
            self.encoder.copy_(self.decoder)

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.div_(self.decoder.norm(dim=1, keepdim=True).clamp_min(1e-6))

    def forward(self, x):
        batch, channels, height, width = x.shape
        centered = x.flatten(2) - self.dec_bias[None, :, None]
        activations = torch.einsum("bcp,fc->bfp", centered, self.encoder)
        activations = F.relu(activations + self.encoder_bias[None, :, None])
        values, indices = torch.topk(activations, self.k, dim=1)
        atoms = self.decoder[indices]
        reconstruction = (values.unsqueeze(-1) * atoms).sum(1).transpose(1, 2)
        reconstruction = reconstruction + self.dec_bias[None, :, None]
        return reconstruction.view(batch, channels, height, width), values, indices

    def feature_ids(self, indices):
        return indices


def feature_ids(sae, indices):
    return sae.feature_ids(indices) if isinstance(sae, ConvTopKSAE) else indices


def transformed_loss(objective, error, powers):
    losses = torch.stack(
        [component_power(component) for component in transform_components(objective, error)],
        dim=1,
    )
    return (losses / powers[None]).mean(1)


def transform_objective_name(objective):
    return objective.removeprefix("weighted_")


@torch.no_grad()
def calibrate(
    autoencoder, objective, raw_latents, indices, batch_size, device,
    base_latents=None,
):
    component_sum = None
    count = 0
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        latent = raw_latents[selected].to(device).float()
        decode_latent = latent
        if base_latents is not None:
            base = base_latents[selected].to(device).float()
            decode_latent = base + latent
            image = (
                autoencoder.decode(decode_latent)
                - autoencoder.decode(base)
            ).div(2)
        else:
            image = autoencoder.decode(decode_latent).add(1).div(2)
        batch_power = torch.stack(
            [component_power(component) for component in transform_components(objective, image)],
            dim=1,
        ).double().cpu()
        if component_sum is None:
            component_sum = torch.zeros(batch_power.shape[1], dtype=torch.float64)
        component_sum += batch_power.sum(0)
        count += len(latent)
    powers = component_sum / count
    floor = torch.quantile(powers[powers > 0], 0.01).clamp_min(1e-10)
    return powers.clamp_min(floor).float().to(device)


@torch.no_grad()
def rescue_dead(sae, inputs, reconstruction, last_fire, step, dead_after, max_rescue):
    dead = torch.where(last_fire < step - dead_after)[0]
    if not len(dead):
        return 0
    residual = inputs - reconstruction
    count = min(len(dead), max_rescue)
    if isinstance(sae, ConvTopKSAE):
        flat = residual.flatten(2)
        errors = flat.square().mean(1).flatten()
        chosen = torch.argsort(errors, descending=True)[:count]
        examples = torch.div(chosen, sae.positions, rounding_mode="floor")
        positions = chosen.remainder(sae.positions)
        directions = F.normalize(flat[examples, :, positions], dim=1)
        centered = inputs.flatten(2)[examples, :, positions] - sae.dec_bias
    else:
        flat = residual.flatten(1)
        examples = torch.argsort(flat.square().mean(1), descending=True)[:count]
        directions = F.normalize(flat[examples], dim=1)
        centered = inputs.flatten(1)[examples] - sae.dec_bias
    count = len(directions)
    features = dead[:count]
    sae.decoder[features].copy_(directions)
    sae.encoder[features].copy_(directions)
    response = (directions * centered).sum(1)
    sae.encoder_bias[features].copy_(-response + 0.1)
    last_fire[features] = step
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("flat", "conv"), required=True)
    parser.add_argument("--autoencoder-type", choices=("spatial", "vector"), default="spatial")
    parser.add_argument("--objective", choices=OBJECTIVES, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--latent-cache", required=True)
    parser.add_argument("--base-latent-cache")
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--feature-count", type=int, default=4096)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--calibration-examples", type=int, default=2048)
    parser.add_argument("--reinit-every", type=int, default=1000)
    parser.add_argument("--dead-after", type=int, default=3000)
    parser.add_argument("--max-rescue", type=int, default=256)
    parser.add_argument("--transform-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--init-seed", type=int)
    parser.add_argument("--data-seed", type=int)
    args = parser.parse_args()

    args.init_seed = args.seed if args.init_seed is None else args.init_seed
    args.data_seed = args.seed if args.data_seed is None else args.data_seed

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.init_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spatial = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
    raw_latents = spatial["latents"]
    train_indices = spatial["train_indices"]
    valid_indices = spatial["valid_indices"]
    base_latents = None
    if args.base_latent_cache:
        base = torch.load(args.base_latent_cache, map_location="cpu", weights_only=False)
        base_latents = base["latents"]
        if len(base_latents) != len(raw_latents):
            raise ValueError("base and target latent caches have different lengths")
    input_shape = tuple(raw_latents.shape[1:])
    autoencoder = (
        load_vector_autoencoder(args.autoencoder, device)
        if args.autoencoder_type == "vector"
        else load_spatial_autoencoder(args.autoencoder, device)
    )

    train_latents = raw_latents[train_indices].float()
    if train_latents.ndim == 2:
        channel_mean = train_latents.mean(dim=0)
        channel_std = train_latents.std(dim=0).clamp_min(1e-6)
        device_mean = channel_mean.to(device)[None, :]
        device_std = channel_std.to(device)[None, :]
    else:
        channel_mean = train_latents.mean(dim=(0, 2, 3))
        channel_std = train_latents.std(dim=(0, 2, 3)).clamp_min(1e-6)
        device_mean = channel_mean.to(device)[None, :, None, None]
        device_std = channel_std.to(device)[None, :, None, None]
    generator = torch.Generator().manual_seed(args.data_seed)
    loader = DataLoader(
        TensorDataset(train_indices), batch_size=args.batch_size, shuffle=True,
        drop_last=True, generator=generator,
    )
    iterator = iter(loader)

    cls = FlatTopKSAE if args.architecture == "flat" else ConvTopKSAE
    sae = cls(input_shape, args.feature_count, args.k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    powers = None
    if args.objective != "latent_mse":
        calibration_indices = train_indices[:min(args.calibration_examples, len(train_indices))]
        powers = calibrate(
            autoencoder, transform_objective_name(args.objective),
            raw_latents, calibration_indices,
            args.batch_size, device, base_latents,
        )
        torch.save(
            {"objective": args.objective, "component_powers": powers.cpu()},
            os.path.join(args.out_dir, "transform_stats.pt"),
        )

    feature_counts = torch.zeros(args.feature_count, dtype=torch.long, device=device)
    last_fire = torch.zeros(args.feature_count, dtype=torch.long, device=device)
    rescued_total = 0
    history = []
    for step in range(1, args.steps + 1):
        try:
            indices = next(iterator)[0]
        except StopIteration:
            iterator = iter(loader)
            indices = next(iterator)[0]
        raw = raw_latents[indices].to(device).float()
        standardized = (raw - device_mean) / device_std
        reconstruction, values, selected = sae(standardized)
        latent_mse = (standardized - reconstruction).square().flatten(1).mean(1)
        if args.objective == "latent_mse":
            per_example_loss = latent_mse
        else:
            reconstructed_raw = reconstruction * device_std + device_mean
            target_decode_latent = raw
            reconstruction_decode_latent = reconstructed_raw
            if base_latents is not None:
                base = base_latents[indices].to(device).float()
                target_decode_latent = base + raw
                reconstruction_decode_latent = base + reconstructed_raw
            with torch.no_grad():
                clean_image = autoencoder.decode(target_decode_latent).add(1).div(2)
            reconstructed_image = autoencoder.decode(reconstruction_decode_latent).add(1).div(2)
            geometry_loss = transformed_loss(
                transform_objective_name(args.objective),
                clean_image - reconstructed_image, powers,
            )
            per_example_loss = (
                latent_mse + args.transform_weight * geometry_loss
                if args.objective.startswith("weighted_") else geometry_loss
            )
        loss = per_example_loss.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip)
        if args.warmup and step <= args.warmup:
            for group in optimizer.param_groups:
                group["lr"] = args.lr * step / args.warmup
        optimizer.step()
        sae.normalize_decoder()

        ids = feature_ids(sae, selected)
        active = values > 0
        feature_counts.scatter_add_(
            0, ids[active], torch.ones_like(ids[active], dtype=torch.long)
        )
        last_fire[ids[active].unique()] = step
        rescued = 0
        if args.reinit_every and step % args.reinit_every == 0:
            rescued = rescue_dead(
                sae, standardized, reconstruction.detach(), last_fire,
                step, args.dead_after, args.max_rescue,
            )
            rescued_total += rescued
        if step == 1 or step % 100 == 0:
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "latent_mse_monitor": float(latent_mse.mean().detach()),
                "active_mean_per_image": float(active.flatten(1).sum(1).float().mean()),
                "active_mean_per_position": (
                    float(active.sum(1).float().mean())
                    if isinstance(sae, ConvTopKSAE) else None
                ),
                "used_features": int((feature_counts > 0).sum()),
                "dead_features": int((last_fire < step - args.dead_after).sum()),
                "rescued": rescued,
                "gradient_norm_before_clip": float(gradient_norm),
            }
            history.append(record)
            print(record, flush=True)

    transformed_sum = pixel_sum = latent_sum = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(valid_indices), args.batch_size):
            indices = valid_indices[start:start + args.batch_size]
            raw = raw_latents[indices].to(device).float()
            standardized = (raw - device_mean) / device_std
            reconstruction, values, selected = sae(standardized)
            reconstructed_raw = reconstruction * device_std + device_mean
            target_decode_latent = raw
            reconstruction_decode_latent = reconstructed_raw
            if base_latents is not None:
                base = base_latents[indices].to(device).float()
                target_decode_latent = base + raw
                reconstruction_decode_latent = base + reconstructed_raw
            clean_image = autoencoder.decode(target_decode_latent).add(1).div(2)
            reconstructed_image = autoencoder.decode(reconstruction_decode_latent).add(1).div(2)
            latent = (standardized - reconstruction).square().flatten(1).mean(1)
            pixel = (clean_image - reconstructed_image).square().flatten(1).mean(1)
            if args.objective == "latent_mse":
                transformed = latent
            else:
                geometry_loss = transformed_loss(
                    transform_objective_name(args.objective),
                    clean_image - reconstructed_image, powers,
                )
                transformed = (
                    latent + args.transform_weight * geometry_loss
                    if args.objective.startswith("weighted_") else geometry_loss
                )
            transformed_sum += float(transformed.sum())
            pixel_sum += float(pixel.sum())
            latent_sum += float(latent.sum())
            count += len(raw)

    metrics = {
        "architecture": args.architecture,
        "objective": args.objective,
        "examples": len(raw_latents),
        "input_shape": input_shape,
        "feature_count": args.feature_count,
        "k": args.k,
        "used_features": int((feature_counts > 0).sum()),
        "dead_features_at_end": int((last_fire < args.steps - args.dead_after).sum()),
        "rescued_total": rescued_total,
        "valid_objective_loss": transformed_sum / count,
        "valid_pixel_mse_monitor": pixel_sum / count,
        "valid_latent_mse_monitor": latent_sum / count,
        "component_powers": powers.cpu().tolist() if powers is not None else None,
        "config": vars(args),
        "history": history,
    }
    torch.save(
        {
            "model": sae.state_dict(),
            "channel_mean": channel_mean,
            "channel_std": channel_std,
            "input_shape": input_shape,
            "config": vars(args),
            "metrics": metrics,
        },
        os.path.join(args.out_dir, "last.pt"),
    )
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
