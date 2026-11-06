#!/usr/bin/env python3
"""Audit pixel, spectral, and semantic SAE objectives before training."""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F


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


def load_sae(path, device):
    from train_zsem_sae_final import TopKSAE

    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state["config"]
    sae = TopKSAE(
        512,
        int(config["dict_mult"]) * 512,
        int(config["k"]),
        int(config.get("aux_k", config["k"])),
    ).to(device)
    sae.load_state_dict(state["model"], strict=True)
    sae.eval()
    for parameter in sae.parameters():
        parameter.requires_grad_(False)
    return sae, state["input_mean"].to(device), state["input_std"].to(device)


def spectral_representation(image):
    spectrum = torch.fft.rfft2(image, norm="ortho")
    return torch.log1p(spectrum.abs().square())


def dino_input(image, mean, std):
    image = F.interpolate(image, size=(224, 224), mode="bicubic", align_corners=False)
    return (image - mean) / std


def cosine_rows(left, right):
    return F.cosine_similarity(left.flatten(1), right.flatten(1), dim=1)


def sample_timestep_batch(start, batch, bins, seed, image_shape, device):
    # A per-batch seed makes the clean-statistics and gradient passes use the
    # exact same timestep/noise pair without retaining a full noise cache.
    generator = torch.Generator().manual_seed(seed + start)
    bin_ids = torch.arange(start, start + batch, device=device) % bins
    offsets = torch.randint(0, 100, (batch,), generator=generator).to(device)
    timesteps = (bin_ids * 100 + offsets).clamp_max(999).long()
    noise = torch.randn(image_shape, generator=generator).to(device)
    return bin_ids, timesteps, noise


def summarize(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sae", required=True)
    parser.add_argument("--pilot-data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    pilot = torch.load(args.pilot_data, map_location="cpu", weights_only=False)
    images = pilot["images_uint8"].float().div_(255)
    codes = pilot["codes"].float()
    lit = load_diffae(args.repo, args.checkpoint, device)
    sae, input_mean, input_std = load_sae(args.sae, device)

    from transformers import AutoImageProcessor, Dinov2Model

    processor = AutoImageProcessor.from_pretrained(args.dino, local_files_only=True)
    dino = Dinov2Model.from_pretrained(args.dino, local_files_only=True).eval().to(device)
    for parameter in dino.parameters():
        parameter.requires_grad_(False)
    dino_mean = torch.tensor(processor.image_mean, device=device)[None, :, None, None]
    dino_std = torch.tensor(processor.image_std, device=device)[None, :, None, None]

    bins = 10
    # Predicted-x0 spectra vary substantially with t. Estimate a separate fixed
    # coefficient scale for each bin so the high-noise bins do not dominate.
    spectrum_sum = [None] * bins
    spectrum_square_sum = [None] * bins
    spectrum_count = [0] * bins
    with torch.no_grad():
        for start in range(0, len(images), args.batch_size):
            clean_image = images[start:start + args.batch_size].to(device)
            raw_code = codes[start:start + args.batch_size].to(device)
            batch = len(clean_image)
            bin_ids, timesteps, noise = sample_timestep_batch(
                start, batch, bins, args.seed, clean_image.shape, device
            )
            x_start = clean_image.mul(2).sub(1)
            x_t = lit.sampler.q_sample(x_start, timesteps, noise=noise)
            prediction = lit.sampler.p_mean_variance(
                lit.ema_model,
                x_t,
                timesteps,
                clip_denoised=True,
                model_kwargs={"cond": raw_code},
            )["pred_xstart"].add(1).div(2)
            spectra = spectral_representation(prediction).double().cpu()
            for bin_id in bin_ids.unique().tolist():
                selected = spectra[bin_ids.cpu() == bin_id]
                value_sum = selected.sum(0)
                square_sum = selected.square().sum(0)
                if spectrum_sum[bin_id] is None:
                    spectrum_sum[bin_id] = value_sum
                    spectrum_square_sum[bin_id] = square_sum
                else:
                    spectrum_sum[bin_id] += value_sum
                    spectrum_square_sum[bin_id] += square_sum
                spectrum_count[bin_id] += len(selected)
            print(f"spectral stats {min(start + batch, len(images))}/{len(images)}", flush=True)
    spectrum_std = []
    for bin_id in range(bins):
        count = spectrum_count[bin_id]
        variance = (
            spectrum_square_sum[bin_id]
            - spectrum_sum[bin_id].square() / count
        ) / max(count - 1, 1)
        std = variance.clamp_min(0).sqrt().float()
        floor = torch.quantile(std.flatten(), 0.05).clamp_min(1e-4)
        spectrum_std.append(std.clamp_min(floor))
    spectrum_std = torch.stack(spectrum_std).to(device)
    torch.save(
        {"spectrum_std": spectrum_std.cpu(), "spectrum_count": spectrum_count},
        os.path.join(args.out_dir, "representation_stats.pt"),
    )

    rows = []
    collected = defaultdict(list)
    for start in range(0, len(images), args.batch_size):
        clean_image = images[start:start + args.batch_size].to(device)
        raw_code = codes[start:start + args.batch_size].to(device)
        batch = len(clean_image)
        bin_ids, timesteps, noise = sample_timestep_batch(
            start, batch, bins, args.seed, clean_image.shape, device
        )
        x_start = clean_image.mul(2).sub(1)
        x_t = lit.sampler.q_sample(x_start, timesteps, noise=noise)

        standardized = (raw_code - input_mean) / input_std
        reconstruction, _, _, _, _ = sae(standardized)
        reconstruction.requires_grad_(True)
        reconstructed_code = reconstruction * input_std + input_mean

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

        latent_per = (standardized - reconstruction).square().mean(1)
        pixel_per = (clean_prediction - reconstructed_prediction).square().flatten(1).mean(1)
        clean_spectrum = spectral_representation(clean_prediction)
        reconstructed_spectrum = spectral_representation(reconstructed_prediction)
        selected_spectrum_std = spectrum_std[bin_ids]
        spectral_per = (((clean_spectrum - reconstructed_spectrum) / selected_spectrum_std)
                        .square().flatten(1).mean(1))

        clean_embedding = dino(
            pixel_values=dino_input(clean_prediction, dino_mean, dino_std)
        ).last_hidden_state[:, 0].detach()
        reconstructed_embedding = dino(
            pixel_values=dino_input(reconstructed_prediction, dino_mean, dino_std)
        ).last_hidden_state[:, 0]
        semantic_per = 1 - F.cosine_similarity(clean_embedding, reconstructed_embedding, dim=1)

        # Parseval sanity check: full complex FFT MSE must equal pixel MSE.
        error = clean_prediction - reconstructed_prediction
        fft_pixel_per = torch.fft.fft2(error, norm="ortho").abs().square().flatten(1).mean(1)

        objectives = {
            "latent": latent_per,
            "pixel": pixel_per,
            "spectral": spectral_per,
            "semantic": semantic_per,
        }
        gradients = {
            name: torch.autograd.grad(value.sum(), reconstruction, retain_graph=True)[0].detach()
            for name, value in objectives.items()
        }
        pairs = (("pixel", "spectral"), ("pixel", "semantic"),
                 ("spectral", "semantic"))

        for row in range(batch):
            record = {
                "index": int(start + row),
                "timestep": int(timesteps[row]),
                "t_bin": int(bin_ids[row]),
                "latent_loss": float(latent_per[row]),
                "pixel_loss": float(pixel_per[row]),
                "spectral_loss": float(spectral_per[row]),
                "semantic_loss": float(semantic_per[row]),
                "parseval_relative_error": float(
                    (fft_pixel_per[row] - pixel_per[row]).abs()
                    / pixel_per[row].clamp_min(1e-12)
                ),
            }
            latent_norm = gradients["latent"][row].norm().clamp_min(1e-12)
            for name, gradient in gradients.items():
                norm = gradient[row].norm()
                record[f"{name}_grad_norm"] = float(norm)
                if name != "latent":
                    record[f"lambda_{name}_equal_grad"] = float(latent_norm / norm.clamp_min(1e-12))
            for left, right in pairs:
                record[f"grad_cos_{left}_{right}"] = float(
                    cosine_rows(gradients[left][row:row + 1], gradients[right][row:row + 1])[0]
                )
            rows.append(record)
            for key, value in record.items():
                if isinstance(value, float):
                    collected[key].append(value)
        print(f"audited {min(start + batch, len(images))}/{len(images)}", flush=True)

    with open(os.path.join(args.out_dir, "per_example.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: summarize(value) for key, value in collected.items()}
    summary["examples"] = len(rows)
    summary["recommended_lambdas"] = {
        name: summary[f"lambda_{name}_equal_grad"]["median"]
        for name in ("pixel", "spectral", "semantic")
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
