#!/usr/bin/env python3
"""Evaluate a matched set of z_sem SAEs under every output representation."""

import argparse
import json
import os
import sys

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
    return sae, state["input_mean"].to(device), state["input_std"].to(device)


def spectrum(image):
    return torch.log1p(torch.fft.rfft2(image, norm="ortho").abs().square())


def dino_input(image, mean, std):
    image = F.interpolate(image, size=(224, 224), mode="bicubic", align_corners=False)
    return (image - mean) / std


def summary(values):
    values = torch.cat(values).double()
    return {"mean": float(values.mean()), "median": float(values.median())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--spectral-stats", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="name=checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--split", choices=("train", "valid", "all"), default="valid")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pilot = torch.load(args.data, map_location="cpu", weights_only=False)
    images = pilot["images_uint8"]
    codes = pilot["codes"].float()
    permutation = torch.randperm(len(images), generator=torch.Generator().manual_seed(args.seed))
    split = int(0.9 * len(images))
    if args.split == "train":
        indices = permutation[:split]
    elif args.split == "valid":
        indices = permutation[split:]
    else:
        indices = permutation
    images, codes = images[indices], codes[indices]
    lit = load_diffae(args.repo, args.checkpoint, device)

    models = {}
    means = {}
    stds = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        models[name], means[name], stds[name] = load_sae(path, device)
    baseline_decoder = models["baseline"].decoder.detach()

    from transformers import AutoImageProcessor, Dinov2Model

    processor = AutoImageProcessor.from_pretrained(args.dino, local_files_only=True)
    dino = Dinov2Model.from_pretrained(args.dino, local_files_only=True).eval().to(device)
    dino_mean = torch.tensor(processor.image_mean, device=device)[None, :, None, None]
    dino_std = torch.tensor(processor.image_std, device=device)[None, :, None, None]
    spectrum_std = torch.load(args.spectral_stats, map_location=device, weights_only=False)["spectrum_std"]

    metrics = {name: {key: [] for key in ("latent", "pixel", "spectral", "semantic")}
               for name in models}
    with torch.inference_mode():
        for start in range(0, len(images), args.batch_size):
            image = images[start:start + args.batch_size].to(device).float().div(255)
            code = codes[start:start + args.batch_size].to(device)
            batch = len(image)
            generator = torch.Generator().manual_seed(args.seed + start)
            bins = torch.arange(start, start + batch, device=device) % 10
            offsets = torch.randint(0, 100, (batch,), generator=generator).to(device)
            timesteps = (bins * 100 + offsets).clamp_max(999).long()
            noise = torch.randn(image.shape, generator=generator).to(device)
            x_t = lit.sampler.q_sample(image.mul(2).sub(1), timesteps, noise=noise)
            clean = lit.sampler.p_mean_variance(
                lit.ema_model, x_t, timesteps, clip_denoised=True,
                model_kwargs={"cond": code}
            )["pred_xstart"].add(1).div(2)
            clean_spectrum = spectrum(clean)
            clean_embedding = dino(
                pixel_values=dino_input(clean, dino_mean, dino_std)
            ).last_hidden_state[:, 0]

            for name, sae in models.items():
                standardized = (code - means[name]) / stds[name]
                reconstructed, _, _, _, _ = sae(standardized)
                reconstructed_code = reconstructed * stds[name] + means[name]
                prediction = lit.sampler.p_mean_variance(
                    lit.ema_model, x_t, timesteps, clip_denoised=True,
                    model_kwargs={"cond": reconstructed_code}
                )["pred_xstart"].add(1).div(2)
                embedding = dino(
                    pixel_values=dino_input(prediction, dino_mean, dino_std)
                ).last_hidden_state[:, 0]
                metrics[name]["latent"].append(
                    (standardized - reconstructed).square().mean(1).cpu()
                )
                metrics[name]["pixel"].append(
                    (clean - prediction).square().flatten(1).mean(1).cpu()
                )
                metrics[name]["spectral"].append(
                    (((clean_spectrum - spectrum(prediction)) / spectrum_std[bins])
                     .square().flatten(1).mean(1).cpu())
                )
                metrics[name]["semantic"].append(
                    (1 - F.cosine_similarity(clean_embedding, embedding, dim=1)).cpu()
                )
            print(f"evaluated {min(start + batch, len(images))}/{len(images)}", flush=True)

    result = {name: {metric: summary(values) for metric, values in by_metric.items()}
              for name, by_metric in metrics.items()}
    for name, model in models.items():
        cosine = F.cosine_similarity(baseline_decoder, model.decoder.detach(), dim=1).cpu().double()
        result[name]["same_index_decoder_cosine_to_baseline"] = {
            "mean": float(cosine.mean()), "median": float(cosine.median()),
            "min": float(cosine.min())
        }
    baseline = result["baseline"]
    for name in result:
        result[name]["relative_change_vs_baseline_percent"] = {
            metric: 100 * (result[name][metric]["mean"] / baseline[metric]["mean"] - 1)
            for metric in ("latent", "pixel", "spectral", "semantic")
        }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
