#!/usr/bin/env python3
"""Fine-tune a TopK z_sem SAE under a frozen DiffAE output geometry."""

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


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
    return sae, state, state["input_mean"].to(device), state["input_std"].to(device)


def spectral_representation(image):
    spectrum = torch.fft.rfft2(image, norm="ortho")
    return torch.log1p(spectrum.abs().square())


def dino_input(image, mean, std):
    image = F.interpolate(image, size=(224, 224), mode="bicubic", align_corners=False)
    return (image - mean) / std


def timestep_batch(step, batch, bins, seed, image_shape, device):
    generator = torch.Generator().manual_seed(seed + step)
    bin_ids = (step * batch + torch.arange(batch, device=device)) % bins
    offsets = torch.randint(0, 100, (batch,), generator=generator).to(device)
    timesteps = (bin_ids * 100 + offsets).clamp_max(999).long()
    noise = torch.randn(image_shape, generator=generator).to(device)
    return bin_ids, timesteps, noise


@torch.no_grad()
def estimate_spectral_std(lit, images, codes, batch_size, seed, device):
    bins = 10
    sums = [None] * bins
    square_sums = [None] * bins
    counts = [0] * bins
    for start in range(0, len(images), batch_size):
        image = images[start:start + batch_size].to(device).float().div(255)
        code = codes[start:start + batch_size].to(device)
        batch = len(image)
        bin_ids, timesteps, noise = timestep_batch(
            start // batch_size, batch, bins, seed, image.shape, device
        )
        x_t = lit.sampler.q_sample(image.mul(2).sub(1), timesteps, noise=noise)
        prediction = lit.sampler.p_mean_variance(
            lit.ema_model,
            x_t,
            timesteps,
            clip_denoised=True,
            model_kwargs={"cond": code},
        )["pred_xstart"].add(1).div(2)
        spectra = spectral_representation(prediction).double().cpu()
        cpu_bins = bin_ids.cpu()
        for bin_id in cpu_bins.unique().tolist():
            selected = spectra[cpu_bins == bin_id]
            value_sum = selected.sum(0)
            square_sum = selected.square().sum(0)
            if sums[bin_id] is None:
                sums[bin_id] = value_sum
                square_sums[bin_id] = square_sum
            else:
                sums[bin_id] += value_sum
                square_sums[bin_id] += square_sum
            counts[bin_id] += len(selected)
        print(f"spectral calibration {min(start + batch, len(images))}/{len(images)}", flush=True)
    output = []
    for bin_id in range(bins):
        count = counts[bin_id]
        variance = (square_sums[bin_id] - sums[bin_id].square() / count) / max(count - 1, 1)
        std = variance.clamp_min(0).sqrt().float()
        floor = torch.quantile(std.flatten(), 0.05).clamp_min(1e-4)
        output.append(std.clamp_min(floor))
    return torch.stack(output).to(device)


def representation_loss(
    objective,
    clean,
    reconstructed,
    bin_ids,
    spectrum_std=None,
    dino=None,
    dino_mean=None,
    dino_std=None,
):
    if objective == "pixel":
        return (clean - reconstructed).square().flatten(1).mean(1)
    if objective == "spectral":
        clean_spectrum = spectral_representation(clean)
        reconstructed_spectrum = spectral_representation(reconstructed)
        return (((clean_spectrum - reconstructed_spectrum) / spectrum_std[bin_ids])
                .square().flatten(1).mean(1))
    if objective == "semantic":
        with torch.no_grad():
            clean_embedding = dino(
                pixel_values=dino_input(clean, dino_mean, dino_std)
            ).last_hidden_state[:, 0]
        reconstructed_embedding = dino(
            pixel_values=dino_input(reconstructed, dino_mean, dino_std)
        ).last_hidden_state[:, 0]
        return 1 - F.cosine_similarity(clean_embedding, reconstructed_embedding, dim=1)
    raise ValueError(objective)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=("pixel", "spectral", "semantic"), required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--init-sae", required=True)
    parser.add_argument("--data", required=True, help="Paired generated images and codes")
    parser.add_argument("--codes", required=True, help="Full z_sem corpus for latent reconstruction")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--dino")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--latent-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--aux-coeff", type=float, default=0.1)
    parser.add_argument("--max-bin-weight", type=float, default=1000.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2028)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    pilot = torch.load(args.data, map_location="cpu", weights_only=False)
    images = pilot["images_uint8"]
    paired_codes = pilot["codes"].float()
    all_codes = torch.load(args.codes, map_location="cpu", weights_only=False)["conds"].float()
    paired_permutation = torch.randperm(
        len(images), generator=torch.Generator().manual_seed(args.seed)
    )
    paired_split = int(0.9 * len(images))
    paired_train_indices = paired_permutation[:paired_split]
    paired_loader = DataLoader(
        TensorDataset(images[paired_train_indices], paired_codes[paired_train_indices]),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    paired_iterator = iter(paired_loader)
    code_permutation = torch.randperm(
        len(all_codes), generator=torch.Generator().manual_seed(args.seed)
    )
    code_split = int(0.9 * len(all_codes))
    code_train_indices, code_valid_indices = (
        code_permutation[:code_split], code_permutation[code_split:]
    )
    code_loader = DataLoader(
        TensorDataset(all_codes[code_train_indices]),
        batch_size=args.latent_batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    code_iterator = iter(code_loader)

    lit = load_diffae(args.repo, args.checkpoint, device)
    sae, initial_state, input_mean, input_std = load_sae(args.init_sae, device)
    sae.train()
    with open(args.calibration) as handle:
        calibration = json.load(handle)
    bin_weights = torch.tensor(
        calibration["bins"][args.objective], device=device, dtype=torch.float32
    ).clamp(min=0, max=args.max_bin_weight)

    dino = dino_mean = dino_std = None
    if args.objective == "semantic":
        if not args.dino:
            parser.error("--dino is required for semantic training")
        from transformers import AutoImageProcessor, Dinov2Model

        processor = AutoImageProcessor.from_pretrained(args.dino, local_files_only=True)
        dino = Dinov2Model.from_pretrained(args.dino, local_files_only=True).eval().to(device)
        for parameter in dino.parameters():
            parameter.requires_grad_(False)
        dino_mean = torch.tensor(processor.image_mean, device=device)[None, :, None, None]
        dino_std = torch.tensor(processor.image_std, device=device)[None, :, None, None]

    spectrum_std = None
    if args.objective == "spectral":
        spectrum_std = estimate_spectral_std(
            lit, images, paired_codes, max(args.batch_size, 2), args.seed, device
        )
        torch.save({"spectrum_std": spectrum_std.cpu()}, os.path.join(args.out_dir, "representation_stats.pt"))

    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    history = []
    feature_counts = torch.zeros(sae.h, dtype=torch.long, device=device)
    for step in range(1, args.steps + 1):
        try:
            image_u8, paired_raw_code = next(paired_iterator)
        except StopIteration:
            paired_iterator = iter(paired_loader)
            image_u8, paired_raw_code = next(paired_iterator)
        try:
            local_raw_code = next(code_iterator)[0]
        except StopIteration:
            code_iterator = iter(code_loader)
            local_raw_code = next(code_iterator)[0]
        image = image_u8.to(device).float().div(255)
        paired_raw_code = paired_raw_code.to(device)
        local_raw_code = local_raw_code.to(device)
        local_standardized = (local_raw_code - input_mean) / input_std
        local_reconstruction, values, indices, aux_reconstruction, _ = sae(local_standardized)
        paired_standardized = (paired_raw_code - input_mean) / input_std
        paired_reconstruction, _, _, _, _ = sae(paired_standardized)
        reconstructed_code = paired_reconstruction * input_std + input_mean
        bin_ids, timesteps, noise = timestep_batch(
            step, len(image), 10, args.seed, image.shape, device
        )
        x_t = lit.sampler.q_sample(image.mul(2).sub(1), timesteps, noise=noise)
        with torch.no_grad():
            clean_prediction = lit.sampler.p_mean_variance(
                lit.ema_model,
                x_t,
                timesteps,
                clip_denoised=True,
                model_kwargs={"cond": paired_raw_code},
            )["pred_xstart"].add(1).div(2)
        reconstructed_prediction = lit.sampler.p_mean_variance(
            lit.ema_model,
            x_t,
            timesteps,
            clip_denoised=True,
            model_kwargs={"cond": reconstructed_code},
        )["pred_xstart"].add(1).div(2)

        latent_loss = F.mse_loss(local_reconstruction, local_standardized)
        residual = (local_standardized - local_reconstruction).detach()
        aux_loss = F.mse_loss(aux_reconstruction, residual)
        rep_per = representation_loss(
            args.objective,
            clean_prediction,
            reconstructed_prediction,
            bin_ids,
            spectrum_std,
            dino,
            dino_mean,
            dino_std,
        )
        rep_loss = (rep_per * bin_weights[bin_ids]).mean()
        loss = latent_loss + args.aux_coeff * aux_loss + rep_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip)
        if args.warmup and step <= args.warmup:
            scale = step / args.warmup
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale
        optimizer.step()
        sae.normalize_decoder()
        feature_counts.scatter_add_(0, indices.flatten(), torch.ones_like(indices).flatten())

        if step == 1 or step % 25 == 0:
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "latent_loss": float(latent_loss.detach()),
                "aux_loss": float(aux_loss.detach()),
                "representation_loss_weighted": float(rep_loss.detach()),
                "representation_loss_raw": float(rep_per.mean().detach()),
                "mean_bin_weight": float(bin_weights[bin_ids].mean()),
                "gradient_norm_before_clip": float(gradient_norm),
            }
            history.append(record)
            print(record, flush=True)

    sae.eval()
    with torch.no_grad():
        mse_sum = 0.0
        value_sum = 0.0
        value_square_sum = 0.0
        element_count = 0
        for start in range(0, len(code_valid_indices), args.latent_batch_size):
            valid = (
                all_codes[code_valid_indices[start:start + args.latent_batch_size]].to(device)
                - input_mean
            ) / input_std
            reconstructed, _, _, _, _ = sae(valid)
            mse_sum += float((reconstructed - valid).square().sum())
            value_sum += float(valid.sum())
            value_square_sum += float(valid.square().sum())
            element_count += valid.numel()
        valid_mse = mse_sum / element_count
        valid_variance = (
            value_square_sum - value_sum * value_sum / element_count
        ) / max(element_count - 1, 1)
        valid_ev = 1 - valid_mse / max(valid_variance, 1e-8)
    result = {
        "objective": args.objective,
        "steps": args.steps,
        "train_examples": len(code_train_indices),
        "valid_examples": len(code_valid_indices),
        "paired_train_examples": len(paired_train_indices),
        "valid_mse": valid_mse,
        "valid_variance": valid_variance,
        "valid_ev": valid_ev,
        "used_features": int((feature_counts > 0).sum()),
        "bin_weights": bin_weights.tolist(),
        "history": history,
        "config": vars(args),
    }
    torch.save(
        {
            "model": sae.state_dict(),
            "input_mean": initial_state["input_mean"],
            "input_std": initial_state["input_std"],
            "config": initial_state["config"],
            "representation_training": result,
        },
        os.path.join(args.out_dir, "last.pt"),
    )
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
