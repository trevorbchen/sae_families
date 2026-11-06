#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from train_dit_cifar import DiT, Diffusion


def parse_ints(s):
    return tuple(int(x) for x in s.split(",") if x.strip())


def load_pt(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(path, device):
    model = DiT(learn_sigma=True).to(device)
    ckpt = load_pt(path, device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/dit_s2_cifar10/last.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs/timestep_eps_norm_curve_eval")
    p.add_argument("--num-images", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--candidate-step", type=int, default=10)
    p.add_argument("--extra-candidates", type=parse_ints, default=(999,))
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = load_model(args.ckpt, device)
    diffusion = Diffusion(args.timesteps, device=device)

    candidates = list(range(0, args.timesteps, args.candidate_step))
    for t in args.extra_candidates:
        if 0 <= t < args.timesteps and t not in candidates:
            candidates.append(t)
    candidates = sorted(candidates)
    candidate_tensor = torch.tensor(candidates, device=device, dtype=torch.long)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    dataset = datasets.CIFAR10(args.data_dir, train=True, transform=transform, download=True)
    gen = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=gen)[: args.num_images].tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    curve_chunks = []
    records = []
    labels_all = []
    true_t_all = []
    pred_t_all = []
    abs_err_all = []
    batch_start = 0
    start = time.time()

    for batch_i, (x0, labels) in enumerate(loader):
        x0 = x0.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        bsz = x0.shape[0]
        local_gen = torch.Generator(device=device).manual_seed(args.seed + 100000 + batch_start)
        true_t = torch.randint(0, args.timesteps, (bsz,), generator=local_gen, device=device)
        noise = torch.randn(x0.shape, generator=local_gen, device=device)
        x_t = diffusion.q_sample(x0, true_t, noise)

        norms = []
        enabled = device.type == "cuda" and args.precision == "bf16"
        for cand_start in range(0, len(candidates), args.batch_size):
            cand = candidate_tensor[cand_start : cand_start + args.batch_size]
            # Evaluate all images for this candidate chunk by repeating the image batch.
            x_rep = x_t.repeat_interleave(cand.numel(), dim=0)
            y_rep = labels.repeat_interleave(cand.numel(), dim=0)
            t_rep = cand.repeat(bsz)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
                out = model(x_rep, t_rep, y_rep)
            eps = torch.chunk(out.float(), 2, dim=1)[0]
            eps_norm = eps.flatten(1).norm(dim=1).reshape(bsz, cand.numel())
            norms.append(eps_norm.cpu())

        norms = torch.cat(norms, dim=1)
        pred_idx = torch.argmin(norms, dim=1)
        pred_t = torch.tensor(candidates, dtype=torch.long)[pred_idx]
        true_t_cpu = true_t.cpu()
        labels_cpu = labels.cpu()
        abs_err = (pred_t - true_t_cpu).abs()

        curve_chunks.append(norms)
        labels_all.append(labels_cpu)
        true_t_all.append(true_t_cpu)
        pred_t_all.append(pred_t)
        abs_err_all.append(abs_err)

        for i in range(bsz):
            records.append({
                "row": batch_start + i,
                "cifar_train_index": indices[batch_start + i],
                "label": int(labels_cpu[i]),
                "true_t": int(true_t_cpu[i]),
                "pred_t": int(pred_t[i]),
                "abs_error": int(abs_err[i]),
                "min_eps_norm": float(norms[i, pred_idx[i]]),
                "true_t_nearest_candidate": int(candidates[min(range(len(candidates)), key=lambda j: abs(candidates[j] - int(true_t_cpu[i])))]),
            })

        batch_start += bsz
        print(f"processed={batch_start}/{args.num_images} elapsed={time.time() - start:.1f}s", flush=True)

    curves = torch.cat(curve_chunks, dim=0)
    labels_all = torch.cat(labels_all)
    true_t_all = torch.cat(true_t_all)
    pred_t_all = torch.cat(pred_t_all)
    abs_err_all = torch.cat(abs_err_all).float()

    # Discretization floor: even a perfect argmin on a step-10 grid has this candidate rounding error.
    cand_cpu = torch.tensor(candidates, dtype=torch.long)
    nearest = torch.stack([(cand_cpu - t).abs() for t in true_t_all]).min(dim=1).values.float()

    summary = {
        "metric": "argmin_t ||eps_theta(x_true_t, candidate_t)||_2",
        "num_images": int(args.num_images),
        "candidate_step": int(args.candidate_step),
        "candidates": candidates,
        "mean_abs_error": float(abs_err_all.mean()),
        "median_abs_error": float(abs_err_all.median()),
        "rmse": float(abs_err_all.pow(2).mean().sqrt()),
        "max_abs_error": int(abs_err_all.max().item()),
        "mean_candidate_rounding_error": float(nearest.mean()),
        "mean_excess_over_candidate_rounding": float((abs_err_all - nearest).mean()),
        "within_10": float((abs_err_all <= 10).float().mean()),
        "within_25": float((abs_err_all <= 25).float().mean()),
        "within_50": float((abs_err_all <= 50).float().mean()),
        "within_100": float((abs_err_all <= 100).float().mean()),
        "config": vars(args),
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "per_image_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        fields = ["row", "cifar_train_index", "label", "true_t", "pred_t", "abs_error", "min_eps_norm", "true_t_nearest_candidate"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    torch.save({
        "curves": curves,
        "candidate_t_values": cand_cpu,
        "true_t": true_t_all,
        "pred_t": pred_t_all,
        "labels": labels_all,
        "cifar_train_indices": torch.tensor(indices, dtype=torch.long),
        "summary": summary,
    }, os.path.join(args.out_dir, "eps_norm_curves.pt"))

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
