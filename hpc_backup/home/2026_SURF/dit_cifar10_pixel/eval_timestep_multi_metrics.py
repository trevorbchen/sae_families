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


def tv_norm(x):
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().flatten(1).mean(dim=1)
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().flatten(1).mean(dim=1)
    return dh + dw


def summarize_metric(name, values, candidate_t, true_t, mode="min"):
    if mode == "min":
        idx = torch.argmin(values, dim=1)
    elif mode == "max":
        idx = torch.argmax(values, dim=1)
    else:
        raise ValueError(mode)
    pred_t = candidate_t[idx]
    err = (pred_t - true_t).abs().float()
    return {
        "metric": name,
        "mode": mode,
        "mean_abs_error": float(err.mean()),
        "median_abs_error": float(err.median()),
        "rmse": float(err.pow(2).mean().sqrt()),
        "max_abs_error": int(err.max().item()),
        "within_10": float((err <= 10).float().mean()),
        "within_25": float((err <= 25).float().mean()),
        "within_50": float((err <= 50).float().mean()),
        "within_100": float((err <= 100).float().mean()),
    }, pred_t, err


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/dit_s2_cifar10/last.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs/timestep_multi_metric_eval")
    p.add_argument("--num-images", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
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
    candidate_t = torch.tensor(candidates, device=device, dtype=torch.long)
    candidate_cpu = candidate_t.cpu()
    sqrt_ac = diffusion.sqrt_ac[candidate_t].reshape(1, -1)
    sigma = diffusion.sqrt_one_minus_ac[candidate_t].reshape(1, -1).clamp_min(1e-8)

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

    metric_chunks = {k: [] for k in [
        "raw_eps_norm",
        "score_norm",
        "oracle_eps_mse",
        "x0_range_violation",
        "x0_mean_abs",
        "x0_std_error",
        "x0_tv",
    ]}
    true_t_all = []
    labels_all = []
    start = time.time()
    offset = 0
    enabled = device.type == "cuda" and args.precision == "bf16"

    for batch_i, (x0, labels) in enumerate(loader):
        x0 = x0.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        bsz = x0.shape[0]
        local_gen = torch.Generator(device=device).manual_seed(args.seed + 100000 + offset)
        true_t = torch.randint(0, args.timesteps, (bsz,), generator=local_gen, device=device)
        noise = torch.randn(x0.shape, generator=local_gen, device=device)
        x_t = diffusion.q_sample(x0, true_t, noise)

        batch_metrics = {k: [] for k in metric_chunks}
        for c0 in range(0, len(candidates), args.batch_size):
            cand = candidate_t[c0 : c0 + args.batch_size]
            chunk = cand.numel()
            x_rep = x_t.repeat_interleave(chunk, dim=0)
            y_rep = labels.repeat_interleave(chunk, dim=0)
            t_rep = cand.repeat(bsz)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
                out = model(x_rep, t_rep, y_rep)
            eps = torch.chunk(out.float(), 2, dim=1)[0]
            eps_flat = eps.flatten(1)
            eps_norm = eps_flat.norm(dim=1).reshape(bsz, chunk)
            noise_rep = noise.repeat_interleave(chunk, dim=0)
            oracle = (eps - noise_rep).pow(2).flatten(1).mean(dim=1).reshape(bsz, chunk)
            cand_sigma = diffusion.sqrt_one_minus_ac[cand].reshape(1, chunk, 1, 1, 1)
            cand_sqrt_ac = diffusion.sqrt_ac[cand].reshape(1, chunk, 1, 1, 1).clamp_min(1e-8)
            x_t_group = x_t.unsqueeze(1)
            eps_group = eps.reshape(bsz, chunk, *eps.shape[1:])
            x0_hat = (x_t_group - cand_sigma * eps_group) / cand_sqrt_ac
            range_violation = (x0_hat.abs() - 1.0).clamp_min(0).flatten(2).mean(dim=2)
            mean_abs = x0_hat.flatten(2).mean(dim=2).abs()
            std_error = (x0_hat.flatten(2).std(dim=2) - 0.5).abs()
            tv = tv_norm(x0_hat.reshape(bsz * chunk, *x0.shape[1:])).reshape(bsz, chunk)

            batch_metrics["raw_eps_norm"].append(eps_norm.cpu())
            batch_metrics["score_norm"].append((eps_norm / diffusion.sqrt_one_minus_ac[cand].reshape(1, chunk).clamp_min(1e-8)).cpu())
            batch_metrics["oracle_eps_mse"].append(oracle.cpu())
            batch_metrics["x0_range_violation"].append(range_violation.cpu())
            batch_metrics["x0_mean_abs"].append(mean_abs.cpu())
            batch_metrics["x0_std_error"].append(std_error.cpu())
            batch_metrics["x0_tv"].append(tv.cpu())

        for k in metric_chunks:
            metric_chunks[k].append(torch.cat(batch_metrics[k], dim=1))
        true_t_all.append(true_t.cpu())
        labels_all.append(labels.cpu())
        offset += bsz
        print(f"processed={offset}/{args.num_images} elapsed={time.time() - start:.1f}s", flush=True)

    metrics = {k: torch.cat(v, dim=0) for k, v in metric_chunks.items()}
    true_t = torch.cat(true_t_all)
    labels = torch.cat(labels_all)

    # Normalize metrics by candidate-t baseline across the sampled dataset.
    normalized = {}
    for k, v in metrics.items():
        baseline = v.mean(dim=0, keepdim=True).clamp_min(1e-12)
        normalized[f"{k}_div_t_mean"] = v / baseline
    metrics.update(normalized)

    summaries = []
    pred_by_metric = {}
    for name, values in metrics.items():
        summary, pred, err = summarize_metric(name, values, candidate_cpu, true_t, mode="min")
        summaries.append(summary)
        pred_by_metric[name] = pred

    summaries.sort(key=lambda r: r["mean_abs_error"])
    with open(os.path.join(args.out_dir, "metric_summary.csv"), "w", newline="", encoding="utf-8") as f:
        fields = list(summaries[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summaries)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "metric_summaries": summaries}, f, indent=2)

    with open(os.path.join(args.out_dir, "per_image_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        fields = ["row", "cifar_train_index", "label", "true_t"] + [f"pred_{s['metric']}" for s in summaries[:8]]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(args.num_images):
            rec = {"row": i, "cifar_train_index": indices[i], "label": int(labels[i]), "true_t": int(true_t[i])}
            for s in summaries[:8]:
                rec[f"pred_{s['metric']}"] = int(pred_by_metric[s["metric"]][i])
            w.writerow(rec)

    torch.save({
        "candidate_t_values": candidate_cpu,
        "true_t": true_t,
        "labels": labels,
        "cifar_train_indices": torch.tensor(indices, dtype=torch.long),
        "metrics": metrics,
        "summaries": summaries,
    }, os.path.join(args.out_dir, "multi_metric_curves.pt"))

    print(json.dumps({"top_metrics": summaries[:10], "out_dir": args.out_dir}, indent=2), flush=True)


if __name__ == "__main__":
    main()
