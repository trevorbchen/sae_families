#!/usr/bin/env python3
import argparse
import csv
import json
import os

import torch


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", default="runs/block0_input_onoff_ablation_16x_k32/block0_input_onoff_ablation.pt")
    p.add_argument("--out-dir", default="runs/block0_input_onoff_ablation_16x_k32/analysis")
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    obj = load_pt(args.result)
    metrics = obj["metrics"]
    t_values = [int(x) for x in obj["t_values"]]
    feature_ids = obj["feature_ids"].long()

    # Shapes: timestep, image, feature.
    pred_x0_delta = metrics["pred_x0_delta_mse"].float()
    eps_delta = metrics["eps_delta_mse"].float()
    clean_delta = metrics["pred_x0_mse_to_clean_delta"].float()
    clean_mse = metrics["pred_x0_mse_to_clean"].float()
    active_frac = metrics["active_frac"].float()
    coeff_abs = metrics["coeff_abs_mean"].float()
    coeff_max = metrics["coeff_max"].float()

    pred_by_t = pred_x0_delta.mean(dim=1)  # t, feature
    eps_by_t = eps_delta.mean(dim=1)
    clean_delta_by_t = clean_delta.mean(dim=1)
    clean_mse_by_t = clean_mse.mean(dim=1)
    active_by_t = active_frac.mean(dim=1)
    coeff_abs_by_t = coeff_abs.mean(dim=1)
    coeff_max_by_t = coeff_max.amax(dim=1)

    overall_pred = pred_by_t.mean(dim=0)
    overall_eps = eps_by_t.mean(dim=0)
    overall_clean_delta = clean_delta_by_t.mean(dim=0)
    overall_active = active_by_t.mean(dim=0)
    overall_coeff_abs = coeff_abs_by_t.mean(dim=0)
    peak_pred, peak_idx = pred_by_t.max(dim=0)

    early_idx = [i for i, t in enumerate(t_values) if t <= 200]
    mid_idx = [i for i, t in enumerate(t_values) if 300 <= t <= 700]
    late_idx = [i for i, t in enumerate(t_values) if t >= 800]

    rows = []
    for fid in torch.argsort(overall_pred, descending=True).tolist():
        rows.append({
            "feature": int(fid),
            "overall_pred_x0_delta_mse": float(overall_pred[fid]),
            "overall_norm_to_mean": float(overall_pred[fid] / overall_pred.mean().clamp_min(1e-12)),
            "overall_eps_delta_mse": float(overall_eps[fid]),
            "overall_pred_x0_clean_delta": float(overall_clean_delta[fid]),
            "overall_active_frac": float(overall_active[fid]),
            "overall_coeff_abs_mean": float(overall_coeff_abs[fid]),
            "peak_t": t_values[int(peak_idx[fid])],
            "peak_pred_x0_delta_mse": float(peak_pred[fid]),
            "peak_norm_to_mean": float(peak_pred[fid] / pred_by_t.mean().clamp_min(1e-12)),
            "early_pred_x0_delta_mse": float(pred_by_t[early_idx, fid].mean()),
            "mid_pred_x0_delta_mse": float(pred_by_t[mid_idx, fid].mean()),
            "late_pred_x0_delta_mse": float(pred_by_t[late_idx, fid].mean()),
        })

    fields_overall = [
        "feature", "overall_pred_x0_delta_mse", "overall_norm_to_mean", "overall_eps_delta_mse",
        "overall_pred_x0_clean_delta", "overall_active_frac", "overall_coeff_abs_mean",
        "peak_t", "peak_pred_x0_delta_mse", "peak_norm_to_mean",
        "early_pred_x0_delta_mse", "mid_pred_x0_delta_mse", "late_pred_x0_delta_mse",
    ]
    write_csv(os.path.join(args.out_dir, "feature_importance_overall.csv"), rows, fields_overall)

    top_rows = []
    by_dir = os.path.join(args.out_dir, "by_timestep")
    os.makedirs(by_dir, exist_ok=True)
    fields_t = [
        "t", "rank", "feature", "pred_x0_delta_mse", "norm_to_t_mean", "eps_delta_mse",
        "pred_x0_clean_delta", "pred_x0_mse_to_clean", "active_frac", "coeff_abs_mean", "coeff_max",
    ]
    for ti, t in enumerate(t_values):
        order = torch.argsort(pred_by_t[ti], descending=True).tolist()
        t_rows = []
        for rank, fid in enumerate(order, start=1):
            rec = {
                "t": t,
                "rank": rank,
                "feature": int(fid),
                "pred_x0_delta_mse": float(pred_by_t[ti, fid]),
                "norm_to_t_mean": float(pred_by_t[ti, fid] / pred_by_t[ti].mean().clamp_min(1e-12)),
                "eps_delta_mse": float(eps_by_t[ti, fid]),
                "pred_x0_clean_delta": float(clean_delta_by_t[ti, fid]),
                "pred_x0_mse_to_clean": float(clean_mse_by_t[ti, fid]),
                "active_frac": float(active_by_t[ti, fid]),
                "coeff_abs_mean": float(coeff_abs_by_t[ti, fid]),
                "coeff_max": float(coeff_max_by_t[ti, fid]),
            }
            t_rows.append(rec)
            if rank <= args.top_k:
                top_rows.append(rec)
        write_csv(os.path.join(by_dir, f"t_{t:03d}_top_features.csv"), t_rows, fields_t)

    write_csv(os.path.join(args.out_dir, "top_features_by_timestep.csv"), top_rows, fields_t)
    torch.save({
        "t_values": t_values,
        "feature_ids": feature_ids,
        "pred_x0_delta_mse_by_t": pred_by_t,
        "eps_delta_mse_by_t": eps_by_t,
        "pred_x0_clean_delta_by_t": clean_delta_by_t,
        "active_frac_by_t": active_by_t,
        "coeff_abs_mean_by_t": coeff_abs_by_t,
    }, os.path.join(args.out_dir, "aggregated_importance.pt"))

    summary = {
        "primary_metric": "pred_x0_delta_mse = mean((pred_x0_ablate - pred_x0_original)^2), averaged over 12 images",
        "secondary_metric": "eps_delta_mse = mean((eps_ablate - eps_original)^2)",
        "t_values": t_values,
        "top_k": args.top_k,
        "top_overall_features": [int(r["feature"]) for r in rows[:args.top_k]],
        "top_feature_by_t": {str(t): int(torch.argmax(pred_by_t[ti]).item()) for ti, t in enumerate(t_values)},
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"out_dir": args.out_dir, "top_overall": summary["top_overall_features"][:10]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
