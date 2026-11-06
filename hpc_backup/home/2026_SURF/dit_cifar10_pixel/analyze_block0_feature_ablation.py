#!/usr/bin/env python3
import argparse
import csv
import json
import os
from glob import glob

import torch

torch.set_num_threads(4)


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def minmax_norm(x):
    lo = x.min()
    hi = x.max()
    return (x - lo) / (hi - lo).clamp_min(1e-12)


def aggregate_shards(result_dir):
    paths = sorted(glob(os.path.join(result_dir, "block0_ablation_features_*.pt")))
    if not paths:
        raise FileNotFoundError(f"no block0_ablation_features_*.pt files found in {result_dir}")

    first = torch_load(paths[0])
    streams = tuple(first["streams"])
    t_values = tuple(int(x) for x in first["t_values"])
    scales = tuple(float(x) for x in first["scales"])
    num_streams = len(streams)
    num_t = len(t_values)
    num_features_total = 6144

    local_mean = torch.empty(num_streams, num_features_total, num_t)
    local_max = torch.empty(num_streams, num_features_total, num_t)
    local_zero = torch.empty(num_streams, num_features_total, num_t)
    local_pos = torch.empty(num_streams, num_features_total, num_t)
    local_neg = torch.empty(num_streams, num_features_total, num_t)
    damage_mean = torch.empty(num_streams, num_features_total, num_t)
    pred_x0_damage_mean = torch.empty(num_streams, num_features_total, num_t)

    scale_t = torch.tensor(scales)
    zero_idx = int(torch.argmin(scale_t.abs()).item())
    pos_idx = torch.where(scale_t > 0)[0]
    neg_idx = torch.where(scale_t < 0)[0]

    for i, path in enumerate(paths, start=1):
        print(f"loading shard {i}/{len(paths)} {os.path.basename(path)}", flush=True)
        shard = torch_load(path)
        feature_ids = shard["feature_ids"].long()
        metrics = shard["metrics"]

        # Shape in shard: stream, timestep, image, feature-in-shard, scale.
        eps_delta = metrics["eps_delta_mse"].float()
        eps_damage = metrics["eps_mse_to_noise_delta"].float()
        x0_damage = metrics["pred_x0_mse_delta"].float()

        # Store as stream, global_feature, timestep.
        local_mean[:, feature_ids, :] = eps_delta.mean(dim=(2, 4)).transpose(1, 2)
        local_max[:, feature_ids, :] = eps_delta.amax(dim=(2, 4)).transpose(1, 2)
        local_zero[:, feature_ids, :] = eps_delta[..., zero_idx].mean(dim=2).transpose(1, 2)
        local_pos[:, feature_ids, :] = eps_delta.index_select(4, pos_idx).mean(dim=(2, 4)).transpose(1, 2)
        local_neg[:, feature_ids, :] = eps_delta.index_select(4, neg_idx).mean(dim=(2, 4)).transpose(1, 2)
        damage_mean[:, feature_ids, :] = eps_damage.mean(dim=(2, 4)).transpose(1, 2)
        pred_x0_damage_mean[:, feature_ids, :] = x0_damage.mean(dim=(2, 4)).transpose(1, 2)
        print(f"aggregated shard {i}/{len(paths)}", flush=True)

    return {
        "streams": streams,
        "t_values": t_values,
        "scales": scales,
        "local_mean": local_mean,
        "local_max": local_max,
        "local_zero": local_zero,
        "local_pos": local_pos,
        "local_neg": local_neg,
        "damage_mean": damage_mean,
        "pred_x0_damage_mean": pred_x0_damage_mean,
    }


def summarize(agg, out_dir, top_k):
    ensure_dir(out_dir)
    streams = agg["streams"]
    t_values = agg["t_values"]
    local = agg["local_mean"]
    local_max = agg["local_max"]
    zero = agg["local_zero"]
    pos = agg["local_pos"]
    neg = agg["local_neg"]
    damage = agg["damage_mean"]
    x0_damage = agg["pred_x0_damage_mean"]

    all_rows = []
    per_t_rows = []
    top_feature_ids = {}

    early_ids = [i for i, t in enumerate(t_values) if t <= 200]
    mid_ids = [i for i, t in enumerate(t_values) if 300 <= t <= 700]
    late_ids = [i for i, t in enumerate(t_values) if t >= 800]

    for si, stream in enumerate(streams):
        stream_local = local[si]
        stream_norm = stream_local / stream_local.mean().clamp_min(1e-12)
        overall = stream_local.mean(dim=1)
        peak_score, peak_t_idx = stream_local.max(dim=1)
        denom = stream_local.sum(dim=1, keepdim=True).clamp_min(1e-12)
        p_t = stream_local / denom
        entropy = -(p_t * (p_t.clamp_min(1e-12).log())).sum(dim=1)
        entropy_norm = entropy / torch.log(torch.tensor(float(len(t_values))))
        specificity = 1.0 - entropy_norm

        early = stream_local[:, early_ids].mean(dim=1)
        mid = stream_local[:, mid_ids].mean(dim=1)
        late = stream_local[:, late_ids].mean(dim=1)
        signed_asym = (pos[si].mean(dim=1) - neg[si].mean(dim=1)) / (
            pos[si].mean(dim=1) + neg[si].mean(dim=1)
        ).clamp_min(1e-12)

        ranked = torch.argsort(overall, descending=True)
        top_feature_ids[stream] = ranked[:top_k].tolist()
        for fid in ranked.tolist():
            all_rows.append(
                {
                    "stream": stream,
                    "feature": fid,
                    "overall_eps_delta_mse": float(overall[fid]),
                    "overall_norm_to_stream_mean": float(overall[fid] / overall.mean().clamp_min(1e-12)),
                    "peak_t": t_values[int(peak_t_idx[fid])],
                    "peak_eps_delta_mse": float(peak_score[fid]),
                    "peak_norm_to_stream_mean": float(peak_score[fid] / stream_local.mean().clamp_min(1e-12)),
                    "time_specificity": float(specificity[fid]),
                    "early_mean": float(early[fid]),
                    "mid_mean": float(mid[fid]),
                    "late_mean": float(late[fid]),
                    "zero_ablation_mean": float(zero[si, fid].mean()),
                    "positive_scale_mean": float(pos[si, fid].mean()),
                    "negative_scale_mean": float(neg[si, fid].mean()),
                    "signed_scale_asymmetry": float(signed_asym[fid]),
                    "global_eps_damage_mean": float(damage[si, fid].mean()),
                    "pred_x0_damage_mean": float(x0_damage[si, fid].mean()),
                }
            )

        for ti, t in enumerate(t_values):
            scores = stream_local[:, ti]
            top = torch.argsort(scores, descending=True)[:top_k]
            for rank, fid_t in enumerate(top.tolist(), start=1):
                per_t_rows.append(
                    {
                        "stream": stream,
                        "t": t,
                        "rank": rank,
                        "feature": fid_t,
                        "eps_delta_mse": float(scores[fid_t]),
                        "norm_to_stream_t_mean": float(scores[fid_t] / scores.mean().clamp_min(1e-12)),
                        "max_over_images_scales": float(local_max[si, fid_t, ti]),
                        "zero_ablation": float(zero[si, fid_t, ti]),
                        "positive_scale_mean": float(pos[si, fid_t, ti]),
                        "negative_scale_mean": float(neg[si, fid_t, ti]),
                        "global_eps_damage": float(damage[si, fid_t, ti]),
                        "pred_x0_damage": float(x0_damage[si, fid_t, ti]),
                    }
                )

    all_rows.sort(key=lambda r: r["overall_eps_delta_mse"], reverse=True)
    write_csv(
        os.path.join(out_dir, "feature_importance_overall.csv"),
        all_rows,
        [
            "stream",
            "feature",
            "overall_eps_delta_mse",
            "overall_norm_to_stream_mean",
            "peak_t",
            "peak_eps_delta_mse",
            "peak_norm_to_stream_mean",
            "time_specificity",
            "early_mean",
            "mid_mean",
            "late_mean",
            "zero_ablation_mean",
            "positive_scale_mean",
            "negative_scale_mean",
            "signed_scale_asymmetry",
            "global_eps_damage_mean",
            "pred_x0_damage_mean",
        ],
    )
    write_csv(
        os.path.join(out_dir, "top_features_by_timestep.csv"),
        per_t_rows,
        [
            "stream",
            "t",
            "rank",
            "feature",
            "eps_delta_mse",
            "norm_to_stream_t_mean",
            "max_over_images_scales",
            "zero_ablation",
            "positive_scale_mean",
            "negative_scale_mean",
            "global_eps_damage",
            "pred_x0_damage",
        ],
    )

    compact = {
        "streams": streams,
        "t_values": t_values,
        "scales": agg["scales"],
        "primary_metric": "eps_delta_mse = mean squared change in epsilon prediction versus original unperturbed model prediction",
        "top_k": top_k,
        "top_features": top_feature_ids,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)

    torch.save(
        {
            "streams": streams,
            "t_values": t_values,
            "scales": agg["scales"],
            "local_mean": local,
            "local_max": local_max,
            "local_zero": zero,
            "local_pos": pos,
            "local_neg": neg,
            "damage_mean": damage,
            "pred_x0_damage_mean": x0_damage,
        },
        os.path.join(out_dir, "aggregated_importance.pt"),
    )

    return top_feature_ids


def plot_results(agg, out_dir, top_feature_ids):
    plt = maybe_import_plotting()
    if plt is None:
        return False

    streams = agg["streams"]
    t_values = agg["t_values"]
    local = agg["local_mean"]
    ensure_dir(out_dir)

    for si, stream in enumerate(streams):
        scores = local[si]
        overall = scores.mean(dim=1)
        top = torch.argsort(overall, descending=True)[:50]

        plt.figure(figsize=(8, 5))
        mean_curve = scores.mean(dim=0)
        p90_curve = scores.quantile(0.90, dim=0)
        p99_curve = scores.quantile(0.99, dim=0)
        plt.plot(t_values, mean_curve, label="mean feature")
        plt.plot(t_values, p90_curve, label="p90 feature")
        plt.plot(t_values, p99_curve, label="p99 feature")
        plt.xlabel("diffusion timestep")
        plt.ylabel("eps_delta_mse")
        plt.title(f"Block 0 {stream} SAE feature perturbation effect")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{stream}_importance_distribution_vs_t.png"), dpi=180)
        plt.close()

        heat = minmax_norm(scores[top])
        plt.figure(figsize=(8, 9))
        plt.imshow(heat.numpy(), aspect="auto", interpolation="nearest")
        plt.xticks(range(len(t_values)), t_values, rotation=45)
        plt.yticks(range(len(top)), [str(int(x)) for x in top])
        plt.xlabel("diffusion timestep")
        plt.ylabel("feature id, top 50 overall")
        plt.colorbar(label="min-max normalized eps_delta_mse")
        plt.title(f"Block 0 {stream}: top feature time profiles")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{stream}_top50_feature_time_heatmap.png"), dpi=180)
        plt.close()

        plt.figure(figsize=(8, 5))
        for fid in top[:10].tolist():
            y = scores[fid]
            plt.plot(t_values, y / y.max().clamp_min(1e-12), label=str(fid))
        plt.xlabel("diffusion timestep")
        plt.ylabel("within-feature normalized effect")
        plt.title(f"Block 0 {stream}: top 10 temporal profiles")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{stream}_top10_temporal_profiles.png"), dpi=180)
        plt.close()

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="runs/block0_feature_ablation_16x_k32")
    parser.add_argument("--out-dir", default="runs/block0_feature_ablation_16x_k32/analysis")
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    print("starting aggregate", flush=True)
    agg = aggregate_shards(args.result_dir)
    print("writing summaries", flush=True)
    top_feature_ids = summarize(agg, args.out_dir, args.top_k)
    print("making plots", flush=True)
    plotted = plot_results(agg, args.out_dir, top_feature_ids)
    print(json.dumps({"out_dir": args.out_dir, "plotted": plotted, "top_k": args.top_k}, indent=2), flush=True)


if __name__ == "__main__":
    main()
