#!/usr/bin/env python3
import csv, json, os
import torch

result = "runs/block0_input_onoff_ablation_16x_k32/block0_input_onoff_ablation.pt"
out_dir = "runs/block0_input_onoff_ablation_16x_k32/analysis"
by_dir = os.path.join(out_dir, "by_timestep")
os.makedirs(by_dir, exist_ok=True)

try:
    obj = torch.load(result, map_location="cpu", weights_only=False)
except TypeError:
    obj = torch.load(result, map_location="cpu")
metrics = obj["metrics"]
t_values = [int(x) for x in obj["t_values"]]

pred = metrics["pred_x0_delta_mse"].float().mean(dim=1)  # t, feature
eps = metrics["eps_delta_mse"].float().mean(dim=1)
clean_delta = metrics["pred_x0_mse_to_clean_delta"].float().mean(dim=1)
active = metrics["active_frac"].float().mean(dim=1)
coeff_abs = metrics["coeff_abs_mean"].float().mean(dim=1)
coeff_max = metrics["coeff_max"].float().amax(dim=1)

fields_t = ["t","rank","feature","pred_x0_delta_mse","norm_to_t_mean","eps_delta_mse","pred_x0_clean_delta","active_frac","coeff_abs_mean","coeff_max"]
def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

top_rows = []
for ti, t in enumerate(t_values):
    order = torch.argsort(pred[ti], descending=True).tolist()
    rows = []
    denom = pred[ti].mean().clamp_min(1e-12)
    for rank, fid in enumerate(order, 1):
        rec = {
            "t": t,
            "rank": rank,
            "feature": int(fid),
            "pred_x0_delta_mse": float(pred[ti, fid]),
            "norm_to_t_mean": float(pred[ti, fid] / denom),
            "eps_delta_mse": float(eps[ti, fid]),
            "pred_x0_clean_delta": float(clean_delta[ti, fid]),
            "active_frac": float(active[ti, fid]),
            "coeff_abs_mean": float(coeff_abs[ti, fid]),
            "coeff_max": float(coeff_max[ti, fid]),
        }
        rows.append(rec)
        if rank <= 50:
            top_rows.append(rec)
    write_csv(os.path.join(by_dir, f"t_{t:03d}_top_features.csv"), rows, fields_t)
write_csv(os.path.join(out_dir, "top_features_by_timestep.csv"), top_rows, fields_t)

overall = pred.mean(dim=0)
overall_eps = eps.mean(dim=0)
overall_clean_delta = clean_delta.mean(dim=0)
overall_active = active.mean(dim=0)
overall_coeff = coeff_abs.mean(dim=0)
peak, peak_idx = pred.max(dim=0)
early = [i for i,t in enumerate(t_values) if t <= 200]
mid = [i for i,t in enumerate(t_values) if 300 <= t <= 700]
late = [i for i,t in enumerate(t_values) if t >= 800]
fields_o = ["feature","overall_pred_x0_delta_mse","overall_norm_to_mean","overall_eps_delta_mse","overall_pred_x0_clean_delta","overall_active_frac","overall_coeff_abs_mean","peak_t","peak_pred_x0_delta_mse","early_pred_x0_delta_mse","mid_pred_x0_delta_mse","late_pred_x0_delta_mse"]
rows = []
denom = overall.mean().clamp_min(1e-12)
for fid in torch.argsort(overall, descending=True).tolist():
    rows.append({
        "feature": int(fid),
        "overall_pred_x0_delta_mse": float(overall[fid]),
        "overall_norm_to_mean": float(overall[fid] / denom),
        "overall_eps_delta_mse": float(overall_eps[fid]),
        "overall_pred_x0_clean_delta": float(overall_clean_delta[fid]),
        "overall_active_frac": float(overall_active[fid]),
        "overall_coeff_abs_mean": float(overall_coeff[fid]),
        "peak_t": t_values[int(peak_idx[fid])],
        "peak_pred_x0_delta_mse": float(peak[fid]),
        "early_pred_x0_delta_mse": float(pred[early, fid].mean()),
        "mid_pred_x0_delta_mse": float(pred[mid, fid].mean()),
        "late_pred_x0_delta_mse": float(pred[late, fid].mean()),
    })
write_csv(os.path.join(out_dir, "feature_importance_overall.csv"), rows, fields_o)

torch.save({"t_values": t_values, "pred_x0_delta_mse_by_t": pred, "eps_delta_mse_by_t": eps, "pred_x0_clean_delta_by_t": clean_delta, "active_frac_by_t": active, "coeff_abs_mean_by_t": coeff_abs}, os.path.join(out_dir, "aggregated_importance.pt"))
summary = {"primary_metric":"pred_x0_delta_mse = mean((pred_x0_ablate - pred_x0_original)^2), averaged over 12 images", "secondary_metric":"eps_delta_mse", "t_values":t_values, "top_overall_features":[int(r["feature"]) for r in rows[:50]], "top_feature_by_t":{str(t): int(torch.argmax(pred[ti]).item()) for ti,t in enumerate(t_values)}}
with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f: json.dump(summary, f, indent=2)
print(json.dumps({"out_dir":out_dir, "top10":summary["top_overall_features"][:10], "top_by_t":summary["top_feature_by_t"]}, indent=2), flush=True)
