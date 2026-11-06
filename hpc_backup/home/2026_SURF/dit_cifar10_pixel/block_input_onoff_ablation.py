#!/usr/bin/env python3
import argparse, csv, json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from train_dit_cifar import DiT, Diffusion

T_VALUES = (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999)

class TopKSAE(nn.Module):
    def __init__(self, input_dim=384, dict_size=6144, k=32):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.k = k
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Parameter(torch.empty(dict_size, input_dim))
        self.dec_bias = nn.Parameter(torch.zeros(input_dim))
    def forward(self, x):
        pre = self.encoder(x - self.dec_bias)
        acts = F.relu(pre)
        vals, idx = torch.topk(acts, self.k, dim=-1)
        recon = (vals.unsqueeze(-1) * self.decoder[idx]).sum(dim=1) + self.dec_bias
        return recon, vals, idx

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

def load_sae(path, block_index, device, input_dim=384, dict_size=6144, k=32):
    ckpt = load_pt(path, device)
    sae = TopKSAE(input_dim=input_dim, dict_size=dict_size, k=k).to(device)
    sae.load_state_dict(ckpt["saes"][f"block{block_index}_input"])
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae

@torch.no_grad()
def make_batch(data_dir, num_images, seed, device):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
    dataset = datasets.CIFAR10(data_dir, train=True, transform=transform, download=True)
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=gen)[:num_images].tolist()
    xs, ys = [], []
    for idx in indices:
        x, y = dataset[idx]
        xs.append(x); ys.append(y)
    return torch.stack(xs).to(device), torch.tensor(ys, device=device, dtype=torch.long), indices

@torch.no_grad()
def get_block_input_and_c(model, diffusion, x0, labels, t, block_index, precision, noise):
    x_t = diffusion.q_sample(x0, t, noise)
    enabled = x_t.device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type=x_t.device.type, dtype=torch.bfloat16, enabled=enabled):
        x = model.x_embedder(x_t).flatten(2).transpose(1, 2) + model.pos_embed
        c = model.t_embedder(t) + model.y_embedder(labels)
        for i in range(block_index):
            x = model.blocks[i](x, c)
    return x.float(), c.float(), x_t, noise

@torch.no_grad()
def finish_from_block(model, x, c, block_index, precision):
    enabled = x.device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=enabled):
        for i in range(block_index, len(model.blocks)):
            x = model.blocks[i](x, c)
        out = model.unpatchify(model.final_layer(x, c))
    return out.float()

@torch.no_grad()
def encode_dense(sae, x_flat):
    pre = sae.encoder(x_flat - sae.dec_bias)
    acts = F.relu(pre)
    vals, idx = torch.topk(acts, sae.k, dim=-1)
    dense = torch.zeros(x_flat.shape[0], sae.dict_size, device=x_flat.device, dtype=x_flat.dtype)
    dense.scatter_(1, idx, vals)
    return dense.reshape(-1, 256, sae.dict_size)

def empty_metrics(num_t, num_images, num_features):
    shape = (num_t, num_images, num_features)
    return {k: torch.empty(shape, dtype=torch.float32) for k in [
        "eps_delta_mse", "eps_delta_l2", "pred_x0_delta_mse", "pred_x0_mse_to_clean",
        "pred_x0_mse_to_clean_delta", "active_frac", "coeff_abs_mean", "coeff_max"]}

def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def analyze(result_path, out_dir):
    obj = load_pt(result_path, "cpu")
    m = obj["metrics"]
    t_values = [int(x) for x in obj["t_values"]]
    pred = m["pred_x0_delta_mse"].float().mean(dim=1)
    eps = m["eps_delta_mse"].float().mean(dim=1)
    clean_delta = m["pred_x0_mse_to_clean_delta"].float().mean(dim=1)
    active = m["active_frac"].float().mean(dim=1)
    coeff_abs = m["coeff_abs_mean"].float().mean(dim=1)
    coeff_max = m["coeff_max"].float().amax(dim=1)
    by_dir = os.path.join(out_dir, "by_timestep")
    os.makedirs(by_dir, exist_ok=True)
    fields_t = ["t","rank","feature","pred_x0_delta_mse","norm_to_t_mean","eps_delta_mse","pred_x0_clean_delta","active_frac","coeff_abs_mean","coeff_max"]
    top_rows = []
    for ti, t in enumerate(t_values):
        order = torch.argsort(pred[ti], descending=True).tolist()
        rows = []
        denom = pred[ti].mean().clamp_min(1e-12)
        for rank, fid in enumerate(order, 1):
            rec = {"t": t, "rank": rank, "feature": int(fid), "pred_x0_delta_mse": float(pred[ti,fid]),
                   "norm_to_t_mean": float(pred[ti,fid] / denom), "eps_delta_mse": float(eps[ti,fid]),
                   "pred_x0_clean_delta": float(clean_delta[ti,fid]), "active_frac": float(active[ti,fid]),
                   "coeff_abs_mean": float(coeff_abs[ti,fid]), "coeff_max": float(coeff_max[ti,fid])}
            rows.append(rec)
            if rank <= 50: top_rows.append(rec)
        write_csv(os.path.join(by_dir, f"t_{t:03d}_top_features.csv"), rows, fields_t)
    write_csv(os.path.join(out_dir, "top_features_by_timestep.csv"), top_rows, fields_t)
    overall = pred.mean(dim=0); overall_eps = eps.mean(dim=0); overall_clean = clean_delta.mean(dim=0)
    overall_active = active.mean(dim=0); overall_coeff = coeff_abs.mean(dim=0); peak, peak_idx = pred.max(dim=0)
    early = [i for i,t in enumerate(t_values) if t <= 200]
    mid = [i for i,t in enumerate(t_values) if 300 <= t <= 700]
    late = [i for i,t in enumerate(t_values) if t >= 800]
    fields_o = ["feature","overall_pred_x0_delta_mse","overall_norm_to_mean","overall_eps_delta_mse","overall_pred_x0_clean_delta","overall_active_frac","overall_coeff_abs_mean","peak_t","peak_pred_x0_delta_mse","early_pred_x0_delta_mse","mid_pred_x0_delta_mse","late_pred_x0_delta_mse"]
    rows = []
    denom = overall.mean().clamp_min(1e-12)
    for fid in torch.argsort(overall, descending=True).tolist():
        rows.append({"feature": int(fid), "overall_pred_x0_delta_mse": float(overall[fid]),
            "overall_norm_to_mean": float(overall[fid]/denom), "overall_eps_delta_mse": float(overall_eps[fid]),
            "overall_pred_x0_clean_delta": float(overall_clean[fid]), "overall_active_frac": float(overall_active[fid]),
            "overall_coeff_abs_mean": float(overall_coeff[fid]), "peak_t": t_values[int(peak_idx[fid])],
            "peak_pred_x0_delta_mse": float(peak[fid]), "early_pred_x0_delta_mse": float(pred[early,fid].mean()),
            "mid_pred_x0_delta_mse": float(pred[mid,fid].mean()), "late_pred_x0_delta_mse": float(pred[late,fid].mean())})
    write_csv(os.path.join(out_dir, "feature_importance_overall.csv"), rows, fields_o)
    torch.save({"t_values": t_values, "pred_x0_delta_mse_by_t": pred, "eps_delta_mse_by_t": eps,
        "pred_x0_clean_delta_by_t": clean_delta, "active_frac_by_t": active, "coeff_abs_mean_by_t": coeff_abs},
        os.path.join(out_dir, "aggregated_importance.pt"))
    summary = {"primary_metric":"pred_x0_delta_mse = mean((pred_x0_ablate - pred_x0_original)^2), averaged over 12 images",
        "secondary_metric":"eps_delta_mse", "t_values": t_values, "top_overall_features":[int(r["feature"]) for r in rows[:50]],
        "top_feature_by_t":{str(t): int(torch.argmax(pred[ti]).item()) for ti,t in enumerate(t_values)}}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f: json.dump(summary, f, indent=2)
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-index", type=int, required=True)
    ap.add_argument("--ckpt", default="runs/dit_s2_cifar10/last.pt")
    ap.add_argument("--sae-ckpt", default="runs/sae_block_inputs_16x_k32_long/last.pt")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-root", default="runs/block_inputs_onoff_ablation_16x_k32")
    ap.add_argument("--num-images", type=int, default=12)
    ap.add_argument("--t-values", type=parse_ints, default=T_VALUES)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--features-per-forward", type=int, default=8)
    ap.add_argument("--forward-batch-size", type=int, default=96)
    ap.add_argument("--precision", choices=["fp32","bf16"], default="bf16")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    b = args.block_index
    out_dir = os.path.join(args.out_root, f"block{b:02d}_input")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f: json.dump(vars(args), f, indent=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    model = load_model(args.ckpt, device); sae = load_sae(args.sae_ckpt, b, device)
    diffusion = Diffusion(args.timesteps, device=device)
    x0, labels, indices = make_batch(args.data_dir, args.num_images, args.seed, device)
    metrics = empty_metrics(len(args.t_values), args.num_images, sae.dict_size)
    baselines = {"eps_mse_to_noise": torch.empty((len(args.t_values), args.num_images)),
                 "pred_x0_mse_to_clean": torch.empty((len(args.t_values), args.num_images))}
    decoder = sae.decoder.float(); gen = torch.Generator(device=device).manual_seed(args.seed + 1000)
    start = time.time()
    for ti, t_int in enumerate(args.t_values):
        t = torch.full((args.num_images,), t_int, device=device, dtype=torch.long)
        noise = torch.randn(x0.shape, device=device, generator=gen)
        x_in, c, x_t, _ = get_block_input_and_c(model, diffusion, x0, labels, t, b, args.precision, noise)
        base_out = finish_from_block(model, x_in, c, b, args.precision)
        base_eps = torch.chunk(base_out, 2, dim=1)[0]
        sqrt_ac = diffusion.extract(diffusion.sqrt_ac, t, x_t.shape); sqrt_om = diffusion.extract(diffusion.sqrt_one_minus_ac, t, x_t.shape)
        base_pred_x0 = (x_t - sqrt_om * base_eps) / sqrt_ac
        base_x0_mse = (base_pred_x0 - x0).pow(2).flatten(1).mean(dim=1)
        baselines["eps_mse_to_noise"][ti] = (base_eps - noise).pow(2).flatten(1).mean(dim=1).cpu()
        baselines["pred_x0_mse_to_clean"][ti] = base_x0_mse.cpu()
        dense = encode_dense(sae, x_in.flatten(0,1))
        for f0 in range(0, sae.dict_size, args.features_per_forward):
            f1 = min(f0 + args.features_per_forward, sae.dict_size)
            states=[]; image_ids=[]; feature_ids=[]
            for ii in range(args.num_images):
                coeff = dense[ii,:,f0:f1].float(); x_img = x_in[ii]
                for lf, fid in enumerate(range(f0,f1)):
                    delta = -coeff[:,lf].unsqueeze(1) * decoder[fid].unsqueeze(0)
                    states.append(x_img + delta); image_ids.append(ii); feature_ids.append(fid)
                    metrics["active_frac"][ti,ii,fid] = (coeff[:,lf] > 0).float().mean().cpu()
                    metrics["coeff_abs_mean"][ti,ii,fid] = coeff[:,lf].abs().mean().cpu()
                    metrics["coeff_max"][ti,ii,fid] = coeff[:,lf].max().cpu()
            states = torch.stack(states, dim=0); ids_t = torch.tensor(image_ids, device=device, dtype=torch.long); c_batch = c[ids_t]
            for s0 in range(0, states.shape[0], args.forward_batch_size):
                s1 = min(s0 + args.forward_batch_size, states.shape[0])
                out = finish_from_block(model, states[s0:s1], c_batch[s0:s1], b, args.precision)
                eps = torch.chunk(out, 2, dim=1)[0]; ids = ids_t[s0:s1]
                local_x_t = x_t[ids]; local_x0 = x0[ids]; local_t = t[ids]
                lsa = diffusion.extract(diffusion.sqrt_ac, local_t, local_x_t.shape); lso = diffusion.extract(diffusion.sqrt_one_minus_ac, local_t, local_x_t.shape)
                pred_x0 = (local_x_t - lso * eps) / lsa
                eps_delta = eps - base_eps[ids]; pred_delta = pred_x0 - base_pred_x0[ids]
                vals = {"eps_delta_mse": eps_delta.pow(2).flatten(1).mean(dim=1).cpu(),
                    "eps_delta_l2": eps_delta.flatten(1).norm(dim=1).cpu(),
                    "pred_x0_delta_mse": pred_delta.pow(2).flatten(1).mean(dim=1).cpu(),
                    "pred_x0_mse_to_clean": (pred_x0 - local_x0).pow(2).flatten(1).mean(dim=1).cpu()}
                vals["pred_x0_mse_to_clean_delta"] = vals["pred_x0_mse_to_clean"] - base_x0_mse[ids].cpu()
                for li, gi in enumerate(range(s0,s1)):
                    ii = image_ids[gi]; fid = feature_ids[gi]
                    for k,v in vals.items(): metrics[k][ti,ii,fid] = v[li]
            if f1 % 1024 == 0 or f1 == sae.dict_size:
                print(f"block={b} t={t_int} features=[0,{f1}) elapsed={time.time()-start:.1f}s", flush=True)
    result_path = os.path.join(out_dir, f"block{b:02d}_input_onoff_ablation.pt")
    torch.save({"metrics": metrics, "baselines": baselines, "t_values": tuple(args.t_values), "feature_ids": torch.arange(sae.dict_size),
        "labels": labels.cpu(), "cifar_train_indices": indices, "config": vars(args),
        "intervention":"set actual TopK coefficient to zero for one block-input SAE feature at all 256 tokens",
        "primary_local_metric":"pred_x0_delta_mse and eps_delta_mse versus original unperturbed prediction"}, result_path)
    summary = analyze(result_path, os.path.join(out_dir, "analysis"))
    print(json.dumps({"block": b, "result": result_path, "top10": summary["top_overall_features"][:10]}, indent=2), flush=True)

if __name__ == "__main__": main()
