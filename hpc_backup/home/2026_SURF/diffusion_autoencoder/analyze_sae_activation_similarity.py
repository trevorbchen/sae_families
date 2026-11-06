#!/usr/bin/env python3
"""Compare SAE features by decoder direction and co-activation on common FFHQ codes."""
import argparse, csv, json, os, sys
import torch


def load_model(path, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    from train_zsem_sae_final import TopKSAE
    cfg = state['config']; d = 512; k = int(cfg['k'])
    model = TopKSAE(d, int(cfg['dict_mult']) * d, k, int(cfg.get('aux_k', k))).to(device)
    model.load_state_dict(state['model'], strict=True); model.eval()
    dec = model.decoder / model.decoder.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return model, dec, state['input_mean'].float().to(device), state['input_std'].float().to(device)


def nearest(a, b, block=1024):
    vals, inds = [], []
    for start in range(0, len(a), block):
        v, i = torch.max(a[start:start + block] @ b.T, dim=1)
        vals.append(v.cpu()); inds.append(i.cpu())
    return torch.cat(vals), torch.cat(inds)


@torch.no_grad()
def activation_match(model_a, mean_a, std_a, model_b, mean_b, std_b, mapping, codes, device, batch_size):
    h_a, h_b = model_a.h, model_b.h
    active_a = torch.zeros(h_a, device=device); active_b = torch.zeros(h_b, device=device); co = torch.zeros(h_a, device=device)
    for start in range(0, len(codes), batch_size):
        raw = codes[start:start + batch_size]
        xa = ((raw - mean_a.cpu()) / std_a.cpu()).to(device)
        xb = ((raw - mean_b.cpu()) / std_b.cpu()).to(device)
        _, va, ia, _ = model_a.encode(xa); _, vb, ib, _ = model_b.encode(xb)
        ma, mb = va > 0, vb > 0
        active_a.scatter_add_(0, ia[ma], torch.ones_like(ia[ma], dtype=torch.float32))
        active_b.scatter_add_(0, ib[mb], torch.ones_like(ib[mb], dtype=torch.float32))
        mapped = mapping.to(device)[ia]
        # Count how often A's matched B feature is active in the same example.
        hit = (mapped.unsqueeze(-1) == ib.unsqueeze(1)).any(-1) & ma
        co.scatter_add_(0, ia[hit], torch.ones_like(ia[hit], dtype=torch.float32))
    matched_b = mapping.to(device)
    union = active_a + active_b[matched_b] - co
    return active_a.cpu(), active_b.cpu(), co.cpu(), (co / union.clamp_min(1)).cpu()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--repo', required=True); p.add_argument('--codes', required=True); p.add_argument('--root', required=True); p.add_argument('--out', required=True); p.add_argument('--batch-size', type=int, default=512)
    args = p.parse_args(); os.makedirs(args.out, exist_ok=True); sys.path.insert(0, os.path.abspath(args.repo))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    codes = torch.load(args.codes, map_location='cpu', weights_only=False)['conds'].float()
    paths = {k: os.path.join(args.root, f'final_32x_k{k}_aux_reinit_merged', 'last.pt') for k in (32, 64, 128)}
    models = {k: load_model(paths[k], device) for k in paths}
    summary = {}
    for ka, kb in ((32, 64), (32, 128), (64, 128)):
        ma, da, mea, sta = models[ka]; mb, db, meb, stb = models[kb]
        cosine, mapping = nearest(da, db)
        reverse_cosine, reverse_mapping = nearest(db, da)
        mutual = torch.tensor([int(reverse_mapping[int(j)]) == i for i, j in enumerate(mapping)])
        aa, ab, co, jaccard = activation_match(ma, mea, sta, mb, meb, stb, mapping, codes, device, args.batch_size)
        valid = aa >= 10
        rows = []
        for i in range(len(mapping)):
            j = int(mapping[i]); rows.append({'feature_a': i, 'feature_b': j, 'cosine': float(cosine[i]), 'mutual_nn': bool(mutual[i]), 'active_a': int(aa[i]), 'active_b': int(ab[j]), 'coactive': int(co[i]), 'activation_jaccard': float(jaccard[i])})
        name = f'k{ka}_k{kb}'
        with open(os.path.join(args.out, f'{name}_activation.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        entry = {'features': len(rows), 'valid_features_active_ge_10': int(valid.sum()), 'mutual_nn_count': int(mutual.sum())}
        for threshold in (.99, .95, .9, .8, .7, .5):
            entry[f'cosine_ge_{threshold}'] = int((cosine >= threshold).sum())
            entry[f'cosine_ge_{threshold}_fraction'] = float((cosine >= threshold).float().mean())
        for threshold in (.9, .75, .5, .25):
            entry[f'jaccard_ge_{threshold}_among_valid'] = int((jaccard[valid] >= threshold).sum())
            entry[f'jaccard_ge_{threshold}_fraction_among_valid'] = float((jaccard[valid] >= threshold).float().mean()) if valid.any() else 0.0
        joint = valid & (cosine >= .8) & (jaccard >= .5)
        entry['joint_cosine_ge_0.8_and_jaccard_ge_0.5'] = int(joint.sum())
        summary[name] = entry
    with open(os.path.join(args.out, 'activation_summary.json'), 'w') as f: json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
