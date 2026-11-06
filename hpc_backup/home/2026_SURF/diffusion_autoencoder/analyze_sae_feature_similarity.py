#!/usr/bin/env python3
"""Match SAE features across dictionary sizes by decoder-direction similarity."""
import argparse, csv, json, os
import torch


def load_decoder(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    decoder = state['model']['decoder'].float()
    return decoder / decoder.norm(dim=1, keepdim=True).clamp_min(1e-8)


def nearest(a, b, topk, block):
    best_values, best_indices = [], []
    for start in range(0, len(a), block):
        sim = a[start:start + block] @ b.T
        values, indices = torch.topk(sim, min(topk, len(b)), dim=1)
        best_values.append(values.cpu()); best_indices.append(indices.cpu())
    return torch.cat(best_values), torch.cat(best_indices)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--out', required=True); p.add_argument('--topk', type=int, default=5)
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    paths = {k: os.path.join(a.root, f'final_32x_k{k}_aux_reinit_merged', 'last.pt') for k in (32, 64, 128)}
    dec = {k: load_decoder(paths[k]) for k in paths}
    results = {}
    for ka, kb in ((32, 64), (32, 128), (64, 128)):
        vals, inds = nearest(dec[ka], dec[kb], a.topk, 1024)
        reverse_vals, reverse_inds = nearest(dec[kb], dec[ka], 1, 1024)
        rows = []
        for i in range(len(dec[ka])):
            for rank in range(a.topk):
                j = int(inds[i, rank]); score = float(vals[i, rank])
                mutual = int(reverse_inds[j, 0]) == i
                rows.append({'feature_a': i, 'feature_b': j, 'rank': rank + 1, 'cosine': score, 'mutual_nn': mutual})
        name = f'k{ka}_k{kb}'
        with open(os.path.join(a.out, f'{name}.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
        top1 = vals[:, 0]
        results[name] = {'a_features': len(dec[ka]), 'b_features': len(dec[kb]), 'top1_mean_cosine': float(top1.mean()), 'top1_median_cosine': float(top1.median()), 'top1_p95_cosine': float(torch.quantile(top1, .95)), 'mutual_top1_count': int(sum(bool(x) for x in [int(reverse_inds[int(inds[i,0]),0]) == i for i in range(len(dec[ka]))]))}
    with open(os.path.join(a.out, 'summary.json'), 'w') as f: json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__': main()
