#!/usr/bin/env python3
"""Save multi-feature SAE interventions using frozen TopK support and Diff-AE sampling."""
import argparse, json, os, sys
from pathlib import Path
import torch
from PIL import Image, ImageDraw


def load_sae(path, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    from train_zsem_sae_final import TopKSAE
    cfg = state['config']; d = 512; k = int(cfg['k']); h = int(cfg['dict_mult']) * d
    sae = TopKSAE(d, h, k, int(cfg.get('aux_k', k))).to(device)
    sae.load_state_dict(state['model'], strict=True); sae.eval()
    return sae, state['input_mean'].float().to(device), state['input_std'].float().to(device)


def save_grid(images, path, labels, cell=128):
    rows, cols = len(images), len(images[0]); out = Image.new('RGB', (cols * cell, rows * (cell + 18)), 'white')
    draw = ImageDraw.Draw(out)
    for r, row in enumerate(images):
        for c, image in enumerate(row):
            image = (image.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype('uint8')
            out.paste(Image.fromarray(image), (c * cell, r * (cell + 18)))
            draw.text((c * cell + 2, r * (cell + 18) + cell), labels[c], fill='black')
    out.save(path)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--repo', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--codes', required=True); p.add_argument('--sae-root', required=True); p.add_argument('--out', required=True)
    p.add_argument('--examples', type=int, default=12); p.add_argument('--seed', type=int, default=2026); p.add_argument('--steps', type=int, default=20)
    args = p.parse_args(); os.makedirs(args.out, exist_ok=True); sys.path.insert(0, os.path.abspath(args.repo))
    from experiment import LitModel
    from templates import ffhq128_autoenc_130M
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conf = ffhq128_autoenc_130M(); conf.pretrain = None; conf.latent_infer_path = None
    lit = LitModel(conf); state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    lit.load_state_dict(state['state_dict'], strict=False); lit.eval().to(device)
    codes = torch.load(args.codes, map_location='cpu', weights_only=False)['conds'].float()
    gen = torch.Generator().manual_seed(args.seed); ids = torch.randperm(len(codes), generator=gen)[:args.examples]
    z_raw = codes[ids].to(device); noise = torch.randn(len(ids), 3, 128, 128, generator=gen).to(device)
    for name, k in [('k32', 32), ('k64', 64), ('k128', 128)]:
        sae_path = os.path.join(args.sae_root, {'k32': 'final_32x_k32_aux_reinit_merged', 'k64': 'final_32x_k64_aux_reinit_merged', 'k128': 'final_32x_k128_aux_reinit_merged'}[name], 'last.pt')
        if not os.path.exists(sae_path): continue
        sae, mean, std = load_sae(sae_path, device)
        x = (z_raw - mean) / std; _, vals, idx, _ = sae.encode(x)
        # Freeze the support selected by the clean input; no feature replacement at intervention time.
        active = vals > 0
        global_sum = torch.zeros(sae.h, device=device); global_count = torch.zeros(sae.h, device=device)
        batch = ((codes - mean.cpu()) / std.cpu())
        for start in range(0, len(batch), 2048):
            _, vv, ii, _ = sae.encode(batch[start:start + 2048].to(device))
            global_sum.scatter_add_(0, ii.reshape(-1), vv.reshape(-1)); global_count.scatter_add_(0, ii.reshape(-1), torch.ones_like(vv).reshape(-1))
        rank = torch.argsort(global_sum / len(codes), descending=True)
        selected = []
        for row in range(len(ids)):
            active_ids = idx[row][active[row]].tolist(); active_ids.sort(key=lambda f: int((rank == f).nonzero()[0]))
            selected.append(active_ids)
        sigma = torch.sqrt((global_sum / len(codes)).new_zeros(sae.h))
        # Estimate all-sample std from the active TopK values, including implicit zeros.
        second = torch.zeros(sae.h, device=device)
        for start in range(0, len(batch), 2048):
            _, vv, ii, _ = sae.encode(batch[start:start + 2048].to(device))
            second.scatter_add_(0, ii.reshape(-1), (vv * vv).reshape(-1))
        mu = global_sum / len(codes); sigma = (second / len(codes) - mu.square()).clamp_min(0).sqrt()
        baseline = lit.render(noise, cond=z_raw, T=args.steps).clamp(0, 1)
        for set_size in (1, 2, 4, 8):
            rows = [baseline]
            labels = ['base']
            for mult in (-10, -4, -2, 0, 2, 4, 10):
                modified = z_raw.clone()
                for row in range(len(ids)):
                    feats = selected[row][:min(set_size, len(selected[row]))]
                    for f in feats:
                        # The original support remains active even for a negative target.
                        clean_value = vals[row, (idx[row] == f).nonzero()[0, 0]]
                        target = torch.clamp(clean_value + mult * sigma[f], min=0)
                        coeff_delta = (target - clean_value) * sae.decoder[f] * std
                        modified[row] += coeff_delta
                rows.append(lit.render(noise, cond=modified, T=args.steps).clamp(0, 1)); labels.append(f'{mult:+d}s')
            images = [[rows[c][r] for c in range(len(rows))] for r in range(len(ids))]
            save_grid(images, os.path.join(args.out, f'{name}_set{set_size}.png'), labels)
        with open(os.path.join(args.out, f'{name}_metadata.json'), 'w') as f:
            json.dump({'ids': ids.tolist(), 'k': k, 'selected_features': selected, 'sigma_definition': 'all-sample std including implicit zeros', 'frozen_support': True}, f, indent=2)
        del sae


if __name__ == '__main__': main()
