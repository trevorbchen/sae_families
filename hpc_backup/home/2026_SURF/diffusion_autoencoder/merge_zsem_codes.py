#!/usr/bin/env python3
import argparse, os, torch

p = argparse.ArgumentParser(); p.add_argument('--ffhq', required=True); p.add_argument('--extra', required=True); p.add_argument('--out', required=True); a = p.parse_args()
ffhq = torch.load(a.ffhq, map_location='cpu', weights_only=False)['conds'].float()
extra = torch.load(a.extra, map_location='cpu', weights_only=False)['conds'].float()
merged = torch.cat([ffhq, extra], dim=0)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save({'conds': merged, 'sources': [a.ffhq, a.extra]}, a.out)
print({'ffhq': len(ffhq), 'extra': len(extra), 'total': len(merged), 'out': a.out}, flush=True)
