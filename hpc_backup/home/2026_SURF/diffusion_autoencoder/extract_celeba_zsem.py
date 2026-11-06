#!/usr/bin/env python3
"""Stream CelebA and extract 512-D semantic codes with the official Diff-AE encoder."""
import argparse, os
import torch
from PIL import Image
from torchvision import transforms
from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True); p.add_argument('--split', default='train')
    p.add_argument('--batch-size', type=int, default=256); p.add_argument('--max-images', type=int, default=0)
    args = p.parse_args()
    import sys
    sys.path.insert(0, os.path.abspath(args.repo))
    from experiment import LitModel
    from templates import ffhq128_autoenc_130M
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conf = ffhq128_autoenc_130M(); conf.pretrain = None; conf.latent_infer_path = None
    lit = LitModel(conf)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    lit.load_state_dict(state['state_dict'], strict=False); lit.eval().to(device)
    transform = transforms.Compose([
        transforms.Resize(128), transforms.CenterCrop(128), transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data = load_dataset('flwrlabs/celeba', split=args.split, streaming=True)
    codes, batch = [], []
    limit = args.max_images or None
    with torch.no_grad():
        for i, row in enumerate(data):
            if limit is not None and i >= limit: break
            image = row['image'].convert('RGB') if isinstance(row['image'], Image.Image) else Image.fromarray(row['image']).convert('RGB')
            batch.append(transform(image))
            if len(batch) == args.batch_size:
                codes.append(lit.ema_model.encoder(torch.stack(batch).to(device)).cpu().float())
                batch.clear()
                if (i + 1) % (args.batch_size * 20) == 0: print(f'extracted {i + 1}', flush=True)
        if batch: codes.append(lit.ema_model.encoder(torch.stack(batch).to(device)).cpu().float())
    out = torch.cat(codes, dim=0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'conds': out, 'source': 'flwrlabs/celeba', 'split': args.split}, args.out)
    print({'count': len(out), 'shape': list(out.shape), 'out': args.out}, flush=True)


if __name__ == '__main__': main()
