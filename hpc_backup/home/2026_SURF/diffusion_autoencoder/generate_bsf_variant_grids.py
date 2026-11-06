#!/usr/bin/env python3
"""Generate controlled Diff-AE image grids for the trained BSF variants."""

import argparse
import os

import torch
from PIL import Image, ImageDraw
from torchvision.utils import make_grid


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--codes", required=True)
    p.add_argument("--variants-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--examples", type=int, default=12)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    import sys
    sys.path.insert(0, os.path.abspath(args.repo))
    from experiment import LitModel
    from templates import ffhq128_autoenc_130M
    from train_zsem_bsf import BlockSparseFeaturizer
    from train_zsem_bsf_variants import GrassmannianBSF

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf = ffhq128_autoenc_130M(); conf.pretrain = None; conf.latent_infer_path = None
    lit = LitModel(conf)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    lit.load_state_dict(state["state_dict"], strict=False); lit.eval().to(device)

    codes_state = torch.load(args.codes, map_location="cpu", weights_only=False)
    codes = codes_state["conds"].float()
    ids = torch.linspace(0, len(codes) - 1, args.examples).round().long()
    raw = codes[ids].to(device)
    x_t = torch.randn(args.examples, 3, 128, 128, device=device,
                       generator=torch.Generator(device=device).manual_seed(args.seed))

    variants = [
        ("raw Diff-AE", None, None),
        ("vanilla d8", "vanilla_d8_k4", "vanilla"),
        ("Grassmannian d8", "grassmannian_d8_k4", "grassmannian"),
        ("vanilla d4", "vanilla_d4_k8", "vanilla"),
        ("vanilla d2", "vanilla_d2_k16", "vanilla"),
    ]
    outputs = []
    model_data = []
    for label, name, kind in variants:
        if name is None:
            cond = raw
        else:
            path = os.path.join(args.variants_root, name, "last.pt")
            s = torch.load(path, map_location="cpu", weights_only=False)
            cfg = s["config"]
            cls = GrassmannianBSF if kind == "grassmannian" else BlockSparseFeaturizer
            model = cls(512, int(cfg["dict_mult"]), int(cfg["block_dim"]), int(cfg["top_blocks"])).to(device)
            model.load_state_dict(s["model"], strict=True); model.eval()
            mean = s["input_mean"].float().to(device); std = s["input_std"].float().to(device)
            x = (raw - mean) / std
            recon, _, _, _ = model(x)
            cond = recon * std + mean
            model_data.append((label, name, model, mean, std, x, recon))
        images = lit.render(x_t, cond=cond, T=args.steps).clamp(0, 1)
        outputs.append((label, images.cpu()))

    # Build a labeled comparison table: rows are the same examples, columns
    # are the raw latent and four BSF variants.
    tile = 128
    header = 28
    canvas = Image.new("RGB", (len(outputs) * tile, args.examples * tile + header), "white")
    draw = ImageDraw.Draw(canvas)
    for col, (label, images) in enumerate(outputs):
        draw.text((col * tile + 3, 6), label, fill="black")
        for row, image in enumerate(images):
            arr = (image.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            canvas.paste(Image.fromarray(arr), (col * tile, header + row * tile))
    canvas.save(os.path.join(args.out_dir, "bsf_variant_comparison.png"))

    # Also save a compact contact sheet for each variant for easier zooming.
    for label, images in outputs:
        grid = make_grid(images, nrow=4, padding=2, pad_value=1.0)
        arr = (grid.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        safe = label.lower().replace(" ", "_")
        Image.fromarray(arr).save(os.path.join(args.out_dir, f"{safe}.png"))

    # For each BSF, perturb its most-used block and label every intervention
    # explicitly. Zero sigma preserves raw z_sem; nonzero values inject only
    # the difference caused by changing that block.
    multipliers = (-10, -4, -2, 0, 2, 4, 10)
    for label, name, model, mean, std, x, recon in model_data:
        with torch.no_grad():
            clean_recon, coeff, norms, selected = model.encode(x)
            counts = torch.bincount(selected.flatten(), minlength=model.n_blocks)
            block = int(counts.argmax())
            mask = torch.zeros_like(norms, dtype=torch.bool)
            mask.scatter_(1, selected, True)
            selected_coeff = coeff * mask.unsqueeze(-1)
            clean = coeff[:, block]
            norm = clean.norm(dim=-1, keepdim=True)
            direction = clean / norm.clamp_min(1e-6)
            fallback = torch.zeros_like(direction); fallback[:, 0] = 1.0
            direction = torch.where(norm > 1e-6, direction, fallback)
            sigma = (norms[:, block].std() if norms[:, block].numel() > 1 else torch.tensor(1.0, device=device)).clamp_min(1e-6)
            generated = []
            for mult in multipliers:
                if mult == 0:
                    cond_variant = raw
                else:
                    modified = selected_coeff.clone()
                    modified[:, block] = clean + mult * sigma * direction
                    delta_norm = (torch.einsum("bng,ngd->bd", modified, model.decoder) + model.dec_bias) - clean_recon
                    cond_variant = raw + delta_norm * std
                generated.append(lit.render(x_t, cond=cond_variant, T=args.steps).clamp(0, 1).cpu())
            tile = 128; header = 34; label_w = 58
            canvas = Image.new("RGB", (label_w + len(multipliers) * tile, header + args.examples * tile), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 8), f"block {block}", fill="black")
            for col, mult in enumerate(multipliers):
                draw.text((label_w + col * tile + 35, 8), f"{mult:+d}σ", fill="black")
            for row in range(args.examples):
                draw.text((5, header + row * tile + 55), f"e{row}", fill="black")
                for col, images in enumerate(generated):
                    image = images[row]
                    arr = (image.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
                    canvas.paste(Image.fromarray(arr), (label_w + col * tile, header + row * tile))
            safe = name.replace(" ", "_")
            canvas.save(os.path.join(args.out_dir, f"{safe}_block_{block}_perturbations.png"))
    print(f"saved grids to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
