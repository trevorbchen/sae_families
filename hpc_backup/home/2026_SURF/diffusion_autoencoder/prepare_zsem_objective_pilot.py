#!/usr/bin/env python3
"""Render a deterministic image cache paired with sampled DiffAE z_sem codes."""

import argparse
import os
import sys

import torch


def load_diffae(repo, checkpoint, device):
    sys.path.insert(0, os.path.abspath(repo))
    from experiment import LitModel
    from templates import ffhq128_autoenc_130M

    conf = ffhq128_autoenc_130M()
    conf.pretrain = None
    conf.latent_infer_path = None
    lit = LitModel(conf)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    lit.load_state_dict(state["state_dict"], strict=False)
    lit.eval().to(device)
    for parameter in lit.parameters():
        parameter.requires_grad_(False)
    return lit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--append-data",
        help="Existing cache to retain while rendering new, non-overlapping codes.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codes_state = torch.load(args.codes, map_location="cpu", weights_only=False)
    all_codes = codes_state["conds"].float()
    generator = torch.Generator().manual_seed(args.seed)
    existing = None
    existing_indices = torch.empty(0, dtype=torch.long)
    existing_images = torch.empty(0, 3, 128, 128, dtype=torch.uint8)
    existing_codes = torch.empty(0, all_codes.shape[1], dtype=all_codes.dtype)
    if args.append_data:
        existing = torch.load(args.append_data, map_location="cpu", weights_only=False)
        existing_indices = existing["indices"].long()
        existing_images = existing["images_uint8"]
        existing_codes = existing["codes"].float()

    target_count = min(args.examples, len(all_codes))
    additional_count = target_count - len(existing_indices)
    if additional_count < 0:
        raise ValueError("--examples cannot be smaller than the appended cache")
    available = torch.ones(len(all_codes), dtype=torch.bool)
    available[existing_indices] = False
    candidates = torch.where(available)[0]
    selection = torch.randperm(len(candidates), generator=generator)[:additional_count]
    indices = candidates[selection]
    codes = all_codes[indices]
    lit = load_diffae(args.repo, args.checkpoint, device)

    images = []
    with torch.inference_mode():
        for start in range(0, additional_count, args.batch_size):
            cond = codes[start:start + args.batch_size].to(device)
            noise = torch.randn(
                len(cond), 3, 128, 128, generator=generator
            ).to(device)
            image = lit.render(noise, cond=cond, T=args.steps).clamp(0, 1)
            images.append((image.cpu() * 255).round().to(torch.uint8))
            print(
                f"rendered {min(start + len(cond), additional_count)}/"
                f"{additional_count} new images",
                flush=True,
            )

    new_images = (
        torch.cat(images)
        if images else torch.empty(0, *existing_images.shape[1:], dtype=torch.uint8)
    )
    combined_images = torch.cat((existing_images, new_images))
    combined_codes = torch.cat((existing_codes, codes))
    combined_indices = torch.cat((existing_indices, indices))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(
        {
            "images_uint8": combined_images,
            "codes": combined_codes,
            "indices": combined_indices,
            "seed": args.seed,
            "render_steps": args.steps,
            "source_codes": args.codes,
            "appended_from": args.append_data,
            "original_examples": len(existing_indices),
            "new_examples": additional_count,
        },
        args.out,
    )
    print(
        {
            "out": args.out,
            "examples": len(combined_images),
            "new_examples": additional_count,
            "shape": list(combined_images.shape[1:]),
        }
    )


if __name__ == "__main__":
    main()
