#!/usr/bin/env python3
"""Generate causal feature panels and label them with local Kimi-VL.

This is deliberately resumable: one JSON object is appended per feature. The
feature intervention is the same raw-latent delta used by the existing GUI.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw


DEFAULT_MULTIPLIERS = (-10.0, -4.0, -2.0, 0.0, 2.0, 4.0, 10.0)


def make_prompt(multipliers, panel_mode):
    strengths = ", ".join(f"{x:+g}s" if x else "0" for x in multipliers)
    presentation = (
        "You receive one image per subject. Each image is a left-to-right, "
        "top-to-bottom sweep"
        if panel_mode == "subjects"
        else "You receive one grid. Each row is one subject and columns are a sweep"
    )
    return f"""You are labeling a causal image-editing feature.
{presentation} through feature strengths: {strengths}. The 0 image is the
untouched DiffAE generation. Compare each subject to itself and report only
changes that repeat across subjects. Extreme strengths can be out of
distribution; distinguish the recurring edit from artifacts.

Check geometry (including head turn or tilt, eye size, ears, teeth and facial
contours), hair, color, cosmetics such as eye shadow, accessories such as hats,
lighting, texture and background. This is a checklist, not a list of effects
that must be present.

Return JSON only with exactly these keys:
primary_effect (specific short phrase), effect_category (geometry/color/texture/
hair/cosmetic/accessory/expression/lighting/background/artifact/unclear),
affected_regions (array of specific regions), direction (what changes from
negative to positive strength), monotonicity (monotonic/non_monotonic/threshold/
saturating/unclear), onset_strength (first clearly visible signed strength or
unclear), secondary_effects (array of short phrases), subject_consistency
(number 0 to 1), identity_preserved (true/false/uncertain), artifact_onset
(signed strength or none), artifact_level_at_extremes (number 0 to 1),
confidence (number 0 to 1), evidence (one concise sentence).
Use primary_effect='unclear' when the effect is weak, inconsistent, or visible
only as an extreme-strength artifact."""


def strength_label(value):
    return f"{value:+g}s" if value else "0"


def make_sheet(images, multipliers, cell=128):
    sheet = Image.new("RGB", (len(multipliers) * cell, len(images) * (cell + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(images):
        for c, image in enumerate(row):
            image = image.detach().float().cpu().clamp(0, 1)
            arr = (image.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            sheet.paste(Image.fromarray(arr).resize((cell, cell)), (c * cell, r * (cell + 18)))
            draw.text((c * cell + 3, r * (cell + 18) + cell), strength_label(multipliers[c]), fill="black")
    return sheet


def make_subject_sheets(images, multipliers, cell=128):
    """Use one nearly square image per subject so VLM resizing preserves detail."""
    sheets = []
    cols = 4
    rows = (len(multipliers) + cols - 1) // cols
    for subject in images:
        sheet = Image.new("RGB", (cols * cell, rows * (cell + 18)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, image in enumerate(subject):
            row, col = divmod(index, cols)
            image = image.detach().float().cpu().clamp(0, 1)
            arr = (image.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            sheet.paste(Image.fromarray(arr).resize((cell, cell)), (col * cell, row * (cell + 18)))
            draw.text((col * cell + 3, row * (cell + 18) + cell), strength_label(multipliers[index]), fill="black")
        sheets.append(sheet)
    return sheets


def load_sae(path, kind, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if kind == "polynomial":
        from train_zsem_polysae import PolyTopKSAE
        sae = PolyTopKSAE(512, int(cfg["dict_mult"]) * 512, int(cfg["k"]), int(cfg.get("aux_k", cfg["k"]))).to(device)
    else:
        from train_zsem_sae import TopKSAE
        sae = TopKSAE(512, int(cfg["dict_mult"]) * 512, int(cfg["k"])).to(device)
    sae.load_state_dict(state["model"], strict=True)
    sae.eval()
    mean = state["input_mean"].float().to(device)
    std = state["input_std"].float().to(device)
    return sae, mean, std, cfg


@torch.no_grad()
def feature_statistics(sae, codes, mean, std, device, batch_size=512):
    h = sae.decoder.shape[1] if sae.decoder.ndim == 3 else sae.decoder.shape[0]
    sums = torch.zeros(h, device=device)
    squares = torch.zeros(h, device=device)
    active = torch.zeros(h, device=device)
    for start in range(0, len(codes), batch_size):
        batch = (codes[start:start + batch_size].to(device) - mean) / std
        acts = torch.relu(sae.encoder(batch - sae.dec_bias))
        vals, idx = torch.topk(acts, sae.k, dim=-1)
        sums.scatter_add_(0, idx.reshape(-1), vals.reshape(-1))
        squares.scatter_add_(0, idx.reshape(-1), vals.square().reshape(-1))
        active.scatter_add_(0, idx.reshape(-1), torch.ones_like(vals).reshape(-1))
    feature_mean = sums / len(codes)
    feature_std = (squares / len(codes) - feature_mean.square()).clamp_min(0).sqrt()
    active_fraction = active / len(codes)
    order = torch.argsort(feature_mean, descending=True)
    return order, feature_mean, feature_std, active_fraction


@torch.no_grad()
def highest_response_examples(sae, codes, mean, std, features, examples, device, batch_size=512):
    features = torch.as_tensor(features, device=device, dtype=torch.long)
    best_values = torch.full((len(features), examples), -torch.inf, device=device)
    best_ids = torch.full((len(features), examples), -1, device=device, dtype=torch.long)
    for start in range(0, len(codes), batch_size):
        batch = (codes[start:start + batch_size].to(device) - mean) / std
        scores = sae.encoder(batch - sae.dec_bias)[:, features].T
        count = min(examples, scores.shape[1])
        values, positions = torch.topk(scores, count, dim=1)
        ids = positions + start
        combined_values = torch.cat((best_values, values), dim=1)
        combined_ids = torch.cat((best_ids, ids), dim=1)
        best_values, keep = torch.topk(combined_values, examples, dim=1)
        best_ids = combined_ids.gather(1, keep)
    return best_ids.cpu().tolist()


@torch.no_grad()
def feature_panel(sae, kind, raw, mean, std, feature, sigma, noise, lit, multipliers):
    x = (raw - mean) / std
    if kind == "polynomial":
        _, vals, idx, acts = sae.encode(x)
    else:
        pre = sae.encoder(x - sae.dec_bias)
        acts = torch.relu(pre)
        vals, idx = torch.topk(acts, sae.k, dim=-1)
    h = sae.decoder.shape[1] if kind == "polynomial" else sae.decoder.shape[0]
    clean = torch.zeros(len(raw), h, device=raw.device)
    clean.scatter_(1, idx, vals)
    active_pos = (idx == feature).nonzero(as_tuple=False)
    clean_value = torch.zeros(len(raw), device=raw.device)
    for row, pos in active_pos.tolist():
        clean_value[row] = vals[row, pos]
    sigma = torch.as_tensor(sigma, device=raw.device).clamp_min(1e-4)
    weakest = vals.argmin(dim=1)
    remove_feature = idx.gather(1, weakest[:, None]).squeeze(1)
    remove_value = vals.gather(1, weakest[:, None]).squeeze(1)
    conditions = []
    for mult in multipliers:
        if mult == 0:
            cond = raw
        else:
            target = torch.clamp(clean_value + mult * sigma, min=0)
            if kind == "polynomial":
                mod_vals = vals.clone()
                present = (idx == feature).nonzero(as_tuple=False)
                for row, pos in present.tolist():
                    mod_vals[row, pos] = target[row]
                # For inactive features, replace the weakest active slot.
                for row in range(len(raw)):
                    if target[row] > 0 and not (idx[row] == feature).any():
                        pos = int(weakest[row])
                        mod_vals[row, pos] = target[row]
                changed = sae.decode_sparse(mod_vals, idx)
                clean_recon = sae.decode_sparse(vals, idx)
            else:
                changed = clean @ sae.decoder
                mod = clean.clone()
                present = (idx == feature).nonzero(as_tuple=False)
                for row, pos in present.tolist():
                    mod[row, feature] = target[row]
                for row in range(len(raw)):
                    if target[row] > 0 and not (idx[row] == feature).any():
                        mod[row, int(remove_feature[row])] = 0
                        mod[row, feature] = target[row]
                changed = mod @ sae.decoder
                clean_recon = clean @ sae.decoder
            cond = raw + (changed - clean_recon) * std
        conditions.append(cond)
    # Render every strength in one diffusion batch while reusing each subject's noise.
    rendered = lit.render(
        noise.repeat(len(multipliers), 1, 1, 1),
        cond=torch.cat(conditions, dim=0),
        T=20,
    ).clamp(0, 1)
    rendered = rendered.reshape(len(multipliers), len(raw), *rendered.shape[1:])
    return [[rendered[c, r] for c in range(len(multipliers))] for r in range(len(raw))], int((clean[:, feature] > 0).sum())


def parse_json(text):
    text = text.strip()
    if "```" in text:
        text = text.split("```", 2)[1]
        text = text.removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model did not return JSON")
    return json.loads(text[start:end + 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sae", required=True)
    p.add_argument("--codes", required=True)
    p.add_argument("--kind", choices=["topk", "polynomial"], default="topk")
    p.add_argument("--out", required=True)
    p.add_argument("--panel-dir")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--examples", type=int, default=3)
    p.add_argument("--multipliers", type=float, nargs="+", default=DEFAULT_MULTIPLIERS)
    p.add_argument("--panel-mode", choices=["grid", "subjects"], default="subjects")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    multipliers = tuple(args.multipliers)
    if 0.0 not in multipliers:
        p.error("--multipliers must include 0 so the untouched generation is shown")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try: done.add(json.loads(line)["feature_id"])
                except Exception: pass
    sys.path.insert(0, os.path.abspath(args.repo))
    from experiment import LitModel
    from templates import ffhq128_autoenc_130M
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf = ffhq128_autoenc_130M(); conf.pretrain = None; conf.latent_infer_path = None
    lit = LitModel(conf)
    lit.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"], strict=False)
    lit.eval().to(device)
    sae, mean, std, cfg = load_sae(args.sae, args.kind, device)
    codes = torch.load(args.codes, map_location="cpu", weights_only=False)["conds"].float()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    order, feature_mean, feature_std, active_fraction = feature_statistics(
        sae, codes, mean, std, device
    )
    h = len(order)
    end = h if args.end < 0 else min(args.end, h)
    ranks = list(range(args.start, end))
    features = order[ranks].tolist()
    example_ids = highest_response_examples(
        sae, codes, mean, std, features, args.examples, device
    )
    panel_dir = Path(args.panel_dir or (os.path.splitext(args.out)[0] + "_panels"))
    panel_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    for rank, feature, ids in zip(ranks, features, example_ids):
        if feature in done:
            continue
        raw = codes[ids].to(device)
        noise = torch.randn(len(ids), 3, 128, 128, generator=gen, device=device)
        sigma = feature_std[feature]
        panels, active = feature_panel(
            sae, args.kind, raw, mean, std, feature, sigma, noise, lit, multipliers
        )
        sheet = make_sheet(panels, multipliers)
        panel_path = panel_dir / f"rank_{rank:05d}_feature_{feature:05d}.png"
        sheet.save(panel_path)
        vlm_images = make_subject_sheets(panels, multipliers) if args.panel_mode == "subjects" else [sheet]
        content = [{"type": "image", "image": image} for image in vlm_images]
        content.append({"type": "text", "text": make_prompt(multipliers, args.panel_mode)})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = processor(images=vlm_images, text=text, return_tensors="pt", padding=True).to(model.device)
        output = model.generate(**inputs, max_new_tokens=384, do_sample=False, use_cache=True)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, output)]
        response = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        try:
            label = parse_json(response); error = None
        except Exception as exc:
            label = {"primary_effect": "unparseable", "raw_response": response}; error = str(exc)
        record = {"variant": Path(args.sae).parent.name, "kind": args.kind,
                  "importance_rank": rank, "feature_id": feature,
                  "dict_size": h, "k": int(cfg["k"]), "example_ids": ids,
                  "dataset_mean_activation": float(feature_mean[feature]),
                  "dataset_active_fraction": float(active_fraction[feature]),
                  "activation_sigma": float(sigma), "selected_active_examples": active,
                  "panel_path": str(panel_path), "multipliers": list(multipliers),
                  "panel_mode": args.panel_mode, "label_schema_version": 2,
                  "label": label, "parse_error": error}
        with open(args.out, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
        print(json.dumps({"rank": rank, "feature": feature, "variant": record["variant"],
                          "label": label.get("primary_effect")}), flush=True)


if __name__ == "__main__":
    main()
