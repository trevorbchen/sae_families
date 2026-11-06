#!/usr/bin/env python3
"""Render the two "chosen for paper" feature galleries as static composite
images, reusing the live GUIs' own FeatureServer classes directly (for
exact visual fidelity to what was actually reviewed and picked in-browser)
rather than reimplementing the multiplier-sweep math separately.

Gallery B (matched512, lambda=100, "objective differences"): rows are
(objective, seed, rank) with a known chosen_example_index, taken straight
from feature_gui_annotations_lambda_100p0.json.

Gallery A (residual general, lambda=1-equivalent, "general features"): rows
are (stage, feature) with a known chosen_example_index, taken straight from
paper_selection.json.

Each row = one multiplier sweep (-10,-4,-2,0,2,4,10 std) for that row's
chosen example, decoded via the same base autoencoder used by the source
GUI, with the row's comment burned in as a label.
"""

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LABEL_WIDTH = 260
CELL = 128
PAD = 4


def decode_b64_png(data_uri: str) -> Image.Image:
    _, b64 = data_uri.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def build_grid(rows: list[dict], out_path: Path, n_cols: int) -> None:
    height = len(rows) * (CELL + PAD) + PAD
    width = LABEL_WIDTH + n_cols * (CELL + PAD) + PAD
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for row_index, row in enumerate(rows):
        y = PAD + row_index * (CELL + PAD)
        draw.multiline_text((PAD, y + CELL // 2 - 20), row["label"], fill="black", font=font)
        for col_index, image in enumerate(row["images"]):
            x = LABEL_WIDTH + PAD + col_index * (CELL + PAD)
            canvas.paste(image.resize((CELL, CELL)), (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"wrote {out_path} ({width}x{height}, {len(rows)} rows)")


def build_gallery_b(args) -> None:
    import matched512_feature_permutation_gui as gui

    server_args = argparse.Namespace(
        checkpoint_root=args.b_checkpoint_root, autoencoder=args.b_autoencoder,
        autoencoder_type="vector", latent_cache=args.b_latent_cache,
        annotations=args.b_annotations, port=0, examples=12, batch_size=512,
        seed=2030, seeds=[2030, 2031, 2032], label="export",
    )
    server = gui.FeatureServer(server_args)

    annotations = json.loads(Path(args.b_annotations).read_text())
    chosen = [v for v in annotations.values() if v.get("chosen")]
    chosen.sort(key=lambda v: (v["objective"], v["seed"], v["rank"]))

    rows = []
    for record in chosen:
        result = server.feature(record["objective"], record["seed"], record["rank"])
        variants = result["images"][record["chosen_example_index"]]["variants"]
        images = [decode_b64_png(v["image"]) for v in variants]
        label = f"{record['objective']}\nseed {record['seed']}\n\"{record.get('comment', '')}\""
        rows.append({"label": label, "images": images})
        print(f"rendered {record['objective']}:{record['seed']}:{record['feature']}", flush=True)

    build_grid(rows, Path(args.out_b), n_cols=len(gui.MULTIPLIERS))


# Short human-written descriptions for gallery A's ten features -- unlike
# gallery B (matched512), this gallery's source annotations never recorded a
# comment, so these were written by inspecting the rendered multiplier sweep
# directly. Keyed by feature index (unique across the ten chosen features).
GENERAL_FEATURE_DESCRIPTIONS = {
    1287: "background red saturation",
    1467: "background stripe fades",
    1725: "background red saturation",
    178: "background warmth",
    3564: "hair color, warm to cool",
    3793: "background/skin warmth",
    2073: "skin tone saturation",
    2292: "glasses frame thickness",
    465: "background texture",
    1308: "background color/brightness",
}


def build_gallery_a(args) -> None:
    import residual_general_feature_shortlist_gui as gui

    server_args = argparse.Namespace(
        checkpoint_root=args.a_checkpoint_root, seed="seed_2030",
        autoencoder=args.a_autoencoder, latent_cache=args.a_latent_cache,
        review_annotations=args.a_review_annotations, choices=args.a_choices,
        port=0, examples=10, rank_examples=4096, batch_size=64,
        shortlist_examples=10, label="export",
    )
    server = gui.FeatureServer(server_args)

    choices = json.loads(Path(args.a_choices).read_text())
    chosen = [dict(v, key=k) for k, v in choices.items() if v.get("chosen")]
    chosen.sort(key=lambda v: v["key"])

    rows = []
    for record in chosen:
        stage = record["source"].removeprefix(gui.SOURCE_PREFIX)
        result = server.render_feature(stage, record["feature"])
        variants = result["images"][record["chosen_example_index"]]["variants"]
        images = [decode_b64_png(v["image"]) for v in variants]
        description = GENERAL_FEATURE_DESCRIPTIONS.get(record["feature"], f"feature {record['feature']}")
        label = f"\"{description}\"\n{gui.STAGE_LABELS.get(stage, stage)} (f{record['feature']})"
        rows.append({"label": label, "images": images})
        print(f"rendered {record['source']}:{record['feature']}", flush=True)

    build_grid(rows, Path(args.out_a), n_cols=len(gui.MULTIPLIERS))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery", choices=("a", "b", "both"), default="both")
    base = "/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
    parser.add_argument("--b-checkpoint-root", default=f"{base}/experiments/vector_16x_k32/checkpoints/lambda_100p0")
    parser.add_argument("--b-autoencoder", default=f"{base}/autoencoders/vector/last.pt")
    parser.add_argument("--b-latent-cache", default=f"{base}/latents/vector_ffhq_32768.pt")
    parser.add_argument("--b-annotations", default="results/family_sae/lambda_sweep/vector_16x_k32/general/feature_gui_annotations_lambda_100p0.json")
    parser.add_argument("--out-b", default="results/family_sae/lambda_sweep/reconstruction_failures/paper_gallery_objective_differences.png")

    parser.add_argument("--a-checkpoint-root", default="results/spatial_map_sae_weighted_seed_replicates/flat")
    parser.add_argument("--a-autoencoder", default="results/spatial_conv_ae/last.pt")
    parser.add_argument("--a-latent-cache", default="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt")
    parser.add_argument("--a-review-annotations", default="/resnick/scratch/tbchen/surf_2026/proper_residual_feature_review/annotations.json")
    parser.add_argument("--a-choices", default="/resnick/scratch/tbchen/surf_2026/proper_residual_feature_review/paper_selection.json")
    parser.add_argument("--out-a", default="results/family_sae/lambda_sweep/reconstruction_failures/paper_gallery_general_features.png")
    args = parser.parse_args()

    if args.gallery in ("b", "both"):
        build_gallery_b(args)
    if args.gallery in ("a", "both"):
        build_gallery_a(args)


if __name__ == "__main__":
    main()
