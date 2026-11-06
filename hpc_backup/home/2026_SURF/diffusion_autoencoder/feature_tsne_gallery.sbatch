#!/bin/bash
#SBATCH --job-name=feat-tsne-gallery
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/logs/feat-tsne-gallery-%j.out
#SBATCH --error=/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/logs/feat-tsne-gallery-%j.err
set -euo pipefail

PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
SCRIPT="/home/tbchen/2026 SURF/diffusion_autoencoder/visualize_feature_tsne_region_gallery.py"
USAGE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/feature_usage_vector_16x_k32_lambda10.pt"
OUTDIR="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/figures"

for SOURCE in ffhq_heldout celeba fairface; do
  "$PY" "$SCRIPT" \
    --usage "$USAGE" \
    --source "$SOURCE" \
    --out "$OUTDIR/${SOURCE}_feature_tsne_region_gallery.png"
done
