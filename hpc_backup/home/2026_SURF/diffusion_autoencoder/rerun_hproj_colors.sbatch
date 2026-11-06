#!/bin/bash
#SBATCH --job-name=tsne-hproj-recolor
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/logs/hproj-recolor-%j.out
#SBATCH --error=/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/logs/hproj-recolor-%j.err
set -euo pipefail

PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
SCRIPT="/home/tbchen/2026 SURF/diffusion_autoencoder/plot_residual_tsne_hprojected.py"
RESIDUALS="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/residuals_vector_16x_k32_lambda10.pt"
OUTDIR="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/figures"

for SOURCE in ffhq_heldout celeba fairface; do
  "$PY" "$SCRIPT" \
    --residuals "$RESIDUALS" \
    --source "$SOURCE" \
    --out-prefix "$OUTDIR/$SOURCE"
done
