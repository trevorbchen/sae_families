#!/bin/bash
#SBATCH --job-name=feat-usage-tsne-highlight
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=00:08:00
#SBATCH --output=/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/logs/feat-usage-tsne-highlight-%j.out
#SBATCH --error=/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/logs/feat-usage-tsne-highlight-%j.err
set -euo pipefail

PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
SCRIPT="/home/tbchen/2026 SURF/diffusion_autoencoder/plot_feature_usage_tsne_joint.py"
USAGE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/feature_usage_vector_16x_k32_lambda10.pt"
OUTDIR="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/figures"

"$PY" "$SCRIPT" \
  --usage "$USAGE" \
  --source ffhq_heldout \
  --only-normalized \
  --with-highlights \
  --out-prefix "$OUTDIR/ffhq_heldout_feature_usage"
