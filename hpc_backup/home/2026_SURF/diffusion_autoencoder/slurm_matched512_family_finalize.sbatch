#!/bin/bash
#SBATCH --job-name=m512-family-final
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/matched512-family-final-%j.out
#SBATCH --error=logs/matched512-family-final-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
cd "$ROOT"
mkdir -p logs "$BASE/overview"
exec "$PY" summarize_matched_512_family.py \
  --family-root "$BASE/experiments" --out-dir "$BASE/overview"
