#!/bin/bash
#SBATCH --job-name=sum-freq500
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/sum-freq500-%j.out
#SBATCH --error=logs/sum-freq500-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT_ROOT="/resnick/scratch/tbchen/surf_2026/frequency_family_labels_top500_v2"

cd "$ROOT"
exec "$PY" summarize_kimi_frequency_labels.py \
  "$OUT_ROOT" \
  --out-dir "$OUT_ROOT/analysis"
