#!/bin/bash
#SBATCH --job-name=m512-1seed-hm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/matched512-v16-1seed-hm-%j.out
#SBATCH --error=logs/matched512-v16-1seed-hm-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
: "${ARM:=vector_16x_k32}"
EXP="$BASE/experiments/$ARM"

case "$ARM" in
  vector_4x_k8|vector_16x_k32) SOURCE=vector ;;
  spatial_4x_k8|spatial_16x_k32) SOURCE=spatial ;;
  *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac

cd "$ROOT"
mkdir -p logs "$EXP/summary/same_seed_vocabulary_heatmaps"
exec "$PY" analyze_matched512_single_seed_lambda_sweep.py \
  --experiment-root "$EXP" \
  --latent-cache "$BASE/latents/${SOURCE}_ffhq_32768.pt" \
  --out-dir "$EXP/summary/same_seed_vocabulary_heatmaps" \
  --seed 2030 --top-features 64 --min-active 10 --batch-size 64
