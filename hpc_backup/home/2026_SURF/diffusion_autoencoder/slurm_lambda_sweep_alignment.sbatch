#!/bin/bash
#SBATCH --job-name=sae-lambda-align
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-4%5
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sae-lambda-align-%A_%a.out
#SBATCH --error=logs/sae-lambda-align-%A_%a.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
EXP="/resnick/scratch/tbchen/surf_2026/sae_lambda_sweep_flat_convae_v1"
LATENTS="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"
TAGS=(0p1 0p3 1p0 3p0 10p0)
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT_ROOT="$EXP/checkpoints/lambda_$TAG"
OUT="$EXP/alignment/lambda_$TAG"

cd "$ROOT"
mkdir -p logs "$OUT/pairwise_categories" "$CHECKPOINT_ROOT/latent_mse"
for SEED in 2030 2031 2032; do
  ln -sfn "$EXP/checkpoints/mse_control/latent_mse/seed_$SEED" \
    "$CHECKPOINT_ROOT/latent_mse/seed_$SEED"
done

"$PY" analyze_spatial_sae_vocabulary_alignment.py \
  --root "$ROOT" --checkpoint-root "$CHECKPOINT_ROOT" \
  --latent-cache "$LATENTS" --architecture flat --weighted \
  --out-dir "$OUT" --top-features 64 --min-active 10 --batch-size 64

exec "$PY" plot_zsem_pairwise_category_matrices.py \
  --input "$OUT/pairwise_alignment.csv" \
  --output-dir "$OUT/pairwise_categories"
