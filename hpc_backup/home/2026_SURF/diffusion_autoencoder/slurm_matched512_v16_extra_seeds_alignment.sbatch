#!/bin/bash
#SBATCH --job-name=m512-v16-hi-aln-extra
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/matched512-v16-hi-aln-extra-%A_%a.out
#SBATCH --error=logs/matched512-v16-hi-aln-extra-%A_%a.err
set -euo pipefail

ARM=vector_16x_k32
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
EXP="$BASE/experiments/$ARM"
LATENTS="$BASE/latents/vector_ffhq_32768.pt"

TAGS=(30p0 100p0)
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
  --latent-cache "$LATENTS" --architecture vector --weighted \
  --out-dir "$OUT" --top-features 64 --min-active 10 --batch-size 64

exec "$PY" plot_cross_seed_similarity_heatmaps.py \
  --input "$OUT/pairwise_alignment.csv" \
  --output-dir "$OUT/pairwise_categories"
