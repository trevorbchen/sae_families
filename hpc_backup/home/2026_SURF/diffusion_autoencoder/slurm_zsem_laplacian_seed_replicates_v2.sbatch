#!/bin/bash
#SBATCH --job-name=lap-seeds-v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/lap-seeds-v2-%A_%a.out
#SBATCH --error=logs/lap-seeds-v2-%A_%a.err
set -euo pipefail

SEEDS=(2030 2031 2032)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt"
OUT="results/zsem_objective_seed_replicates_v2/laplacian/seed_$SEED"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_zsem_transformed_sae.py \
  --objective laplacian \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --data "$DATA" \
  --out-dir "$OUT" \
  --dict-mult 32 --k 32 \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --grad-clip 1.0 --reinit-every 1000 --dead-after 3000 \
  --max-rescue 256 --split-seed 2030 --seed "$SEED"
