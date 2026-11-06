#!/bin/bash
#SBATCH --job-name=specialist-sae-v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --output=logs/specialist-sae-v2-%A_%a.out
#SBATCH --error=logs/specialist-sae-v2-%A_%a.err
set -euo pipefail

MODES=(grouped grouped_pure)
NAMES=(proper_loss transformed_loss)
MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="/resnick/scratch/tbchen/surf_2026/grouped_objective_sae_bookkeeping_v2/${NAME}_32x_k64_seed2026"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_grouped_objective_sae.py \
  --mode "$MODE" --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --data /resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt \
  --out-dir "$OUT" --steps 20000 --batch-size 32 \
  --lr 3e-4 --warmup 500 --grad-clip 1.0 --transform-weight 1.0 \
  --calibration-examples 2048 --aux-coeff 0.1 \
  --reinit-every 1000 --dead-after 3000 --max-rescue-per-group 64 \
  --validate-every 250 --checkpoint-every 1000 --state-every 500 \
  --trace-flush-every 100 --seed 2026 --split-seed 2026
