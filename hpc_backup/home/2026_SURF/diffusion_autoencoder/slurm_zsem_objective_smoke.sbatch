#!/bin/bash
#SBATCH --job-name=zsem-obj-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=hpc-93-36
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/zsem-obj-smoke-%j.out
#SBATCH --error=logs/zsem-obj-smoke-%j.err
set -euo pipefail

: "${OBJECTIVE:?Set OBJECTIVE=pixel, spectral, or semantic}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_objective_audit/pilot_1024.pt"
DINO="/resnick/scratch/tbchen/models/dinov2-base"
STEPS="${STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-2}"
cd "$ROOT"
mkdir -p logs "results/zsem_objective_smoke/$OBJECTIVE"

exec "$PY" train_zsem_representation_sae.py \
  --objective "$OBJECTIVE" \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --init-sae results/zsem_sae/final_32x_k128_aux_reinit_merged/last.pt \
  --data "$DATA" \
  --calibration results/zsem_objective_audit/pilot_1024/calibration.json \
  --dino "$DINO" \
  --out-dir "results/zsem_objective_smoke/$OBJECTIVE" \
  --steps "$STEPS" --batch-size "$BATCH_SIZE" --lr 3e-5 --warmup 50 \
  --aux-coeff 0.1 --max-bin-weight 1000 --seed 2028
