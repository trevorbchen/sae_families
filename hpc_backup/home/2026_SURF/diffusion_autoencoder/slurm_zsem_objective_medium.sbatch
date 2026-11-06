#!/bin/bash
#SBATCH --job-name=zsem-obj-10k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/zsem-obj-10k-%A_%a.out
#SBATCH --error=logs/zsem-obj-10k-%A_%a.err
set -euo pipefail

OBJECTIVES=(pixel spectral semantic)
OBJECTIVE="${OBJECTIVES[$SLURM_ARRAY_TASK_ID]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_objective_audit/pilot_1024.pt"
DINO="/resnick/scratch/tbchen/models/dinov2-base"
OUT_ROOT="results/zsem_objective_medium_10k"

cd "$ROOT"
mkdir -p logs "$OUT_ROOT/$OBJECTIVE"

exec "$PY" train_zsem_representation_sae.py \
  --objective "$OBJECTIVE" \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --init-sae results/zsem_sae/final_32x_k128_aux_reinit_merged/last.pt \
  --data "$DATA" \
  --codes results/zsem_codes/ffhq_celeba_train.pt \
  --calibration results/zsem_objective_audit/pilot_1024/calibration.json \
  --dino "$DINO" \
  --out-dir "$OUT_ROOT/$OBJECTIVE" \
  --steps 10000 --batch-size 2 --latent-batch-size 1024 \
  --lr 3e-5 --warmup 500 --aux-coeff 0.1 --max-bin-weight 1000 \
  --grad-clip 1.0 --seed 2028
