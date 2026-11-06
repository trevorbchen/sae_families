#!/bin/bash
#SBATCH --job-name=constr-joint-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/constr-joint-smoke-%A_%a.out
#SBATCH --error=logs/constr-joint-smoke-%A_%a.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT_ROOT="/resnick/scratch/tbchen/surf_2026/constrained_joint_changed_objectives_smoke"
INITIAL="/resnick/scratch/tbchen/surf_2026/grouped_spatial_conv_ae_bookkeeping_v1/original_16k_k64_seed2026/best_valid.pt"
cd "$ROOT"
OUT="$OUT_ROOT/task_$SLURM_ARRAY_TASK_ID"
mkdir -p logs "$OUT"

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
  OBJECTIVES=(laplacian)
else
  OBJECTIVES=(laplacian fourier_low fourier_mid fourier_high)
fi

exec "$PY" train_constrained_grouped_spatial_sae.py \
  --objectives "${OBJECTIVES[@]}" \
  --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --autoencoder results/spatial_conv_ae/last.pt --initial-sae "$INITIAL" \
  --out-dir "$OUT" --steps 20 --mse-adapt-steps 5 \
  --batch-size 8 --lr 1e-4 --lr-warmup 2 --transform-warmup 2 \
  --transform-gradient-ratio 0.1 --mse-tolerance 0.01 \
  --aux-coeff 0.1 --grad-clip 1.0 --trust-every 5 \
  --geometry-every 10 --seed 2026
