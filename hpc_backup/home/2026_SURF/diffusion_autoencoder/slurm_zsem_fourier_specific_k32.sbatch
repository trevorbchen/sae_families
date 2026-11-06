#!/bin/bash
#SBATCH --job-name=zsem-band-k32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/zsem-band-k32-%A_%a.out
#SBATCH --error=logs/zsem-band-k32-%A_%a.err
set -euo pipefail

OBJECTIVES=(fourier_low fourier_mid fourier_high)
OBJECTIVE="${OBJECTIVES[$SLURM_ARRAY_TASK_ID]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt"
cd "$ROOT"
mkdir -p logs "results/zsem_fourier_specific_pure_k32/$OBJECTIVE"

exec "$PY" train_zsem_transformed_sae.py \
  --objective "$OBJECTIVE" \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --data "$DATA" \
  --out-dir "results/zsem_fourier_specific_pure_k32/$OBJECTIVE" \
  --dict-mult 32 --k 32 \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --grad-clip 1.0 --reinit-every 1000 --dead-after 3000 \
  --max-rescue 256 --seed 2030
