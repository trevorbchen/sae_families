#!/bin/bash
#SBATCH --job-name=zsem-k32-gui
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/zsem-k32-gui-%A_%a.out
#SBATCH --error=logs/zsem-k32-gui-%A_%a.err
set -euo pipefail

LABELS=(laplacian-k32 fourier-all-k32 fourier-low-k32 fourier-mid-k32 fourier-high-k32)
CHECKPOINTS=(
  results/zsem_transformed_pure_k32/laplacian/last.pt
  results/zsem_transformed_pure_k32/fourier_bands/last.pt
  results/zsem_fourier_specific_pure_k32/fourier_low/last.pt
  results/zsem_fourier_specific_pure_k32/fourier_mid/last.pt
  results/zsem_fourier_specific_pure_k32/fourier_high/last.pt
)
PORTS=(8790 8791 8792 8793 8794)

INDEX="$SLURM_ARRAY_TASK_ID"
LABEL="${LABELS[$INDEX]}"
SAE="${CHECKPOINTS[$INDEX]}"
PORT="${PORTS[$INDEX]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"

cd "$ROOT"
test -f "$SAE"
exec "$PY" feature_permutation_gui.py \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --sae "$SAE" \
  --codes official_diffae/checkpoints/ffhq128_autoenc/latent.pkl \
  --port "$PORT" --examples 10 --batch-size 512 --seed 2030 \
  --label "$LABEL"
