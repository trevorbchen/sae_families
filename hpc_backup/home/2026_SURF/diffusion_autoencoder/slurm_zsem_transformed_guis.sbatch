#!/bin/bash
#SBATCH --job-name=zsem-xform-gui
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/zsem-xform-gui-%A_%a.out
#SBATCH --error=logs/zsem-xform-gui-%A_%a.err
set -euo pipefail

LABELS=(gradient laplacian fourier-bands steerable)
DIRECTORIES=(gradient laplacian fourier_bands steerable)
PORTS=(8780 8781 8782 8783)

INDEX="$SLURM_ARRAY_TASK_ID"
LABEL="${LABELS[$INDEX]}"
DIRECTORY="${DIRECTORIES[$INDEX]}"
PORT="${PORTS[$INDEX]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
SAE="results/zsem_transformed_pure/$DIRECTORY/last.pt"

cd "$ROOT"
test -f "$SAE"
exec "$PY" feature_permutation_gui.py \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --sae "$SAE" \
  --codes official_diffae/checkpoints/ffhq128_autoenc/latent.pkl \
  --port "$PORT" --examples 10 --batch-size 512 --seed 2030 \
  --label "pure $LABEL objective"
