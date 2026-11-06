#!/bin/bash
#SBATCH --job-name=zsem-obj-gui
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/zsem-obj-gui-%A_%a.out
#SBATCH --error=logs/zsem-obj-gui-%A_%a.err
set -euo pipefail

LABELS=(baseline pixel spectral semantic)
CHECKPOINTS=(
  results/zsem_sae/final_32x_k128_aux_reinit_merged/last.pt
  results/zsem_objective_medium_10k/pixel/last.pt
  results/zsem_objective_medium_10k/spectral/last.pt
  results/zsem_objective_medium_10k/semantic/last.pt
)
PORTS=(8770 8771 8772 8773)

INDEX="$SLURM_ARRAY_TASK_ID"
LABEL="${LABELS[$INDEX]}"
SAE="${CHECKPOINTS[$INDEX]}"
PORT="${PORTS[$INDEX]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"

cd "$ROOT"
exec "$PY" feature_permutation_gui.py \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --sae "$SAE" \
  --codes official_diffae/checkpoints/ffhq128_autoenc/latent.pkl \
  --port "$PORT" --examples 10 --batch-size 512 --seed 2026 \
  --label "$LABEL"
