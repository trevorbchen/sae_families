#!/bin/bash
#SBATCH --job-name=spatial-sae-gui
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/spatial-sae-gui-%A_%a.out
#SBATCH --error=logs/spatial-sae-gui-%A_%a.err
set -euo pipefail

ARCHITECTURES=(flat conv)
LABELS=("Flat whole-map" "Convolutional position-shared")
PORTS=(8800 8801)
INDEX="$SLURM_ARRAY_TASK_ID"
ARCHITECTURE="${ARCHITECTURES[$INDEX]}"
LABEL="${LABELS[$INDEX]}"
PORT="${PORTS[$INDEX]}"

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
LATENTS="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"
cd "$ROOT"
mkdir -p logs

exec "$PY" spatial_sae_feature_gui.py \
  --architecture "$ARCHITECTURE" \
  --checkpoint-root results/spatial_map_saes \
  --autoencoder results/spatial_conv_ae/last.pt \
  --latent-cache "$LATENTS" \
  --port "$PORT" --examples 10 --rank-examples 4096 --batch-size 32 \
  --label "$LABEL"
