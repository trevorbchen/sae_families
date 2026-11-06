#!/bin/bash
#SBATCH --job-name=sae-probe-data-cpu
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sae-probe-data-cpu-%j.out
#SBATCH --error=logs/sae-probe-data-cpu-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="/resnick/scratch/tbchen/surf_2026/sae_linear_probes/celeba_32768.pt"

cd "$ROOT"
mkdir -p logs "$(dirname "$OUT")" /resnick/scratch/tbchen/huggingface
export HF_HOME=/resnick/scratch/tbchen/huggingface
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

exec "$PY" prepare_celeba_spatial_probe_cache.py \
  --autoencoder results/spatial_conv_ae/last.pt \
  --output "$OUT" \
  --max-images 32768 \
  --batch-size 64 \
  --seed 260818
