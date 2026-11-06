#!/bin/bash
#SBATCH --job-name=zsem-xform-data
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/zsem-xform-data-%j.out
#SBATCH --error=logs/zsem-xform-data-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt"
cd "$ROOT"
mkdir -p logs "$(dirname "$OUT")"

if [[ ! -f "$OUT" ]]; then
  "$PY" prepare_zsem_objective_pilot.py \
    --repo official_diffae \
    --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
    --codes official_diffae/checkpoints/ffhq128_autoenc/latent.pkl \
    --out "$OUT" --examples 16384 --batch-size 64 --steps 20 --seed 2030
fi

sbatch --dependency=afterok:"$SLURM_JOB_ID" slurm_zsem_transformed_objectives.sbatch
