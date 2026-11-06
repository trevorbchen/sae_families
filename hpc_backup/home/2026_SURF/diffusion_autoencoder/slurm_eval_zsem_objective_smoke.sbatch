#!/bin/bash
#SBATCH --job-name=zsem-obj-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/zsem-obj-eval-%j.out
#SBATCH --error=logs/zsem-obj-eval-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
cd "$ROOT"

exec "$PY" eval_zsem_representation_saes.py \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --data /resnick/scratch/tbchen/surf_2026/zsem_objective_audit/pilot_1024.pt \
  --dino /resnick/scratch/tbchen/models/dinov2-base \
  --spectral-stats results/zsem_objective_smoke/spectral/representation_stats.pt \
  --models \
    baseline=results/zsem_sae/final_32x_k128_aux_reinit_merged/last.pt \
    pixel=results/zsem_objective_smoke/pixel/last.pt \
    spectral=results/zsem_objective_smoke/spectral/last.pt \
    semantic=results/zsem_objective_smoke/semantic/last.pt \
  --out results/zsem_objective_smoke/cross_objective_eval.json \
  --batch-size 2 --seed 2028
