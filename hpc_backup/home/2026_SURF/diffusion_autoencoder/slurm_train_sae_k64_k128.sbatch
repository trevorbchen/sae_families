#!/bin/bash
#SBATCH --job-name=zsem-sae-k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=hpc-93-36
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/zsem-sae-k-%A_%a.out
#SBATCH --error=logs/zsem-sae-k-%A_%a.err
set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs results/zsem_sae
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then K=64; else K=128; fi
exec "$PY" train_zsem_sae_final.py \
  --codes results/zsem_codes/ffhq_celeba_train.pt \
  --out-dir "results/zsem_sae/final_32x_k${K}_aux_reinit_merged" \
  --dict-mult 32 --k "$K" --aux-k "$K" --aux-coeff 0.1 \
  --steps 100000 --batch-size 1024 --lr 3e-4 --warmup 2000 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 256
