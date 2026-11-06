#!/bin/bash
#SBATCH --job-name=dit-cifar-debug
#SBATCH --account=perona
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs data runs

module purge
module load python/3.10.12-gcc-11.3.1-djdhgrq
module load cuda/11.8.0-gcc-11.3.1-i5witig

if [ ! -d .venv ]; then
  python -m venv .venv
  . .venv/bin/activate
  python -m ensurepip --upgrade
  python -m pip install --upgrade pip
  python -m pip install torch torchvision tqdm numpy --index-url https://download.pytorch.org/whl/cu118
else
  . .venv/bin/activate
fi

python train_dit_cifar.py \
  --data-dir data \
  --out-dir runs/debug \
  --epochs 1 \
  --max-steps 200 \
  --batch-size 128 \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --log-every 10 \
  --save-every 100 \
  --sample-every 100 \
  --learn-sigma \
  --precision bf16
