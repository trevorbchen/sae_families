#!/bin/bash
#SBATCH --job-name=dit-cifar-noadaln
#SBATCH --account=perona
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs data runs

module purge
module load python/3.10.12-gcc-11.3.1-djdhgrq
module load cuda/11.8.0-gcc-11.3.1-i5witig

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -d .venv ]; then
  python -m venv .venv
  . .venv/bin/activate
  python -m ensurepip --upgrade
  python -m pip install --upgrade pip
  python -m pip install torch torchvision tqdm numpy --index-url https://download.pytorch.org/whl/cu118
else
  . .venv/bin/activate
fi

python train_dit_cifar_no_adaln.py   --data-dir data   --out-dir runs/dit_s2_cifar10_no_adaln   --epochs 400   --batch-size 128   --num-workers "${SLURM_CPUS_PER_TASK}"   --log-every 50   --save-every 5000   --sample-every 10000   --learn-sigma   --precision bf16
