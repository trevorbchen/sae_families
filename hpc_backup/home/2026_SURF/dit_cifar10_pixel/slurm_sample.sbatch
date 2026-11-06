#!/bin/bash
#SBATCH --job-name=dit-cifar-sample
#SBATCH --account=perona
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs runs/sanity_samples

module purge
module load python/3.10.12-gcc-11.3.1-djdhgrq
module load cuda/11.8.0-gcc-11.3.1-i5witig

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

. .venv/bin/activate

python sample_dit_cifar.py \
  --ckpt runs/dit_s2_cifar10/last.pt \
  --out-dir runs/sanity_samples \
  --num-samples 20 \
  --timesteps 1000 \
  --seed 2026 \
  --learn-sigma
