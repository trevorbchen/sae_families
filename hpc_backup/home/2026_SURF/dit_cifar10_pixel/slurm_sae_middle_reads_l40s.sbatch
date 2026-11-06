#!/bin/bash
#SBATCH --job-name=sae-mid-read-l40s
#SBATCH --account=perona
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs runs/sae_middle_reads_16x_k32_long
module purge
module load python/3.10.12-gcc-11.3.1-djdhgrq
module load cuda/11.8.0-gcc-11.3.1-i5witig
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate

python train_middle_read_saes.py \
  --ckpt runs/dit_s2_cifar10/last.pt \
  --data-dir data \
  --out-dir runs/sae_middle_reads_16x_k32_long \
  --block-start 0 \
  --block-end 11 \
  --batch-size 32 \
  --num-workers 8 \
  --max-steps 10000 \
  --log-every 50 \
  --save-every 2000 \
  --lr 3e-4 \
  --dict-mult 16 \
  --k 32 \
  --precision bf16 \
  --seed 2026
