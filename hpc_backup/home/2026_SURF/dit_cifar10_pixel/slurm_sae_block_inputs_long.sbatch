#!/bin/bash
#SBATCH --job-name=sae-block-inputs
#SBATCH --account=perona
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs runs/sae_block_inputs_16x_k32_long

module purge
module load python/3.10.12-gcc-11.3.1-djdhgrq
module load cuda/11.8.0-gcc-11.3.1-i5witig

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

. .venv/bin/activate

python train_block_input_saes.py \
  --ckpt runs/dit_s2_cifar10/last.pt \
  --data-dir data \
  --out-dir runs/sae_block_inputs_16x_k32_long \
  --batch-size 32 \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --max-steps 50000 \
  --log-every 50 \
  --save-every 5000 \
  --lr 3e-4 \
  --dict-mult 16 \
  --k 32 \
  --timesteps 1000 \
  --precision bf16 \
  --seed 2026
