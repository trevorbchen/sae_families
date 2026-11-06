#!/bin/bash
#SBATCH --job-name=write-recon
#SBATCH --account=perona
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs runs/write_sae_reconstruction_eval_1k

module purge
module load python/3.10.12-gcc-11.3.1-djdhgrq
module load cuda/11.8.0-gcc-11.3.1-i5witig

. .venv/bin/activate

python eval_write_sae_reconstruction.py \
  --ckpt runs/dit_s2_cifar10/last.pt \
  --block0-sae-ckpt runs/sae_block0_writes_16x_k32/last.pt \
  --remaining-sae-ckpt runs/sae_remaining_block_writes_16x_k32/last.pt \
  --data-dir data \
  --out-dir runs/write_sae_reconstruction_eval_1k \
  --num-samples 1000 \
  --batch-size 32 \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --precision bf16 \
  --seed 2026
