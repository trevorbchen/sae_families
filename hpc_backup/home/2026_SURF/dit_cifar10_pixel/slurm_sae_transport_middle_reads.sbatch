#!/bin/bash
#SBATCH --job-name=sae-transport-middle
#SBATCH --account=perona
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs runs
module purge
module load python/3.10.12-gcc-11.3.1-djdhgrq
module load cuda/11.8.0-gcc-11.3.1-i5witig
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate

python train_transport_saes.py \
  --site middle \
  --ckpt runs/dit_s2_cifar10_consistency_transport/last.pt \
  --out-dir runs/sae_transport_middle_reads_16x_k32 \
  --data-dir data \
  --block-start 0 \
  --block-end 11 \
  --batch-size 32 \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --max-steps 10000 \
  --log-every 50 \
  --save-every 20000 \
  --dict-mult 16 \
  --k 32 \
  --precision bf16 \
  --seed 2026
