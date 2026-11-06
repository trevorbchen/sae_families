#!/bin/bash
#SBATCH --job-name=t-eps-curve
#SBATCH --account=perona
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs runs/timestep_eps_norm_curve_eval

module purge
module load python/3.10.12-gcc-11.3.1-djdhgrq
module load cuda/11.8.0-gcc-11.3.1-i5witig

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

. .venv/bin/activate

python eval_timestep_eps_norm_curve.py \
  --ckpt runs/dit_s2_cifar10/last.pt \
  --data-dir data \
  --out-dir runs/timestep_eps_norm_curve_eval \
  --num-images 512 \
  --batch-size 8 \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --candidate-step 10 \
  --extra-candidates 999 \
  --precision bf16 \
  --seed 2026
