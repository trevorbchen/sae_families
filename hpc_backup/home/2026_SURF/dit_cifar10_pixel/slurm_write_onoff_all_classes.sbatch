#!/bin/bash
#SBATCH --job-name=write-onoff-cls
#SBATCH --account=perona
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-239%12
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs
module purge
module load python/3.10.12-gcc-11.3.1-djdhgrq
module load cuda/11.8.0-gcc-11.3.1-i5witig
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate

CLASS_IDS=(0 1 2 3 4 5 6 7 8 9)
CLASS_NAMES=(airplane automobile bird cat deer dog frog horse ship truck)
STREAMS=(attn mlp)
TASK_ID=${SLURM_ARRAY_TASK_ID}
CLASS_SLOT=$((TASK_ID / 24))
REM=$((TASK_ID % 24))
BLOCK_ID=$((REM / 2))
STREAM=${STREAMS[$((REM % 2))]}
CLASS_ID=${CLASS_IDS[${CLASS_SLOT}]}
CLASS_NAME=${CLASS_NAMES[${CLASS_SLOT}]}
OUT_ROOT="runs/block_${STREAM}_writes_onoff_ablation_16x_k32_class_${CLASS_ID}_${CLASS_NAME}"
BLOCK_DIR="${OUT_ROOT}/block$(printf "%02d" "${BLOCK_ID}")_${STREAM}"
RESULT="${BLOCK_DIR}/block$(printf "%02d" "${BLOCK_ID}")_${STREAM}_write_onoff_ablation.pt"
mkdir -p "${OUT_ROOT}"

if [ -f "${RESULT}" ]; then
  echo "skip existing result: ${RESULT}"
  exit 0
fi

echo "task=${TASK_ID} class_id=${CLASS_ID} class_name=${CLASS_NAME} block=${BLOCK_ID} stream=${STREAM} out=${OUT_ROOT}"
python write_onoff_ablation.py \
  --block-index "${BLOCK_ID}" \
  --stream "${STREAM}" \
  --ckpt runs/dit_s2_cifar10/last.pt \
  --block0-sae-ckpt runs/sae_block0_writes_16x_k32/last.pt \
  --remaining-sae-ckpt runs/sae_remaining_block_writes_16x_k32/last.pt \
  --data-dir data \
  --out-root "${OUT_ROOT}" \
  --num-images 12 \
  --class-id "${CLASS_ID}" \
  --t-values 0,100,200,300,400,500,600,700,800,900,999 \
  --features-per-forward 8 \
  --forward-batch-size 96 \
  --precision bf16 \
  --seed 2026
