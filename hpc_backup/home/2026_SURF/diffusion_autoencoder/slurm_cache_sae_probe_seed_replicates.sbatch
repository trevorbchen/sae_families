#!/bin/bash
#SBATCH --job-name=probe-cache-seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --nice=10000
#SBATCH --output=logs/probe-cache-seeds-%A_%a.out
#SBATCH --error=logs/probe-cache-seeds-%A_%a.err
set -euo pipefail

SEEDS=(2026 2031 2032)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
SCRATCH="/resnick/scratch/tbchen/surf_2026"
EXP="$SCRATCH/sae_linear_probe_seed_replicates_v1"
OUT="$EXP/representations/seed_$SEED"

if [[ "$SEED" == 2026 ]]; then
  DIRECT="$SCRATCH/grouped_spatial_conv_ae_bookkeeping_v1/original_16k_k64_seed2026/best_valid.pt"
  UNION="$SCRATCH/independent_routed_union_v1/runs"
  direct_path="$DIRECT"
  path_for() {
    case "$1" in
      all_mse_2branch) label=all_mse_control_2_branches ;;
      plus_laplacian) label=mse_plus_laplacian ;;
      plus_fourier_low) label=mse_plus_fourier_low ;;
      plus_fourier_mid) label=mse_plus_fourier_mid ;;
      plus_fourier_high) label=mse_plus_fourier_high ;;
      all_mse_5branch) label=all_mse_control_5_branches ;;
      tandem_all4) label=mse_plus_all_four ;;
    esac
    printf '%s/%s/last.pt' "$UNION" "$label"
  }
else
  RUNS="$EXP/runs"
  direct_path="$RUNS/direct_mse/seed_$SEED/best_valid.pt"
  path_for() { printf '%s/%s/seed_%s/last.pt' "$RUNS" "$1" "$SEED"; }
fi

cd "$ROOT"
mkdir -p logs "$OUT"
exec "$PY" cache_sae_probe_representations.py \
  --data "$SCRATCH/sae_linear_probes/celeba_32768.pt" \
  --output-dir "$OUT" --batch-size 128 \
  --direct "direct_mse=$direct_path" \
  --union "all_mse_2branch=$(path_for all_mse_2branch)" \
  --union "plus_laplacian=$(path_for plus_laplacian)" \
  --union "plus_fourier_low=$(path_for plus_fourier_low)" \
  --union "plus_fourier_mid=$(path_for plus_fourier_mid)" \
  --union "plus_fourier_high=$(path_for plus_fourier_high)" \
  --union "all_mse_5branch=$(path_for all_mse_5branch)" \
  --union "tandem_all4=$(path_for tandem_all4)"
