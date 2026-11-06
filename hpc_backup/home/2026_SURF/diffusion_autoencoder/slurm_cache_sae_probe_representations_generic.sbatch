#!/bin/bash
#SBATCH --job-name=sae-probe-reps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/sae-probe-reps-%j.out
#SBATCH --error=logs/sae-probe-reps-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
SCRATCH="/resnick/scratch/tbchen/surf_2026"
DATA="$SCRATCH/sae_linear_probes/celeba_32768.pt"
OUT="$SCRATCH/sae_linear_probes/representations"
UNION_V1="$SCRATCH/independent_routed_union_v1/runs"

cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" cache_sae_probe_representations.py \
  --data "$DATA" \
  --output-dir "$OUT" \
  --batch-size 128 \
  --direct "direct_mse=$SCRATCH/grouped_spatial_conv_ae_bookkeeping_v1/original_16k_k64_seed2026/best_valid.pt" \
  --union "all_mse_2branch=$UNION_V1/all_mse_control_2_branches/last.pt" \
  --union "plus_laplacian=$UNION_V1/mse_plus_laplacian/last.pt" \
  --union "plus_fourier_low=$UNION_V1/mse_plus_fourier_low/last.pt" \
  --union "plus_fourier_mid=$UNION_V1/mse_plus_fourier_mid/last.pt" \
  --union "plus_fourier_high=$UNION_V1/mse_plus_fourier_high/last.pt" \
  --union "all_mse_5branch=$UNION_V1/all_mse_control_5_branches/last.pt" \
  --union "tandem_all4=$SCRATCH/independent_routed_union_v2/runs/mse_plus_all_four_seed2030/last.pt" \
  --flat "standalone_laplacian=results/spatial_map_sae_weighted_seed_replicates/flat/weighted_laplacian/seed_2030/last.pt" \
  --flat "standalone_fourier_low=results/spatial_map_sae_weighted_seed_replicates/flat/weighted_fourier_low/seed_2030/last.pt" \
  --flat "standalone_fourier_mid=results/spatial_map_sae_weighted_seed_replicates/flat/weighted_fourier_mid/seed_2030/last.pt" \
  --flat "standalone_fourier_high=results/spatial_map_sae_weighted_seed_replicates/flat/weighted_fourier_high/seed_2030/last.pt"
