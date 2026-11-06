#!/bin/bash
set -euo pipefail

BASE="/resnick/scratch/tbchen/surf_2026/sae_lambda_sweep_matched_32768_v1"
SPATIAL_SOURCE="/resnick/scratch/tbchen/surf_2026/spatial_ae_experiment_4x_k8_32768/results/flat/proper_loss"
VECTOR_SOURCE="/resnick/scratch/tbchen/surf_2026/vector_ae_experiment_4x_k8_32768/results/proper_loss"
mkdir -p "$BASE"

link_constrained_sources() {
  local variant="$1"
  local source="$2"
  local exp="$BASE/$variant"
  mkdir -p "$exp/checkpoints/mse_control/latent_mse" "$exp/checkpoints/lambda_1p0"
  for seed in 2030 2031 2032; do
    ln -sfn "$source/latent_mse/seed_$seed" \
      "$exp/checkpoints/mse_control/latent_mse/seed_$seed"
  done
  for objective in laplacian fourier_low fourier_mid fourier_high; do
    for seed in 2030 2031 2032; do
      local target="$exp/checkpoints/lambda_1p0/$objective/seed_$seed"
      mkdir -p "$(dirname "$target")"
      ln -sfn "$source/$objective/seed_$seed" "$target"
    done
  done
}

link_constrained_sources spatial_flat_k8 "$SPATIAL_SOURCE"
link_constrained_sources vector_1d_k8 "$VECTOR_SOURCE"

submit_variant() {
  local variant="$1"
  local memory="$2"
  local time_limit="$3"
  local short="$4"
  local train geometry alignment final
  train=$(sbatch --parsable --job-name="lf-$short-tr" --mem="$memory" \
    --time="$time_limit" --export="ALL,VARIANT=$variant" \
    slurm_lambda_family_train.sbatch)
  train="${train%%;*}"
  geometry=$(sbatch --parsable --job-name="lf-$short-geo" --mem="$memory" \
    --time="$time_limit" \
    --dependency="afterok:$train" --export="ALL,VARIANT=$variant" \
    slurm_lambda_family_geometry.sbatch)
  geometry="${geometry%%;*}"
  alignment=$(sbatch --parsable --job-name="lf-$short-aln" \
    --dependency="afterok:$train" --export="ALL,VARIANT=$variant" \
    slurm_lambda_family_alignment.sbatch)
  alignment="${alignment%%;*}"
  final=$(sbatch --parsable --job-name="lf-$short-fin" \
    --dependency="afterok:$geometry:$alignment" --export="ALL,VARIANT=$variant" \
    slurm_lambda_family_finalize.sbatch)
  final="${final%%;*}"
  printf '%s,%s,%s,%s,%s\n' "$variant" "$train" "$geometry" "$alignment" "$final"
}

{
  echo "variant,train,geometry,alignment,final"
  submit_variant spatial_flat_k32 48G 04:00:00 sf32
  submit_variant spatial_flat_k8 64G 04:00:00 sf8
  submit_variant vector_1d_k32 32G 03:00:00 v32
  submit_variant vector_1d_k8 32G 03:00:00 v8
} | tee "$BASE/submission_jobs.csv"
