#!/bin/bash
set -euo pipefail

BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
mkdir -p "$BASE" logs

submit_id() {
  local output
  output=$(sbatch --parsable "$@")
  printf '%s' "${output%%;*}"
}

# The vector AE and its constrained lambda=10 family have normal priority.
VECTOR_AE=$(submit_id --job-name=m512-vector-ae \
  --export=ALL,LATENT_SOURCE=vector slurm_matched512_autoencoder.sbatch)
PRIORITY_TRAIN=$(submit_id --job-name=m512-v4-l10-tr --array=0-14%4 \
  --dependency="afterok:$VECTOR_AE" \
  --export=ALL,ARM=vector_4x_k8,RUN_MODE=priority_10 \
  slurm_matched512_sae_train.sbatch)
PRIORITY_GEOMETRY=$(submit_id --job-name=m512-v4-l10-geo --array=0-14%4 \
  --dependency="afterok:$PRIORITY_TRAIN" \
  --export=ALL,ARM=vector_4x_k8,RUN_MODE=priority_10 \
  slurm_matched512_geometry.sbatch)
PRIORITY_ALIGNMENT=$(submit_id --job-name=m512-v4-l10-aln --array=0 \
  --dependency="afterok:$PRIORITY_TRAIN" \
  --export=ALL,ARM=vector_4x_k8,RUN_MODE=priority_10 \
  slurm_matched512_alignment.sbatch)
PRIORITY_FINAL=$(submit_id --job-name=m512-v4-l10-fin \
  --dependency="afterok:$PRIORITY_GEOMETRY:$PRIORITY_ALIGNMENT" \
  --export=ALL,ARM=vector_4x_k8,RUN_MODE=priority_10 \
  slurm_matched512_finalize.sbatch)

# Everything else is deliberately niced so the priority wave wins scheduling ties.
SPATIAL_AE=$(submit_id --job-name=m512-spatial-ae --nice=10000 \
  --export=ALL,LATENT_SOURCE=spatial slurm_matched512_autoencoder.sbatch)

submit_arm() {
  local arm="$1"
  local dependency="$2"
  local short="$3"
  local train geometry alignment final
  train=$(submit_id --job-name="m512-$short-tr" --array=0-62%4 --nice=10000 \
    --dependency="afterok:$dependency" --export="ALL,ARM=$arm,RUN_MODE=full" \
    slurm_matched512_sae_train.sbatch)
  geometry=$(submit_id --job-name="m512-$short-geo" --array=0-62%4 --nice=10000 \
    --dependency="afterok:$train" --export="ALL,ARM=$arm,RUN_MODE=full" \
    slurm_matched512_geometry.sbatch)
  alignment=$(submit_id --job-name="m512-$short-aln" --array=0-4%1 --nice=10000 \
    --dependency="afterok:$train" --export="ALL,ARM=$arm,RUN_MODE=full" \
    slurm_matched512_alignment.sbatch)
  final=$(submit_id --job-name="m512-$short-fin" --nice=10000 \
    --dependency="afterok:$geometry:$alignment" \
    --export="ALL,ARM=$arm,RUN_MODE=full" slurm_matched512_finalize.sbatch)
  printf '%s,%s,%s,%s,%s\n' "$arm" "$train" "$geometry" "$alignment" "$final"
}

VECTOR4_ROW=$(submit_arm vector_4x_k8 "$PRIORITY_TRAIN" v4)
VECTOR16_ROW=$(submit_arm vector_16x_k32 "$VECTOR_AE" v16)
SPATIAL4_ROW=$(submit_arm spatial_4x_k8 "$SPATIAL_AE" s4)
SPATIAL16_ROW=$(submit_arm spatial_16x_k32 "$SPATIAL_AE" s16)

IFS=, read -r _ _ _ _ VECTOR4_FINAL <<< "$VECTOR4_ROW"
IFS=, read -r _ _ _ _ VECTOR16_FINAL <<< "$VECTOR16_ROW"
IFS=, read -r _ _ _ _ SPATIAL4_FINAL <<< "$SPATIAL4_ROW"
IFS=, read -r _ _ _ _ SPATIAL16_FINAL <<< "$SPATIAL16_ROW"
FAMILY_FINAL=$(submit_id --job-name=m512-overview --nice=10000 \
  --dependency="afterok:$VECTOR4_FINAL:$VECTOR16_FINAL:$SPATIAL4_FINAL:$SPATIAL16_FINAL" \
  slurm_matched512_family_finalize.sbatch)

{
  echo "priority_component,job_id"
  echo "vector_autoencoder,$VECTOR_AE"
  echo "vector_4x_lambda10_train,$PRIORITY_TRAIN"
  echo "vector_4x_lambda10_geometry,$PRIORITY_GEOMETRY"
  echo "vector_4x_lambda10_alignment,$PRIORITY_ALIGNMENT"
  echo "vector_4x_lambda10_finalize,$PRIORITY_FINAL"
  echo "spatial_autoencoder,$SPATIAL_AE"
  echo
  echo "arm,train,geometry,alignment,final"
  echo "$VECTOR4_ROW"
  echo "$VECTOR16_ROW"
  echo "$SPATIAL4_ROW"
  echo "$SPATIAL16_ROW"
  echo "family_overview,,,,$FAMILY_FINAL"
} | tee "$BASE/submission_jobs.csv"

cat > "$BASE/EXPERIMENT.md" <<'EOF'
# Matched 512-dimensional SAE lambda sweep

- Data: the same 32,768 FFHQ images for both autoencoders and every SAE.
- Split: autoencoder seed 2030; SAE data-order seed 2026.
- Latents: spatial ConvAE `8x8x8=512` and conventional vector AE `512`.
- SAE capacities: constrained `2048/k8` and unconstrained `8192/k32`.
- Objectives: latent MSE, MSE + Laplacian, and MSE + Fourier low/mid/high.
- Transform weights: 0.1, 0.3, 1.0, 3.0, and 10.0.
- SAE seeds: 2030, 2031, and 2032.
- Evaluation: held-out latent EV, objective/pixel/latent losses, gradient histories,
  used/dead features, pairwise vocabulary categories, edited-output geometry, and
  causal-delta geometry.
- Visualization: centered `coolwarm` heatmaps relative to the five-vocabulary mean.

The vector `2048/k8`, lambda=10 family is submitted first at normal priority. All
remaining jobs are niced so that this qualitative-inspection wave schedules first.
EOF
