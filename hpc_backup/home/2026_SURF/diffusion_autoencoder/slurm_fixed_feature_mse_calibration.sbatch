#!/bin/bash
#SBATCH --job-name=fixed-mse-cal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-5%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/fixed-mse-cal-%A_%a.out
#SBATCH --error=logs/fixed-mse-cal-%A_%a.err
set -euo pipefail

AMPLITUDES=(0.25 0.5 0.75 1.0 1.5 2.0)
TAGS=(a025 a050 a075 a100 a150 a200)
INDEX="$SLURM_ARRAY_TASK_ID"
AMPLITUDE="${AMPLITUDES[$INDEX]}"
TAG="${TAGS[$INDEX]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
RUN_ROOT="/resnick/scratch/tbchen/surf_2026/fixed_feature_spatial/calibration/$TAG"
CACHE="$RUN_ROOT/ffhq_fixed_high.pt"
SAE_ROOT="$RUN_ROOT/saes"
cd "$ROOT"
mkdir -p logs "$RUN_ROOT" "$SAE_ROOT/latent_mse/seed_2030" "$RUN_ROOT/analysis"

"$PY" prepare_fixed_feature_spatial_latents.py \
  --data /resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt \
  --clean-latent-cache /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --autoencoder results/spatial_conv_ae/last.pt \
  --output "$CACHE" --preview "$RUN_ROOT/injection_preview.png" \
  --frequency-x 17 --frequency-y 17 --amplitude "$AMPLITUDE" --prevalence 0.05

"$PY" train_spatial_map_sae.py \
  --architecture flat --objective latent_mse \
  --data /resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt \
  --latent-cache "$CACHE" --autoencoder results/spatial_conv_ae/last.pt \
  --out-dir "$SAE_ROOT/latent_mse/seed_2030" \
  --feature-count 4096 --k 32 --steps 20000 --batch-size 32 \
  --lr 3e-4 --warmup 500 --grad-clip 1.0 \
  --calibration-examples 2048 --reinit-every 1000 \
  --dead-after 3000 --max-rescue 256 --seed 2030

exec "$PY" analyze_fixed_feature_sae_detection.py \
  --cache "$CACHE" --autoencoder results/spatial_conv_ae/last.pt \
  --sae-root "$SAE_ROOT" --output-dir "$RUN_ROOT/analysis" \
  --objectives latent_mse --seeds 2030
