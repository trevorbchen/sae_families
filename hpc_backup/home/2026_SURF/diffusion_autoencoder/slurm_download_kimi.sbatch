#!/bin/bash
#SBATCH --job-name=kimi-vl-download
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/kimi-vl-download-%j.out
#SBATCH --error=logs/kimi-vl-download-%j.err
set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs "/resnick/scratch/tbchen/models"
export HF_HOME="/resnick/scratch/tbchen/hf_cache"
export HF_HUB_CACHE="/resnick/scratch/tbchen/hf_cache"
export SSL_CERT_FILE="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/lib/python3.10/site-packages/certifi/cacert.pem"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
"$PY" - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="moonshotai/Kimi-VL-A3B-Instruct",
    local_dir="/resnick/scratch/tbchen/models/Kimi-VL-A3B-Instruct",
    max_workers=4,
)
print(f"downloaded to {path}", flush=True)
PY
