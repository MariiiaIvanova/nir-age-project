#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_KERNEL_CACHE_PATH=/tmp/nir-torch-kernels
mkdir -p "$PYTORCH_KERNEL_CACHE_PATH" logs
log_file="logs/train-whisper-adamw-val-$(date +%Y%m%d-%H%M%S).log"
set +e
../nir_venv/bin/python -u app/train.py \
  --splits output/native/validation_seed42 \
  --weights weights/tiny.pt \
  --run output/native/runs/whisper_tiny_adamw_val_seed42 \
  --epochs 30 --batch-size 32 --seed 42 2>&1 | tee "$log_file"
train_status=${PIPESTATUS[0]}
printf 'Код завершения обучения: %s\n' "$train_status" | tee -a "$log_file"
exit "$train_status"
