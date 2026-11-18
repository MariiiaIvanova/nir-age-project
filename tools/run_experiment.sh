#!/usr/bin/env bash
# Usage: bash tools/run_experiment.sh EXTRACTOR WEIGHTS NEW_RUN_DIRECTORY [train arguments]
set -euo pipefail
if (( $# < 3 )); then
  echo 'Usage: run_experiment.sh EXTRACTOR WEIGHTS NEW_RUN_DIRECTORY [train arguments]' >&2
  exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
extractor_name=$1
weights_path=$2
run_path=$3
shift 3
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-/tmp/nir-torch-kernels}"
mkdir -p "$PYTORCH_KERNEL_CACHE_PATH" logs
log_file="logs/train-${extractor_name}-$(date +%Y%m%d-%H%M%S).log"
python_bin="${NIR_PYTHON:-../nir_venv/bin/python}"
set +e
"$python_bin" -u app/train.py --extractor "$extractor_name" --weights "$weights_path" \
  --splits output/native/validation_seed42 --run "$run_path" "$@" 2>&1 | tee "$log_file"
train_status=${PIPESTATUS[0]}
printf 'Код завершения обучения: %s\n' "$train_status" | tee -a "$log_file"
exit "$train_status"
