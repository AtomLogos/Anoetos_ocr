#!/bin/bash
set -euo pipefail

echo "Starting ancient character inference (YOLO + EfficientNet)..."

echo "===== Runtime paths ====="
echo "PWD=$(pwd)"
echo "INPUT_DIR=${INPUT_DIR:-<default>}"
echo "OUTPUT_FILE=${OUTPUT_FILE:-<default>}"
for path in /saisdata /saisresult /app; do
  if [ -e "${path}" ]; then
    echo "${path}: present"
    ls -la "${path}" | head -n 40 || true
  else
    echo "${path}: missing"
  fi
done
echo "===== End runtime paths ====="

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${NVIDIA_VISIBLE_DEVICES:-}" ] \
  && [ "${NVIDIA_VISIBLE_DEVICES}" != "all" ] && [ "${NVIDIA_VISIBLE_DEVICES}" != "void" ]; then
  export CUDA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}"
fi

echo "===== GPU diagnostics ====="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
ls -l /dev/nvidia* 2>/dev/null || echo "/dev/nvidia* not found"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA devices:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"
echo "===== End GPU diagnostics ====="

if [ ! -d "/saisdata" ]; then
  echo "Warning: /saisdata not found; continuing so prediction.json can still be produced"
fi

mkdir -p /saisresult

python3 /app/src/run_inference.py

PREDICTION_FILE="${OUTPUT_FILE:-/saisresult/prediction.json}"
if [ ! -f "${PREDICTION_FILE}" ]; then
  echo "Error: ${PREDICTION_FILE} not found"
  exit 1
fi

echo "Done!"
