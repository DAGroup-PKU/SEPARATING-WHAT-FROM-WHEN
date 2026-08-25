#!/usr/bin/env bash
set -euo pipefail
# TCR training. Recipe: configs/tcr_av_lora.yaml
#
#   bash scripts/train.sh \
#     --model-path /path/to/ltx-2.3-22b-dev.safetensors \
#     --text-encoder-path /path/to/gemma-3-12b-it \
#     --dataset /path/to/dataset.json \
#     --output-dir ./outputs/tcr_av_lora
#
#   bash scripts/train.sh \
#     --model-path /path/to/ltx-2.3-22b-dev.safetensors \
#     --text-encoder-path /path/to/gemma-3-12b-it \
#     --data-root /path/to/preprocessed \
#     --output-dir ./outputs/tcr_av_lora

source "$(cd "$(dirname "$0")" && pwd)/_env.sh"

MODEL_PATH="" TEXT_ENCODER="" DATASET="" DATA_ROOT=""
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tcr_av_lora}"

usage() { echo "Usage: bash scripts/train.sh --model-path CKPT --text-encoder-path GEMMA (--dataset JSON | --data-root DIR) --output-dir DIR"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --text-encoder-path) TEXT_ENCODER="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${MODEL_PATH}" || -z "${TEXT_ENCODER}" || -z "${OUTPUT_DIR}" ]]; then
  usage; exit 1
fi
if [[ -z "${DATA_ROOT}" && -z "${DATASET}" ]]; then
  usage; exit 1
fi

if [[ -z "${DATA_ROOT}" ]]; then
  DATA_ROOT="$(cd "$(dirname "${DATASET}")" && pwd)/.precomputed"
  if [[ ! -d "${DATA_ROOT}/latents" || ! -d "${DATA_ROOT}/conditions" || ! -d "${DATA_ROOT}/audio_latents" ]]; then
    cd "${REPO_ROOT}/packages/ltx-trainer"
    "${PYTHON}" scripts/process_dataset.py \
      "${DATASET}" \
      --resolution-buckets 704x1280x113 \
      --model-path "${MODEL_PATH}" \
      --text-encoder-path "${TEXT_ENCODER}" \
      --output-dir "${DATA_ROOT}" \
      --with-audio
  fi
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}/packages/ltx-trainer"
CMD=(
  "${PYTHON}" scripts/train.py "${REPO_ROOT}/configs/tcr_av_lora.yaml"
  --model-path "${MODEL_PATH}"
  --text-encoder-path "${TEXT_ENCODER}"
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
)
# Optional: STEPS=2 bash scripts/train.sh ...  (smoke / debug; not a public flag)
if [[ -n "${STEPS:-}" ]]; then
  CMD+=(--steps "${STEPS}")
fi
"${CMD[@]}"
echo "Checkpoints: ${OUTPUT_DIR}/checkpoints/"
