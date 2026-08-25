#!/usr/bin/env bash
set -euo pipefail
# TCR inference. Built-in prompt: examples/clip_32845.json (10 s @ 24 fps).
#
#   bash scripts/infer.sh \
#     --checkpoint /path/to/ltx-2.3-22b-dev.safetensors \
#     --text-encoder-path /path/to/gemma-3-12b-it \
#     --lora-path /path/to/separating-what-from-when.safetensors \
#     [--output outputs/tcr_infer.mp4]

source "$(cd "$(dirname "$0")" && pwd)/_env.sh"

CHECKPOINT="" TEXT_ENCODER="" LORA_PATH=""
OUTPUT="${OUTPUT:-${REPO_ROOT}/outputs/tcr_infer.mp4}"

usage() { echo "Usage: bash scripts/infer.sh --checkpoint CKPT --text-encoder-path GEMMA --lora-path LORA [--output OUT]"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --text-encoder-path) TEXT_ENCODER="$2"; shift 2 ;;
    --lora-path) LORA_PATH="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${CHECKPOINT}" || -z "${TEXT_ENCODER}" || -z "${LORA_PATH}" ]]; then
  usage; exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"
cd "${REPO_ROOT}/packages/ltx-trainer"
"${PYTHON}" scripts/inference.py \
  --checkpoint "${CHECKPOINT}" \
  --text-encoder-path "${TEXT_ENCODER}" \
  --lora-path "${LORA_PATH}" \
  --prompt-file "${REPO_ROOT}/examples/clip_32845.json" \
  --temporal-mode tcr --tcr-beta 5.0 \
  --width 704 --height 1280 --num-frames 241 --frame-rate 24 \
  --num-inference-steps 30 --guidance-scale 4.0 \
  --stg-scale 1.0 --stg-blocks 29 --stg-mode stg_av \
  --seed 42 --output "${OUTPUT}"
echo "Done: ${OUTPUT}"
