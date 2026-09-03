#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

UGNS_DATA_ROOT="${UGNS_DATA_ROOT:-${REPO_ROOT}/data}"
UGNS_OUTPUT_DIR="${UGNS_OUTPUT_DIR:-${REPO_ROOT}/outputs/training}"
UGNS_PRETRAINED="${UGNS_PRETRAINED:-${REPO_ROOT}/checkpoints/256x256_diffusion_uncond.pt}"
GPU_ID="${GPU_ID:-0}"

mkdir -p "${UGNS_OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
UGNS_DATA_ROOT="${UGNS_DATA_ROOT}" \
python "${SCRIPT_DIR}/train_256_with_guided_diffusion.py" \
    --data_dir "${UGNS_DATA_ROOT}/ultrasound_drus_train_256" \
    --output_dir "${UGNS_OUTPUT_DIR}" \
    --pretrained_path "${UGNS_PRETRAINED}" \
    --batch_size 2 \
    --grad_accum_steps 4 \
    --use_checkpoint True \
    --use_amp False \
    "$@"
