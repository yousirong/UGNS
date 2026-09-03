#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

UGNS_INPUT_DIR="${UGNS_INPUT_DIR:-${REPO_ROOT}/data/input}"
UGNS_OUTPUT_DIR="${UGNS_OUTPUT_DIR:-${REPO_ROOT}/outputs/inference}"
UGNS_CHECKPOINT="${UGNS_CHECKPOINT:-${REPO_ROOT}/checkpoints/ugns_prior.pt}"
UGNS_CONFIG="${UGNS_CONFIG:-${REPO_ROOT}/checkpoints/config.json}"
GPU_ID="${GPU_ID:-0}"

if [[ ! -e "${UGNS_INPUT_DIR}" ]]; then
    echo "Input path does not exist: ${UGNS_INPUT_DIR}" >&2
    echo "Set UGNS_INPUT_DIR to an image or directory." >&2
    exit 2
fi

mkdir -p "${UGNS_OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
UGNS_CHECKPOINT="${UGNS_CHECKPOINT}" \
UGNS_CONFIG="${UGNS_CONFIG}" \
python "${SCRIPT_DIR}/wn_ddnm_speckle_denoising_best_fusion.py" \
    --preset paper \
    --input "${UGNS_INPUT_DIR}" \
    --output_dir "${UGNS_OUTPUT_DIR}" \
    "$@"
