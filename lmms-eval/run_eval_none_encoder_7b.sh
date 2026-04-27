#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Evaluate Tuna-2 (no encoder, 7B) on multimodal understanding benchmarks.
set -e
export TOKENIZERS_PARALLELISM=false

CKPT_PATH="${CKPT_PATH:-/path/to/tuna_2_pixel_7b.pt}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU="${GPU:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12347}"
CONFIG_FILE="lmms_eval/models/configs/tuna_none_encoder_7b.yaml"
TASKS="${TASKS:-ai2d,gqa,ocrbench,vstar_bench,realworldqa,chartqa,mmvet,seedbench_2_plus,countbench,mmvp,visulogic}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval/none_encoder_7b}"

cd "$(dirname "$0")"
mkdir -p log

CKPT_NAME=$(basename "${CKPT_PATH}" .pt)
LOG_FILE="log/eval_none_encoder_7b_${CKPT_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Model: tuna_none_encoder | Config: ${CONFIG_FILE}"
echo "Checkpoint: ${CKPT_PATH} | Tasks: ${TASKS}"

CUDA_VISIBLE_DEVICES=${GPU} accelerate launch \
    --main_process_port ${MAIN_PROCESS_PORT} \
    --num_processes ${NUM_GPUS} \
    -m lmms_eval \
    --model tuna_none_encoder \
    --model_args config_file=${CONFIG_FILE},ckpt_path=${CKPT_PATH},precision=bf16 \
    --tasks ${TASKS} \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix tuna_none_encoder \
    --output_path ${OUTPUT_DIR} \
    2>&1 | tee "${LOG_FILE}"
