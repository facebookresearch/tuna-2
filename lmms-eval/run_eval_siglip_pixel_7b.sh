#!/bin/bash
# Evaluate Tuna-R (SigLIP pixel, 7B) on multimodal understanding benchmarks.
set -e
export TOKENIZERS_PARALLELISM=false

CKPT_PATH="${CKPT_PATH:-/path/to/tuna_2r_pixel_7b.pt}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU="${GPU:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12346}"
CONFIG_FILE="lmms_eval/models/configs/tuna_siglip_pixel_7b.yaml"
TASKS="${TASKS:-ai2d,gqa,ocrbench,vstar_bench,realworldqa,chartqa,mmvet,seedbench_2_plus,countbench,mmvp,visulogic,mmmu_val}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval/siglip_pixel_7b}"

cd "$(dirname "$0")"
mkdir -p log

CKPT_NAME=$(basename "${CKPT_PATH}" .pt)
LOG_FILE="log/eval_siglip_pixel_7b_${CKPT_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Model: tuna_siglip_pixel | Config: ${CONFIG_FILE}"
echo "Checkpoint: ${CKPT_PATH} | Tasks: ${TASKS}"

CUDA_VISIBLE_DEVICES=${GPU} accelerate launch \
    --main_process_port ${MAIN_PROCESS_PORT} \
    --num_processes ${NUM_GPUS} \
    -m lmms_eval \
    --model tuna_siglip_pixel \
    --model_args config_file=${CONFIG_FILE},ckpt_path=${CKPT_PATH},precision=bf16 \
    --tasks ${TASKS} \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix tuna_siglip_pixel \
    --output_path ${OUTPUT_DIR} \
    2>&1 | tee "${LOG_FILE}"
