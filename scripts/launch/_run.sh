#!/usr/bin/env bash
# Generic Tuna predict launcher. Shared by all demo_predict_*.sh wrappers.
#
# Usage (called by demos):
#   bash scripts/launch/_run.sh <CONFIG_NAME> <MODEL_NAME>
#
# Required env (via wrapping demo or caller):
#   GPU         GPU index, default 0.
#   CKPT        Local path to the .pt checkpoint.
# Optional env:
#   PROMPT      For t2i: text prompt.
#   IMAGE_PATH  For edit/mmu: source image path.
#   INSTRUCTION For edit: edit instruction text.
# Any extra positional args are forwarded as Hydra overrides.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -d .venv && -z "${VIRTUAL_ENV:-}" ]] && source .venv/bin/activate

# Force HF Hub offline so Qwen / SigLIP / Wan VAE are loaded from local cache
# without round-tripping HF metadata APIs (avoids hangs on flaky networks).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

CONFIG="${1:?usage: _run.sh CONFIG MODEL [hydra_overrides...]}"
MODEL="${2:?usage: _run.sh CONFIG MODEL [hydra_overrides...]}"
shift 2

GPU="${GPU:-0}"
CKPT="${CKPT:-./checkpoints/${MODEL}.pt}"

EXTRA=()
[[ -n "${PROMPT:-}" ]]      && EXTRA+=("prompt='$PROMPT'")
[[ -n "${IMAGE_PATH:-}" ]]  && EXTRA+=("image_path=$IMAGE_PATH")
[[ -n "${INSTRUCTION:-}" ]] && EXTRA+=("instruction='$INSTRUCTION'")

CUDA_VISIBLE_DEVICES="$GPU" python -m tuna.scripts.predict \
    --config-name "$CONFIG" \
    "model=$MODEL" \
    "inference.ckpt_path=$CKPT" \
    "${EXTRA[@]}" \
    "$@"
