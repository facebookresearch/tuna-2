#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# ============================================================================
# Tuna unified inference script.
#
# Usage:
#   bash scripts/launch/predict.sh [OPTIONS]
#
# Required:
#   --ckpt PATH          Path to the model checkpoint (.pt)
#   --prompt TEXT         Text prompt (for t2i) or instruction (for edit)
#
# Optional:
#   --task t2i|edit       Inference task (default: t2i)
#   --variant vae|siglip_pixel|none_encoder
#                         Model variant (default: none_encoder)
#   --resolution HxW     Output resolution, e.g. 512x512, 1024x1024,
#                         448x576, 576x448, 384x672, 672x384 (default: 512x512)
#   --gpu ID              GPU device index (default: 0)
#   --image PATH          Source image path (required for edit)
#   --output DIR          Output directory (default: ./outputs/predictions)
#   --steps N             Number of inference steps (default: 50)
#   --guidance FLOAT      Guidance scale (default: from config)
#   --seed INT            Random seed (default: 42)
#   --negative TEXT       Negative prompt (default: from config)
#
# Examples:
#   # Text-to-image with Tuna-2 (no encoder) at 512px
#   bash scripts/launch/predict.sh \
#       --ckpt /path/to/tuna_2_pixel_7b.pt \
#       --prompt "a photo of a cat sitting on a windowsill"
#
#   # Text-to-image with Tuna-R (SigLIP pixel) at 1024px
#   bash scripts/launch/predict.sh \
#       --variant siglip_pixel --resolution 1024 \
#       --ckpt /path/to/tuna_2r_pixel_7b.pt \
#       --prompt "a highly realistic beauty portrait"
#
#   # Text-to-image with Tuna (VAE) at 512px
#   bash scripts/launch/predict.sh \
#       --variant vae \
#       --ckpt /path/to/tuna_7b.pt \
#       --prompt "an oil painting of a coastal village"
#
#   # Image editing
#   bash scripts/launch/predict.sh \
#       --task edit --variant vae \
#       --ckpt /path/to/edit_ckpt.pt \
#       --image /path/to/source.jpg \
#       --prompt "replace the people with robots"
# ============================================================================

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -d .venv && -z "${VIRTUAL_ENV:-}" ]] && source .venv/bin/activate

# ---- Defaults ----
TASK="t2i"
VARIANT="none_encoder"
SIZE="7b"
RESOLUTION="512x512"
GPU="0"
CKPT=""
PROMPT=""
IMAGE=""
OUTPUT=""
EXTRA_ARGS=()

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)       TASK="$2"; shift 2 ;;
        --variant)    VARIANT="$2"; shift 2 ;;
        --size)       SIZE="$2"; shift 2 ;;
        --resolution) RESOLUTION="$2"; shift 2 ;;
        --gpu)        GPU="$2"; shift 2 ;;
        --ckpt)       CKPT="$2"; shift 2 ;;
        --prompt)     PROMPT="$2"; shift 2 ;;
        --image)      IMAGE="$2"; shift 2 ;;
        --output)     OUTPUT="$2"; shift 2 ;;
        --steps)      EXTRA_ARGS+=("inference.num_inference_steps=$2"); shift 2 ;;
        --guidance)   EXTRA_ARGS+=("inference.guidance_scale=$2"); shift 2 ;;
        --seed)       EXTRA_ARGS+=("inference.seed=$2"); shift 2 ;;
        --negative)   EXTRA_ARGS+=("inference.negative_prompt='$2'"); shift 2 ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

[[ -z "$CKPT" ]] && { echo "Error: --ckpt is required"; exit 1; }
[[ -z "$PROMPT" ]] && { echo "Error: --prompt is required"; exit 1; }
[[ "$TASK" == "edit" && -z "$IMAGE" ]] && { echo "Error: --image is required for edit"; exit 1; }

# ---- Resolve config + model ----
case "${VARIANT}_${SIZE}" in
    none_encoder_7b) MODEL="tuna_2_pixel_7b" ;;
    none_encoder_*)  MODEL="tuna_2_pixel_7b" ;;
    siglip_pixel_7b) MODEL="tuna_2r_pixel_7b" ;;
    siglip_pixel_*)  MODEL="tuna_2r_pixel_7b" ;;
    vae_2b)          MODEL="tuna_2b" ;;
    vae_7b)          MODEL="tuna_7b" ;;
    vae_*)           MODEL="tuna_7b" ;;
    *)               echo "Error: --variant must be vae|siglip_pixel|none_encoder"; exit 1 ;;
esac

# Parse HxW resolution
RES_H="${RESOLUTION%%x*}"
RES_W="${RESOLUTION##*x}"
[[ "$RES_H" == "$RESOLUTION" ]] && { RES_H="$RESOLUTION"; RES_W="$RESOLUTION"; }

EXTRA_ARGS+=("inference.height=$RES_H" "inference.width=$RES_W")

if [[ "$TASK" == "t2i" ]]; then
    CONFIG="t2i"
    case "$VARIANT" in
        none_encoder)
            EXTRA_ARGS+=("inference.pipe=Tuna2PixelPipeline"
                         "inference.generation_mode=t2i_pixel"
                         "inference.guidance_scale=3"
                         "inference.sampling_method=euler") ;;
        siglip_pixel)
            EXTRA_ARGS+=("inference.pipe=Tuna2RPixelPipeline"
                         "inference.generation_mode=t2i_pixel"
                         "inference.guidance_scale=4"
                         "inference.sampling_method=heun") ;;
        vae)
            EXTRA_ARGS+=("inference.pipe=TunaPipeline"
                         "inference.generation_mode=t2i"
                         "inference.guidance_scale=7.5") ;;
    esac
elif [[ "$TASK" == "edit" ]]; then
    CONFIG="edit"
    case "$VARIANT" in
        none_encoder)
            EXTRA_ARGS+=("inference.pipe=Tuna2PixelPipeline"
                         "inference.guidance_scale=2"
                         "inference.sampling_method=euler") ;;
        siglip_pixel)
            EXTRA_ARGS+=("inference.pipe=Tuna2RPixelPipeline"
                         "inference.guidance_scale=5"
                         "inference.sampling_method=heun") ;;
        vae)
            EXTRA_ARGS+=("inference.pipe=TunaPipeline"
                         "inference.weight_dtype=float32"
                         "inference.guidance_scale=7.5") ;;
    esac
else
    echo "Error: --task must be t2i|edit"; exit 1
fi

[[ -n "$OUTPUT" ]] && EXTRA_ARGS+=("inference.output_dir=$OUTPUT")

# ---- Build hydra command ----
HYDRA_ARGS=(
    --config-name "$CONFIG"
    "model=$MODEL"
    "inference.ckpt_path=$CKPT"
    "prompt='$PROMPT'"
)

if [[ "$TASK" == "edit" ]]; then
    HYDRA_ARGS+=("image_path=$IMAGE" "instruction='$PROMPT'")
    # For edit, prompt is the instruction
    unset 'HYDRA_ARGS[3]'  # remove the prompt= line
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "Task: $TASK | Variant: $VARIANT | Resolution: $RESOLUTION | GPU: $GPU"
echo "Config: $CONFIG | Model: $MODEL"

CUDA_VISIBLE_DEVICES="$GPU" python -m tuna.scripts.predict \
    "${HYDRA_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
