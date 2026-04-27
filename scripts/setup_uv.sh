#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Tuna environment setup via uv.
#
# This is a thin wrapper around `uv sync` plus a few extras that don't sit
# cleanly in pyproject.toml (CUDA-specific torch wheels, optional flash-attn).
#
# Usage:
#   bash scripts/setup_uv.sh                   # default: cu121 wheels
#   bash scripts/setup_uv.sh cu124             # pin a different CUDA wheel
#   bash scripts/setup_uv.sh cpu               # CPU-only (small smoke tests)
set -euo pipefail

CUDA_TAG="${1:-cu121}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[tuna] Setting up environment in $REPO_ROOT (cuda=$CUDA_TAG)"

# 1. Install uv if missing.
if ! command -v uv >/dev/null 2>&1; then
    echo "[tuna] uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env" 2>/dev/null || source "$HOME/.local/bin/env" 2>/dev/null || true
fi

# 2. Sync the locked environment.
echo "[tuna] uv sync ..."
uv sync

# 3. Re-install torch / torchvision against the requested CUDA wheel index
#    (uv.lock pins versions but not the CUDA tag, so we override here).
echo "[tuna] Installing torch wheels for $CUDA_TAG ..."
case "$CUDA_TAG" in
    cpu)
        INDEX_URL="https://download.pytorch.org/whl/cpu"
        ;;
    cu*)
        INDEX_URL="https://download.pytorch.org/whl/$CUDA_TAG"
        ;;
    *)
        echo "[tuna] Unknown CUDA tag: $CUDA_TAG (expected cpu, cu118, cu121, cu124, ...)" >&2
        exit 1
        ;;
esac
uv pip install --upgrade torch torchvision --index-url "$INDEX_URL"

# 4. Editable install of tuna itself so `python -m tuna.scripts.predict` works.
uv pip install -e .

echo
echo "[tuna] Done. Activate the environment with:"
echo "    source .venv/bin/activate"
echo
echo "Then try:"
echo "    python -c 'import tuna; print(tuna.__version__)'"
echo "    CUDA_VISIBLE_DEVICES=0 python -m tuna.scripts.predict --config-name t2i_1k \\"
echo "        prompt='a photo of a cat' inference.ckpt_path=/path/to/ckpt.pt"
