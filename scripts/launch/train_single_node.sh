#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Launch single-node multi-GPU Tuna training via torchrun.
#
# Usage:
#   bash scripts/launch/train_single_node.sh
#   bash scripts/launch/train_single_node.sh --config-name stage2_edit
#   bash scripts/launch/train_single_node.sh training.batch_size=8

set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CONFIG_NAME="${CONFIG_NAME:-stage1_t2i}"

torchrun \
    --standalone \
    --nproc-per-node="${NPROC_PER_NODE}" \
    -m tuna.scripts.train \
    --config-name "${CONFIG_NAME}" \
    "$@"
