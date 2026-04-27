#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Launch multi-node Tuna training via torchrun.
#
# Required env vars:
#   NNODES         — number of nodes
#   NODE_RANK      — rank of *this* node (0..NNODES-1)
#   MASTER_ADDR    — hostname of node 0
#   MASTER_PORT    — port (default 29500)
#
# Optional:
#   NPROC_PER_NODE (default 8)
#   CONFIG_NAME    (default stage1_t2i)
#
# Usage on each node:
#   NNODES=4 NODE_RANK=0 MASTER_ADDR=node0 \
#       bash scripts/launch/train_multi_node.sh
#   NNODES=4 NODE_RANK=1 MASTER_ADDR=node0 \
#       bash scripts/launch/train_multi_node.sh

set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CONFIG_NAME="${CONFIG_NAME:-stage1_t2i}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -z "${NNODES:-}" || -z "${NODE_RANK:-}" || -z "${MASTER_ADDR:-}" ]]; then
    echo "ERROR: NNODES, NODE_RANK and MASTER_ADDR must all be set." >&2
    exit 1
fi

torchrun \
    --nnodes="${NNODES}" \
    --node-rank="${NODE_RANK}" \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    -m tuna.scripts.train \
    --config-name "${CONFIG_NAME}" \
    "$@"
