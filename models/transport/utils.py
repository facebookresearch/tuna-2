# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import math
from typing import Any

import torch as th


class EasyDict:
    def __init__(self, sub_dict):
        for k, v in sub_dict.items():
            setattr(self, k, v)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def mean_flat(x):
    """Take the mean over all non-batch dimensions."""
    return th.mean(x, dim=list(range(1, len(x.size()))))


def log_state(state: dict[str, Any]) -> str:
    result = []
    sorted_state = dict(sorted(state.items()))
    for key, value in sorted_state.items():
        if "<object" in str(value) or "object at" in str(value):
            result.append(f"{key}: [{value.__class__.__name__}]")
        else:
            result.append(f"{key}: {value}")
    return "\n".join(result)


def time_shift(mu: float, sigma: float, t: th.Tensor):
    t = 1 - t
    t = math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
    t = 1 - t
    return t


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def expand_dims(v, dims):
    """Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dims`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]
