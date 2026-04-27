# pyre-unsafe
# (c) Meta Platforms, Inc. and affiliates. Apache-2.0.
"""
Multi-resolution iterator wrapper for Tuna training.

Wraps a DataLoader and, for each batch, randomly picks a resolution from a
set of buckets. All images in the batch are resized to the chosen resolution.
This matches the showme ``multiresolution_iterator_wrapper`` behavior.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)

DEFAULT_RESOLUTION_BUCKETS = [
    [512, 512],
    [448, 576],
    [576, 448],
    [384, 672],
    [672, 384],
]


class MultiResolutionIterator:
    """Wraps a DataLoader and resizes image tensors per-batch to a random resolution.

    Args:
        dataloader: The underlying DataLoader to wrap.
        resolution_buckets: List of ``[H, W]`` resolution options.
        image_keys: Batch dict keys containing image tensors to resize.
    """

    def __init__(
        self,
        dataloader,
        resolution_buckets: list[list[int]] | None = None,
        image_keys: tuple[str, ...] = ("images", "images_clip"),
    ) -> None:
        self.dataloader = dataloader
        self.resolution_buckets = resolution_buckets or DEFAULT_RESOLUTION_BUCKETS
        self.image_keys = image_keys

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for batch in self.dataloader:
            # Pick a random resolution for this batch.
            target_h, target_w = random.choice(self.resolution_buckets)

            for key in self.image_keys:
                if key not in batch or batch[key] is None:
                    continue
                tensor = batch[key]
                if tensor.dim() < 3:
                    continue
                # Get current spatial dims (last 2 dims).
                cur_h, cur_w = tensor.shape[-2], tensor.shape[-1]
                if cur_h == target_h and cur_w == target_w:
                    continue
                # Resize: bicubic for images, handles [B, C, H, W] or [B, C, T, H, W].
                orig_shape = tensor.shape
                if tensor.dim() == 5:
                    b, c, t, h, w = tensor.shape
                    tensor = tensor.reshape(b * t, c, h, w)
                tensor = F.interpolate(
                    tensor.float(), size=(target_h, target_w),
                    mode="bicubic", align_corners=False,
                ).to(batch[key].dtype)
                if len(orig_shape) == 5:
                    tensor = tensor.reshape(b, c, t, target_h, target_w)
                batch[key] = tensor

            yield batch

    def __len__(self) -> int:
        return len(self.dataloader)
