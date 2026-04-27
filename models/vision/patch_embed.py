# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from __future__ import annotations

import torch
import torch.nn as nn


class SimplePatchEmbedding(nn.Module):
    """Simple Conv2D-based patchifier with RMSNorm.

    Used by the "no encoder" Tuna variant (variant C). Takes raw RGB pixel
    values, splits them into non-overlapping patches via a strided Conv2d,
    and applies RMSNorm on the channel dimension.
    """

    def __init__(
        self,
        patch_size: int = 16,
        hidden_size: int = 1152,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.in_channels = in_channels

        # Conv2d patchify: (B, C, H, W) -> (B, hidden_size, H/p, W/p)
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        # RMSNorm on the channel/feature dim
        self.norm = nn.RMSNorm(hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: Tensor of shape (B, 3, H, W) of RGB pixel values.

        Returns:
            Tensor of shape (B, num_patches, hidden_size) where
            num_patches = (H // patch_size) * (W // patch_size).
        """
        # (B, C, H, W) -> (B, hidden_size, H/p, W/p)
        embeddings = self.patch_embedding(pixel_values)

        # (B, hidden_size, H/p, W/p) -> (B, num_patches, hidden_size)
        b, c, h, w = embeddings.shape
        embeddings = embeddings.reshape(b, c, h * w).transpose(1, 2)

        # RMSNorm on the channel dim
        embeddings = self.norm(embeddings)

        return embeddings
