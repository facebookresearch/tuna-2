# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers shared across the three Tuna model variants.

Pure module-level utilities lifted out of `tuna.py`, `tuna_2_pixel.py`, and
`tuna_2r_pixel.py`. Kept here so the variant modules don't drift out of sync.
"""

from __future__ import annotations

# pyre-unsafe

from typing import Union

import torch
from einops import rearrange
from torch import Tensor

from tuna.models.vision.patch_embed import SimplePatchEmbedding


TorchDevice = Union[str, torch.device]


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("n,d->nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "n d (i j) -> n d i j", i=2, j=2)
    return out.float()


def build_rope(latent_shape, patch_size, attention_head_dim):
    dim_t = attention_head_dim // 4
    dim_h = attention_head_dim // 8 * 3
    dim_w = attention_head_dim // 8 * 3
    assert dim_t + dim_h + dim_w == attention_head_dim, (
        f"{dim_t + dim_h + dim_w} != {attention_head_dim}"
    )

    latent_t = latent_shape[0]
    latent_h = latent_shape[1] // patch_size
    latent_w = latent_shape[2] // patch_size
    visual_ids = torch.zeros(latent_t, latent_h, latent_w, 3)
    visual_ids[..., 0] = visual_ids[..., 0] + torch.arange(latent_t)[:, None, None]
    visual_ids[..., 1] = visual_ids[..., 1] + torch.arange(latent_h)[None, :, None]
    visual_ids[..., 2] = visual_ids[..., 2] + torch.arange(latent_w)[None, None, :]
    visual_ids = rearrange(visual_ids, "t h w c -> (t h w) c")

    rope_3d = torch.cat(
        [
            rope(visual_ids[..., 0], dim_t, 10_000),
            rope(visual_ids[..., 1], dim_h, 10_000),
            rope(visual_ids[..., 2], dim_w, 10_000),
        ],
        dim=-3,
    )
    return rope_3d


def _patchify_5d(
    pixel_values: torch.Tensor,
    patch_embed: SimplePatchEmbedding,
    reshape_frame_to_batch_dim: bool,
) -> torch.Tensor:
    """Apply a 4D :class:`SimplePatchEmbedding` to a 5D ``[B, C, T, H, W]`` tensor.

    Either folds frames into the batch dim or processes them sequentially and
    concatenates along the patch dim. The OSS SimplePatchEmbedding is image-
    only (4D), so this helper bridges to the original 5D interface.
    """
    b, c, T, h, w = pixel_values.shape
    if reshape_frame_to_batch_dim:
        pixel_values = rearrange(pixel_values, "b c t h w -> (b t) c h w")
        return patch_embed(pixel_values)
    # Process each frame separately and concatenate along the patch dim
    frame_embeddings = []
    for t_idx in range(T):
        frame = pixel_values[:, :, t_idx, :, :]
        frame_embeddings.append(patch_embed(frame))
    return torch.cat(frame_embeddings, dim=1)
