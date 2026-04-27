# coding=utf-8
# (c) Meta Platforms, Inc. and affiliates. Apache-2.0.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""Task-dispatch wrapper for image reconstruction.

Reconstruction is text-to-image generation seeded with the source pixels and
an empty prompt. Each pipeline's ``t2i`` method honors ``reca_mode=True`` by
using ``pixel_values`` as the noise initialization.
"""

from __future__ import annotations

from typing import List

import torch
from PIL import Image


def reconstruct(
    pipeline,
    images: torch.Tensor,
    **kwargs,
) -> List[Image.Image]:
    """
    Reconstruct images by feeding them back through the model with an empty
    prompt and ``reca_mode=True``.

    This dispatches to ``pipeline.t2i(prompts=[""] * B, pixel_values=images,
    reca_mode=True, ...)``. See ``tuna/inference.py`` (the original
    reference: ``reca_mode = True`` is set on the inference object and the
    pipeline routes ``pixel_values`` through to seed the noise).

    Args:
        pipeline: A Tuna pipeline instance with a ``t2i`` method that accepts
            a ``reca_mode`` flag.
        images: Source-image tensor (B, C, H, W) or (B, C, T, H, W). One PIL
            image will be returned per example along the batch axis.
        **kwargs: Additional keyword arguments forwarded to ``pipeline.t2i`` —
            common ones include ``num_inference_steps``, ``guidance_scale``,
            ``sampling_method``, ``time_shifting_factor``.

    Returns:
        A list of reconstructed PIL images.
    """
    batch_size = images.shape[0]
    return pipeline.t2i(
        prompts=[""] * batch_size,
        pixel_values=images,
        reca_mode=True,
        **kwargs,
    )
