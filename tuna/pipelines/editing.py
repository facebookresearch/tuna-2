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

"""Task-dispatch wrapper for image editing (image + instruction -> image)."""

from __future__ import annotations

from typing import List, Union

import torch
from PIL import Image


def edit(
    pipeline,
    images: torch.Tensor,
    instructions: Union[str, List[str]],
    **kwargs,
) -> List[Image.Image]:
    """
    Edit images according to text instructions.

    Dispatches to the underlying ``pipeline.t2i_edit`` method, which is
    implemented by all three Tuna pipeline variants.

    Args:
        pipeline: A Tuna pipeline instance with a ``t2i_edit`` method.
        images: Source-image tensor of shape (B, N, C, H, W) where N is the
            number of conditioning images per example (typically 1). The model
            uses these as the "before" image for the edit.
        instructions: A single edit instruction or a list of instructions, one
            per example in the batch.
        **kwargs: Additional keyword arguments forwarded to ``pipeline.t2i_edit`` —
            common ones include ``num_inference_steps``, ``guidance_scale``,
            ``sampling_method``, ``time_shifting_factor``.

    Returns:
        A list of edited PIL images.
    """
    if isinstance(instructions, str):
        instructions = [instructions]
    return pipeline.t2i_edit(prompts=instructions, pixel_values=images, **kwargs)
