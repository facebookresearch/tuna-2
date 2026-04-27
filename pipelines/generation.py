# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# coding=utf-8
# pyre-unsafe

"""Task-dispatch wrapper for text-to-image generation."""

from __future__ import annotations

from typing import List, Union

from PIL import Image


def generate(
    pipeline,
    prompts: Union[str, List[str]],
    **kwargs,
) -> List[Image.Image]:
    """
    Generate images from text prompts.

    Dispatches to the underlying ``pipeline.t2i`` method, which is implemented
    by all three Tuna pipeline variants (``TunaPipeline``, ``Tuna2RPixelPipeline``,
    ``Tuna2PixelPipeline``).

    Args:
        pipeline: A Tuna pipeline instance with a ``t2i`` method.
        prompts: A single text prompt or a list of prompts.
        **kwargs: Additional keyword arguments forwarded to ``pipeline.t2i`` —
            common ones include ``num_inference_steps``, ``guidance_scale``,
            ``sampling_method``, ``time_shifting_factor``, ``noise_level``.

    Returns:
        A list of generated PIL images, one per prompt.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    return pipeline.t2i(prompts=prompts, **kwargs)
