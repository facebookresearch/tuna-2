# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# coding=utf-8
# pyre-unsafe

"""Task-dispatch wrapper for multimodal understanding (image + prompt -> text)."""

from __future__ import annotations

from typing import List, Optional, Union

import torch


def understand(
    pipeline,
    images: Optional[Union[torch.Tensor, List[torch.Tensor]]],
    prompts: Union[str, List[str]],
    **kwargs,
) -> List[str]:
    """
    Generate text answers for image+prompt pairs.

    Dispatches to the underlying ``pipeline.mmu`` method, which currently
    expects a single ``pixel_values`` tensor and a single ``prompt`` string.
    This wrapper iterates over batched inputs and concatenates results.

    Args:
        pipeline: A Tuna pipeline instance with an ``mmu`` method.
        images: A single pixel-value tensor (B, C, H, W) or a list of such
            tensors, one per prompt. Pass ``None`` for text-only understanding.
        prompts: A single prompt or a list of prompts.
        **kwargs: Additional keyword arguments forwarded to ``pipeline.mmu`` —
            common ones include ``do_sample``, ``temperature``, ``top_k``,
            ``top_p``, ``max_new_tokens``, ``height``, ``width``.

    Returns:
        A list of generated text answers, one per prompt.
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    # Normalize images to a parallel list aligned with prompts.
    if images is None:
        images_list: List[Optional[torch.Tensor]] = [None] * len(prompts)
    elif isinstance(images, torch.Tensor):
        # Treat a (B, ...) tensor as a stack of per-prompt examples.
        if images.dim() >= 4 and images.shape[0] == len(prompts):
            images_list = [images[i : i + 1] for i in range(len(prompts))]
        else:
            images_list = [images for _ in prompts]
    else:
        images_list = list(images)

    answers: List[str] = []
    for img, prompt in zip(images_list, prompts):
        answers.append(pipeline.mmu(pixel_values=img, prompt=prompt, **kwargs))
    return answers
