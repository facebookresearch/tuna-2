# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""
Tokenization & sequence-formatting utilities for Tuna.

These utilities build the *unified* text-image token sequence consumed by the
Qwen2.5 backbone (and similar autoregressive decoders) used in Tuna.

Only depends on: ``torch`` and stdlib.

The "unified sequence" layout used by the model wrappers is:

T2I (text-to-image, generation):
    [bos] [text_tokens] [boi] [img_pad] * N [eoi] [eos]

MMU (multimodal understanding, image+text -> text):
    [bos] [boi] [img_pad] * N [eoi] [prompt_tokens] [response_tokens] [eos]

EDIT (image editing, raw_image + instruction -> target_image):
    [bos] [boi] [img_pad] * N [eoi] [text_tokens] [boi] [img_pad] * N [eoi] [eos]

The functions below produce ``(text_tokens, text_labels, modality_positions,
text_mask, image_mask)`` tuples where:

* ``text_tokens`` is the LongTensor input sequence padded to ``max_seq_len``.
* ``text_labels`` mirrors ``text_tokens`` but with ``-100`` (the standard
  ``nn.CrossEntropyLoss`` ignore index) at positions that should not contribute
  to the loss.
* ``modality_positions`` is a LongTensor of ``[start, length]`` rows for each
  image span, used by the model to slot image features into the text stream.
* ``text_mask`` is 1 at positions that contain real text tokens.
* ``image_mask`` is 1 at positions that contain image-pad tokens (i.e. where
  image features are spliced in).
"""

from __future__ import annotations

import copy

import torch


# Field-name constants (record-key conventions used by the JSONL datasets).
IMAGE_FIELD: str = "image"
INPUT_IMAGE: str = "image"


def remove_prefix(caption: str) -> str:
    """Strip a long list of common, low-information caption prefixes.

    Many machine-generated captions begin with redundant lead-ins like
    ``"The image shows ..."``. These add no information and only inflate the
    sequence length, so we delete them before tokenizing.
    """
    caption = (
        caption.replace("The image features ", "")
        .replace("The image presents ", "")
        .replace("The image you've sent is, ", "")
        .replace("In the center of the image, ", "")
        .replace("The image showcases ", "")
        .replace("The image is ", "")
        .replace("The image captures ", "")
        .replace("In the given image ", "")
        .replace("The image portrays ", "")
        .replace("In the image, ", "")
        .replace("In this image, we see ", "")
        .replace("The image depicts ", "")
        .replace("This is ", "")
        .replace("In this image, ", "")
        .replace("This image captures ", "")
        .replace("This image showcases ", "")
        .replace("This suggests ", "")
        .replace("In the photo, we see ", "")
        .replace("This is ", "")
        .replace("This image is ", "")
        .replace("In the photo, we have ", "")
        .replace("The photo features ", "")
        .replace("The photo depicts ", "")
        .replace("The photo appears to be ", "")
    )
    return caption


# Unified-sequence formatters ------------------------------------------------


def format_sequence_gen_qwen2_5(
    text_tokens: list[int],
    system_tokens: list[list[int]] | None,
    bos_id: int,
    eos_id: int,
    boi_id: int,
    eoi_id: int,
    pad_id: int,
    img_pad_id: int,
    num_image_tokens: int,
    max_seq_len: int,
    system_token_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Format a T2I (generation) sequence for Qwen2.5.

    For T2I we *do not* train a language-model loss over the text — the text
    is purely conditioning. The image tokens are also masked from the loss
    here; they are recovered downstream by the diffusion / flow loss head,
    which reads ``modality_positions`` to find the image span.
    """
    if system_token_len == 0:
        modality_positions = torch.tensor(
            [[len(text_tokens) + 1 + 1, num_image_tokens]]
        )
        text_labels = (
            [-100]
            + [-100] * len(text_tokens)
            + [-100]
            + [-100] * num_image_tokens
            + [-100]
            + [-100]
        )
        text_tokens = (
            [bos_id]
            + text_tokens
            + [boi_id]
            + [img_pad_id] * num_image_tokens
            + [eoi_id]
            + [eos_id]
        )
    else:
        assert system_tokens is not None
        modality_positions = torch.tensor(
            [[1 + system_token_len + len(text_tokens) + 1 + 1, num_image_tokens]]
        )
        text_labels = (
            [bos_id]
            + [-100] * len(system_tokens[0] + system_tokens[1] + text_tokens)
            + [eos_id]
            + [-100] * len(system_tokens[2])
            + [boi_id]
            + [-100] * num_image_tokens
            + [eoi_id]
            + [eos_id]
        )
        text_tokens = (
            [bos_id]
            + system_tokens[0]
            + system_tokens[1]
            + text_tokens
            + [eos_id]
            + system_tokens[2]
            + [boi_id]
            + [img_pad_id] * num_image_tokens
            + [eoi_id]
            + [eos_id]
        )

    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))
    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
    text_tokens_t = torch.tensor(text_tokens)
    text_labels_t = torch.tensor(text_labels)

    text_mask = torch.where(
        (text_tokens_t != img_pad_id) & (text_tokens_t != pad_id),
        torch.ones_like(text_tokens_t),
        torch.zeros_like(text_tokens_t),
    )
    image_mask = torch.where(
        text_tokens_t == img_pad_id,
        torch.ones_like(text_tokens_t),
        torch.zeros_like(text_tokens_t),
    )

    return text_tokens_t, text_labels_t, modality_positions, text_mask, image_mask


def format_sequence_gen_qwen2_5_edit(
    text_tokens: list[int],
    system_tokens: list[list[int]] | None,
    bos_id: int,
    eos_id: int,
    boi_id: int,
    eoi_id: int,
    pad_id: int,
    img_pad_id: int,
    num_image_tokens: int,
    max_seq_len: int,
    system_token_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Format an EDIT sequence: ``[bos][raw_img][instruction][target_img][eos]``.

    The ``image_mask`` is 1 only on the *second* image span (the target). The
    raw image is conditioning, not a loss target.
    """
    if system_token_len == 0:
        modality_positions = torch.tensor(
            [
                [2, num_image_tokens],
                [len(text_tokens) + 4 + num_image_tokens, num_image_tokens],
            ]
        )
        text_labels = (
            [-100]
            + [-100]
            + [-100] * num_image_tokens
            + [-100]
            + [-100] * len(text_tokens)
            + [-100]
            + [-100] * num_image_tokens
            + [-100]
            + [-100]
        )
        text_tokens = (
            [bos_id]
            + [boi_id]
            + [img_pad_id] * num_image_tokens
            + [eoi_id]
            + text_tokens
            + [boi_id]
            + [img_pad_id] * num_image_tokens
            + [eoi_id]
            + [eos_id]
        )
    else:
        assert system_tokens is not None
        modality_positions = torch.tensor(
            [[1 + system_token_len + len(text_tokens) + 1 + 1, num_image_tokens]]
        )
        text_labels = (
            [bos_id]
            + [-100] * len(system_tokens[0] + system_tokens[1] + text_tokens)
            + [eos_id]
            + [-100] * len(system_tokens[2])
            + [boi_id]
            + [-100] * num_image_tokens
            + [eoi_id]
            + [eos_id]
        )
        text_tokens = (
            [bos_id]
            + system_tokens[0]
            + system_tokens[1]
            + text_tokens
            + [eos_id]
            + system_tokens[2]
            + [boi_id]
            + [img_pad_id] * num_image_tokens
            + [eoi_id]
            + [eos_id]
        )

    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))
    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
    text_tokens_t = torch.tensor(text_tokens)
    text_labels_t = torch.tensor(text_labels)

    text_mask = torch.where(
        (text_tokens_t != img_pad_id) & (text_tokens_t != pad_id),
        torch.ones_like(text_tokens_t),
        torch.zeros_like(text_tokens_t),
    )

    # image_mask: only mark the *second* image span (the target image).
    image_mask = torch.zeros_like(text_tokens_t)
    boi_positions = (text_tokens_t == boi_id).nonzero(as_tuple=True)[0]
    if len(boi_positions) >= 2:
        second_boi_pos = boi_positions[1].item()
        eoi_positions_after_second_boi = (
            text_tokens_t[second_boi_pos:] == eoi_id
        ).nonzero(as_tuple=True)[0]
        if len(eoi_positions_after_second_boi) > 0:
            second_eoi_pos = (
                second_boi_pos + eoi_positions_after_second_boi[0].item()
            )
            mask_region = text_tokens_t[second_boi_pos : second_eoi_pos + 1]
            mask_indices = (
                mask_region == img_pad_id
            ).nonzero(as_tuple=True)[0] + second_boi_pos
            image_mask[mask_indices] = 1

    return text_tokens_t, text_labels_t, modality_positions, text_mask, image_mask


def format_sequence_und(
    text_tokens: list[int],
    bos_id: int,
    eos_id: int,
    boi_id: int,
    eoi_id: int,
    pad_id: int,
    img_pad_id: int,
    num_image_tokens: int,
    max_seq_len: int,
    prompt_tokens: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Format an MMU (image-conditioned text generation) sequence.

    Layout: ``[bos][boi][img_pad]*N[eoi][prompt][response][eos]`` where the
    response is the loss target (everything before is masked to ``-100``).
    """
    if prompt_tokens is None:
        prompt_tokens = []
    modality_positions = torch.tensor([[1 + 1, num_image_tokens]])

    text_labels = (
        [bos_id]
        + [boi_id]
        + [-100] * num_image_tokens
        + [eoi_id]
        + [-100] * len(prompt_tokens)
        + text_tokens
        + [eos_id]
    )
    text_tokens = (
        [bos_id]
        + [boi_id]
        + [img_pad_id] * num_image_tokens
        + [eoi_id]
        + prompt_tokens
        + text_tokens
        + [eos_id]
    )

    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))
    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
    text_tokens_t = torch.tensor(text_tokens)
    text_labels_t = torch.tensor(text_labels)

    text_mask = torch.where(
        (text_tokens_t != img_pad_id) & (text_tokens_t != pad_id),
        torch.ones_like(text_tokens_t),
        torch.zeros_like(text_tokens_t),
    )
    image_mask = torch.where(
        text_tokens_t == img_pad_id,
        torch.ones_like(text_tokens_t),
        torch.zeros_like(text_tokens_t),
    )

    return text_tokens_t, text_labels_t, modality_positions, text_mask, image_mask


# Chat-template formatters ---------------------------------------------------


def format_conversation_und(
    text_tokens: list[int],
    eos_id: int,
    boi_id: int,
    eoi_id: int,
    pad_id: int,
    img_id: int,
    img_pad_id: int,
    num_image_tokens: int,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace the single ``img_id`` placeholder in a chat-templated token list
    with a proper ``[boi][img_pad]*N[eoi]`` block, then mark the
    user/assistant turns appropriately for loss masking (only assistant turns
    contribute to the LM loss).
    """
    img_idx = text_tokens.index(img_id)
    text_tokens = (
        text_tokens[:img_idx]
        + [boi_id]
        + [img_pad_id] * num_image_tokens
        + [eoi_id]
        + text_tokens[img_idx + 1 :]
    )
    text_labels = copy.deepcopy(text_tokens)

    all_sep_ids = [i for i, num in enumerate(text_tokens) if num == eos_id]
    sys_sep_idx = all_sep_ids[0]
    text_labels[: sys_sep_idx + 1] = [-100] * (sys_sep_idx + 1)
    curr_sep_idx = all_sep_ids[0]
    i = 1
    for i in range(1, len(all_sep_ids)):
        prev_sep_idx = all_sep_ids[i - 1]
        curr_sep_idx = all_sep_ids[i]
        if i % 2 == 1:
            text_labels[prev_sep_idx + 1 : curr_sep_idx + 1] = [-100] * (
                curr_sep_idx - prev_sep_idx
            )
    if curr_sep_idx != len(text_tokens) - 2 and i % 2 == 1:
        text_labels[curr_sep_idx + 1 :] = [-100] * len(
            text_labels[curr_sep_idx + 1 :]
        )

    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))

    modality_positions = torch.tensor([[img_idx + 1, num_image_tokens]])

    text_tokens_t = torch.tensor(text_tokens)
    text_labels_t = torch.tensor(text_labels)

    text_mask = torch.where(
        (text_tokens_t != img_pad_id) & (text_tokens_t != pad_id),
        torch.ones_like(text_tokens_t),
        torch.zeros_like(text_tokens_t),
    )
    image_mask = torch.where(
        text_tokens_t == img_pad_id,
        torch.ones_like(text_tokens_t),
        torch.zeros_like(text_tokens_t),
    )

    return text_tokens_t, text_labels_t, modality_positions, text_mask, image_mask


def format_conversation_und_text(
    text_tokens: list[int],
    eos_id: int,
    boi_id: int,
    eoi_id: int,
    pad_id: int,
    img_id: int,
    img_pad_id: int,
    num_image_tokens: int,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Text-only conversation (no image). Same loss-masking pattern as
    ``format_conversation_und`` but with no image-span splice.
    """
    text_labels = copy.deepcopy(text_tokens)

    all_sep_ids = [i for i, num in enumerate(text_tokens) if num == eos_id]
    sys_sep_idx = all_sep_ids[0]
    text_labels[: sys_sep_idx + 1] = [-100] * (sys_sep_idx + 1)
    curr_sep_idx = all_sep_ids[0]
    i = 1
    for i in range(1, len(all_sep_ids)):
        prev_sep_idx = all_sep_ids[i - 1]
        curr_sep_idx = all_sep_ids[i]
        if i % 2 == 1:
            text_labels[prev_sep_idx + 1 : curr_sep_idx + 1] = [-100] * (
                curr_sep_idx - prev_sep_idx
            )
    if curr_sep_idx != len(text_tokens) - 2 and i % 2 == 1:
        text_labels[curr_sep_idx + 1 :] = [-100] * len(
            text_labels[curr_sep_idx + 1 :]
        )

    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))

    modality_positions = torch.tensor([[-1, -1]])

    text_tokens_t = torch.tensor(text_tokens)
    text_labels_t = torch.tensor(text_labels)

    text_mask = torch.where(
        (text_tokens_t != img_pad_id) & (text_tokens_t != pad_id),
        torch.ones_like(text_tokens_t),
        torch.zeros_like(text_tokens_t),
    )
    image_mask = torch.where(
        text_tokens_t == img_pad_id,
        torch.ones_like(text_tokens_t),
        torch.zeros_like(text_tokens_t),
    )

    return text_tokens_t, text_labels_t, modality_positions, text_mask, image_mask


def format_conversation_gen(
    text_tokens: list[int],
    eos_id: int,
    boi_id: int,
    eoi_id: int,
    pad_id: int,
    img_id: int,
    img_pad_id: int,
    num_image_tokens: int,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Conversational T2I: the assistant turn ends in an image. Loss covers
    the assistant text (the prompt rewrite, etc.) plus the trailing eos.
    """
    img_idx = text_tokens.index(img_id)
    text_tokens = (
        text_tokens[:img_idx]
        + [boi_id]
        + [img_pad_id] * num_image_tokens
        + [eoi_id]
        + text_tokens[img_idx + 1 :]
    )
    text_labels = [-100] * len(text_tokens)
    all_sep_ids = [i for i, num in enumerate(text_tokens) if num == eos_id]
    assistant_sep_idx = all_sep_ids[-2]
    final_sep_idx = all_sep_ids[-1]
    text_labels[assistant_sep_idx + 1 : img_idx + 1] = text_tokens[
        assistant_sep_idx + 1 : img_idx + 1
    ]
    text_labels[final_sep_idx:] = text_tokens[final_sep_idx:]

    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))

    modality_positions = torch.tensor([[img_idx + 1, num_image_tokens]])

    text_tokens_t = torch.tensor(text_tokens)
    text_labels_t = torch.tensor(text_labels)

    text_mask = torch.where(
        (text_tokens_t != img_pad_id) & (text_tokens_t != pad_id),
        torch.ones_like(text_tokens_t),
        torch.zeros_like(text_tokens_t),
    )
    image_mask = torch.where(
        text_tokens_t == img_pad_id,
        torch.ones_like(text_tokens_t),
        torch.zeros_like(text_tokens_t),
    )

    return text_tokens_t, text_labels_t, modality_positions, text_mask, image_mask


# Caption prompt templates ---------------------------------------------------

captioning_templates: dict[str, list[str]] = {
    "user_short": [
        "Describe what is happening in this {}.",
        "What is shown in the {}?",
        "Explain the content of this {}.",
        "What can be seen in the {}?",
        "Give a brief description of the {}.",
        "Briefly describe the {}.",
        "Provide a short description of the {}.",
        "Describe the contents in the {}.",
        "What can you observe in the {}?",
        "Explain the scenario in the {}.",
        "What is the main focus of the {}?",
        "Outline what happens in the {}.",
        "What is the highlight of the {}?",
        "Describe the activity in the {}.",
        "What is the {} showing?",
        "Give an overview of the {} content.",
        "What is captured in the {}?",
    ],
    "user_long": [
        "Describe the {} in detail.",
        "Give a detailed description of the {}.",
        "Describe the {} thoroughly.",
        "Tell me in detail what is shown in the {}.",
        "Provide a full account of the {}.",
        "Provide a detailed description of the {}.",
        "Give a comprehensive description of the {}.",
        "What does the {} show? Describe it in detail.",
        "Explain the scene in the {} with as much detail as possible.",
    ],
    "assistant": [
        "The {} shows {}.",
        "In the {}, {}.",
        "The content of the {} is {}.",
        "The {} illustrates {}.",
        "The scene in the {} describes {}.",
        "The {} depicts {}.",
        "According to the {}, {}.",
        "The {} portrays {}.",
        "The {} demonstrates {}.",
        "The {} reveals {}.",
        "The {} conveys {}.",
        "The {} captures {}.",
        "The {} narrates {}.",
        "As shown in the {}, {}.",
        "The {} tells the story of {}.",
        "The {} represents {}.",
        "From the {}, we see {}.",
        "The {} describes {}.",
        "The {} presents {}.",
        "The {} features {}.",
    ],
}
