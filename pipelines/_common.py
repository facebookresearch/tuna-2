"""Helpers shared across the three Tuna pipeline variants.

Lifted from `tuna_pipeline.py`, `tuna_2_pixel_pipeline.py`, and
`tuna_2r_pixel_pipeline.py` so the variant-specific files don't drift.
"""

# (c) Meta Platforms, Inc. and affiliates. Apache-2.0.

from __future__ import annotations

# pyre-unsafe

import numpy as np
import torch


def denorm(images):
    """Denormalize images from [-1, 1] to [0, 255] and convert to numpy.

    Args:
        images: Tensor of shape (B, C, H, W) with values in [-1, 1]

    Returns:
        Numpy array of shape (B, H, W, C) with values in [0, 255]
    """
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return images


def prepare_gen_input_chat(
    prompts,
    text_tokenizer,
    num_image_tokens,
    bos_id,
    eos_id,
    boi_id,
    eoi_id,
    pad_id,
    img_pad_id,
    max_text_len,
    max_seq_len,
    device,
):
    batch_text_tokens = []
    batch_modality_positions = []
    batch_text_tokens_null = []
    batch_modality_positions_null = []
    for prompt in prompts:
        text_tokens = text_tokenizer(prompt, add_special_tokens=False)["input_ids"][
            :(max_text_len)
        ]
        prompt = text_tokenizer.decode(text_tokens)

        conversation = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "<image>"},
        ]
        conv_prompt = text_tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        text_tokens = text_tokenizer(conv_prompt, add_special_tokens=False).input_ids
        img_id = text_tokenizer("<image>", add_special_tokens=False).input_ids[0]
        img_idx = text_tokens.index(img_id)

        modality_positions = torch.tensor([img_idx + 1, num_image_tokens]).unsqueeze(0)
        text_tokens = (
            text_tokens[:img_idx]
            + [boi_id]
            + [img_pad_id] * num_image_tokens
            + [eoi_id]
            + text_tokens[img_idx + 1 :]
        )
        text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
        batch_text_tokens.append(torch.tensor(text_tokens))
        batch_modality_positions.append(modality_positions)
        text_tokens_null = text_tokenizer(
            "ugly, distorted, deformed, disfigured, low quality, worst quality, blurry, noisy, pixelated, overexposed, underexposed, bad anatomy, bad proportions, extra limbs, missing limbs, fused fingers, extra fingers, poorly drawn hands, poorly drawn face, asymmetrical eyes, messed up face, disfigured mouth, unnatural lighting, strange reflections, artifact, jpeg artifacts, watermark, text, subtitle, logo, frame border, over-saturated, color bleeding, unrealistic colors, low-res, low resolution, bad composition, messy background, cluttered, cropped head, cut-off body, unnatural pose, broken limbs, wrong perspective, out of frame, duplicated parts",
            add_special_tokens=False,
        )["input_ids"][:(max_text_len)]
        prompt_null = text_tokenizer.decode(text_tokens_null)
        conversation_null = conversation.copy()
        conversation_null[1]["content"] = prompt_null
        conv_prompt_null = text_tokenizer.apply_chat_template(
            conversation_null, tokenize=False, add_generation_prompt=False
        )
        text_tokens_null = text_tokenizer(
            conv_prompt_null, add_special_tokens=False
        ).input_ids

        img_idx = text_tokens_null.index(img_id)
        modality_positions_null = torch.tensor(
            [img_idx + 1, num_image_tokens]
        ).unsqueeze(0)
        text_tokens_null = (
            text_tokens_null[:img_idx]
            + [boi_id]
            + [img_pad_id] * num_image_tokens
            + [eoi_id]
            + text_tokens_null[img_idx + 1 :]
        )
        text_tokens_null = text_tokens_null + [pad_id] * (
            max_seq_len - len(text_tokens_null)
        )

        batch_text_tokens_null.append(torch.tensor(text_tokens_null))
        batch_modality_positions_null.append(modality_positions_null)

    batch_text_tokens = torch.stack(batch_text_tokens, dim=0).to(device)
    batch_modality_positions = torch.stack(batch_modality_positions, dim=0).to(device)

    batch_text_tokens_null = torch.stack(batch_text_tokens_null, dim=0).to(device)
    batch_modality_positions_null = torch.stack(
        batch_modality_positions_null, dim=0
    ).to(device)

    return (
        batch_text_tokens,
        batch_text_tokens_null,
        batch_modality_positions,
        batch_modality_positions_null,
    )


# Path to LLM name mapping
path_to_llm_name = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen2_5",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2_5",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2_5",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen2_5",
}
