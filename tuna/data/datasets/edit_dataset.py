# pyre-unsafe
# (c) Meta Platforms, Inc. and affiliates. Apache-2.0.
"""
Map-style image-editing dataset for Tuna.

JSONL format - one object per line:

    {"raw_image": "imgs/000.jpg",
     "out_image": "imgs/000_edit.jpg",
     "instruction": "make it sunset"}

Each ``__getitem__`` returns the unified-sequence layout used by the editing
pipeline:

    [text_prompt][raw_image][target_image]

Concretely, the dict has:

* ``images``: ``list[Tensor]`` of length 2: ``[raw, target]`` each ``(3, H, W)``
  in ``[-1, 1]``.
* ``text_tokens`` / ``text_labels`` / ``text_masks`` / ``image_masks`` /
  ``modality_positions``: produced by
  ``format_sequence_gen_qwen2_5_edit``. ``image_masks`` is set only on the
  *target* image span (the raw image is conditioning).
* ``data_type``: ``"edit_interleaved"``.
* ``sentence``: the raw instruction string.
* (Optional) SigLIP2 inputs for the raw image, when ``siglip_processor_id``
  is set.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from tuna.data.tokenize_utils import format_sequence_gen_qwen2_5_edit
from tuna.data.transforms import build_image_transform, build_siglip_transform


logger = logging.getLogger(__name__)


_BOI_TOKEN = "<|img_start|>"
_EOI_TOKEN = "<|img_end|>"
_IMG_PAD_TOKEN = "<|img_pad|>"


def _resolve_special(tokenizer: Any, token: str, fallback_attr: str | None) -> int:
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid is None or tid == tokenizer.unk_token_id:
        if fallback_attr is not None:
            tid = getattr(tokenizer, fallback_attr, None)
    if tid is None:
        raise ValueError(
            f"Could not resolve special token {token!r} on the supplied tokenizer; "
            "make sure it has been added before constructing the dataset."
        )
    return int(tid)


class EditDataset(Dataset):
    """Map-style image-editing dataset over a local JSONL manifest.

    Args:
        jsonl_path: Path to a UTF-8 JSONL file with one record per line.
        image_root: Directory used to resolve relative image paths.
        tokenizer: A HuggingFace ``AutoTokenizer`` already extended with the
            Tuna special tokens.
        image_size: Target ``(H, W)`` for both raw and target images.
        max_text_length: Maximum length of the unified token sequence.
        raw_image_field / out_image_field / instruction_field: JSON keys.
        siglip_processor_id: If set, also emit SigLIP2 inputs for the raw
            image (the conditioning side).
        center_crop: Passed through to ``build_image_transform``.
        num_image_tokens: How many ``<|img_pad|>`` tokens per image. Defaults
            to ``(H/16) * (W/16)``.
    """

    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        tokenizer,
        image_size: int | tuple[int, int] = 256,
        max_text_length: int = 256,
        raw_image_field: str = "raw_image",
        out_image_field: str = "out_image",
        instruction_field: str = "instruction",
        siglip_processor_id: str | None = None,
        center_crop: bool = True,
        num_image_tokens: int | None = None,
        clip_image_size: int | tuple[int, int] = 384,
        tuna_token_ids: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.jsonl_path = jsonl_path
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_size = (
            (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        )
        self.max_text_length = max_text_length
        self.raw_image_field = raw_image_field
        self.out_image_field = out_image_field
        self.instruction_field = instruction_field

        self.records: list[dict[str, Any]] = self._load_jsonl(jsonl_path)
        self.image_transform = build_image_transform(self.image_size, center_crop)

        if num_image_tokens is None:
            patch = 16
            num_image_tokens = (self.image_size[0] // patch) * (
                self.image_size[1] // patch
            )
        self.num_image_tokens = num_image_tokens

        if tuna_token_ids is not None:
            self.bos_id = tuna_token_ids["bos_id"]
            self.eos_id = tuna_token_ids["eos_id"]
            self.pad_id = tokenizer.pad_token_id or self.eos_id
            self.boi_id = tuna_token_ids["boi_id"]
            self.eoi_id = tuna_token_ids["eoi_id"]
            self.img_pad_id = tuna_token_ids["img_pad_id"]
        else:
            self.bos_id = _resolve_special(tokenizer, tokenizer.bos_token or "<|bos|>", "bos_token_id")
            self.eos_id = _resolve_special(tokenizer, tokenizer.eos_token or "<|eos|>", "eos_token_id")
            self.pad_id = _resolve_special(
                tokenizer,
                tokenizer.pad_token or tokenizer.eos_token or "<|pad|>",
                "pad_token_id",
            )
            self.boi_id = _resolve_special(tokenizer, _BOI_TOKEN, None)
            self.eoi_id = _resolve_special(tokenizer, _EOI_TOKEN, None)
            self.img_pad_id = _resolve_special(tokenizer, _IMG_PAD_TOKEN, None)

        self.siglip_processor_id = siglip_processor_id
        self.siglip_transform = (
            build_siglip_transform(siglip_processor_id)
            if siglip_processor_id is not None
            else None
        )
        # Fallback `images_clip` transform for the no-SigLIP variant.
        self.clip_image_size = (
            (clip_image_size, clip_image_size)
            if isinstance(clip_image_size, int)
            else tuple(clip_image_size)
        )
        self.clip_image_transform = build_image_transform(self.clip_image_size, center_crop)

        logger.info(
            f"EditDataset: loaded {len(self.records)} records from {jsonl_path} "
            f"(image_size={self.image_size}, num_image_tokens={self.num_image_tokens}, "
            f"siglip={siglip_processor_id})"
        )

    @staticmethod
    def _load_jsonl(path: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"{path}:{ln}: skipping malformed JSON: {e}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, rel_or_abs: str) -> str:
        if os.path.isabs(rel_or_abs):
            return rel_or_abs
        return os.path.join(self.image_root, rel_or_abs)

    def _load_image(self, rel_or_abs: str) -> Image.Image:
        path = self._resolve_image_path(rel_or_abs)
        with Image.open(path) as img:
            img.load()
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img

    def _tokenize_instruction(self, instruction: str) -> list[int]:
        # Reserve room for: bos + 2 * (boi + eoi) + 2 * num_image_tokens + eos
        reserve = 1 + 2 * (1 + 1) + 2 * self.num_image_tokens + 1
        max_text = max(0, self.max_text_length - reserve)
        ids = self.tokenizer(
            instruction,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, max_text),
        )["input_ids"]
        return list(ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]

        raw_pil = self._load_image(record[self.raw_image_field])
        out_pil = self._load_image(record[self.out_image_field])
        raw_tensor = self.image_transform(raw_pil)
        out_tensor = self.image_transform(out_pil)

        instruction = str(record.get(self.instruction_field, ""))
        text_token_ids = self._tokenize_instruction(instruction)

        tt, tl, mp, tm, im = format_sequence_gen_qwen2_5_edit(
            text_tokens=text_token_ids,
            system_tokens=None,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            boi_id=self.boi_id,
            eoi_id=self.eoi_id,
            pad_id=self.pad_id,
            img_pad_id=self.img_pad_id,
            num_image_tokens=self.num_image_tokens,
            max_seq_len=self.max_text_length,
            system_token_len=0,
        )

        out: dict[str, Any] = {
            "images": torch.stack([raw_tensor, out_tensor]),
            "text_tokens": tt.long(),
            "text_labels": tl.long(),
            "text_masks": tm.bool(),
            "image_masks": im.bool(),
            "modality_positions": mp.long(),
            "data_type": "edit_interleaved",
            "sentence": instruction,
        }

        # `images_clip` is the low-res / SigLIP-resized version of the *raw*
        # (source) image, used for clean-image-embedding conditioning across
        # all three Tuna variants. See TIDataset for the full rationale.
        if self.siglip_transform is not None:
            sig = self.siglip_transform(raw_pil)
            out["images_clip"] = sig["pixel_values"]
            out["siglip_pixel_attention_mask"] = sig["pixel_attention_mask"]
            out["siglip_spatial_shapes"] = sig["spatial_shapes"]
        else:
            out["images_clip"] = self.clip_image_transform(raw_pil)

        return out
