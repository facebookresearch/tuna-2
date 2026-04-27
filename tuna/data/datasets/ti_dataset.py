# pyre-unsafe
# (c) Meta Platforms, Inc. and affiliates. Apache-2.0.
"""
Map-style text-image dataset for Tuna training.

This is the OSS replacement for the Hive-backed enriched dataset used in the
internal tuna codebase. It reads from a local **JSONL** file, where each
line is a JSON object describing one sample. Two record shapes are supported:

* Caption form (T2I or single-turn captioning):

      {"image": "imgs/000.jpg", "caption": "a photo of a cat"}

* Conversation form (multi-turn understanding / chat):

      {"image": "imgs/000.jpg",
       "conversations": [
           {"from": "human", "value": "What's in this image?"},
           {"from": "gpt",   "value": "A cat."}
       ]}

Each ``__getitem__`` returns a dict keyed by the names the Tuna model
wrappers expect:

* ``images``: ``(3, H, W)`` float tensor in ``[-1, 1]``
* ``text_tokens`` / ``text_labels`` / ``text_masks`` / ``image_masks``: the
  unified text-image token sequence (see ``tuna.data.tokenize_utils``)
* ``data_type``: ``"t2i"`` or ``"mmu"``
* ``sentence``: the raw text (handy for debug logs)

In addition, every sample emits ``images_clip`` — the SigLIP-resized (or
plain low-res) companion image read by all three Tuna variants. When
``siglip_processor_id`` is set, the SigLIP2 spatial-shape / attention-mask
companions (``siglip_spatial_shapes`` / ``siglip_pixel_attention_mask``)
are emitted alongside.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from tuna.data.tokenize_utils import (
    captioning_templates,
    format_sequence_gen_qwen2_5,
    format_sequence_und,
)
from tuna.data.transforms import build_image_transform, build_siglip_transform


logger = logging.getLogger(__name__)


# Token names we look up on the tokenizer's vocabulary. These are the
# standard "special" sentinels added when fine-tuning Qwen2.5 for unified
# text+image generation. They must already exist on the supplied tokenizer
# (the model wrappers do this at construction time).
_BOI_TOKEN = "<|img_start|>"
_EOI_TOKEN = "<|img_end|>"
_IMG_PAD_TOKEN = "<|img_pad|>"


def _resolve_special(tokenizer: Any, token: str, fallback_attr: str | None) -> int:
    """Resolve a special token id off the tokenizer.

    Tries ``convert_tokens_to_ids(token)`` first; if that comes back with the
    unknown-token id, falls back to ``getattr(tokenizer, fallback_attr)`` (used
    for ids like ``bos_token_id`` / ``eos_token_id`` / ``pad_token_id`` that
    the tokenizer exposes directly).
    """
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


class TIDataset(Dataset):
    """Map-style text-image dataset over a local JSONL manifest.

    Args:
        jsonl_path: Path to a UTF-8 JSONL file with one record per line.
        image_root: Directory used to resolve relative ``image`` paths.
            Absolute paths in the JSONL are kept as-is.
        tokenizer: A HuggingFace ``AutoTokenizer`` (or anything with the same
            interface) that already has the Tuna special tokens registered.
        image_size: Target ``(H, W)`` for the WAN-VAE / patch input.
        data_type: ``"t2i"`` for text-to-image, ``"mmu"`` for image
            understanding.
        max_text_length: Maximum length of the *unified* token sequence,
            including image-pad tokens.
        image_field / text_field / conversations_field: JSON keys to pull
            from each line.
        center_crop: Passed through to ``build_image_transform``.
        num_image_tokens: How many ``<|img_pad|>`` tokens to splice in for
            each image. Defaults to ``(H/16) * (W/16)`` (one per 16x16 patch).
        siglip_processor_id: If set, also produce SigLIP2 inputs using the
            named HF processor (e.g. ``"google/siglip2-so400m-patch16-384"``).
    """

    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        tokenizer,
        image_size: int | tuple[int, int] = 256,
        data_type: str = "t2i",
        max_text_length: int = 256,
        image_field: str = "image",
        text_field: str = "caption",
        conversations_field: str = "conversations",
        center_crop: bool = True,
        num_image_tokens: int | None = None,
        siglip_processor_id: str | None = None,
        clip_image_size: int | tuple[int, int] = 384,
        tuna_token_ids: dict[str, int] | None = None,
        multi_resolution: bool = False,
        resolution_buckets: list[list[int]] | None = None,
    ) -> None:
        super().__init__()
        if data_type not in {"t2i", "mmu", "mmu_text"}:
            raise ValueError(f"data_type must be 't2i', 'mmu', or 'mmu_text', got {data_type!r}")

        self.jsonl_path = jsonl_path
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_size = (
            (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        )
        self.multi_resolution = multi_resolution
        self.resolution_buckets = resolution_buckets or [
            [512, 512],
            [448, 576],
            [576, 448],
            [384, 672],
            [672, 384],
        ]
        self.data_type = data_type
        self.max_text_length = max_text_length
        self.image_field = image_field
        self.text_field = text_field
        self.conversations_field = conversations_field

        self.records: list[dict[str, Any]] = self._load_jsonl(jsonl_path)
        self.image_transform = build_image_transform(self.image_size, center_crop)

        if num_image_tokens is None:
            # 16x16 patch size is the WAN-VAE/Tuna default.
            patch = 16
            num_image_tokens = (self.image_size[0] // patch) * (
                self.image_size[1] // patch
            )
        self.num_image_tokens = num_image_tokens

        # Resolve special token ids. When `tuna_token_ids` is provided (from
        # the model wrapper), use those directly — they are already resolved
        # for the correct LLM variant (Qwen2.5 uses <|vision_start|> etc.).
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
        # `images_clip` fallback transform for the no-SigLIP variant (Tuna2Pixel).
        # When SigLIP is enabled, `images_clip` comes from the SigLIP processor;
        # otherwise we just emit a resized RGB tensor.
        self.clip_image_size = (
            (clip_image_size, clip_image_size)
            if isinstance(clip_image_size, int)
            else tuple(clip_image_size)
        )
        self.clip_image_transform = build_image_transform(self.clip_image_size, center_crop)

        logger.info(
            f"TIDataset: loaded {len(self.records)} records from {jsonl_path} "
            f"(data_type={data_type}, image_size={self.image_size}, "
            f"num_image_tokens={self.num_image_tokens}, siglip={siglip_processor_id})"
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

    # --- core sample loading -----------------------------------------------

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

    def _extract_text(self, record: dict[str, Any]) -> str:
        """Extract the user-visible string for this record.

        For the conversation shape, we use the *last* assistant turn as the
        target text in MMU mode, and the *last* human turn as the prompt in
        T2I mode (matches the tuna convention).
        """
        if self.text_field in record:
            return str(record[self.text_field])

        convs = record.get(self.conversations_field)
        if isinstance(convs, list) and convs:
            if self.data_type == "mmu":
                # Pull the last "gpt" / "assistant" turn.
                for turn in reversed(convs):
                    role = turn.get("from") or turn.get("role")
                    if role in {"gpt", "assistant"}:
                        return str(turn.get("value") or turn.get("content") or "")
                # Fallback: last turn regardless of role.
                return str(convs[-1].get("value") or convs[-1].get("content") or "")
            else:  # t2i
                # Pull the last "human" / "user" turn as the prompt.
                for turn in reversed(convs):
                    role = turn.get("from") or turn.get("role")
                    if role in {"human", "user"}:
                        return str(turn.get("value") or turn.get("content") or "")
                return str(convs[0].get("value") or convs[0].get("content") or "")
        return ""

    def _has_chat_template(self, record: dict[str, Any]) -> bool:
        return (
            self.conversations_field in record
            and isinstance(record[self.conversations_field], list)
            and len(record[self.conversations_field]) > 0
            and self.text_field not in record
        )

    def _tokenize_text(
        self, record: dict[str, Any], sentence: str
    ) -> list[int]:
        """Return the token-id list for the text portion of this sample."""
        if self._has_chat_template(record) and hasattr(
            self.tokenizer, "apply_chat_template"
        ):
            # Convert "human"/"gpt" to "user"/"assistant" for HF chat templates.
            messages = []
            role_map = {"human": "user", "gpt": "assistant"}
            for turn in record[self.conversations_field]:
                role = turn.get("from") or turn.get("role") or "user"
                role = role_map.get(role, role)
                content = turn.get("value") or turn.get("content") or ""
                messages.append({"role": role, "content": content})
            try:
                ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=True,
                )
                return list(ids)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"apply_chat_template failed ({e}); falling back to raw")
        # Plain text fallback: tokenize the bare sentence.
        # We avoid adding special tokens because format_sequence_* adds bos/eos.
        ids = self.tokenizer(
            sentence, add_special_tokens=False, truncation=True, max_length=self.max_text_length
        )["input_ids"]
        return list(ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]

        # 1. Load image (or create dummy for text-only).
        has_image = self.image_field in record and record[self.image_field]
        if has_image:
            image_path = record[self.image_field]
            pil = self._load_image(image_path)
        else:
            pil = None

        # Multi-resolution: pick the bucket closest to the image's native
        # aspect ratio (matches showme's multiresolution_iterator_wrapper).
        if self.multi_resolution and pil is not None:
            src_w, src_h = pil.size
            src_ar = src_w / max(src_h, 1)
            best_bucket = min(
                self.resolution_buckets,
                key=lambda b: abs(b[1] / max(b[0], 1) - src_ar),
            )
            cur_size = tuple(best_bucket)
            cur_transform = build_image_transform(cur_size, center_crop=True)
            patch = 16
            cur_num_image_tokens = (cur_size[0] // patch) * (cur_size[1] // patch)
        else:
            cur_size = self.image_size
            cur_transform = self.image_transform
            cur_num_image_tokens = self.num_image_tokens

        if pil is not None:
            image_tensor = cur_transform(pil)
        else:
            image_tensor = torch.zeros(3, cur_size[0], cur_size[1])
            pil = None

        # 2. Build the text token list.
        sentence = self._extract_text(record)
        if not sentence and self.data_type == "mmu":
            # Without a target string for MMU, synthesize a generic prompt.
            sentence = random.choice(captioning_templates["user_short"]).format("image")
        text_token_ids = self._tokenize_text(record, sentence)
        # Hard-cap text length so the unified sequence fits in max_text_length.
        # Reserve space for bos/eos/boi/eoi (4) + img_pad tokens.
        reserve = 4 + cur_num_image_tokens
        max_text = max(0, self.max_text_length - reserve)
        if len(text_token_ids) > max_text:
            text_token_ids = text_token_ids[:max_text]

        # 3. Format the unified sequence.
        if self.data_type == "t2i":
            tt, tl, mp, tm, im = format_sequence_gen_qwen2_5(
                text_tokens=text_token_ids,
                system_tokens=None,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                boi_id=self.boi_id,
                eoi_id=self.eoi_id,
                pad_id=self.pad_id,
                img_pad_id=self.img_pad_id,
                num_image_tokens=cur_num_image_tokens,
                max_seq_len=self.max_text_length,
                system_token_len=0,
            )
        else:  # mmu
            tt, tl, mp, tm, im = format_sequence_und(
                text_tokens=text_token_ids,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                boi_id=self.boi_id,
                eoi_id=self.eoi_id,
                pad_id=self.pad_id,
                img_pad_id=self.img_pad_id,
                num_image_tokens=cur_num_image_tokens,
                max_seq_len=self.max_text_length,
            )

        out: dict[str, Any] = {
            "images": image_tensor,
            "text_tokens": tt.long(),
            "text_labels": tl.long(),
            "text_masks": tm.bool(),
            "image_masks": im.bool(),
            "modality_positions": mp.long(),
            "data_type": self.data_type,
            "sentence": sentence,
        }

        # 4. `images_clip` — companion image for SigLIP / patch embed.
        if pil is not None:
            if self.siglip_transform is not None:
                sig = self.siglip_transform(pil)
                out["images_clip"] = sig["pixel_values"]
                out["siglip_pixel_attention_mask"] = sig["pixel_attention_mask"]
                out["siglip_spatial_shapes"] = sig["spatial_shapes"]
            else:
                out["images_clip"] = self.clip_image_transform(pil)
        else:
            # Text-only: dummy clip image.
            out["images_clip"] = torch.zeros(3, self.clip_image_size[0], self.clip_image_size[1])

        return out
