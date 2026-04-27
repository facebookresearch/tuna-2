# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""
Map-style video dataset for Tuna training.

Each JSONL line describes one video clip::

    {"video": "videos/video.mp4", "caption": "a panda drinking coffee"}

The ``video`` field points to either:
  - A video file (``.mp4``, ``.avi``, ``.mov``, ``.mkv``, ``.webm``), decoded
    with OpenCV and uniformly sampled to ``num_frames``.
  - A directory of frame images (``frame_000.jpg``, ``frame_001.jpg``, …),
    sorted alphabetically.

The dataset loads up to ``num_frames`` frames, resizes them to
``image_size``, and stacks into a ``[C, T, H, W]`` tensor. The VAE encoder
(inside the model wrapper) converts this to latent space during training.

Returns the same dict schema as :class:`TIDataset` so the training loop and
weighted sampler can mix video and image batches seamlessly.
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

from tuna.data.tokenize_utils import format_sequence_gen_qwen2_5
from tuna.data.transforms import build_image_transform


logger = logging.getLogger(__name__)


class VideoDataset(Dataset):
    """Map-style video-text dataset over a local JSONL manifest.

    Args:
        jsonl_path: JSONL file — one record per line with ``video`` (video file
            path or frame directory) and ``caption`` fields.
        image_root: Root directory for resolving relative paths.
        tokenizer: HuggingFace tokenizer with Tuna special tokens registered.
        image_size: Target ``(H, W)`` per frame.
        num_frames: Number of frames to load per clip.
        max_text_length: Max unified-sequence length including image-pad tokens.
        tuna_token_ids: Token ID dict from the model wrapper.
    """

    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        tokenizer,
        image_size: int | tuple[int, int] = (384, 672),
        num_frames: int = 13,
        max_text_length: int = 512,
        video_field: str = "video",
        text_field: str = "caption",
        center_crop: bool = True,
        tuna_token_ids: dict[str, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.jsonl_path = jsonl_path
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_size = (
            (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        )
        self.num_frames = num_frames
        self.max_text_length = max_text_length
        self.video_field = video_field
        self.text_field = text_field

        self.records = self._load_jsonl(jsonl_path)
        self.image_transform = build_image_transform(self.image_size, center_crop)

        # Wan2.2 VAE downsamples 16x spatial and 4x temporal (with causal conv).
        # Latent shape: [B, 48, T_latent, H/16, W/16].
        spatial_ds = 16
        temporal_ds = 4
        latent_h = self.image_size[0] // spatial_ds
        latent_w = self.image_size[1] // spatial_ds
        latent_t = (num_frames + temporal_ds - 2) // temporal_ds + 1  # causal conv formula
        self.num_tokens_per_frame = latent_h * latent_w
        self.num_image_tokens = self.num_tokens_per_frame * latent_t + 1

        if tuna_token_ids is not None:
            self.bos_id = tuna_token_ids["bos_id"]
            self.eos_id = tuna_token_ids["eos_id"]
            self.pad_id = tokenizer.pad_token_id or self.eos_id
            self.boi_id = tuna_token_ids["boi_id"]
            self.eoi_id = tuna_token_ids["eoi_id"]
            self.img_pad_id = tuna_token_ids["img_pad_id"]
        else:
            raise ValueError("tuna_token_ids is required for VideoDataset")

        logger.info(
            f"VideoDataset: loaded {len(self.records)} records from {jsonl_path} "
            f"(num_frames={num_frames}, image_size={self.image_size}, "
            f"num_image_tokens={self.num_image_tokens})"
        )

    @staticmethod
    def _load_jsonl(path: str) -> list[dict[str, Any]]:
        records = []
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

    def _resolve_path(self, rel_or_abs: str) -> str:
        if os.path.isabs(rel_or_abs):
            return rel_or_abs
        return os.path.join(self.image_root, rel_or_abs)

    _VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

    def _load_frames(self, video_path: str) -> list[Image.Image]:
        """Load frames from a video file or a directory of frame images."""
        full_path = self._resolve_path(video_path)

        if os.path.isfile(full_path) and full_path.lower().endswith(self._VIDEO_EXTENSIONS):
            return self._load_frames_from_video(full_path)
        return self._load_frames_from_dir(full_path)

    def _load_frames_from_video(self, video_file: str) -> list[Image.Image]:
        """Decode a video file with OpenCV and uniformly sample num_frames."""
        import cv2

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_file}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"Video has no frames: {video_file}")

        if total_frames >= self.num_frames:
            indices = [
                int(i * total_frames / self.num_frames)
                for i in range(self.num_frames)
            ]
        else:
            indices = list(range(total_frames))
            while len(indices) < self.num_frames:
                indices.append(total_frames - 1)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, bgr = cap.read()
            if not ret:
                if frames:
                    frames.append(frames[-1].copy())
                    continue
                raise RuntimeError(f"Failed to read frame {idx} from {video_file}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        cap.release()
        return frames

    def _load_frames_from_dir(self, frame_dir: str) -> list[Image.Image]:
        """Load frames from a directory of images, sorted alphabetically."""
        frame_files = sorted(
            f for f in os.listdir(frame_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if len(frame_files) > self.num_frames:
            indices = [
                int(i * len(frame_files) / self.num_frames)
                for i in range(self.num_frames)
            ]
            frame_files = [frame_files[i] for i in indices]
        elif len(frame_files) < self.num_frames:
            while len(frame_files) < self.num_frames:
                frame_files.append(frame_files[-1])

        frames = []
        for fname in frame_files:
            path = os.path.join(frame_dir, fname)
            with Image.open(path) as img:
                img.load()
                if img.mode != "RGB":
                    img = img.convert("RGB")
                frames.append(img)
        return frames

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]

        # 1. Load video frames → [C, T, H, W]
        frames = self._load_frames(record[self.video_field])
        frame_tensors = [self.image_transform(f) for f in frames]
        # Stack: [T, C, H, W] → permute to [C, T, H, W]
        video_tensor = torch.stack(frame_tensors).permute(1, 0, 2, 3)

        # 2. Tokenize caption
        caption = record.get(self.text_field, "")
        text_token_ids = self.tokenizer(
            caption, add_special_tokens=False, truncation=True,
            max_length=self.max_text_length,
        )["input_ids"]

        reserve = 4 + self.num_image_tokens
        max_text = max(0, self.max_text_length - reserve)
        if len(text_token_ids) > max_text:
            text_token_ids = text_token_ids[:max_text]

        # 3. Format unified sequence (same as t2i but with video token count)
        tt, tl, mp, tm, im = format_sequence_gen_qwen2_5(
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

        return {
            "images": video_tensor,          # [C, T, H, W]
            "text_tokens": tt.long(),
            "text_labels": tl.long(),
            "text_masks": tm.bool(),
            "image_masks": im.bool(),
            "modality_positions": mp.long(),
            "data_type": "t2i",              # model treats video as t2i with latent_frames > 1
            "sentence": caption,
        }
