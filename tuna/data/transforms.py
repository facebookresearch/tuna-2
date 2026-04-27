# pyre-unsafe
# (c) Meta Platforms, Inc. and affiliates. Apache-2.0.
"""
Image transforms and aspect-ratio bucketing for Tuna.

Three things live here:

1. ``build_image_transform``: turns a ``PIL.Image`` into a
   ``[-1, 1]``-normalized ``torch.FloatTensor (3, H, W)``. This is the *target*
   pixel representation that the WAN-VAE / patch-embedding sees.
2. ``build_siglip_transform``: a thin wrapper around HF
   ``AutoProcessor`` that yields the SigLIP2 inputs (pixel values, attention
   mask, spatial shapes) used by the SigLIP2 vision encoder branch.
3. ``AspectRatioBucketSampler``: groups samples by aspect ratio so that
   variable-resolution batches stay homogeneous (otherwise we'd have to pad
   the whole batch to the largest H/W).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pixel transforms
# ---------------------------------------------------------------------------


def _as_hw(image_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, int):
        return (image_size, image_size)
    return (int(image_size[0]), int(image_size[1]))


def build_image_transform(
    image_size: int | tuple[int, int],
    center_crop: bool = True,
) -> Callable[[Image.Image], torch.Tensor]:
    """Build the standard Tuna pixel transform.

    Resize (preserving aspect ratio) so that the shorter side equals
    ``min(H, W)``, optionally center-crop to ``(H, W)``, convert to a tensor,
    then normalize to ``[-1, 1]`` (the WAN-VAE / patch-embedding input range).

    Args:
        image_size: Either an int (square image) or a ``(H, W)`` tuple.
        center_crop: If True, center-crop after resizing. If False, just
            resize directly to ``(H, W)`` (which may distort aspect ratio).

    Returns:
        A function ``PIL.Image -> torch.FloatTensor`` of shape ``(3, H, W)``.
    """
    h, w = _as_hw(image_size)
    short_side = min(h, w)

    ops: list[Callable[..., Any]] = []
    if center_crop:
        ops.append(
            transforms.Resize(short_side, interpolation=InterpolationMode.BICUBIC)
        )
        ops.append(transforms.CenterCrop((h, w)))
    else:
        ops.append(transforms.Resize((h, w), interpolation=InterpolationMode.BICUBIC))
    ops.append(transforms.ToTensor())  # -> [0, 1]
    # Normalize to [-1, 1]: (x - 0.5) * 2 == (x - 0.5) / 0.5
    ops.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    composed = transforms.Compose(ops)

    def _transform(img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return composed(img)

    return _transform


def build_siglip_transform(
    processor_id: str = "google/siglip2-so400m-patch16-384",
) -> Callable[[Image.Image], dict[str, torch.Tensor]]:
    """Build a SigLIP2-compatible image transform.

    Lazily imports ``transformers`` and constructs the HF ``AutoProcessor``;
    this avoids a hard dependency at module-import time.

    Returns a function ``PIL.Image -> dict`` with keys:

    * ``pixel_values``: ``(3, H, W)`` float tensor
    * ``pixel_attention_mask``: bool tensor over patches
    * ``spatial_shapes``: long tensor ``(2,)`` = ``(H_patches, W_patches)``
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(processor_id)

    def _transform(img: Image.Image) -> dict[str, torch.Tensor]:
        if img.mode != "RGB":
            img = img.convert("RGB")
        out = processor(images=img, return_tensors="pt")
        # AutoProcessor returns batched tensors; drop the batch dim.
        result: dict[str, torch.Tensor] = {}
        if "pixel_values" in out:
            result["pixel_values"] = out["pixel_values"][0]
        if "pixel_attention_mask" in out:
            result["pixel_attention_mask"] = out["pixel_attention_mask"][0].bool()
        else:
            # Build an all-True mask if the processor didn't supply one.
            pv = result.get("pixel_values")
            if pv is not None:
                # Best-effort: assume one mask token per image (full coverage).
                result["pixel_attention_mask"] = torch.ones(1, dtype=torch.bool)
        if "spatial_shapes" in out:
            ss = out["spatial_shapes"]
            # spatial_shapes from the processor can be (B, 2) or (2,).
            if ss.dim() == 2:
                ss = ss[0]
            result["spatial_shapes"] = ss.long()
        else:
            # Derive patches from pixel_values + processor patch_size if known.
            pv = result.get("pixel_values")
            patch_size = getattr(processor, "patch_size", None) or 16
            if pv is not None:
                _, ph, pw = pv.shape
                result["spatial_shapes"] = torch.tensor(
                    [ph // patch_size, pw // patch_size], dtype=torch.long
                )
        return result

    return _transform


# ---------------------------------------------------------------------------
# Aspect-ratio bucketing
# ---------------------------------------------------------------------------


# Reasonable defaults covering common training resolutions (256/384/512).
DEFAULT_AR_BUCKETS: list[tuple[int, int]] = [
    (256, 256),
    (256, 384),
    (384, 256),
    (256, 512),
    (512, 256),
    (384, 384),
    (384, 512),
    (512, 384),
    (512, 512),
    (384, 640),
    (640, 384),
    (512, 768),
    (768, 512),
]


class AspectRatioBucketSampler(Sampler[list[int]]):
    """Yield batches of indices whose source images share the same AR bucket.

    Each input sample's native ``(H, W)`` is matched to the nearest bucket by
    aspect ratio (``W/H``). Indices within each bucket are then chunked into
    batches of ``batch_size``. With ``shuffle=True`` (default), bucket order
    and within-bucket order are reshuffled every epoch.

    Notes:
        * The ``aspect_ratios`` argument should be an iterable of float
          ``W/H`` values, one per item in the underlying dataset.
        * The dataset itself is responsible for actually resizing each image
          to the matched bucket dimensions; this sampler only groups indices.
        * If ``drop_last=False``, the final partial batch in each bucket is
          yielded as-is. Set ``drop_last=True`` for fully-uniform batches.
    """

    def __init__(
        self,
        aspect_ratios: list[float],
        batch_size: int,
        buckets: list[tuple[int, int]] | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.aspect_ratios = list(aspect_ratios)
        self.batch_size = batch_size
        self.buckets = list(buckets) if buckets is not None else list(DEFAULT_AR_BUCKETS)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Precompute bucket aspect ratios once.
        self._bucket_ars: list[float] = [w / h for h, w in self.buckets]
        # Group each sample index by its closest bucket.
        self._index_buckets: list[list[int]] = self._assign_buckets()

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch (controls the RNG seed for shuffling)."""
        self.epoch = epoch

    def _assign_buckets(self) -> list[list[int]]:
        groups: list[list[int]] = [[] for _ in self.buckets]
        for idx, ar in enumerate(self.aspect_ratios):
            best = self._closest_bucket(ar)
            groups[best].append(idx)
        return groups

    def _closest_bucket(self, ar: float) -> int:
        # Use log-AR distance so a 4:3 / 3:4 pair is symmetric.
        log_ar = math.log(max(ar, 1e-6))
        best_idx = 0
        best_d = float("inf")
        for i, b_ar in enumerate(self._bucket_ars):
            d = abs(log_ar - math.log(b_ar))
            if d < best_d:
                best_d = d
                best_idx = i
        return best_idx

    def __iter__(self) -> Iterator[list[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        all_batches: list[list[int]] = []
        for bucket_indices in self._index_buckets:
            if not bucket_indices:
                continue
            indices = list(bucket_indices)
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in perm]
            # Chunk into batches.
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in perm]

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        total = 0
        for bucket_indices in self._index_buckets:
            n = len(bucket_indices)
            if n == 0:
                continue
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total
