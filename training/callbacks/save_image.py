# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Save generated images during prediction to a local directory.

Replaces the original ``SaveImageCallback`` which uploaded to Manifold. This
OSS variant writes PNG files (and ``.txt`` prompt files alongside) to a local
folder, with one subdirectory per rank to avoid name collisions.
"""

from __future__ import annotations

# pyre-unsafe

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TPredictUnit
from torchtnt.utils.distributed import get_global_rank


logger: logging.Logger = logging.getLogger(__name__)


def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """Convert a single ``[C, H, W]`` tensor in ``[0, 1]`` or ``[0, 255]`` to PIL."""
    # Squeeze any leading singleton dims (batch, temporal).
    while img_tensor.dim() > 3 and img_tensor.shape[0] == 1:
        img_tensor = img_tensor.squeeze(0)
    if img_tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor [C, H, W], got shape {tuple(img_tensor.shape)}")
    arr = img_tensor.detach().to(torch.float32).cpu().numpy()
    if arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))  # [H, W, C]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr)


def _tensor_to_pil_frames(video_tensor: torch.Tensor) -> list[Image.Image]:
    """Convert a video tensor with multiple frames to a list of PIL images.

    Handles shapes like ``[T, C, H, W]``, ``[T, H, W, C]``,
    ``[B, T, C, H, W]``, ``[B, C, T, H, W]``, etc.
    """
    t = video_tensor.detach().cpu()
    # Squeeze leading batch dims of size 1.
    while t.dim() > 4 and t.shape[0] == 1:
        t = t.squeeze(0)
    # Now expecting [T, C, H, W] or [T, H, W, C] or [C, T, H, W].
    if t.dim() == 4:
        # Determine if channels-first or channels-last or time-first.
        if t.shape[1] in (1, 3):
            # [T, C, H, W] — iterate over T.
            return [_tensor_to_pil(t[i]) for i in range(t.shape[0])]
        elif t.shape[-1] in (1, 3):
            # [T, H, W, C] — permute to [T, C, H, W].
            t = t.permute(0, 3, 1, 2)
            return [_tensor_to_pil(t[i]) for i in range(t.shape[0])]
        elif t.shape[0] in (1, 3):
            # [C, T, H, W] — permute to [T, C, H, W].
            t = t.permute(1, 0, 2, 3)
            return [_tensor_to_pil(t[i]) for i in range(t.shape[0])]
    # Fallback: try to squeeze into single image.
    return [_tensor_to_pil(t)]


class SaveImageCallback(Callback):
    """Write images produced during prediction to ``output_dir``.

    Args:
        output_dir: Local directory under which to write PNGs.
        task_type: Just a tag used in log lines / filenames.
    """

    def __init__(
        self,
        output_dir: str,
        task_type: str = "generation",
    ) -> None:
        self.output_dir = output_dir
        self.task_type = task_type
        self.rank: int = (
            dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        )
        self.num_saved: int = 0
        if get_global_rank() == 0:
            os.makedirs(self.output_dir, exist_ok=True)

    # ---- Output writers ----------------------------------------------------
    def _save_pil(self, pil_image: Image.Image, fname: str, prompt: str) -> None:
        path = os.path.join(self.output_dir, fname)
        os.makedirs(os.path.dirname(path) or self.output_dir, exist_ok=True)
        pil_image.save(path)
        with open(path.rsplit(".", 1)[0] + ".txt", "w", encoding="utf-8") as f:
            f.write(prompt)

    def _save_video_mp4(
        self, frames: list[Image.Image], fname: str, prompt: str, fps: int = 16
    ) -> None:
        """Save a list of PIL frames as video.

        Tries MP4 via torchvision/PyAV first; falls back to animated GIF
        (PIL-only, no extra deps) if PyAV is not installed.
        """
        path = os.path.join(self.output_dir, fname)
        os.makedirs(os.path.dirname(path) or self.output_dir, exist_ok=True)
        try:
            import torchvision.io

            video_tensor = torch.stack(
                [torch.from_numpy(np.array(f)) for f in frames]
            )
            torchvision.io.write_video(path, video_tensor, fps=fps)
        except ImportError:
            path = path.rsplit(".", 1)[0] + ".gif"
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=1000 // fps,
                loop=0,
            )
        with open(path.rsplit(".", 1)[0] + ".txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        logger.info(f"Saved {len(frames)}-frame video to {path}")

    def _batch_save(self, data: dict[str, Any], step: int) -> None:
        if "generated_image" not in data:
            logger.warning("No generated_image in prediction output; skipping save.")
            return
        generated_images = data["generated_image"]
        generated_path = data.get("save_path")
        prompts: list[str] = data.get("prompts", [])  # type: ignore[assignment]

        # Normalise the input to a flat list of PIL images.
        # For video outputs (multi-frame tensors), each frame becomes a
        # separate image saved as ``frame_00.png``, ``frame_01.png``, etc.
        pil_images: list[Image.Image] = []
        is_video = False
        if isinstance(generated_images, list):
            for img in generated_images:
                if isinstance(img, Image.Image):
                    pil_images.append(img)
                elif isinstance(img, torch.Tensor):
                    if img.dim() > 4 or (img.dim() == 4 and img.shape[0] > 8):
                        frames = _tensor_to_pil_frames(img)
                        pil_images.extend(frames)
                        is_video = len(frames) > 1
                    else:
                        pil_images.append(_tensor_to_pil(img))
                else:  # pragma: no cover
                    logger.warning(f"Unsupported generated_image element type: {type(img)}")
        elif isinstance(generated_images, torch.Tensor):
            t = generated_images
            if t.dim() > 4:
                pil_images = _tensor_to_pil_frames(t)
                is_video = len(pil_images) > 1
            elif t.dim() == 3:
                pil_images.append(_tensor_to_pil(t))
            else:
                for i in range(t.shape[0]):
                    pil_images.append(_tensor_to_pil(t[i]))

        if is_video and len(pil_images) > 1:
            prompt = prompts[0] if prompts else ""
            fname = f"{self.task_type}/step_{step}_rank_{self.rank}_video_{self.num_saved}.mp4"
            self._save_video_mp4(pil_images, fname, prompt)
            self.num_saved += 1
        else:
            for idx, pil_image in enumerate(pil_images):
                if (
                    isinstance(generated_path, list)
                    and idx < len(generated_path)
                    and generated_path[idx]
                ):
                    fname = f"{generated_path[idx]}.png"
                elif isinstance(generated_path, str) and generated_path:
                    fname = f"{generated_path}_rank{self.rank}_idx{idx}.png"
                else:
                    fname = (
                        f"{self.task_type}/step_{step}_rank_{self.rank}_img_{self.num_saved}.png"
                    )
                prompt = (
                    prompts[idx % len(prompts)] if prompts else f"image_{self.num_saved}"
                )
                self._save_pil(pil_image, fname, prompt)
                self.num_saved += 1

        if "generated_text" in data:
            text_outputs = data["generated_text"]
            if isinstance(text_outputs, str):
                text_outputs = [text_outputs]
            for idx, text in enumerate(text_outputs):
                fname = f"{self.task_type}/step_{step}_rank_{self.rank}_txt_{idx}.txt"
                path = os.path.join(self.output_dir, fname)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)

    def _save_originals(self, data: dict[str, Any], step: int) -> None:
        original_images = data["original_image"]
        if not isinstance(original_images, torch.Tensor):
            return
        # Denormalise [-1, 1] -> [0, 1].
        images = torch.clamp((original_images + 1.0) / 2.0, 0.0, 1.0)
        if images.dim() == 4:
            for idx in range(images.shape[0]):
                pil = _tensor_to_pil(images[idx])
                fname = f"{self.task_type}/step_{step}_rank_{self.rank}_img_{idx}_orig.png"
                self._save_pil(pil, fname, prompt="")

    # ---- TorchTNT hooks ----------------------------------------------------
    def on_predict_step_end(self, state: State, unit: TPredictUnit) -> None:
        step_output = state.predict_state.step_output  # type: ignore[union-attr]
        if step_output is None:
            return
        # PredictUnit returns (loss, data); TunaPredUnit always returns
        # (None, dict).
        if isinstance(step_output, tuple) and len(step_output) == 2:
            _loss, data = step_output
        else:
            data = step_output
        if not isinstance(data, dict):
            return
        step = unit.predict_progress.num_steps_completed
        self._batch_save(data, step)
        if "original_image" in data:
            self._save_originals(data, step)
