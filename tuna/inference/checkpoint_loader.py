"""Local-file / HuggingFace checkpoint loading utilities for Tuna inference."""

# (c) Meta Platforms, Inc. and affiliates. Apache-2.0.

from __future__ import annotations

# pyre-unsafe

import logging
import os
from typing import Any

import torch
import torch.nn as nn


logger: logging.Logger = logging.getLogger(__name__)


# Module-level imports kept lazy so the runner does not require huggingface_hub
# unless an HF id is actually used.


def _looks_like_hf_repo(path: str) -> bool:
    """Heuristic: an HF repo id is ``org/name`` with no path separators or ext."""
    if os.path.exists(path):
        return False
    if path.startswith(("./", "/", "~", "..")):
        return False
    return "/" in path and not path.endswith((".pt", ".pth", ".bin", ".safetensors"))


def download_from_hf(repo_id: str, filename: str | None = None) -> str:
    """Download a file (or full snapshot) from HuggingFace Hub.

    Args:
        repo_id: ``"org/name"`` repo identifier.
        filename: If given, download just that one file via ``hf_hub_download``;
            otherwise snapshot the whole repo and return the snapshot dir.
    """
    if filename is None:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(repo_id=repo_id)
        logger.info(f"Snapshot of {repo_id} downloaded to {local_dir}")
        return local_dir

    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    logger.info(f"Downloaded {repo_id}/{filename} to {local_path}")
    return local_path


def _load_state_dict_from_file(path: str) -> dict[str, Any]:
    """Read a state dict from a local ``.safetensors``, ``.pt``, ``.pth`` or ``.bin``."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    obj = torch.load(path, weights_only=True, map_location="cpu")
    # Many checkpoints wrap the state dict in {"model": ..., "step": ...}.
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unexpected checkpoint payload type at {path}: {type(obj)}")


def load_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    strict: bool = False,
) -> nn.Module:
    """Load weights into ``model`` from a local file or HuggingFace repo.

    Supported sources:

    * Local ``.pt`` / ``.pth`` / ``.bin`` (``torch.save``-style)
    * Local ``.safetensors``
    * HuggingFace repo id ``"org/name"`` — looks for ``model.safetensors``
      then ``pytorch_model.bin``.

    Args:
        model: Module to load into.
        ckpt_path: Local path or HF repo id.
        strict: Forwarded to :meth:`torch.nn.Module.load_state_dict`.
    """
    if _looks_like_hf_repo(ckpt_path):
        try:
            local_path = download_from_hf(ckpt_path, filename="model.safetensors")
        except Exception:
            local_path = download_from_hf(ckpt_path, filename="pytorch_model.bin")
    else:
        local_path = ckpt_path

    state_dict = _load_state_dict_from_file(local_path)
    # Drop frozen Wan VAE keys (`vision_model.vae.*`) — Tuna loads the VAE
    # separately via `Wan2_2_VAE.from_pretrained`.
    state_dict = {k: v for k, v in state_dict.items() if ".vae." not in k}

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        logger.warning(f"Missing keys when loading checkpoint: {missing[:10]} (truncated)")
    if unexpected:
        logger.warning(f"Unexpected keys when loading checkpoint: {unexpected[:10]} (truncated)")
    logger.info(f"Loaded checkpoint from {local_path}")
    return model
