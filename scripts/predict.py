"""Hydra entry point for Tuna prediction.

Usage::

    python -m tuna.scripts.predict --config-name t2i_1k prompt="a cat"
    python -m tuna.scripts.predict --config-name edit_vae_512 \\
        image_path=./img.jpg instruction="make it sunset"
"""

# (c) Meta Platforms, Inc. and affiliates. Apache-2.0.

from __future__ import annotations

# pyre-unsafe

import logging
import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from tuna.inference.runner import TunaInference
from tuna.training.callbacks.save_image import SaveImageCallback


logger: logging.Logger = logging.getLogger(__name__)


def _make_data_for_mode(cfg: DictConfig) -> dict:
    """Materialise the per-mode input dict the runner expects."""
    mode = cfg.inference.inference_mode
    if mode == "t2i":
        prompt = cfg.get("prompt", "a photo of a cat")
        return {"text": prompt if isinstance(prompt, list) else [prompt]}
    if mode == "edit":
        instruction = cfg.get("instruction", "")
        image_path = cfg.get("image_path", None)
        if image_path is None:
            raise ValueError("predict (edit) requires an `image_path` override.")
        image = Image.open(image_path).convert("RGB")
        return {"text": [instruction], "image": [image]}
    if mode == "mmu":
        image_path = cfg.get("image_path", None)
        if image_path is None:
            raise ValueError("predict (mmu) requires an `image_path` override.")
        image = Image.open(image_path).convert("RGB")
        return {"image": [image]}
    raise ValueError(f"Unknown inference_mode: {mode!r}")


def _save_outputs(cfg: DictConfig, outputs: dict) -> None:
    """Persist results using the SaveImageCallback's helper logic."""
    output_dir = cfg.inference.get("output_dir", "./outputs/predictions")
    os.makedirs(output_dir, exist_ok=True)
    saver = SaveImageCallback(output_dir=output_dir)
    saver._batch_save(outputs, step=0)
    if "original_image" in outputs:
        saver._save_originals(outputs, step=0)
    logger.info(f"Wrote prediction outputs to {output_dir}")


@hydra.main(version_base=None, config_path="../../configs/predict", config_name="t2i")
def main(cfg: DictConfig) -> None:
    logger.info("Instantiating model from config...")
    model = instantiate(cfg.model)
    inference_kwargs = OmegaConf.to_container(cfg.inference, resolve=True)
    inference_kwargs.pop("output_dir", None)
    call_kwargs = {}
    if "seed" in inference_kwargs:
        call_kwargs["seed"] = inference_kwargs.pop("seed")
    runner = TunaInference(model=model, **inference_kwargs)

    data = _make_data_for_mode(cfg)
    with torch.no_grad():
        outputs = runner(data, **call_kwargs)

    _save_outputs(cfg, outputs)


if __name__ == "__main__":
    main()
