# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Top-level training entrypoint that wraps :func:`torchtnt.framework.train`.

Top-level training wrapper around :func:`torchtnt.framework.train`.
"""

from __future__ import annotations

# pyre-unsafe

import logging
import os
from typing import Any, Iterable

import torch
import torchtnt.framework as tnt_framework
from omegaconf import DictConfig
from torchtnt.framework.auto_unit import AutoUnit
from torchtnt.framework.callback import Callback
from torchtnt.framework.callbacks.tqdm_progress_bar import TQDMProgressBar
from torchtnt.utils.distributed import get_global_rank
from torchtnt.utils.loggers.tensorboard import TensorBoardLogger

from tuna.training.callbacks.checkpoint import LocalCheckpointCallback


logger: logging.Logger = logging.getLogger(__name__)


def build_default_callbacks(cfg: DictConfig) -> list[Callback]:
    """Standard set of callbacks: tqdm progress + local checkpoint saver + GC."""
    callbacks: list[Callback] = []
    if get_global_rank() == 0:
        callbacks.append(TQDMProgressBar())
    output_dir = cfg.training.output_dir
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    save_every = int(cfg.training.get("save_every", 1000))
    keep_last = int(cfg.training.get("keep_last", 3))
    callbacks.append(
        LocalCheckpointCallback(
            checkpoint_dir=ckpt_dir,
            save_every_n_steps=save_every,
            keep_last=keep_last,
        )
    )
    # Periodic garbage collection to free GPU memory fragmentation.
    gc_interval = int(cfg.training.get("gc_interval", 5001))
    callbacks.append(GarbageCollectorCallback(step_interval=gc_interval))
    return callbacks


class GarbageCollectorCallback(Callback):
    """Run ``gc.collect()`` + ``torch.cuda.empty_cache()`` every N steps."""

    def __init__(self, step_interval: int = 5001) -> None:
        self.step_interval = step_interval

    def on_train_step_end(self, state, unit) -> None:
        step = unit.train_progress.num_steps_completed
        if step > 0 and step % self.step_interval == 0:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def build_tb_logger(cfg: DictConfig) -> TensorBoardLogger:
    output_dir = cfg.training.output_dir
    tb_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    return TensorBoardLogger(path=tb_dir)


def train(
    cfg: DictConfig,
    unit: AutoUnit,
    train_dataloader: Iterable[Any],
    callbacks: list[Callback] | None = None,
) -> None:
    """Direct call to :func:`torchtnt.framework.train` with sensible defaults.

    Args:
        cfg: Hydra config. Looks up ``cfg.training.{output_dir, max_epochs,
            max_steps_per_epoch, save_every, keep_last}``.
        unit: A :class:`torchtnt.framework.auto_unit.AutoUnit` (typically
            :class:`tuna.training.unit.TunaUnit`).
        train_dataloader: Anything iterable; usually a
            :class:`torch.utils.data.DataLoader`.
        callbacks: Optional callback list. If ``None``, a default progress bar
            + local checkpoint saver is constructed.
    """
    if callbacks is None:
        callbacks = build_default_callbacks(cfg)
    max_epochs = int(cfg.training.get("max_epochs", 1))
    max_steps_per_epoch = cfg.training.get("max_steps_per_epoch", None)
    if max_steps_per_epoch is not None:
        max_steps_per_epoch = int(max_steps_per_epoch)

    # Resume from the latest checkpoint if one exists.
    resume_path = cfg.training.get("resume_from", None)
    if resume_path is None:
        ckpt_dir = os.path.join(cfg.training.output_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            existing = sorted(
                [d for d in os.listdir(ckpt_dir) if d.startswith("step_")],
                key=lambda x: int(x.split("_")[1]),
            )
            if existing:
                resume_path = os.path.join(ckpt_dir, existing[-1])
                logger.info(f"Auto-resuming from {resume_path}")

    logger.info(
        f"Starting training: max_epochs={max_epochs}, "
        f"max_steps_per_epoch={max_steps_per_epoch}, "
        f"output_dir={cfg.training.output_dir}"
    )
    tnt_framework.train(
        unit,
        train_dataloader,
        max_steps_per_epoch=max_steps_per_epoch,
        max_epochs=max_epochs,
        callbacks=callbacks,
    )
    logger.info("Training finished.")
