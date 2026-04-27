"""TorchTNT AutoUnit subclass for Tuna training.

Cleaned-up port of the original tuna training unit. The OSS version drops
the AIX logger entirely and uses ``torchtnt.utils.loggers.tensorboard.TensorBoardLogger``
directly. All FSDP / EMA / training-step logic is preserved.
"""

# (c) Meta Platforms, Inc. and affiliates. Apache-2.0.

from __future__ import annotations

# pyre-unsafe

import logging
import time
from typing import Any, Callable, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchtnt.framework.auto_unit import AutoUnit, TrainStepResults
from torchtnt.framework.state import State
from torchtnt.utils.distributed import get_global_rank, get_world_size
from torchtnt.utils.loggers.tensorboard import TensorBoardLogger
from torchtnt.utils.lr_scheduler import TLRScheduler
from torchtnt.utils.prepare_module import Strategy


logger: logging.Logger = logging.getLogger(__name__)


class TunaUnit(AutoUnit):
    """AutoUnit implementation for Tuna training.

    Wraps a Tuna training-wrapper module (``TunaModel`` / ``Tuna2RPixelModel``
    / ``Tuna2PixelModel``) and a TensorBoard logger. Computes the per-step
    loss by calling ``self.module(data)`` and unpacking ``"loss"`` from the
    output dict (the Tuna model wrappers all return ``{"loss": ..., "loss_ntp":
    ..., "loss_flow": ..., ...}``).
    """

    def __init__(
        self,
        model: nn.Module,
        tb_logger: TensorBoardLogger | None = None,
        strategy: Strategy | str | None = None,
        optim_fn: Callable[[list[nn.Parameter]], Optimizer] | None = None,
        lr_scheduler_fn: Callable[[Optimizer], TLRScheduler] | None = None,
        step_lr_interval: Literal["step", "epoch"] = "step",
        precision: str | torch.dtype | None = None,
        clip_grad_norm: float | None = None,
        swa_params: Any | None = None,
        gradient_accumulation_steps: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            module=model,
            strategy=strategy,
            precision=precision,
            step_lr_interval=step_lr_interval,
            clip_grad_norm=clip_grad_norm,
            swa_params=swa_params,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self._step_time: float = 0.0
        self._tb_logger = tb_logger
        self._world_size: int = get_world_size()
        self._optim_fn = optim_fn
        self._lr_scheduler_fn = lr_scheduler_fn
        self.train_losses: list[int | float | torch.Tensor] = []
        self.val_losses: list[int | float | torch.Tensor] = []

    # ---- Optim / scheduler -------------------------------------------------
    def configure_optimizers_and_lr_scheduler(
        self, module: nn.Module
    ) -> tuple[Optimizer, TLRScheduler | None]:
        params = [p for p in module.parameters() if p.requires_grad]
        if self._optim_fn is None:
            raise ValueError(
                "TunaUnit needs an `optim_fn` (a partial that returns an Optimizer)."
            )
        optimizer = self._optim_fn(params)
        lr_scheduler = self._lr_scheduler_fn(optimizer) if self._lr_scheduler_fn else None
        return optimizer, lr_scheduler

    # ---- Logging helper ----------------------------------------------------
    def _log_scalar(
        self, name: str, value: int | float | torch.Tensor, step: int
    ) -> None:
        if self._tb_logger is not None:
            self._tb_logger.log(name, value, step)

    # ---- Lifecycle ---------------------------------------------------------
    def on_train_start(self, state: State) -> None:
        self._step_time = time.perf_counter()

    def on_train_end(self, state: State) -> None:
        if self._tb_logger is not None:
            self._tb_logger.close()
        logger.info("Training complete")

    def on_train_step_end(
        self,
        state: State,
        data: dict[str, Any],
        step: int,
        results: TrainStepResults,
    ) -> None:
        if get_global_rank() != 0:
            return
        loss, outputs, total_grad_norm = (
            results.loss,
            results.outputs,
            results.total_grad_norm,
        )
        self._log_scalar("Train Step Loss", loss, step)
        self.train_losses.append(loss.item())
        if total_grad_norm is not None:
            self._log_scalar("Train grad norm", total_grad_norm, step)

        if isinstance(outputs, dict):
            if "loss_disp" in outputs:
                self._log_scalar("Train Step Disploss", outputs["loss_disp"], step)
            if "loss_ntp" in outputs:
                self._log_scalar("Train Step ntploss", outputs["loss_ntp"], step)
            if "loss_flow" in outputs:
                self._log_scalar("Train Step mseloss", outputs["loss_flow"], step)
        # Log learning rate.
        if hasattr(self, "optimizer") and self.optimizer is not None:
            for i, pg in enumerate(self.optimizer.param_groups):
                self._log_scalar(f"Learning Rate/group_{i}", pg["lr"], step)

        delta = time.perf_counter() - self._step_time
        # Datasets may be wrapped in a single-key dict (e.g. {"image_t2i": batch}).
        maybe_dataset_names = list(data.keys())
        if len(maybe_dataset_names) == 1:
            dataset_key = maybe_dataset_names[0]
            if any(
                tok in dataset_key
                for tok in ("image_", "video_", "text_", "edit")
            ):
                data = data[dataset_key]
        if "texts" in data:
            self._log_scalar(
                "Train Throughput",
                len(data["texts"]) * self._world_size / max(delta, 1e-6),
                step,
            )
        self._step_time = time.perf_counter()

    def on_train_epoch_end(self, state: State) -> None:
        super().on_train_epoch_end(state)
        if get_global_rank() == 0 and self.train_losses:
            self._log_scalar(
                "Train Epoch Loss",
                float(np.mean([float(x) for x in self.train_losses])),
                self.train_progress.num_epochs_completed,
            )
            self.train_losses = []

    def on_eval_step_end(
        self,
        state: State,
        data: dict[str, Any],
        step: int,
        loss: torch.Tensor,
        outputs: torch.Tensor,
    ) -> None:
        if get_global_rank() == 0:
            self.val_losses.append(loss.item())

    def on_eval_epoch_end(self, state: State) -> None:
        if get_global_rank() == 0 and self.val_losses:
            self._log_scalar(
                "Val Epoch Loss",
                float(np.mean([float(x) for x in self.val_losses])),
                self.train_progress.num_epochs_completed,
            )
            self.val_losses = []

    # ---- Loss --------------------------------------------------------------
    def compute_loss(
        self, state: State, data: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        maybe_dataset_names = list(data.keys())
        if len(maybe_dataset_names) == 1:
            dataset_key = maybe_dataset_names[0]
            if any(
                tok in dataset_key
                for tok in ("image_", "video_", "text_", "edit")
            ):
                data = data[dataset_key]
        model_outputs = self.module(data)
        loss = model_outputs["loss"]
        return loss, model_outputs
