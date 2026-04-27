"""Local-filesystem checkpoint callback.

Local checkpoint saving callback for Tuna training.
Writes ``model_step_<N>.pt`` (or
``.safetensors``) under a local directory and prunes old checkpoints to keep
the most recent ``keep_last`` files.
"""

# (c) Meta Platforms, Inc. and affiliates. Apache-2.0.

from __future__ import annotations

# pyre-unsafe

import logging
import os
import re
from typing import Any

import torch
import torch.distributed as dist
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TTrainUnit
from torchtnt.utils.distributed import get_global_rank


logger: logging.Logger = logging.getLogger(__name__)

try:
    from safetensors.torch import save_file as _safetensors_save_file
except Exception:  # pragma: no cover - safetensors is in requirements but be safe
    _safetensors_save_file = None  # type: ignore[assignment]


class LocalCheckpointCallback(Callback):
    """Save model + optimizer state to a local directory every ``N`` steps.

    Args:
        checkpoint_dir: Directory under which to write ``step_<N>/model.pt``
            (or ``model.safetensors``).
        save_every_n_steps: Save frequency in *training* steps. ``<=0``
            disables periodic saves (epoch-end save still runs).
        keep_last: Maximum number of historical checkpoints to keep on disk.
            Older ones are deleted after each successful save.
        use_safetensors: If ``True``, write ``model.safetensors`` (only the
            model weights — no optimizer state).
        save_optimizer: If ``True`` and not using safetensors, also include
            optimizer / lr-scheduler / progress state in the checkpoint.
    """

    _STEP_RE = re.compile(r"^step_(\d+)$")

    def __init__(
        self,
        checkpoint_dir: str,
        save_every_n_steps: int = 1000,
        keep_last: int = 3,
        use_safetensors: bool = False,
        save_optimizer: bool = True,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_steps = int(save_every_n_steps)
        self.keep_last = int(keep_last)
        self.use_safetensors = use_safetensors
        self.save_optimizer = save_optimizer

        if get_global_rank() == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ---- TorchTNT hooks ----------------------------------------------------
    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        if self.save_every_n_steps <= 0:
            return
        step = unit.train_progress.num_steps_completed
        if step == 0 or step % self.save_every_n_steps != 0:
            return
        self._save(unit, step)

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        step = unit.train_progress.num_steps_completed
        self._save(unit, step)

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        step = unit.train_progress.num_steps_completed
        self._save(unit, step)

    # ---- Internals ---------------------------------------------------------
    def _save(self, unit: TTrainUnit, step: int) -> None:
        # FSDP shard collection happens collectively; gather across all ranks
        # via the AutoUnit's state_dict (SHARDED_STATE_DICT vs FULL_STATE_DICT
        # is decided by the FSDPStrategy on the unit).
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        rank = get_global_rank()
        step_dir = os.path.join(self.checkpoint_dir, f"step_{step}")
        if rank == 0:
            os.makedirs(step_dir, exist_ok=True)

        module = getattr(unit, "module", None)
        if module is None:
            logger.warning("LocalCheckpointCallback: unit has no .module; skipping.")
            return
        try:
            state_dict = module.state_dict()
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to gather state_dict for save: {e}")
            return

        if rank == 0:
            if self.use_safetensors and _safetensors_save_file is not None:
                # safetensors only supports tensors; cast on cpu to avoid issues.
                cpu_state = {
                    k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in state_dict.items()
                }
                _safetensors_save_file(cpu_state, os.path.join(step_dir, "model.safetensors"))
            else:
                payload: dict[str, Any] = {"model": state_dict, "step": step}
                if self.save_optimizer:
                    optimizer = getattr(unit, "optimizer", None)
                    if optimizer is not None:
                        try:
                            payload["optimizer"] = optimizer.state_dict()
                        except Exception as e:  # pragma: no cover
                            logger.warning(f"Could not save optimizer state: {e}")
                    lr_scheduler = getattr(unit, "lr_scheduler", None)
                    if lr_scheduler is not None:
                        try:
                            payload["lr_scheduler"] = lr_scheduler.state_dict()
                        except Exception:  # pragma: no cover
                            pass
                torch.save(payload, os.path.join(step_dir, "model.pt"))
            logger.info(f"Saved checkpoint to {step_dir}")
            self._prune_old()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _prune_old(self) -> None:
        if self.keep_last <= 0:
            return
        try:
            entries = os.listdir(self.checkpoint_dir)
        except FileNotFoundError:
            return
        kept = []
        for name in entries:
            m = self._STEP_RE.match(name)
            if m is not None:
                kept.append((int(m.group(1)), name))
        kept.sort(reverse=True)
        for _, name in kept[self.keep_last:]:
            path = os.path.join(self.checkpoint_dir, name)
            try:
                for root, _dirs, files in os.walk(path, topdown=False):
                    for f in files:
                        os.remove(os.path.join(root, f))
                    os.rmdir(root)
                logger.info(f"Pruned old checkpoint: {path}")
            except OSError as e:
                logger.warning(f"Could not prune {path}: {e}")
