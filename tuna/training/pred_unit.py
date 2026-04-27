"""TorchTNT PredictUnit subclass for Tuna inference.

Wraps a callable inference factory (typically :class:`tuna.inference.runner.TunaInference`)
and forwards each batch to ``self.inference(data)``.
"""

# (c) Meta Platforms, Inc. and affiliates. Apache-2.0.

from __future__ import annotations

# pyre-unsafe

import logging
from typing import Any, Callable

import torch
from torchtnt.framework import PredictUnit
from torchtnt.framework.state import State


logger: logging.Logger = logging.getLogger(__name__)

TorchDevice = str | torch.device


class TunaPredUnit(PredictUnit):
    """Prediction Unit responsible for running inference from a data loader."""

    def __init__(
        self,
        inference_fn: Callable[..., Any] | None = None,
        device: TorchDevice | int | None = None,
    ) -> None:
        super().__init__()
        if inference_fn is None:
            raise ValueError(
                "TunaPredUnit requires an `inference_fn` callable that returns a "
                "TunaInference instance when called with `device=...`."
            )
        self.inference = inference_fn(device=device)

    def predict_step(
        self, state: State, data: dict[str, Any]
    ) -> tuple[None, dict[str, Any]]:
        """Generate outputs (images / text) for each batch during prediction."""
        with torch.no_grad():
            outputs = self.inference(data)
        data.update(outputs)
        return None, data
