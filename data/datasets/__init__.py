# pyre-unsafe
# (c) Meta Platforms, Inc. and affiliates. Apache-2.0.
"""Map-style datasets backed by local JSONL manifests."""

from __future__ import annotations

from tuna.data.datasets.edit_dataset import EditDataset
from tuna.data.datasets.ti_dataset import TIDataset


__all__ = ["EditDataset", "TIDataset"]
