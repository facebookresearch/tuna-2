# pyre-unsafe
# (c) Meta Platforms, Inc. and affiliates. Apache-2.0.
"""
Weighted multi-dataloader sampler for Tuna.

Combines several dataloaders (e.g. T2I, MMU, edit) into a single iterator
that draws each batch from one of the underlying dataloaders according to a
weight distribution. Optionally synchronizes the per-step choice across all
distributed ranks so that every rank consumes from the same underlying
dataloader on every step (useful when the dataloaders have different
collators / output schemas).
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

BatchDict = dict[str, Any]

logger = logging.getLogger(__name__)


class WeightedDataLoaderSampler:
    """A sampler that randomly selects from multiple dataloaders based on
    sampling weights.

    Automatically re-initializes dataloaders when they raise ``StopIteration``,
    so iteration is effectively infinite.
    """

    def __init__(
        self,
        dataloaders: dict[str, Any],
        sampling_weights: dict[str, float],
        sync_sampling: bool = False,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            dataloaders: Dictionary mapping dataloader names to dataloader
                objects.
            sampling_weights: Dictionary mapping dataloader names to their
                sampling weights (any positive floats; will be normalized).
            sync_sampling: If True, rank 0 picks the dataloader for each step
                and broadcasts the choice so all ranks consume from the same
                source on every step.
            seed: Random seed for reproducible sampling.
        """
        self.dataloaders = dataloaders
        self.sampling_weights = sampling_weights
        self.dataloader_iterators: dict[str, Iterator] = {}
        self.sync_sampling = sync_sampling

        # Validate inputs.
        if set(dataloaders.keys()) != set(sampling_weights.keys()):
            raise ValueError("Dataloader names and sampling weight keys must match")

        # Normalize weights.
        total_weight = sum(sampling_weights.values())
        self.normalized_weights = {
            name: weight / total_weight for name, weight in sampling_weights.items()
        }

        # Set up the random number generator.
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize all iterators.
        self._initialize_all_iterators()

        logger.info(
            f"Initialized WeightedDataLoaderSampler with {len(dataloaders)} dataloaders: "
            f"{list(dataloaders.keys())}, weights: {self.normalized_weights}"
        )

    def _initialize_all_iterators(self) -> None:
        """Initialize iterators for all dataloaders."""
        for name, dataloader in self.dataloaders.items():
            self.dataloader_iterators[name] = iter(dataloader)
            logger.debug(f"Initialized iterator for dataloader: {name}")

    def _reinitialize_iterator(self, name: str) -> None:
        """Re-initialize a specific dataloader iterator."""
        logger.debug(f"Re-initializing iterator for dataloader: {name}")
        self.dataloader_iterators[name] = iter(self.dataloaders[name])

    def _sample_dataloader(self) -> str:
        """Sample a dataloader name based on the weights."""
        names = list(self.normalized_weights.keys())
        weights = list(self.normalized_weights.values())
        return str(np.random.choice(names, p=weights))

    def _sample_dataloader_sync(self) -> str:
        """Sample a dataloader name on rank 0 and broadcast to all ranks."""
        # Rank 0 samples.
        if dist.get_rank() == 0:
            probs = torch.tensor(
                list(self.normalized_weights.values()), dtype=torch.float32
            )
            idx = int(torch.multinomial(probs, num_samples=1).item())
        else:
            idx = -1
        # Broadcast the index.
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        idx_tensor = torch.tensor([idx], dtype=torch.int64, device=device)
        dist.broadcast(idx_tensor, src=0)
        idx = int(idx_tensor.item())
        names = list(self.normalized_weights.keys())
        return names[idx]

    def __iter__(self) -> Iterator[BatchDict]:
        """Iterate indefinitely, sampling from dataloaders based on weights.
        Re-initializes dataloaders when they raise ``StopIteration``.
        """
        use_sync = (
            self.sync_sampling
            and dist.is_available()
            and dist.is_initialized()
        )
        while True:
            selected_name = (
                self._sample_dataloader_sync()
                if use_sync
                else self._sample_dataloader()
            )
            selected_iterator = self.dataloader_iterators[selected_name]

            try:
                batch = next(selected_iterator)
                logger.info(f"Yielding batch from dataloader: {selected_name}")
                yield batch
            except StopIteration:
                # Re-initialize the exhausted dataloader and try again.
                logger.info(f"Dataloader {selected_name} exhausted, re-initializing...")
                self._reinitialize_iterator(selected_name)

                try:
                    batch = next(self.dataloader_iterators[selected_name])
                    logger.debug(
                        f"Yielding batch from re-initialized dataloader: {selected_name}"
                    )
                    yield batch
                except StopIteration:
                    # If it still raises StopIteration the dataloader is empty.
                    logger.warning(
                        f"Dataloader {selected_name} appears to be empty, skipping..."
                    )
                    continue


def weighted_dataloader_iterator(
    dataloaders: dict[str, Any] | list[Any],
    sync_sampling: bool = False,
    sampling_weights: dict[str, float] | list[float] | None = None,
    seed: int | None = None,
) -> WeightedDataLoaderSampler:
    """Convenience constructor for ``WeightedDataLoaderSampler``.

    Args:
        dataloaders: Either a dict mapping names to dataloaders, or a list of
            dataloaders (which will be auto-named ``dataloader_0``, ...).
        sync_sampling: See ``WeightedDataLoaderSampler``.
        sampling_weights: Either a dict, a list (matching the order of
            ``dataloaders``), or ``None`` (uniform weights).
        seed: Random seed for reproducible sampling.

    Returns:
        A configured ``WeightedDataLoaderSampler``.
    """
    # Convert list inputs to dictionaries.
    if isinstance(dataloaders, list):
        dataloaders = {f"dataloader_{i}": dl for i, dl in enumerate(dataloaders)}

    if sampling_weights is None:
        sampling_weights = {name: 1.0 for name in dataloaders.keys()}
    elif isinstance(sampling_weights, list):
        if len(sampling_weights) != len(dataloaders):
            raise ValueError("Number of weights must match number of dataloaders")
        dataloader_names = list(dataloaders.keys())
        sampling_weights = {
            name: weight
            for name, weight in zip(dataloader_names, sampling_weights, strict=False)
        }

    return WeightedDataLoaderSampler(dataloaders, sampling_weights, sync_sampling, seed)
