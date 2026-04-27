"""Hydra entry point for Tuna training.

Usage::

    python -m tuna.scripts.train --config-name stage1_t2i
    torchrun --standalone --nproc-per-node=8 -m tuna.scripts.train \\
        --config-name stage1_t2i training.batch_size=8
"""

# (c) Meta Platforms, Inc. and affiliates. Apache-2.0.

from __future__ import annotations

# pyre-unsafe

import functools
import logging

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchtnt.utils.prepare_module import DDPStrategy

from tuna.training.fsdp_utils import create_fsdp_strategy
from tuna.training.trainer import build_default_callbacks, build_tb_logger, train
from tuna.training.unit import TunaUnit


logger: logging.Logger = logging.getLogger(__name__)


def _build_strategy(cfg: DictConfig):
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    fsdp_cfg = cfg.training.get("fsdp", None)
    if fsdp_cfg is not None and fsdp_cfg.get("enable", False) and distributed:
        mp = fsdp_cfg.get("mixed_precision", None)
        return create_fsdp_strategy(
            sharding_strategy=fsdp_cfg.get("sharding_strategy", None),
            state_dict_type=fsdp_cfg.get("state_dict_type", None),
            mixed_precision=dict(mp) if mp is not None else None,
        )
    if distributed:
        return DDPStrategy(find_unused_parameters=False)
    return None


def _build_optimizer_partial(cfg: DictConfig):
    lr = float(cfg.training.get("learning_rate", 1.0e-4))
    weight_decay = float(cfg.training.get("weight_decay", 1.0e-2))
    return functools.partial(
        torch.optim.AdamW,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=weight_decay,
    )


def _build_single_dataloader(
    cfg: DictConfig,
    dataset,
    batch_size_override: int | None = None,
    num_workers_override: int | None = None,
) -> torch.utils.data.DataLoader:
    batch_size = batch_size_override or int(cfg.training.get("batch_size", 4))
    num_workers = num_workers_override or int(cfg.training.get("num_workers", 4))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )


def _instantiate_dataset(data_cfg: DictConfig, model) -> torch.utils.data.Dataset:
    """Instantiate one dataset, injecting the model's tokenizer + token IDs."""
    from hydra.utils import get_class
    from omegaconf import OmegaConf

    kwargs = OmegaConf.to_container(data_cfg, resolve=True)
    cls = get_class(kwargs.pop("_target_"))
    kwargs.pop("tokenizer", None)
    kwargs.pop("batch_size", None)
    kwargs.pop("num_workers", None)
    tuna_token_ids = getattr(model, "tuna_token_ids", None)
    if tuna_token_ids is not None:
        kwargs["tuna_token_ids"] = tuna_token_ids
    return cls(tokenizer=model.text_tokenizer, **kwargs)


def _build_dataloaders(cfg: DictConfig, model):
    """Build either a single DataLoader or a weighted multi-stream sampler.

    Config shapes:
      (a) ``cfg.data._target_`` exists → single dataset / DataLoader.
      (b) ``cfg.data.streams`` exists → one dataset per stream, mixed via
          :class:`WeightedDataLoaderSampler` with ``cfg.data.sampling_weights``.
    """
    from tuna.data.weighted_sampler import weighted_dataloader_iterator

    if "streams" in cfg.data:
        dataloaders = {}
        for name, stream_cfg in cfg.data.streams.items():
            ds = _instantiate_dataset(stream_cfg, model)
            bs = stream_cfg.get("batch_size", None)
            nw = stream_cfg.get("num_workers", None)
            dataloaders[name] = _build_single_dataloader(cfg, ds, bs, nw)
        weights = dict(cfg.data.sampling_weights)
        sync = cfg.data.get("sync_sampling", False)
        logger.info(f"Multi-stream data: {list(dataloaders.keys())}, weights={weights}")
        return weighted_dataloader_iterator(
            dataloaders=dataloaders,
            sampling_weights=weights,
            sync_sampling=sync,
        )
    else:
        ds = _instantiate_dataset(cfg.data, model)
        return _build_single_dataloader(cfg, ds)


@hydra.main(version_base=None, config_path="../../configs/train", config_name="train")
def main(cfg: DictConfig) -> None:
    # Global seed for reproducibility.
    if "seed" in cfg:
        import random
        import numpy as np

        seed = int(cfg.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Set global seed to {seed}")

    logger.info("Instantiating model + dataset from config...")
    model = instantiate(cfg.model)

    # Build dataloader(s). Supports two config shapes:
    #   (a) Single dataset: cfg.data has _target_ → one DataLoader.
    #   (b) Multi-stream: cfg.data has `streams` + `sampling_weights`
    #       → multiple DataLoaders mixed via WeightedDataLoaderSampler
    #       (matches the internal 4-stream weighted-sampling pattern).
    dataloader = _build_dataloaders(cfg, model)

    # Multi-resolution is handled at the dataset level (TIDataset picks a
    # random resolution bucket per sample when multi_resolution=True).
    # Requires batch_size=1 since different samples have different sizes.

    strategy = _build_strategy(cfg)
    tb_logger = build_tb_logger(cfg)

    precision = cfg.training.get("mixed_precision", None)
    if precision == "bf16":
        precision_arg: str | None = "bf16"
    elif precision == "fp16":
        precision_arg = "fp16"
    else:
        precision_arg = None

    # LR scheduler: linear warmup then constant.
    warmup_steps = int(cfg.training.get("warmup_steps", 0))
    lr_scheduler_fn = None
    if warmup_steps > 0:
        from torch.optim.lr_scheduler import LambdaLR
        from tuna.training.fsdp_utils import constant_lr_with_warmup

        lr_scheduler_fn = functools.partial(
            LambdaLR,
            lr_lambda=functools.partial(
                constant_lr_with_warmup, warmup_steps=warmup_steps
            ),
        )

    # EMA (Stochastic Weight Averaging with EMA mode).
    swa_params = None
    ema_decay = cfg.training.get("ema_decay", None)
    if ema_decay is not None:
        from torchtnt.framework.auto_unit import SWAParams

        swa_params = SWAParams(
            warmup_steps_or_epochs=0,
            step_or_epoch_update_freq=1,
            averaging_method="ema",
            ema_decay=float(ema_decay),
            use_lit=True,
        )

    # Gradient accumulation.
    grad_accum = int(cfg.training.get("gradient_accumulation_steps", 1))

    unit = TunaUnit(
        model=model,
        tb_logger=tb_logger,
        strategy=strategy,
        optim_fn=_build_optimizer_partial(cfg),
        lr_scheduler_fn=lr_scheduler_fn,
        precision=precision_arg,
        clip_grad_norm=1.0,
        swa_params=swa_params,
        gradient_accumulation_steps=grad_accum,
    )

    callbacks = build_default_callbacks(cfg)
    train(cfg, unit, dataloader, callbacks=callbacks)


if __name__ == "__main__":
    main()
