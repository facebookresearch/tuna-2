# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""FSDP / activation-checkpointing helper functions for Tuna training.

Hydra config convenience wrappers ported from the original tuna code.
"""

from __future__ import annotations

# pyre-unsafe

import logging
from typing import Any

import hydra
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
)
from torch.distributed.fsdp import ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torchtnt.utils import FSDPStrategy
from torchtnt.utils.fsdp_utils import MixedPrecision
from torchtnt.utils.prepare_module import ActivationCheckpointParams


logger: logging.Logger = logging.getLogger(__name__)


# ---- Hydra utilities -------------------------------------------------------


def get_objects(paths: list[str]) -> list[Any]:
    """Reference objects by name without constructing or calling them."""
    return [hydra.utils.get_object(path) for path in paths]


# ---- FSDP strategy ---------------------------------------------------------

# Default Tuna wrap classes for FSDP auto-wrapping.
DEFAULT_TUNA_WRAP_CLASSES: list[str] = [
    "tuna.models.backbones.modules.ModulatedAttentionBlock",
    "tuna.models.backbones.modules_new_mm.ModulatedAttentionBlock",
    "tuna.models.backbones.qwen2.Qwen2DecoderLayer",
    # SigLIP2's encoder layer is inherited from the base HF SigLIP class.
    "transformers.models.siglip.modeling_siglip.SiglipEncoderLayer",
    "tuna.models.vae.wan22_vae.ResidualBlock",
    "tuna.models.vae.wan22_vae.AttentionBlock",
]


def create_fsdp_strategy(
    class_paths: list[str] | None = None,
    sharding_strategy: str | None = None,
    state_dict_type: str | None = None,
    mixed_precision: dict[str, str | bool] | None = None,
) -> FSDPStrategy:
    """Hydra config convenience target for constructing :class:`FSDPStrategy`.

    Args:
        class_paths: Dotted paths of module classes to wrap with FSDP. Defaults
            to :data:`DEFAULT_TUNA_WRAP_CLASSES`.
        sharding_strategy: One of the names on
            :class:`torch.distributed.fsdp.ShardingStrategy`.
        state_dict_type: One of the names on
            :class:`torch.distributed.fsdp.StateDictType`.
        mixed_precision: Kwargs for :class:`torchtnt.utils.fsdp_utils.MixedPrecision`.
    """
    if class_paths is None:
        class_paths = DEFAULT_TUNA_WRAP_CLASSES
    classes = get_objects(class_paths)
    strategy = ShardingStrategy[sharding_strategy] if sharding_strategy else None
    return FSDPStrategy(
        sharding_strategy=strategy,
        auto_wrap_policy=ModuleWrapPolicy(classes),
        use_orig_params=True,
        state_dict_type=StateDictType[state_dict_type] if state_dict_type else None,
        mixed_precision=MixedPrecision(**mixed_precision) if mixed_precision else None,
    )


# ---- Activation checkpointing ---------------------------------------------


def layer_based_auto_wrap_policy(auto_wrap_layer_cls: list[str]) -> ModuleWrapPolicy:
    classes = get_objects(auto_wrap_layer_cls)
    return ModuleWrapPolicy(module_classes=classes)


def build_activation_checkpoint_params(
    auto_wrap_layer_cls: list[str],
    auto_wrap_policy: str,
    reentrant: bool = False,
) -> ActivationCheckpointParams:
    if reentrant:
        checkpoint_impl = CheckpointImpl.REENTRANT
    else:
        checkpoint_impl = CheckpointImpl.NO_REENTRANT

    if auto_wrap_policy == "layer_based_auto_wrap_policy":
        policy_func = layer_based_auto_wrap_policy(auto_wrap_layer_cls)
    else:
        raise NotImplementedError(f"{auto_wrap_policy} is not supported yet")

    return ActivationCheckpointParams(
        checkpoint_impl=checkpoint_impl,
        auto_wrap_policy=policy_func,
    )


# ---- LR schedule -----------------------------------------------------------


def constant_lr_with_warmup(current_step: int, warmup_steps: int = 1000) -> float:
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0
