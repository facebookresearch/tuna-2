# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared base + mixins for the three Tuna wrapper classes.

`TunaWrapperBase` owns the methods that are byte-for-byte identical across
all three wrappers (`TunaModel`, `Tuna2PixelModel`, `Tuna2RPixelModel`).
`JiTWrapperMixin` owns the JiT-style `prepare_latents_and_labels` shared by
the two pixel wrappers (B and C). The variant-A wrapper does latent-space
diffusion via the WAN VAE and uses a different transport-based latent
preparation; it does not inherit the mixin.

The aim is to eliminate ~150 lines of duplication while keeping each
variant's surprising behavior (variant init, build_models specifics,
forward(batch) plumbing) in its own module.
"""

from __future__ import annotations

# pyre-unsafe

from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from tuna.models.omni_attention import (
    omni_attn_mask_flexattention,
    omni_attn_mask_naive,
)


class TunaWrapperBase(nn.Module):
    """Base class for the Hydra-instantiable wrapper around an inner Tuna model.

    Subclasses must populate ``self.attention_backend`` (and, depending on the
    variant, ``self.tuna_model``, ``self.text_tokenizer``, etc. via
    ``build_models``) before any of the methods on this base are called.
    """

    def _freeze_params(
        self, model: nn.Module, frozen_params: Optional[List[str]] = None
    ) -> None:
        if frozen_params is None:
            return
        for n, p in model.named_parameters():
            for name in frozen_params:
                if name in n:
                    p.requires_grad = False

    @torch.no_grad()
    def create_attention_mask(
        self,
        batch_size: int,
        seq_length: int,
        modality_positions: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Build the omni attention mask (sdpa or flexattention)."""
        if self.attention_backend == "sdpa":
            attention_mask = omni_attn_mask_naive(
                batch_size, seq_length, modality_positions, device
            ).to(dtype)
            return attention_mask, None
        elif self.attention_backend == "flexattention":
            attention_mask = omni_attn_mask_flexattention(
                modality_positions,
                seq_length,
                self.tuna_model.tuna.config.num_attention_heads,
                device=device,
            )
            attention_mask_diffhead = omni_attn_mask_flexattention(
                modality_positions,
                seq_length,
                self.tuna_model.diffusion_head_config.num_attention_heads,
                device=device,
            )
            return attention_mask, attention_mask_diffhead
        else:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")


class JiTWrapperMixin:
    """JiT-style latent-prep, shared by `Tuna2PixelModel` (C) and
    `Tuna2RPixelModel` (B).

    Both variants run diffusion directly in pixel space (no VAE) and feed the
    pixel tensor to a JiT noise scheduler. Variant A (`TunaModel`) runs in
    VAE-latent space and overrides `prepare_latents_and_labels` with a
    transport-based version, so it does NOT inherit this mixin.

    Subclasses must populate ``self.jit_noise_scheduler``,
    ``self.und_max_t0``, ``self.mmu_noise_prob``, ``self.mmu_noise_level``.
    """

    @torch.no_grad()
    def prepare_latents_and_labels(
        self,
        pixel_values: torch.Tensor,
        data_type: List[str],
        image_masks: torch.Tensor,
        use_feature: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """JiT formulation: ``z_t = t * x + (1-t) * noise``.

        Returns ``(xt, t, x0_clean, masks, pixel_values)``.
        """
        from tuna.models.misc import prepare_jit_training_batch

        if len(pixel_values.shape) == 4:
            pixel_values = pixel_values.unsqueeze(2)

        t_list, xt_list, x0_list, masks = [], [], [], []

        for i, tp in enumerate(data_type):
            if tp == "edit_interleaved":
                is_first_image = i % 2 == 0
                max_t0 = self.und_max_t0 if is_first_image else None
            elif tp in [
                "mmu",
                "mmu_vid",
                "mmu_interleaved",
                "mmu_text",
                            ]:
                import random

                if self.mmu_noise_prob > 0 and random.random() < self.mmu_noise_prob:
                    noise_level = random.uniform(0, self.mmu_noise_level)
                    max_t0 = 1.0 - noise_level
                else:
                    max_t0 = self.und_max_t0
            else:
                max_t0 = None

            x0_input = pixel_values[i][None]
            zt, t, x0 = prepare_jit_training_batch(
                x0_input, self.jit_noise_scheduler, max_t0
            )

            t_list.append(t)
            xt_list.append(zt)
            x0_list.append(x0)

            if image_masks is None:
                # Inference paths (t2i_edit) pass no image_masks; nothing to
                # accumulate, the final masks output stays None.
                continue

            if (
                tp in ["mmu", "mmu_vid", "mmu_interleaved", "mmu_text"]
                and self.und_max_t0 == 1.0
            ):
                if i < image_masks.shape[0]:
                    masks.append(image_masks[i][None] * 0.0)
            elif tp == "edit_interleaved" and i % 2 == 0:
                masks.append(image_masks[i // 2][None])
            elif tp == "edit_interleaved" and i % 1 == 0:
                pass
            else:
                masks.append(image_masks[i][None])

        t = torch.stack(t_list, dim=0).squeeze(-1)
        xt = torch.cat(xt_list, dim=0)
        x0_clean = torch.cat(x0_list, dim=0)
        masks = torch.cat(masks, dim=0) if masks else image_masks

        return xt, t, x0_clean, masks, pixel_values
