# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared base + mixin for the three Tuna inner-model classes.

`TunaInnerBase` owns the methods that are byte-for-byte identical across
`Tuna`, `Tuna2Pixel`, and `Tuna2RPixel` — namely
`_set_gradient_checkpointing`, `disp_loss`, `unpatchify`, `_prepare_input`,
and `mmu_generate`.

`SiglipMixin` owns `encode_pixels_with_siglip`, used by the SigLIP-based
variants A and B (not C, which has no vision encoder).

Each subclass keeps its own variant-specific ``__init__``,
``reset_parameters``, ``forward``, ``_prepare_embeds``, ``t2i_generate``,
``t2i_generate_edit``, and any extra hooks (e.g.
``adaptive_normalize_cfg_feat`` for B).
"""

from __future__ import annotations

# pyre-unsafe

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from torch.nn.attention.flex_attention import BlockMask

from tuna.models.omni_attention import step_block_mask_from_old


class TunaInnerBase(ModelMixin, ConfigMixin):
    """Base class for the three Tuna inner-model variants.

    Subclasses must populate:
      * ``self.config`` (provided by ``ConfigMixin`` + ``register_to_config``)
        with at least ``image_latent_dim``, ``patch_size``, ``add_time_embeds``,
        and ``add_aspect_ratio_embeds``;
      * ``self.tuna`` — the Qwen2 LLM backbone.
    """

    _supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value

    def disp_loss(self, z):
        """Dispersive Loss (InfoNCE-L2 variant)."""
        z = z.reshape((z.shape[0], -1))
        diff = torch.nn.functional.pdist(z).pow(2) / z.shape[1]
        diff = torch.concat(
            (diff, diff, torch.zeros(z.shape[0]).cuda())
        )
        return torch.log(torch.exp(-diff).mean())

    def unpatchify(self, x, h, w, T=0):
        """Inverse of the patch embedding.

        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.config.image_latent_dim
        p = self.config.patch_size
        if T == 0:
            x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
            imgs = x.reshape(shape=(x.shape[0], h * p * w * p, c))
        else:
            x = x.reshape(shape=(x.shape[0], T, h, w, p, p, c))
            imgs = x.reshape(shape=(x.shape[0], T, h * p * w * p, c))
        return imgs

    def _prepare_input(
        self,
        input_embeds,
        image_embeds,
        image_labels,
        image_masks,
        new_image_labels,
        modality_positions,
        height_embeds_proj,
        width_embeds_proj,
        time_embeds_proj,
    ):
        # Vision token format: <BOI><height><width><time><img><img>...<img><EOI>
        for i, modality_batch in enumerate(modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                if offset < 0 and length < 0:
                    continue
                if self.config.add_time_embeds and self.config.add_aspect_ratio_embeds:
                    input_embeds[i, offset] = height_embeds_proj[
                        i * modality_positions.size(1) + j
                    ]
                    input_embeds[i, offset + 1] = width_embeds_proj[
                        i * modality_positions.size(1) + j
                    ]
                    input_embeds[i, offset + 2] = time_embeds_proj[
                        i * modality_positions.size(1) + j
                    ]
                    input_embeds[i, offset + 3 : offset + length] = image_embeds[
                        i * modality_positions.size(1) + j, : max(length - 3, 0)
                    ]
                    if image_labels is not None:
                        image_masks[i, offset] = 0
                        image_masks[i, offset + 1] = 0
                        image_masks[i, offset + 2] = 0
                        new_image_labels[i, offset + 3 : offset + length] = (
                            image_labels[
                                i * modality_positions.size(1) + j, : max(length - 3, 0)
                            ]
                        )
                elif self.config.add_time_embeds:
                    input_embeds[i, offset] = time_embeds_proj[
                        i * modality_positions.size(1) + j
                    ]
                    input_embeds[i, offset + 1 : offset + 1 + length - 1] = (
                        image_embeds[
                            i * modality_positions.size(1) + j, : max(length - 1, 0)
                        ]
                    )
                    if image_labels is not None:
                        image_masks[i, offset] = 0
                        new_image_labels[i, offset + 1 : offset + 1 + length - 1] = (
                            image_labels[
                                i * modality_positions.size(1) + j, : max(length - 1, 0)
                            ]
                        )
                else:
                    input_embeds[i, offset : offset + length] = image_embeds[
                        i * modality_positions.size(1) + j, :length
                    ]
                    if image_labels is not None:
                        new_image_labels[i, offset : offset + length] = image_labels[
                            i * modality_positions.size(1) + j, :length
                        ]
        return input_embeds, new_image_labels, image_masks

    @torch.no_grad()
    def mmu_generate(
        self,
        input_embeds=None,
        attention_mask=None,
        max_new_tokens=100,
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
        eos_token=None,
    ):
        """Autoregressive decode for multimodal-understanding."""
        device = input_embeds.device

        result = []
        idx_next_embeds = input_embeds
        for i in range(max_new_tokens):
            if i == 0:
                model_output = self.tuna(
                    inputs_embeds=input_embeds, attention_mask=attention_mask
                )
                logits = model_output.logits
                past_key_values = model_output.past_key_values
            else:
                model_output = self.tuna(
                    inputs_embeds=idx_next_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = model_output.logits
                past_key_values = model_output.past_key_values

            if isinstance(attention_mask, BlockMask):
                attention_mask = step_block_mask_from_old(
                    attention_mask, attention_mask.seq_lengths[1] + 1
                )
            else:
                attention_mask = attention_mask.squeeze([0, 1])
                attention_mask = torch.hstack(
                    [attention_mask[-1, :], torch.tensor([0]).to(device)]
                ).unsqueeze(0)
                attention_mask = attention_mask.expand(1, 1, -1, -1)

            if not do_sample:
                idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            else:
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                        :, :-1
                    ].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    logits[sorted_indices[sorted_indices_to_remove]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            result.append(idx_next[0][0])
            idx_next_embeds = self.tuna.model.embed_tokens(idx_next)
            input_embeds = torch.cat([input_embeds, idx_next_embeds], dim=1)

            if eos_token is not None and idx_next.cpu() == eos_token:
                break

        return result


class SiglipMixin:
    """Mixin for variants that use a SigLIP vision encoder (A and B).

    Subclasses must populate ``self.vision_model`` with a callable that
    accepts ``(pixel_values, interpolate_pos_encoding=...)`` and returns an
    object with ``last_hidden_state``.
    """

    def encode_pixels_with_siglip(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: Optional[torch.LongTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run SigLIP on pixel/latent values and return last hidden states.

        Accepts 5D ``[B, C, T, H, W]`` (collapses T into batch) or 4D
        ``[B, C, H, W]``. Uses position-embedding interpolation so spatial
        sizes other than the SigLIP pretrain size are supported.

        ``spatial_shapes`` and ``pixel_attention_mask`` are accepted for API
        symmetry with NaFlex SigLIP2 but ignored — the underlying model is
        classic SigLIP (Conv2d patch embed, fixed-grid positions).
        """
        if pixel_values.dim() == 5:
            b, c, t, h, w = pixel_values.shape
            pixel_values = pixel_values.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        outputs = self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=True,
        )
        return outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
