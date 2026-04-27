# coding=utf-8
# (c) Meta Platforms, Inc. and affiliates. Apache-2.0.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe
from __future__ import annotations

import functools
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig

from tuna.models.backbones.modules_new_mm import (
    DiffusionHeadConfig,
    FinalLayer,
    ModulatedAttentionBlock,
    TimestepEmbedder,
)
from tuna.models._common import _patchify_5d, build_rope
from tuna.models._inner_base import TunaInnerBase
from tuna.models._wrapper_base import JiTWrapperMixin, TunaWrapperBase
from tuna.models.backbones.qwen2 import Qwen2ForCausalLM
from tuna.models.misc import get_text_tokenizer, next_token_prediction
from tuna.models.vision.patch_embed import SimplePatchEmbedding

logger: logging.Logger = logging.getLogger(__name__)


class Tuna2Pixel(TunaInnerBase):
    """Pure patchify Tuna variant (variant C, 7B).

    Architecture:
      - No vision encoder (no SigLIP, no VAE).
      - A simple Conv2d-based :class:`SimplePatchEmbedding` patchifies raw RGB
        pixels into the LLM hidden size directly.
      - Diffusion head learns to predict ``x0`` (clean pixels) via JiT-style
        noise scheduling.
      - Optional learnable ``mask_token`` (DeTok integration) for masked-image
        modelling.
    """

    @register_to_config
    def __init__(
        self,
        llm_vocab_size=None,
        llm_model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        init_llm_from_config=False,
        image_latent_dim=16,
        image_latent_height=16,
        image_latent_width=16,
        video_latent_height=16,
        video_latent_width=16,
        reshape_frame_to_batch_dim=False,
        num_attention_heads=24,
        num_key_value_heads=8,
        patch_size=2,
        hidden_size=2048,
        num_diffusion_layers=10,
        add_aspect_ratio_embeds=True,
        add_time_embeds=True,
        use_disp=False,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        enable_mask_token: bool = False,
        masked_image_ratio_min=0.0,
        masked_image_ratio=0.75,
        **kwargs,
    ):
        super().__init__()
        self.use_disp = use_disp

        llm_config = AutoConfig.from_pretrained(llm_model_path)
        if init_llm_from_config:
            self.tuna = Qwen2ForCausalLM(llm_config)
        else:
            self.tuna = Qwen2ForCausalLM.from_pretrained(
                llm_model_path, attn_implementation="sdpa"
            )
        self.tuna.resize_token_embeddings(llm_vocab_size)

        # Simple vision encoder: Conv2d patchify only (no SigLIP).
        self.vision_encoder = SimplePatchEmbedding(
            patch_size=patch_size,
            hidden_size=hidden_size,
            in_channels=image_latent_dim,
        )

        self.register_buffer(
            "image_position_ids",
            torch.arange(image_latent_height * image_latent_width).expand((1, -1)),
            persistent=False,
        )

        # Diffusion head for generation
        self.diffusion_head_config = DiffusionHeadConfig(
            hidden_size=self.tuna.config.hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=self.tuna.config.intermediate_size,
            max_position_embeddings=self.tuna.config.max_position_embeddings,
        )
        self.time_embed = TimestepEmbedder(self.diffusion_head_config.hidden_size)
        if add_aspect_ratio_embeds:
            self.aspect_ratio_embed = TimestepEmbedder(
                self.diffusion_head_config.hidden_size
            )
        if hidden_size != self.diffusion_head_config.hidden_size:
            self.diff_proj = nn.Sequential(
                nn.Linear(hidden_size, self.diffusion_head_config.hidden_size),
                nn.GELU(),
                nn.Linear(
                    self.diffusion_head_config.hidden_size,
                    self.diffusion_head_config.hidden_size,
                ),
            )
            self.time_embed_proj = nn.Linear(
                self.diffusion_head_config.hidden_size, hidden_size
            )
            if add_aspect_ratio_embeds:
                self.ar_embed_proj = nn.Linear(
                    self.diffusion_head_config.hidden_size, hidden_size
                )
        self.diffusion_head_a = nn.ModuleList(
            [
                ModulatedAttentionBlock(self.diffusion_head_config, layer_idx)
                for layer_idx in range(num_diffusion_layers)
            ]
        )
        self.diffusion_head_b = FinalLayer(
            self.diffusion_head_config.hidden_size, patch_size, image_latent_dim
        )

        # Masked Image (DeTok integration) — Learnable mask token.
        # Only the training-time random masking path is kept; inpainting was
        # dropped per Tuna spec.
        self.enable_mask_token = enable_mask_token
        self.masked_image_ratio = masked_image_ratio
        self.masked_image_ratio_min = masked_image_ratio_min
        if self.enable_mask_token:
            scale = hidden_size**-0.5
            self.mask_token = nn.Parameter(scale * torch.randn(1, 1, hidden_size))
            logger.info(
                f"[MaskedImage] Enabled with ratio={masked_image_ratio}, "
                f"mask_token initialized with shape {self.mask_token.shape}"
            )

        self.gradient_checkpointing = False
        if gradient_checkpointing:
            self.gradient_checkpointing = True
            self._gradient_checkpointing_func = functools.partial(
                checkpoint, **gradient_checkpointing_kwargs
            )
            self.tuna.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize patch embedding
        w = self.vision_encoder.patch_embedding.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.vision_encoder.patch_embedding.bias, 0)

        # Initialize projection layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        _basic_init(self.diffusion_head_a)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out output layers
        nn.init.constant_(self.diffusion_head_b.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.diffusion_head_b.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.diffusion_head_b.linear.weight, 0)
        nn.init.constant_(self.diffusion_head_b.linear.bias, 0)

    def forward(
        self,
        text_tokens=None,
        image_latents=None,
        t=None,
        attention_mask=None,
        diffhead_attention_mask=None,
        text_masks=None,
        image_masks=None,
        text_labels=None,
        image_labels=None,
        modality_positions=None,
        first_frame_as_cond=False,
        only_denoise_last_image=False,
        guidance_scale=0.0,
        output_hidden_states=True,
        max_seq_len=None,
        device="cuda:0",
        label=None,
        return_input_embeds=False,
        image_grid_thw=None,
        **kwargs,
    ):
        # multimodal understanding and generation
        clean_image_embeds = kwargs.get("clean_image_embeds", None)
        input_embeds = self.tuna.model.embed_tokens(text_tokens)
        dtype = input_embeds.dtype

        # Replace text embeddings with clean image embeddings for reconstruction
        if clean_image_embeds is not None and modality_positions is not None:
            clean_seq_len = clean_image_embeds.shape[1]
            for i in range(input_embeds.shape[0]):
                img_offset = modality_positions[i][0][0]
                replace_start = img_offset - clean_seq_len - 1
                if replace_start >= 0:
                    input_embeds[
                        i, replace_start : replace_start + clean_seq_len, :
                    ] = clean_image_embeds[i].to(dtype)

        (
            image_embeds,
            time_embeds,
            time_embeds_proj,
            height_embeds_proj,
            width_embeds_proj,
            rope_3d,
        ) = self._prepare_embeds(image_latents, t, device, dtype)
        # Prepare image labels for training
        # Structure text and image embeddings into sequences
        if image_latents is not None:
            b, c, T, h, w = image_latents.shape
            p = self.config.patch_size
            h_, w_ = h // p, w // p

        new_image_labels = None
        if image_labels is not None:
            image_labels = rearrange(image_labels, "b c t h w -> b (t h w) c")
            image_labels = image_labels.reshape(shape=(b, T, h_, w_, p, p, c))
            image_labels = image_labels.reshape(shape=(b, T * h_ * w_, p * p * c))
            new_image_labels = torch.zeros(
                [image_embeds.shape[0], max_seq_len, p * p * c],
                device=device,
                dtype=dtype,
            )
            image_masks = image_masks[:, :, None].repeat(1, 1, p * p * c)

        if image_embeds is not None:
            input_embeds, new_image_labels, image_masks = self._prepare_input(
                input_embeds,
                image_embeds,
                image_labels,
                image_masks,
                new_image_labels,
                modality_positions,
                height_embeds_proj,
                width_embeds_proj,
                time_embeds_proj,
            )

        if return_input_embeds:
            return input_embeds

        outputs = self.tuna(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        logits, last_hidden_states = outputs["logits"], outputs["hidden_states"][-1]
        position_ids = torch.arange(
            last_hidden_states.shape[1], device=last_hidden_states.device
        ).unsqueeze(0)
        # Diffusion head to predict x0 (clean images) - JiT style
        if hasattr(self, "diff_proj"):
            last_hidden_states = self.diff_proj(last_hidden_states)

        if diffhead_attention_mask is None:
            diffhead_attention_mask = attention_mask
        act = []
        for layer in self.diffusion_head_a:
            if self.gradient_checkpointing and self.training:
                last_hidden_states = self._gradient_checkpointing_func(
                    layer,
                    hidden_states=last_hidden_states,
                    adaln_input=time_embeds,
                    attention_mask=diffhead_attention_mask,
                    position_ids=position_ids,
                    rope_3d=rope_3d,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=None,
                    position_embeddings=None,
                    modality_positions=modality_positions,
                )[0]
            else:
                last_hidden_states = layer(
                    hidden_states=last_hidden_states,
                    adaln_input=time_embeds,
                    attention_mask=diffhead_attention_mask,
                    position_ids=position_ids,
                    rope_3d=rope_3d,
                    modality_positions=modality_positions,
                )[0]
            act.append(last_hidden_states)

        # Predict x0 (clean image) directly - JiT style
        x0_pred = self.diffusion_head_b(
            last_hidden_states, time_embeds, modality_positions
        )
        loss_disp = torch.tensor(0.0, device=device)
        if image_latents is None:
            loss_ntp = next_token_prediction(
                logits, text_labels, self.config.llm_vocab_size
            )
            loss_flow = torch.tensor(0.0, device=device)
            return logits, loss_ntp, loss_flow, loss_disp

        if text_labels is not None and image_labels is not None:
            from tuna.models.misc import jit_x0_prediction_loss

            loss_ntp = next_token_prediction(
                logits, text_labels, self.config.llm_vocab_size
            )
            # JiT-style loss: directly predict clean images (x0)
            if t.shape[0] == x0_pred.shape[0]:
                t_for_loss = t
            elif t.shape[0] == x0_pred.shape[0] * 2:
                t_for_loss = t[1::2]
            else:
                t_for_loss = None
            loss_flow = jit_x0_prediction_loss(
                x0_pred, new_image_labels[: x0_pred.shape[0]], t_for_loss, image_masks
            )
            if self.use_disp:
                loss_disp = self.disp_loss(act[-1])
            return logits, loss_ntp, loss_flow, loss_disp

        else:
            # Inference mode - return x0 predictions (JiT style)
            x0_pred_ = []
            num_imgs = 0
            for i, modality_batch in enumerate(modality_positions):
                for _, (offset, length) in enumerate(modality_batch):
                    if length == 0:
                        break
                    else:
                        x0_pred_.append(x0_pred[i, offset : offset + length])
                        num_imgs += 1
            x0_pred_ = torch.stack(x0_pred_)

            # Remove the time embedding
            if self.config.add_time_embeds and self.config.add_aspect_ratio_embeds:
                x0_pred_ = x0_pred_[:, 3:, :]
            elif self.config.add_time_embeds:
                x0_pred_ = x0_pred_[:, 1:, :]

            # Unpatchify
            x0_pred_ = self.unpatchify(x0_pred_, h_, w_, T=T)

            x0_pred_ = x0_pred_.permute(0, 3, 1, 2)
            x0_pred_ = x0_pred_.reshape(
                num_imgs,
                self.config.image_latent_dim,
                T,
                h_ * self.config.patch_size,
                w_ * self.config.patch_size,
            )

            return logits, x0_pred_

    def _prepare_embeds(self, image_latents, t, device, dtype):
        image_embeds = None
        rope_3d = None
        h_ = w_ = None
        b = None
        if image_latents is not None:
            b, c, T, h, w = image_latents.shape
            rope_3d = (
                build_rope(latent_shape=[T, h, w], patch_size=16, attention_head_dim=64)
                .to(device)
                .to(dtype)
            )
            p = self.config.patch_size
            h_, w_ = h // p, w // p

            # Use simple patch embedding (no SigLIP, no VAE).
            image_embeds = _patchify_5d(
                image_latents.to(dtype),
                self.vision_encoder,
                self.config.reshape_frame_to_batch_dim,
            )

            # Apply masked image (DeTok-style) after patch embedding
            if self.enable_mask_token and self.training:
                import random

                import numpy as np

                # Apply random masking to image embeddings (DeTok-style)
                B, N, D = image_embeds.shape

                # Generate random mask for each sample in batch
                for batch_idx in range(B):
                    mask_ratio = random.uniform(
                        self.config.masked_image_ratio_min, self.masked_image_ratio
                    )
                    num_masked = int(N * mask_ratio)

                    if num_masked > 0:
                        mask_indices = np.random.choice(N, num_masked, replace=False)
                        image_embeds[batch_idx, mask_indices] = self.mask_token

        time_embeds = self.time_embed(t, dtype)
        if hasattr(self, "time_embed_proj"):
            time_embeds_proj = self.time_embed_proj(time_embeds)
        else:
            time_embeds_proj = time_embeds

        height_embeds_proj = None
        width_embeds_proj = None
        if hasattr(self, "aspect_ratio_embed") and h_ is not None:
            latent_height = torch.tensor(h_, device=device).repeat(b)
            latent_width = torch.tensor(w_, device=device).repeat(b)
            height_embeds = self.aspect_ratio_embed(latent_height, dtype)
            width_embeds = self.aspect_ratio_embed(latent_width, dtype)
            if hasattr(self, "ar_embed_proj"):
                height_embeds_proj = self.ar_embed_proj(height_embeds)
                width_embeds_proj = self.ar_embed_proj(width_embeds)
            else:
                height_embeds_proj = height_embeds
                width_embeds_proj = width_embeds
        return (
            image_embeds,
            time_embeds,
            time_embeds_proj,
            height_embeds_proj,
            width_embeds_proj,
            rope_3d,
        )

    @torch.no_grad()
    def t2i_generate(
        self,
        image_latents=None,
        t=None,
        text_tokens=None,
        attention_mask=None,
        diffhead_attention_mask=None,
        modality_positions=None,
        first_frame_as_cond=False,
        only_denoise_last_image=False,
        max_seq_len=None,
        guidance_scale=0.0,
        label=None,
        image_masks=None,
        image_labels=None,
        second_time=False,
        **kwargs,
    ):
        clean_image_embeds = kwargs.get("clean_image_embeds", None)

        if guidance_scale > 0.0:
            if t.shape[-1] != text_tokens.shape[0]:
                t_cond, t_uncond = torch.chunk(t, 2)
                t_cond[:-1] = 1.0
                t_uncond[:-1] = 1.0
                t = torch.cat([t_cond, t_uncond])

            if second_time:
                # Split into two separate forward passes
                t_cond, t_uncond = torch.chunk(t, 2)
                text_tokens_cond, text_tokens_uncond = torch.chunk(text_tokens, 2)
                image_latents_cond, image_latents_uncond = torch.chunk(image_latents, 2)

                clean_image_embeds_cond = None
                clean_image_embeds_uncond = None
                if clean_image_embeds is not None:
                    clean_image_embeds_cond, clean_image_embeds_uncond = torch.chunk(
                        clean_image_embeds, 2
                    )

                # First forward pass (conditional)
                _, v_cond = self(
                    text_tokens_cond,
                    image_latents=image_latents_cond,
                    t=t_cond,
                    attention_mask=attention_mask,
                    diffhead_attention_mask=diffhead_attention_mask,
                    modality_positions=modality_positions,
                    first_frame_as_cond=first_frame_as_cond,
                    only_denoise_last_image=only_denoise_last_image,
                    guidance_scale=guidance_scale,
                    output_hidden_states=True,
                    max_seq_len=max_seq_len,
                    clean_image_embeds=clean_image_embeds_cond,
                )

                # Second forward pass (unconditional)
                _, v_uncond = self(
                    text_tokens_uncond,
                    image_latents=image_latents_uncond,
                    t=t_uncond,
                    attention_mask=attention_mask,
                    diffhead_attention_mask=diffhead_attention_mask,
                    modality_positions=modality_positions,
                    first_frame_as_cond=first_frame_as_cond,
                    only_denoise_last_image=only_denoise_last_image,
                    guidance_scale=guidance_scale,
                    output_hidden_states=True,
                    max_seq_len=max_seq_len,
                    clean_image_embeds=clean_image_embeds_uncond,
                )
            else:
                # Original single forward pass (batch_size=2)
                _, v = self(
                    text_tokens,
                    image_latents=image_latents,
                    t=t,
                    attention_mask=attention_mask,
                    diffhead_attention_mask=diffhead_attention_mask,
                    modality_positions=modality_positions,
                    first_frame_as_cond=first_frame_as_cond,
                    only_denoise_last_image=only_denoise_last_image,
                    guidance_scale=guidance_scale,
                    output_hidden_states=True,
                    max_seq_len=max_seq_len,
                    clean_image_embeds=clean_image_embeds,
                )
                v_cond, v_uncond = torch.chunk(v, 2)

            # Apply classifier-free guidance
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            return torch.cat([v, v], dim=0)

    @torch.no_grad()
    def t2i_generate_edit(
        self,
        image_latents=None,
        t=None,
        text_tokens=None,
        attention_mask=None,
        diffhead_attention_mask=None,
        modality_positions=None,
        first_frame_as_cond=False,
        only_denoise_last_image=False,
        max_seq_len=None,
        guidance_scale=0.0,
        label=None,
        image_masks=None,
        image_labels=None,
        image_edit_original=None,
        **kwargs,
    ):
        if guidance_scale > 0.0:
            if t.shape[-1] != text_tokens.shape[0]:
                t_cond, t_uncond = torch.chunk(t, 2)
                t_cond[:-1] = 1.0
                t_uncond[:-1] = 1.0
                t = torch.cat([t_cond, t_uncond])

            batch_size = image_edit_original.shape[0]
            image_latents_final = torch.cat(
                [
                    torch.stack([image_edit_original[i], image_latents[i]], dim=0)
                    for i in range(batch_size)
                ],
                dim=0,
            )
            new_t = torch.tensor([1, t[0], 1, t[1]]).to(t[0].device).to(t[0].dtype)
            _, v = self(
                text_tokens,
                image_latents=image_latents_final,
                t=new_t,
                attention_mask=attention_mask,
                diffhead_attention_mask=diffhead_attention_mask,
                modality_positions=modality_positions,
                first_frame_as_cond=first_frame_as_cond,
                only_denoise_last_image=only_denoise_last_image,
                guidance_scale=guidance_scale,
                output_hidden_states=True,
                max_seq_len=max_seq_len,
            )

            # Select only the 2nd and 4th elements (indices 1 and 3) from the first dimension
            v = v[[1, 3]]  # 4*48*1*h*w -> 2*48*1*h*w

            v_cond, v_uncond = torch.chunk(v, 2)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            return torch.cat([v, v], dim=0)

class Tuna2PixelModel(JiTWrapperMixin, TunaWrapperBase):
    """High-level training wrapper for :class:`Tuna2Pixel` (variant C).

    Hydra-instantiable. Trains pure pixel-space x0 prediction with no vision
    encoder and no VAE — just a Conv2d patchifier and the Qwen2.5 LLM.
    """

    def __init__(
        self,
        # Tuna configuration
        llm_model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        load_stage1_model: Optional[str] = None,
        frozen_params: Optional[List[str]] = None,
        hidden_size: int = 1536,
        image_latent_dim: int = 16,
        image_latent_height: int = 27,
        image_latent_width: int = 27,
        hq_image_latent_height: int = 64,
        hq_image_latent_width: int = 64,
        mixed_modal_latent_height: int = 27,
        mixed_modal_latent_width: int = 27,
        patch_size: int = 2,
        add_time_embeds: bool = True,
        add_aspect_ratio_embeds: bool = True,
        mrope_type: str = "dit_3drope_mm",
        reshape_frame_to_batch_dim: bool = False,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 8,
        attention_backend: str = "sdpa",
        # Training configuration
        ntp_coeff: float = 1.0,
        flow_coeff: float = 1.0,
        und_max_t0: float = 1.0,
        use_disp: bool = False,
        gradient_checkpointing: bool = False,
        gradient_checkpointing_kwargs: Optional[DictConfig] = None,
        # Transport configuration
        path_type: str = "Linear",
        prediction: str = "velocity",
        loss_weight: Optional[str] = None,
        train_eps: Optional[float] = 1e-5,
        sample_eps: Optional[float] = 1e-3,
        snr_type: str = "uniform",
        do_shift: bool = False,
        # Sampling configuration
        sampling_method: str = "euler",
        guidance_scale: float = 5.0,
        num_inference_steps: int = 50,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        reverse: bool = False,
        time_shifting_factor: float = 3.0,
        # Data configuration
        dtype: str = "bf16",
        flow_head_num: int = 10,
        # Masked Image (DeTok integration). Inpainting was dropped per Tuna spec.
        enable_mask_token: bool = False,
        masked_image_ratio_min: float = 0.0,
        masked_image_ratio: float = 0.75,
        # MMU (Multimodal Understanding) noise configuration
        mmu_noise_prob: float = 0.0,
        mmu_noise_level: float = 0.0,
        # JiT noise configuration
        noise_scale: float = 1.0,
    ):
        super().__init__()

        # Tuna configuration
        self.llm_model_path = llm_model_path
        self.frozen_params = frozen_params
        self.hidden_size = hidden_size
        self.image_latent_dim = image_latent_dim
        self.image_latent_height = image_latent_height
        self.image_latent_width = image_latent_width
        self.hq_image_latent_height = hq_image_latent_height
        self.hq_image_latent_width = hq_image_latent_width
        self.mixed_modal_latent_height = mixed_modal_latent_height
        self.mixed_modal_latent_width = mixed_modal_latent_width
        self.patch_size = patch_size
        self.add_time_embeds = add_time_embeds
        self.add_aspect_ratio_embeds = add_aspect_ratio_embeds
        self.load_stage1_model = load_stage1_model
        self.flow_head_num = flow_head_num
        self.mrope_type = mrope_type
        self.reshape_frame_to_batch_dim = reshape_frame_to_batch_dim
        self.attention_backend = attention_backend

        # Training coefficients
        self.ntp_coeff = ntp_coeff
        self.flow_coeff = flow_coeff
        self.und_max_t0 = und_max_t0
        self.use_disp = use_disp
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_checkpointing_kwargs = (
            OmegaConf.to_container(gradient_checkpointing_kwargs)
            if gradient_checkpointing_kwargs is not None
            else None
        )

        # Transport configuration
        self.path_type = path_type
        self.prediction = prediction
        self.loss_weight = loss_weight
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.snr_type = snr_type
        self.do_shift = do_shift

        # Sampling configuration
        self.sampling_method = sampling_method
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.atol = atol
        self.rtol = rtol
        self.reverse = reverse
        self.time_shifting_factor = time_shifting_factor

        # Masked Image (DeTok) configuration
        self.enable_mask_token = enable_mask_token
        self.masked_image_ratio_min = masked_image_ratio_min
        self.masked_image_ratio = masked_image_ratio

        # MMU noise configuration
        self.mmu_noise_prob = mmu_noise_prob
        self.mmu_noise_level = mmu_noise_level

        # Device and dtype
        self.dtype = dtype

        # Build models
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.noise_scale = noise_scale
        self.build_models()

    def build_models(self):
        """Initialize all model components"""
        from tuna.models.misc import JiTNoiseScheduler
        from tuna.models.transport.define import create_transport
        from tuna.models.transport.transport import Sampler

        # Initialize text tokenizer
        self.text_tokenizer, self.tuna_token_ids = get_text_tokenizer(
            self.llm_model_path,
            add_tuna_tokens=True,
            return_tuna_token_ids=True,
        )
        self.llm_vocab_size = len(self.text_tokenizer)

        # Initialize Tuna pure-patchify model (no vision encoder)
        model_config = {
            "llm_vocab_size": self.llm_vocab_size,
            "llm_model_path": self.llm_model_path,
            "init_llm_from_config": False,
            "image_latent_dim": self.image_latent_dim,
            "image_latent_height": self.image_latent_height,
            "image_latent_width": self.image_latent_width,
            "video_latent_height": self.image_latent_height,
            "video_latent_width": self.image_latent_width,
            "reshape_frame_to_batch_dim": self.reshape_frame_to_batch_dim,
            "hidden_size": self.hidden_size,
            "patch_size": self.patch_size,
            "add_time_embeds": self.add_time_embeds,
            "add_aspect_ratio_embeds": self.add_aspect_ratio_embeds,
            "num_diffusion_layers": self.flow_head_num,
            "gradient_checkpointing": self.gradient_checkpointing,
            "gradient_checkpointing_kwargs": self.gradient_checkpointing_kwargs,
            "use_disp": self.use_disp,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "enable_mask_token": self.enable_mask_token,
            "masked_image_ratio_min": self.masked_image_ratio_min,
            "masked_image_ratio": self.masked_image_ratio,
        }

        self.tuna_model = Tuna2Pixel(**model_config)
        if self.load_stage1_model is not None and self.load_stage1_model != "no":
            checkpoint = torch.load(self.load_stage1_model, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)

            model_dict = self.tuna_model.state_dict()

            # Only keep parameters with matching shapes
            filtered_dict = {
                k: v
                for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }

            logger.info(
                f"Loaded {len(filtered_dict)} / {len(model_dict)} keys from checkpoint."
            )

            self.tuna_model.load_state_dict(filtered_dict, strict=False)
        self._freeze_params(self.tuna_model, self.frozen_params)

        # Initialize transport for flow matching (kept for API compat)
        self.transport = create_transport(
            path_type=self.path_type,
            prediction=self.prediction,
            loss_weight=self.loss_weight,
            train_eps=self.train_eps,
            sample_eps=self.sample_eps,
            snr_type=self.snr_type,
            do_shift=self.do_shift,
        )
        logger.info("loaded all pretrained model!")
        self.sampler = Sampler(self.transport)

        # Initialize JiT-style noise scheduler
        self.jit_noise_scheduler = JiTNoiseScheduler(
            P_mean=-0.8,
            P_std=0.8,
            noise_scale=self.noise_scale,
            t_eps=5e-2,
        )

    @torch.no_grad()
    def prepare_clean_image_embeds(
        self, pixel_values_low: torch.Tensor
    ) -> torch.Tensor:
        """Encode degraded/low-res images through the patch embedding to get
        clean image embeddings used for reconstruction conditioning."""
        if len(pixel_values_low.shape) == 4:
            pixel_values_low = pixel_values_low.unsqueeze(2)

        dtype = next(self.tuna_model.parameters()).dtype
        clean_image_embeds = _patchify_5d(
            pixel_values_low.to(dtype),
            self.tuna_model.vision_encoder,
            self.tuna_model.config.reshape_frame_to_batch_dim,
        )
        return clean_image_embeds

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass for training"""
        # Extract batch data
        weight_type = torch.bfloat16 if self.dtype == "bf16" else torch.float32
        text_tokens = batch["text_tokens"]
        text_labels = batch["text_labels"]
        pixel_values = batch["images"]
        pixel_values_low = batch.get("images_clip", None)
        text_masks = batch["text_masks"]
        image_masks = batch["image_masks"]
        modality_positions = batch["modality_positions"]
        data_type = batch["data_type"]

        # Handle interleaved data
        if data_type[0] == "mmu_interleaved" or data_type[0] == "edit_interleaved":
            b, n = pixel_values.shape[:2]
            pixel_values = rearrange(pixel_values, "b n c h w -> (b n) c h w")
            data_type = data_type * n
        if data_type[0] != "mmu_text":
            # Prepare image latents and labels
            image_latents, t, image_labels, image_masks, image_latents_clean = (
                self.prepare_latents_and_labels(pixel_values, data_type, image_masks)
            )
        else:
            image_latents = None
            t = torch.tensor([0.0] * text_tokens.shape[0], device=text_tokens.device)
            image_labels = None
            image_masks = None

        # Prepare clean image embeddings for reconstruction conditioning
        clean_image_embeds = None
        if pixel_values_low is not None:
            clean_image_embeds = self.prepare_clean_image_embeds(pixel_values_low)

        # Create attention mask
        block_mask, block_mask_diffhead = self.create_attention_mask(
            text_tokens.size(0),
            text_tokens.size(1),
            modality_positions,
            text_tokens.device,
            weight_type,
        )

        # Forward pass through the model
        logits, loss_ntp, loss_flow, loss_disp = self.tuna_model(
            text_tokens=text_tokens,
            image_latents=image_latents,
            t=t.to(weight_type),
            attention_mask=block_mask,
            diffhead_attention_mask=block_mask_diffhead,
            text_masks=text_masks,
            image_masks=image_masks,
            text_labels=text_labels,
            image_labels=image_labels,
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=text_tokens.size(1),
            device=text_tokens.device,
            clean_image_embeds=clean_image_embeds,
        )

        # Compute total loss
        total_loss = self.flow_coeff * loss_flow + self.ntp_coeff * loss_ntp
        if self.use_disp:
            total_loss += 0.25 * loss_disp

        outputs = {
            "loss": total_loss,
            "loss_ntp": loss_ntp,
            "loss_flow": loss_flow,
            "loss_disp": loss_disp,
            "logits": logits,
            "recons_images": None,
        }

        return outputs
