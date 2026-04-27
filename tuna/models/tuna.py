# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# coding=utf-8
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
    RMSNorm,
    TimestepEmbedder,
)
from tuna.models._common import build_rope
from tuna.models._inner_base import SiglipMixin, TunaInnerBase
from tuna.models._wrapper_base import TunaWrapperBase
from tuna.models.backbones.qwen2 import Qwen2ForCausalLM
from tuna.models.misc import (
    get_text_tokenizer,
    next_token_prediction,
    velocity_prediction,
)
from tuna.models.vae.wan22_vae import Wan2_2_VAE
from transformers import SiglipVisionConfig, SiglipVisionModel
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer

logger: logging.Logger = logging.getLogger(__name__)


class Tuna(SiglipMixin, TunaInnerBase):
    """Latent-diffusion Tuna variant (variant A).

    Architecture:
      - SigLIP2 vision encoder runs on raw pixel values (frozen image features).
      - WAN 2.2 VAE encodes pixels into a 48-channel latent and decodes back.
      - Latents are the diffusion target; SigLIP features condition the LLM.

    Note: the original tuna code bundled SigLIP and WAN-VAE behind a single
    ``SiglipModelWithVAE`` wrapper that fed VAE latents directly into SigLIP.
    Tuna decouples the two: SigLIP works on pixels while the VAE handles the
    diffusion latent space independently.
    """

    @register_to_config
    def __init__(
        self,
        siglip_model_id: str = "google/siglip2-so400m-patch16-384",
        # Override siglip config to match the trained checkpoint shape.
        # For tuna post-VAE training (e.g. uni_siglip2_wan2_2_512_7b):
        #   siglip_image_size=32, siglip_patch_size=1, siglip_num_channels=48
        # → patch_embed [1152, 48, 1, 1], position_embed [1024, 1152].
        siglip_image_size: int = 384,
        siglip_patch_size: int = 16,
        siglip_num_channels: int = 3,
        siglip_feature_layer: int = -1,
        vae_model_id: str = "Wan-AI/Wan2.2-VAE",
        llm_vocab_size=None,
        llm_model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
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
        clip_latent_dim=1152,
        num_diffusion_layers=10,
        add_aspect_ratio_embeds=True,
        add_time_embeds=True,
        use_disp=False,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        model_args=None,
        **kwargs,
    ):
        super().__init__()
        self.use_disp = use_disp
        self.model_args = model_args
        self.siglip_model_id = siglip_model_id
        self.vae_model_id = vae_model_id

        # LLM backbone (Qwen2.5)
        llm_config = AutoConfig.from_pretrained(llm_model_path)
        if init_llm_from_config:
            self.tuna = Qwen2ForCausalLM(llm_config)
        else:
            self.tuna = Qwen2ForCausalLM.from_pretrained(
                llm_model_path, attn_implementation="sdpa"
            )
        self.tuna.resize_token_embeddings(llm_vocab_size)

        # Vision encoder — bare SigLIP transformer (no outer wrapper layer in
        # state_dict prefix), shaped to match the tuna training config
        # (see Tuna2RPixelModel for the full rationale).
        siglip_config = SiglipVisionConfig.from_pretrained(siglip_model_id)
        siglip_config.image_size = siglip_image_size
        siglip_config.patch_size = siglip_patch_size
        siglip_config.num_channels = siglip_num_channels
        siglip_config.vision_use_head = False
        self.vision_model = SiglipVisionTransformer(siglip_config)
        self.vision_model.post_layernorm = nn.Identity()
        n_keep = (
            siglip_feature_layer + 1
            if siglip_feature_layer >= 0
            else siglip_config.num_hidden_layers + siglip_feature_layer + 1
        )
        self.vision_model.encoder.layers = self.vision_model.encoder.layers[:n_keep]
        try:
            pretrained = SiglipVisionModel.from_pretrained(
                siglip_model_id, ignore_mismatched_sizes=True
            )
            own_sd = self.vision_model.state_dict()
            warm = {
                k: v
                for k, v in pretrained.vision_model.state_dict().items()
                if k in own_sd and own_sd[k].shape == v.shape
            }
            self.vision_model.load_state_dict(warm, strict=False)
            logger.info(
                f"SigLIP warm-init: loaded {len(warm)}/{len(own_sd)} matching keys"
            )
            del pretrained
        except Exception as e:
            logger.warning(f"Could not warm-init SigLIP from pretrained: {e}")

        # WAN 2.2 VAE — latent encoder/decoder
        self.vae = Wan2_2_VAE.from_pretrained(vae_model_id)

        self.register_buffer(
            "image_position_ids",
            torch.arange(image_latent_height * image_latent_width).expand((1, -1)),
            persistent=False,
        )

        self.siglip_proj = nn.Sequential(
            RMSNorm(clip_latent_dim),
            nn.Linear(clip_latent_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        tuna_cfg = (
            getattr(model_args, "tuna_config", model_args)
            if model_args is not None
            else None
        )
        # Diffusion head for generation
        separate_diffusion_head_config = (
            getattr(tuna_cfg, "separate_diffusion_head_config", False)
            if tuna_cfg is not None
            else False
        )
        if separate_diffusion_head_config and tuna_cfg is not None:
            self.diffusion_head_config = DiffusionHeadConfig(
                hidden_size=tuna_cfg.diffusion_head_hidden_size,
                num_attention_heads=tuna_cfg.diffusion_head_num_attention_heads,
                num_key_value_heads=tuna_cfg.diffusion_head_num_key_value_heads,
                intermediate_size=tuna_cfg.diffusion_head_intermediate_size,
                max_position_embeddings=self.tuna.config.max_position_embeddings,
                head_dim=tuna_cfg.diffusion_head_attention_head_dim,
            )
        else:
            self.diffusion_head_config = DiffusionHeadConfig(
                hidden_size=self.tuna.config.hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=self.tuna.config.intermediate_size,
                max_position_embeddings=self.tuna.config.max_position_embeddings,
                head_dim=getattr(self.tuna.config, "head_dim", None),
            )
        self.time_embed = TimestepEmbedder(self.diffusion_head_config.hidden_size)
        self.share_adaln = (
            getattr(tuna_cfg, "share_adaln", False) if tuna_cfg is not None else False
        )
        if self.share_adaln:
            # Shared adaLN projection for diffusion head blocks (replaces per-block adaLN_modulation)
            self.shared_adaln = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    self.diffusion_head_config.hidden_size,
                    6 * self.diffusion_head_config.hidden_size,
                    bias=True,
                ),
            )
            nn.init.zeros_(self.shared_adaln[1].weight)
            nn.init.zeros_(self.shared_adaln[1].bias)
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
                ModulatedAttentionBlock(
                    self.diffusion_head_config, layer_idx, model_args=model_args
                )
                for layer_idx in range(num_diffusion_layers)
            ]
        )

        self.diffusion_head_b = FinalLayer(
            self.diffusion_head_config.hidden_size, patch_size, image_latent_dim
        )

        tuna_total = sum(p.numel() for p in self.tuna.parameters())
        diffusion_head_a_total = sum(
            p.numel() for p in self.diffusion_head_a.parameters()
        )
        diffusion_head_b_total = sum(
            p.numel() for p in self.diffusion_head_b.parameters()
        )
        logger.info(
            f"tuna total parameters: {tuna_total:,} ({tuna_total / 1e6:.2f}M)"
        )
        logger.info(
            f"diffusion_head_a total parameters: {diffusion_head_a_total:,} ({diffusion_head_a_total / 1e6:.2f}M)"
        )
        logger.info(
            f"diffusion_head_b total parameters: {diffusion_head_b_total:,} ({diffusion_head_b_total / 1e6:.2f}M)"
        )

        self.gradient_checkpointing = False
        if gradient_checkpointing:
            self.gradient_checkpointing = True
            self._gradient_checkpointing_func = functools.partial(
                checkpoint, **gradient_checkpointing_kwargs
            )
            if hasattr(self.vision_model, "gradient_checkpointing_enable"):
                self.vision_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
            self.tuna.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize projection layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Don't reset SigLIP parameters - keep pretrained weights
        _basic_init(self.siglip_proj)
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
        pixel_values=None,
        siglip_spatial_shapes=None,
        siglip_pixel_attention_mask=None,
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
        input_embeds = self.tuna.model.embed_tokens(text_tokens)
        dtype = input_embeds.dtype

        (
            image_embeds,
            time_embeds,
            time_embeds_proj,
            height_embeds_proj,
            width_embeds_proj,
            rope_3d,
        ) = self._prepare_embeds(
            image_latents,
            pixel_values,
            siglip_spatial_shapes,
            siglip_pixel_attention_mask,
            t,
            device,
            dtype,
        )

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
        # Diffusion head to predict vector fields
        if hasattr(self, "diff_proj"):
            last_hidden_states = self.diff_proj(last_hidden_states)

        if diffhead_attention_mask is None:
            diffhead_attention_mask = attention_mask
        # Compute adaLN input for diffusion head blocks
        if self.share_adaln:
            adaln_input = self.shared_adaln(time_embeds)  # (B*num_imgs, 6*D)
        else:
            adaln_input = time_embeds
        act = []
        for layer in self.diffusion_head_a:
            if self.gradient_checkpointing and self.training:
                last_hidden_states = self._gradient_checkpointing_func(
                    layer,
                    hidden_states=last_hidden_states,
                    adaln_input=adaln_input,
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
                    adaln_input=adaln_input,
                    attention_mask=diffhead_attention_mask,
                    position_ids=position_ids,
                    rope_3d=rope_3d,
                    modality_positions=modality_positions,
                )[0]
            act.append(last_hidden_states)
        v_pred = self.diffusion_head_b(
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
            loss_ntp = next_token_prediction(
                logits, text_labels, self.config.llm_vocab_size
            )
            loss_flow = velocity_prediction(
                v_pred, new_image_labels[: v_pred.shape[0]], image_masks
            )
            if self.use_disp:
                loss_disp = self.disp_loss(act[-1])
            return logits, loss_ntp, loss_flow, loss_disp

        else:
            # Inference mode - return velocity predictions
            v_pred_ = []
            num_imgs = 0
            for i, modality_batch in enumerate(modality_positions):
                for _, (offset, length) in enumerate(modality_batch):
                    if length == 0:
                        break
                    else:
                        v_pred_.append(v_pred[i, offset : offset + length])
                        num_imgs += 1
            v_pred_ = torch.stack(v_pred_)

            # Remove the time embedding
            if self.config.add_time_embeds and self.config.add_aspect_ratio_embeds:
                v_pred_ = v_pred_[:, 3:, :]
            elif self.config.add_time_embeds:
                v_pred_ = v_pred_[:, 1:, :]

            # Unpatchify
            v_pred_ = self.unpatchify(v_pred_, h_, w_, T=T)

            v_pred_ = v_pred_.permute(0, 3, 1, 2)
            v_pred_ = v_pred_.reshape(
                num_imgs,
                self.config.image_latent_dim,
                T,
                h_ * self.config.patch_size,
                w_ * self.config.patch_size,
            )

            return logits, v_pred_

    def _prepare_embeds(
        self,
        image_latents,
        pixel_values,
        siglip_spatial_shapes,
        siglip_pixel_attention_mask,
        t,
        device,
        dtype,
    ):
        image_embeds = None
        rope_3d = None
        h_ = w_ = None
        b = None
        if image_latents is not None:
            b, c, T, h, w = image_latents.shape
            rope_3d = (
                build_rope(
                    latent_shape=[T, h, w],
                    patch_size=1,
                    attention_head_dim=self.diffusion_head_config.head_dim,
                )
                .to(device)
                .to(dtype)
            )
            p = self.config.patch_size
            h_, w_ = h // p, w // p

            # Drive SigLIP with image_latents (tuna post-VAE convention):
            # the diffusion target/noise IS the SigLIP input. Falls back to
            # explicit pixel_values when the caller provides one (e.g.
            # conditioning on a separate clean image). Without this path the
            # t2i-from-noise inference would crash with image_embeds=None.
            siglip_input = pixel_values if pixel_values is not None else image_latents
            image_embeds_siglip = self.encode_pixels_with_siglip(
                siglip_input.to(dtype),
                spatial_shapes=siglip_spatial_shapes,
                pixel_attention_mask=siglip_pixel_attention_mask,
            )
            image_embeds = self.siglip_proj(image_embeds_siglip)
            # For video (T>1), encode_pixels_with_siglip returns [B*T, N, D]
            # (each frame processed independently). Reshape back to [B, T*N, D]
            # so _prepare_input can index by the full video token count.
            if T > 1 and image_embeds.shape[0] == b * T:
                image_embeds = image_embeds.reshape(b, T * image_embeds.shape[1], -1)

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
        pixel_values=None,
        siglip_spatial_shapes=None,
        siglip_pixel_attention_mask=None,
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
                pixel_values_cond = pixel_values_uncond = None
                if pixel_values is not None:
                    pixel_values_cond, pixel_values_uncond = torch.chunk(pixel_values, 2)

                # First forward pass (conditional)
                _, v_cond = self(
                    text_tokens_cond,
                    image_latents=image_latents_cond,
                    pixel_values=pixel_values_cond,
                    siglip_spatial_shapes=siglip_spatial_shapes,
                    siglip_pixel_attention_mask=siglip_pixel_attention_mask,
                    t=t_cond,
                    attention_mask=attention_mask,
                    diffhead_attention_mask=diffhead_attention_mask,
                    modality_positions=modality_positions,
                    first_frame_as_cond=first_frame_as_cond,
                    only_denoise_last_image=only_denoise_last_image,
                    guidance_scale=guidance_scale,
                    output_hidden_states=True,
                    max_seq_len=max_seq_len,
                )

                # Second forward pass (unconditional)
                _, v_uncond = self(
                    text_tokens_uncond,
                    image_latents=image_latents_uncond,
                    pixel_values=pixel_values_uncond,
                    siglip_spatial_shapes=siglip_spatial_shapes,
                    siglip_pixel_attention_mask=siglip_pixel_attention_mask,
                    t=t_uncond,
                    attention_mask=attention_mask,
                    diffhead_attention_mask=diffhead_attention_mask,
                    modality_positions=modality_positions,
                    first_frame_as_cond=first_frame_as_cond,
                    only_denoise_last_image=only_denoise_last_image,
                    guidance_scale=guidance_scale,
                    output_hidden_states=True,
                    max_seq_len=max_seq_len,
                )
            else:
                # Original single forward pass (batch_size=2)
                _, v = self(
                    text_tokens,
                    image_latents=image_latents,
                    pixel_values=pixel_values,
                    siglip_spatial_shapes=siglip_spatial_shapes,
                    siglip_pixel_attention_mask=siglip_pixel_attention_mask,
                    t=t,
                    attention_mask=attention_mask,
                    diffhead_attention_mask=diffhead_attention_mask,
                    modality_positions=modality_positions,
                    first_frame_as_cond=first_frame_as_cond,
                    only_denoise_last_image=only_denoise_last_image,
                    guidance_scale=guidance_scale,
                    output_hidden_states=True,
                    max_seq_len=max_seq_len,
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
        pixel_values=None,
        siglip_spatial_shapes=None,
        siglip_pixel_attention_mask=None,
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
                pixel_values=pixel_values,
                siglip_spatial_shapes=siglip_spatial_shapes,
                siglip_pixel_attention_mask=siglip_pixel_attention_mask,
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

class TunaModel(TunaWrapperBase):
    """High-level training wrapper for :class:`Tuna` (variant A).

    Hydra-instantiable. Encodes pixel values to latents via the WAN VAE, then
    runs a flow-matching training step over the latent space with SigLIP2
    pixel features as conditioning.
    """

    def __init__(
        self,
        # SigLIP2 + VAE configuration
        siglip_model_id: str = "google/siglip2-so400m-patch16-384",
        siglip_image_size: int = 384,
        siglip_patch_size: int = 16,
        siglip_num_channels: int = 3,
        siglip_feature_layer: int = -1,
        vae_model_id: str = "Wan-AI/Wan2.2-VAE",
        # Tuna / LLM configuration
        llm_model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
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
        clip_latent_dim: int = 1152,
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
        # Precision options - cast model components to bf16 for FSDP compatibility
        model_to_bf16: bool = False,
        llm_to_bf16: bool = False,
        diffusion_head_to_bf16: bool = False,
        vision_encoder_to_bf16: bool = False,
        vae_to_bf16: bool = False,
        # Model args for passing to inner models
        model_args: Optional[Any] = None,
    ) -> None:
        super().__init__()

        # Store model_args for passing to inner models
        self.model_args = model_args

        # SigLIP2 + VAE
        self.siglip_model_id = siglip_model_id
        self.siglip_image_size = siglip_image_size
        self.siglip_patch_size = siglip_patch_size
        self.siglip_num_channels = siglip_num_channels
        self.siglip_feature_layer = siglip_feature_layer
        self.vae_model_id = vae_model_id

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
        self.clip_latent_dim = clip_latent_dim
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
        self.gradient_checkpointing_kwargs: dict[str, Any] | None = (
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

        # Device and dtype
        self.dtype = dtype
        # Precision options for bf16 casting
        self.model_to_bf16 = model_to_bf16
        self.llm_to_bf16 = llm_to_bf16
        self.diffusion_head_to_bf16 = diffusion_head_to_bf16
        self.vision_encoder_to_bf16 = vision_encoder_to_bf16
        self.vae_to_bf16 = vae_to_bf16

        # Build models
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.build_models()

    def build_models(self) -> None:
        """Initialize all model components"""
        from tuna.models.transport.define import create_transport
        from tuna.models.transport.transport import Sampler

        # Initialize text tokenizer
        self.text_tokenizer, self.tuna_token_ids = get_text_tokenizer(
            self.llm_model_path,
            add_tuna_tokens=True,
            return_tuna_token_ids=True,
        )
        self.llm_vocab_size = len(self.text_tokenizer)

        # Initialize Tuna single-path model
        model_config = {
            "siglip_model_id": self.siglip_model_id,
            "siglip_image_size": self.siglip_image_size,
            "siglip_patch_size": self.siglip_patch_size,
            "siglip_num_channels": self.siglip_num_channels,
            "siglip_feature_layer": self.siglip_feature_layer,
            "vae_model_id": self.vae_model_id,
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
            "clip_latent_dim": self.clip_latent_dim,
            "add_time_embeds": self.add_time_embeds,
            "add_aspect_ratio_embeds": self.add_aspect_ratio_embeds,
            "num_diffusion_layers": self.flow_head_num,
            "gradient_checkpointing": self.gradient_checkpointing,
            "gradient_checkpointing_kwargs": self.gradient_checkpointing_kwargs,
            "use_disp": self.use_disp,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "model_args": self.model_args,
        }
        self.tuna_model = Tuna(**model_config)
        if self.load_stage1_model is not None and self.load_stage1_model != "no":
            checkpoint = torch.load(self.load_stage1_model, map_location="cpu")
            missing_keys, unexpected_keys = self.tuna_model.load_state_dict(
                checkpoint, strict=False
            )
            if missing_keys:
                logger.warning(
                    f"Missing keys when loading stage1 model (these will use default initialization): {missing_keys}"
                )
            if unexpected_keys:
                logger.warning(
                    f"Unexpected keys when loading stage1 model (these will be ignored): {unexpected_keys}"
                )
            logger.info(
                f"Loaded stage1 model from {self.load_stage1_model} "
                f"(missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)})"
            )

        # Cast entire model to bf16 for FSDP compatibility
        if self.model_to_bf16:
            logger.info(
                "Casting entire model (LLM, diffusion head, vision encoder, VAE) to bf16"
            )
            self.tuna_model.to(torch.bfloat16)
        else:
            # Granular bf16 casting for individual components
            if self.llm_to_bf16:
                logger.info("Casting LLM decoder to bf16")
                self.tuna_model.tuna.to(torch.bfloat16)

            if self.diffusion_head_to_bf16:
                logger.info("Casting diffusion head components to bf16")
                self.tuna_model.diffusion_head_a.to(torch.bfloat16)
                self.tuna_model.diffusion_head_b.to(torch.bfloat16)
                self.tuna_model.time_embed.to(torch.bfloat16)
                self.tuna_model.siglip_proj.to(torch.bfloat16)
                if hasattr(self.tuna_model, "diff_proj"):
                    self.tuna_model.diff_proj.to(torch.bfloat16)
                if hasattr(self.tuna_model, "time_embed_proj"):
                    self.tuna_model.time_embed_proj.to(torch.bfloat16)
                if hasattr(self.tuna_model, "aspect_ratio_embed"):
                    self.tuna_model.aspect_ratio_embed.to(torch.bfloat16)
                if hasattr(self.tuna_model, "ar_embed_proj"):
                    self.tuna_model.ar_embed_proj.to(torch.bfloat16)
                if hasattr(self.tuna_model, "shared_adaln"):
                    self.tuna_model.shared_adaln.to(torch.bfloat16)

            if self.vision_encoder_to_bf16:
                logger.info("Casting vision encoder (SigLIP) to bf16")
                self.tuna_model.vision_model.to(torch.bfloat16)

            if self.vae_to_bf16:
                logger.info("Casting VAE to bf16")
                self.tuna_model.vae.to(torch.bfloat16)

        self._freeze_params(self.tuna_model, self.frozen_params)

        # Initialize transport for flow matching
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

    @torch.no_grad()
    def prepare_latents_and_labels(
        self,
        pixel_values: torch.Tensor,
        data_type: List[str],
        image_masks: torch.Tensor,
        use_feature: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare image latents and labels for training."""

        if len(pixel_values.shape) == 4:
            pixel_values = pixel_values.unsqueeze(2)

        # Cast pixel_values to bf16 if model is in bf16 mode to match VAE weights
        weight_dtype = (
            torch.bfloat16
            if (self.vae_to_bf16 or self.model_to_bf16)
            else torch.float32
        )
        pixel_values = pixel_values.to(weight_dtype)

        if use_feature:
            image_latents = pixel_values
        else:
            # Wan2_2_VAE.sample produces a sample from the posterior; use
            # deterministic=False to match the original training-time behaviour.
            image_latents = self.tuna_model.vae.sample(
                pixel_values, deterministic=False
            )

        # Prepare timesteps, noise, and targets
        t_list, xt_list, ut_list, masks = [], [], [], []

        for i, tp in enumerate(data_type):
            # Special handling for edit_interleaved: first image no noise, second image with noise
            if tp == "edit_interleaved":
                is_first_image = i % 2 == 0
                max_t0 = self.und_max_t0 if is_first_image else None
            else:
                max_t0 = (
                    self.und_max_t0
                    if tp
                    in ["mmu", "mmu_vid", "mmu_interleaved", "mmu_text"]
                    else None
                )

            # Sample timestep and noise
            t, x0, x1 = self.transport.sample(image_latents[i][None], max_t0)
            # Get noisy latents and velocity targets
            t, xt, ut = self.transport.path_sampler.plan(t, x0, x1)

            t_list.append(t)
            xt_list.append(xt)
            ut_list.append(ut)

            if image_masks is None:
                # Inference paths (t2i_edit) pass no image_masks; nothing to
                # accumulate, the final masks output stays None.
                continue

            # Handle masks for understanding tasks
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
        ut = torch.cat(ut_list, dim=0)
        masks = torch.cat(masks, dim=0) if masks else image_masks

        # Always return both clean and noisy latents for consistency
        return xt, t, ut, masks, image_latents

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass for training"""
        # Extract batch data
        weight_type = torch.bfloat16 if self.dtype == "bf16" else torch.float32
        text_tokens = batch["text_tokens"]
        text_labels = batch["text_labels"]
        pixel_values = batch["images"]
        text_masks = batch["text_masks"]
        image_masks = batch["image_masks"]
        modality_positions = batch["modality_positions"]
        data_type = batch["data_type"]
        # SigLIP-resized image (matches tuna `images_clip` convention) plus
        # the optional SigLIP2 spatial shape / attention mask companions.
        images_clip = batch.get("images_clip", None)
        siglip_spatial_shapes = batch.get("siglip_spatial_shapes", None)
        siglip_pixel_attention_mask = batch.get("siglip_pixel_attention_mask", None)

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

        # Variant A: SigLIP expects VAE-latent input (48ch), not RGB.
        # Encode images_clip through the frozen VAE before passing to SigLIP.
        if images_clip is not None and hasattr(self.tuna_model, "vae"):
            with torch.no_grad():
                clip_input = images_clip.to(weight_type)
                if clip_input.dim() == 4:
                    clip_input = clip_input.unsqueeze(2)
                images_clip = self.tuna_model.vae.sample(
                    clip_input, deterministic=True
                )

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
            pixel_values=images_clip,
            siglip_spatial_shapes=siglip_spatial_shapes,
            siglip_pixel_attention_mask=siglip_pixel_attention_mask,
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
