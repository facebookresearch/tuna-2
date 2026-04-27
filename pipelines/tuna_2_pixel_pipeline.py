# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# coding=utf-8
# pyre-unsafe

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from tuna.models.misc import prepare_gen_input, prepare_gen_input_edit
from tuna.pipelines._common import denorm
from tuna.pipelines._pipeline_base import TunaPipelineBase


def get_hyper_params(
    text_tokenizer,
    tuna_token_ids,
    use_chat_template=False,
    add_aspect_ratio_embeds=False,
    height=512,
    width=512,
    generation_mode="t2i",
    latent_frames=1,
):
    """
    Extract hyperparameters from config.

    Args:
        text_tokenizer: Text tokenizer
        tuna_token_ids: Tuna token IDs dictionary

    Returns:
        Tuple of hyperparameters
    """
    # Extract basic parameters
    if width == "auto":
        width = 512
    if height == "auto":
        height = 512
    latent_width = width // 16
    latent_height = height // 16
    num_image_tokens = (
        latent_width * latent_height + int(add_aspect_ratio_embeds) * 2 + 1
    )
    num_video_tokens = (
        latent_width * latent_height * latent_frames
        + int(add_aspect_ratio_embeds) * 2
        + 1
    )
    if generation_mode == "t2i":
        calculated_seq_len = (
            latent_width * latent_height * 1 + int(add_aspect_ratio_embeds) * 2 + 1024
        )
        # Choose between 2048 or 5129 based on which is closer to calculated value
        diff_2048 = abs(calculated_seq_len - 2048)
        diff_5129 = abs(calculated_seq_len - 5129)
        max_seq_len = 2048 if diff_2048 <= diff_5129 else 5129
        max_text_len = (
            max_seq_len - num_image_tokens - 33
            if use_chat_template
            else max_seq_len - num_image_tokens - 4
        )
        image_latent_dim = 48
        patch_size = 1
    elif generation_mode == "t2i_pixel":
        calculated_seq_len = (
            latent_width * latent_height * 1 + int(add_aspect_ratio_embeds) * 2 + 1024
        )
        # Choose between 2048 or 5129 based on which is closer to calculated value
        diff_2048 = abs(calculated_seq_len - 2048)
        diff_5129 = abs(calculated_seq_len - 5129)
        max_seq_len = 2048 if diff_2048 <= diff_5129 else 5129
        max_text_len = (
            max_seq_len - num_image_tokens - 33
            if use_chat_template
            else max_seq_len - num_image_tokens - 4
        )
        image_latent_dim = 3
        patch_size = 16
    elif generation_mode == "mmu":
        max_seq_len = num_image_tokens - 1 + 1024
        max_text_len = (
            max_seq_len - num_image_tokens - 33
            if use_chat_template
            else max_seq_len - num_image_tokens - 4
        )
        image_latent_dim = 48
        patch_size = 1
    elif generation_mode == "edit":
        max_seq_len = 4096
        max_text_len = max_seq_len - num_image_tokens * 2 - 6
        image_latent_dim = 3
        patch_size = 1
    else:
        max_seq_len = num_video_tokens - 1 + 1024
        max_text_len = (
            max_seq_len - num_video_tokens - 33
            if use_chat_template
            else max_seq_len - num_video_tokens - 4
        )

        image_latent_dim = 48
        patch_size = 1

    # Token IDs
    pad_id = text_tokenizer.pad_token_id
    bos_id = tuna_token_ids["bos_id"]
    eos_id = tuna_token_ids["eos_id"]
    boi_id = tuna_token_ids["boi_id"]
    eoi_id = tuna_token_ids["eoi_id"]
    bov_id = tuna_token_ids["bov_id"]
    eov_id = tuna_token_ids["eov_id"]
    img_pad_id = tuna_token_ids["img_pad_id"]
    vid_pad_id = tuna_token_ids["vid_pad_id"]

    # Guidance scale
    guidance_scale = 7.5
    return (
        num_image_tokens,
        num_video_tokens,
        max_seq_len,
        max_text_len,
        image_latent_dim,
        patch_size,
        latent_width,
        latent_height,
        pad_id,
        bos_id,
        eos_id,
        boi_id,
        eoi_id,
        bov_id,
        eov_id,
        img_pad_id,
        vid_pad_id,
        guidance_scale,
    )


class Tuna2PixelPipeline(TunaPipelineBase):
    """
    Pipeline for the Tuna2Pixel (variant C) model: pure patchify, no separate
    vision encoder and no VAE. The model operates directly on pixel patches.

    Exposes:
        * ``t2i`` — text-to-image generation (also drives reconstruction when
          ``reca_mode=True`` is passed via the ``reconstruct`` wrapper).
        * ``t2i_edit`` — image editing conditioned on a source image plus an
          instruction prompt.
        * ``mmu`` — multimodal understanding (image + text -> text answer).
    """

    def __init__(
        self,
        model,
        text_tokenizer=None,
        tuna_token_ids=None,
        config=None,
        weight_dtype=torch.float32,
        device="cuda",
        use_tf32=True,
        use_chat_template=True,
        add_aspect_ratio_embeds=False,
        second_time=False,
        height=512,
        width=512,
        latent_frames=1,
        generation_mode="t2i",
    ):
        self.model = model
        self.latent_frames = latent_frames
        self.generation_mode = generation_mode
        self.text_tokenizer = text_tokenizer
        self.tuna_token_ids = tuna_token_ids
        self.config = config
        self.device = device
        self.weight_dtype = weight_dtype
        self.use_chat_template = use_chat_template
        self.add_aspect_ratio_embeds = add_aspect_ratio_embeds
        self.num_visual_tokens = 1009
        self.height = height
        self.width = width
        self.second_time = second_time
        self.conversation = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
        ]

        # Enable TF32 for faster training on Ampere GPUs (A100 and RTX 30 series).
        if use_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

        # Initialize hyperparameters if config is provided
        if text_tokenizer and tuna_token_ids:
            self._init_hyperparams(self.height, self.width)

    def _init_hyperparams(self, height, width):
        """Initialize hyperparameters from config"""
        (
            self.num_image_tokens,
            self.num_video_tokens,
            self.max_seq_len,
            self.max_text_len,
            self.image_latent_dim,
            self.patch_size,
            self.latent_width,
            self.latent_height,
            self.pad_id,
            self.bos_id,
            self.eos_id,
            self.boi_id,
            self.eoi_id,
            self.bov_id,
            self.eov_id,
            self.img_pad_id,
            self.vid_pad_id,
            self.guidance_scale,
        ) = get_hyper_params(
            self.text_tokenizer,
            self.tuna_token_ids,
            self.use_chat_template,
            self.add_aspect_ratio_embeds,
            height,
            width,
            self.generation_mode,
            self.latent_frames,
        )
        if self.generation_mode == "t2i":
            self.num_visual_tokens = self.num_image_tokens
        else:
            self.num_visual_tokens = self.num_video_tokens

    @torch.no_grad()
    def t2i(
        self,
        prompts: Union[str, List[str]],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        transport=None,
        sampler=None,
        sampling_method: str = "euler",
        atol: float = 1e-6,
        rtol: float = 1e-3,
        reverse: bool = False,
        time_shifting_factor: float = 3.0,
        noise_level: float = 1.0,
        noise_scale: float = 1.0,
        input_latents: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        reca_mode: bool = False,
        **kwargs,
    ) -> List[Image.Image]:
        """
        Generate images from text prompts.

        Args:
            prompts: Text prompt(s) for image generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            generator: Random generator for reproducible generation
            transport: Transport object for sampling
            sampler: Sampler object
            sampling_method: Sampling method (e.g., "dopri5", "euler")
            atol: Absolute tolerance for ODE solver
            rtol: Relative tolerance for ODE solver
            reverse: Whether to reverse the sampling process
            time_shifting_factor: Time shifting factor for sampling
            noise_level: Noise level to start from (0.0=clean, 1.0=pure noise, 0.5=halfway)
            input_latents: Optional input latents to add noise to. If None, uses random noise.
            pixel_values: Source pixels. Used as a clean conditioning signal during
                          generation, and (when ``reca_mode=True``) as the noise init.
            negative_prompt: Optional negative prompt for CFG.
            reca_mode: If True, treat ``pixel_values`` as the noise initialization
                       (reconstruction). When False, ``pixel_values`` (if provided)
                       only contributes via ``prepare_clean_image_embeds``.

        Returns:
            List of generated PIL images
        """
        # Handle single prompt
        if isinstance(prompts, str):
            prompts = [prompts]

        batch_size = len(prompts)

        # Prepare text tokens and modality positions
        reca_pad_len = self.num_visual_tokens if pixel_values is not None else 0
        max_text_len = self.max_text_len
        max_seq_len = self.max_seq_len
        if reca_pad_len > 0:
            # Additive padding: text_tokens = prompt + [pad]*(reca_pad_len-1)
            # max_text_len must accommodate original prompt budget + reca padding
            max_text_len = self.max_text_len + reca_pad_len - 1
            max_seq_len = max_text_len + self.num_visual_tokens + 4

        (
            batch_text_tokens,
            batch_text_tokens_null,
            batch_modality_positions,
            batch_modality_positions_null,
        ) = prepare_gen_input(
            prompts,
            self.text_tokenizer,
            self.num_visual_tokens,
            self.bos_id,
            self.eos_id,
            self.boi_id,
            self.eoi_id,
            self.pad_id,
            self.img_pad_id,
            max_text_len,
            self.device,
            reca_pad_len=reca_pad_len,
            negative_prompt=negative_prompt,
        )

        from tuna.models.jit_utils import JiTSampler

        my_jit_sampler = JiTSampler(self.device, noise_scale=noise_scale)
        sampler = my_jit_sampler
        transport = my_jit_sampler
        if sampler is not None and transport is not None:
            sample_fn, t_start = sampler.sample_ode(
                sampling_method=sampling_method,
                num_steps=num_inference_steps,
                atol=atol,
                rtol=rtol,
                reverse=reverse,
                time_shifting_factor=time_shifting_factor,
                noise_level=noise_level,
            )

        # Initialize latents with controlled noise level
        if input_latents is not None:
            # Use provided latents as starting point
            x1 = input_latents.to(self.weight_dtype).to(self.device)
            if x1.shape[0] != batch_size:
                x1 = x1.repeat(batch_size, 1, 1, 1)

        # Reconstruction mode: pure-pixel variant has no encoder/VAE so use the
        # source pixels directly as the noise init.
        if reca_mode and pixel_values is not None:
            init = pixel_values.to(self.device, self.weight_dtype)
            if init.dim() == 4:
                init = init.unsqueeze(2)  # (B, C, 1, H, W)
            z = noise_scale * init
        else:
            z = noise_scale * torch.randn(
                (
                    batch_size,
                    self.image_latent_dim,
                    self.latent_frames,
                    self.latent_height * self.patch_size,
                    self.latent_width * self.patch_size,
                )
            ).to(self.weight_dtype).to(self.device)

        # Prepare inputs for classifier-free guidance
        if guidance_scale > 0:
            z = torch.cat([z, z], dim=0)
            text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
            modality_positions = torch.cat(
                [batch_modality_positions, batch_modality_positions_null], dim=0
            )
            bs = text_tokens.size(0)
            if self.second_time:
                modality_positions = batch_modality_positions
                bs = text_tokens.size(0) // 2
            attention_mask, diffhead_attention_mask = self.model.create_attention_mask(
                bs,
                max_seq_len,
                modality_positions,
                self.device,
                self.weight_dtype,
            )
        else:
            text_tokens = batch_text_tokens
            modality_positions = batch_modality_positions

            # Create attention mask
            attention_mask, diffhead_attention_mask = self.model.create_attention_mask(
                text_tokens.size(0),
                max_seq_len,
                modality_positions,
                self.device,
                self.weight_dtype,
            )

        # Prepare clean image embeddings for reconstruction conditioning
        clean_image_embeds = None
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device, self.weight_dtype)
            clean_image_embeds = self.model.prepare_clean_image_embeds(pixel_values)
            if guidance_scale > 0:
                clean_image_embeds = torch.cat(
                    [clean_image_embeds, clean_image_embeds], dim=0
                )

        model_kwargs = {
            "text_tokens": text_tokens,
            "attention_mask": attention_mask,
            "diffhead_attention_mask": diffhead_attention_mask,
            "modality_positions": modality_positions,
            "output_hidden_states": True,
            "max_seq_len": max_seq_len,
            "guidance_scale": guidance_scale,
            "second_time": self.second_time,
            "cfg_interval": (0, 1.0),
            "clean_image_embeds": clean_image_embeds,
        }

        # Sample using transport
        samples = sample_fn(
            z, self.model.tuna_model.t2i_generate, **model_kwargs
        )[-1]

        # Handle classifier-free guidance
        if guidance_scale > 0:
            samples = torch.chunk(samples, 2)[0]
        # Decode latents to images
        images = self._decode_latents(samples)
        return images

    @torch.no_grad()
    def t2i_edit(
        self,
        prompts: Union[str, List[str]],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        transport=None,
        sampler=None,
        sampling_method: str = "euler",
        atol: float = 1e-6,
        rtol: float = 1e-3,
        reverse: bool = False,
        time_shifting_factor: float = 3.0,
        noise_level: float = 1.0,
        noise_scale: float = 1.0,
        input_latents: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        **kwargs,
    ) -> List[Image.Image]:
        if isinstance(prompts, str):
            prompts = [prompts]
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device, self.weight_dtype)
        batch_size = len(prompts)
        b, n, c, h, w = pixel_values.shape
        self.num_visual_tokens = h // 16 * w // 16 + 1
        self.max_text_len = self.max_seq_len - self.num_visual_tokens * 2 - 6
        (
            batch_text_tokens,
            batch_text_tokens_null,
            batch_modality_positions,
            batch_modality_positions_null,
        ) = prepare_gen_input_edit(
            prompts,
            self.text_tokenizer,
            self.num_visual_tokens,
            self.bos_id,
            self.eos_id,
            self.boi_id,
            self.eoi_id,
            self.pad_id,
            self.img_pad_id,
            self.max_text_len,
            self.device,
            negative_prompt=negative_prompt,
        )

        from tuna.models.jit_utils import JiTSampler

        my_jit_sampler = JiTSampler(self.device, noise_scale=noise_scale)
        sampler = my_jit_sampler
        transport = my_jit_sampler
        if sampler is not None and transport is not None:
            sample_fn, t_start = sampler.sample_ode(
                sampling_method=sampling_method,
                num_steps=num_inference_steps,
                atol=atol,
                rtol=rtol,
                reverse=reverse,
                time_shifting_factor=time_shifting_factor,
                noise_level=noise_level,
            )
        if input_latents is not None:
            x1 = input_latents.to(self.weight_dtype).to(self.device)
            if x1.shape[0] != batch_size:
                x1 = x1.repeat(batch_size, 1, 1, 1)
        data_type = ["edit_interleaved"]
        pixel_values = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        data_type = data_type * n
        image_latents, t, image_labels, image_masks, image_latents = (
            self.model.prepare_latents_and_labels(pixel_values, data_type, image_masks)
        )
        _, _, _, latent_h, latent_w = image_latents.shape
        z = noise_scale * torch.randn(
            (
                batch_size,
                self.image_latent_dim,
                self.latent_frames,
                latent_h * self.patch_size,
                latent_w * self.patch_size,
            )
        ).to(self.weight_dtype).to(self.device)

        if guidance_scale > 0:
            z = torch.cat([z, z], dim=0)
            text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
            modality_positions = torch.cat(
                [batch_modality_positions, batch_modality_positions_null], dim=0
            )

            attention_mask, diffhead_attention_mask = self.model.create_attention_mask(
                text_tokens.size(0),
                self.max_seq_len,
                modality_positions,
                self.device,
                self.weight_dtype,
            )
        else:
            text_tokens = batch_text_tokens
            modality_positions = batch_modality_positions

            attention_mask, diffhead_attention_mask = self.model.create_attention_mask(
                text_tokens.size(0),
                self.max_seq_len,
                modality_positions,
                self.device,
                self.weight_dtype,
            )

        model_kwargs = {
            "text_tokens": text_tokens,
            "attention_mask": attention_mask,
            "diffhead_attention_mask": diffhead_attention_mask,
            "modality_positions": modality_positions,
            "output_hidden_states": True,
            "max_seq_len": self.max_seq_len,
            "guidance_scale": guidance_scale,
            "image_edit_original": image_latents[0:1].repeat(2, 1, 1, 1, 1),
        }

        samples = sample_fn(
            z, self.model.tuna_model.t2i_generate_edit, **model_kwargs
        )[-1]

        if guidance_scale > 0:
            samples = torch.chunk(samples, 2)[0]
        images = self._decode_latents(samples)
        return images

    def _decode_latents(self, latents: torch.Tensor) -> List[Image.Image]:
        """
        Decode latents to PIL images. The pure-pixel variant has no VAE, so the
        model output is already in pixel space.

        Args:
            latents: Latent representations to decode

        Returns:
            List of PIL images
        """

        if self.generation_mode == "t2i_pixel":
            images = latents
        elif hasattr(self, "vae_model") and self.vae_model is not None:
            device, dtype = latents.device, latents.dtype
            scale = self.model.tuna_model.vision_encoder.get_vae_scale(device, dtype)
            images = self.vae_model.decode(latents, scale=scale)
        else:
            images = latents

        if images.shape[2] == 1:
            images = images.squeeze(2)

        # Convert to PIL images
        if (
            self.generation_mode == "t2i"
            or self.generation_mode == "edit"
            or self.generation_mode == "t2i_pixel"
        ):
            images = denorm(images)
            pil_images = [Image.fromarray(image) for image in images]
            return pil_images
        else:
            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).to(
                torch.float32
            )
            images *= 255.0
            images = (
                images.permute(0, 2, 3, 4, 1).cpu().numpy().astype(np.uint8)
            )  # [B, T, H, W, C]

            frames = [images]

            return frames

