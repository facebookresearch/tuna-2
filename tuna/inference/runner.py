"""High-level inference runner for Tuna.

Builds a model + pipeline and dispatches to one of three modes:

* ``t2i``  — text-to-image generation (also drives reconstruction when
  ``reca_mode=True`` is set on the runner).
* ``edit`` — image editing conditioned on a source image plus an instruction.
* ``mmu``  — multimodal understanding (image + text -> text answer).

This is the OSS replacement for the tuna ``SimpleInference`` runner. All
Manifold / Model Store / video / inpainting code paths have been dropped;
only local-file checkpoint loading is supported.
"""

# (c) Meta Platforms, Inc. and affiliates. Apache-2.0.

from __future__ import annotations

# pyre-unsafe

import logging
import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as tvF
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from tuna.inference.checkpoint_loader import load_checkpoint
from tuna.models.vae.wan22_vae import Wan2_2_VAE
from tuna.pipelines.tuna_2_pixel_pipeline import Tuna2PixelPipeline
from tuna.pipelines.tuna_2r_pixel_pipeline import Tuna2RPixelPipeline
from tuna.pipelines.tuna_pipeline import TunaPipeline


logger: logging.Logger = logging.getLogger(__name__)

TorchDevice = str | torch.device


# ---------- Image utilities -------------------------------------------------


def _resize_center_crop(image: Image.Image, resolution: tuple[int, int]) -> Image.Image:
    """Resize to cover ``resolution`` (h, w) and center-crop the excess."""
    target_h, target_w = resolution
    src_w, src_h = image.size
    if src_w == 0 or src_h == 0:
        raise ValueError("Source image has zero dimension.")
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    image = image.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return image.crop((left, top, left + target_w, top + target_h))


def image_transform(
    image: Image.Image,
    resolution: tuple[int, int] = (512, 512),
    normalize: bool = True,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    centercrop: bool = True,
) -> torch.Tensor:
    if centercrop:
        image = _resize_center_crop(image, resolution)
    else:
        image = tvF.resize(
            image,
            [resolution[0], resolution[1]],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    tensor = transforms.ToTensor()(image)
    if normalize:
        tensor = transforms.Normalize(mean=mean, std=std, inplace=True)(tensor)
    return tensor


def tensor_to_pil(
    images: torch.Tensor | np.ndarray,
) -> list[Image.Image] | Image.Image:
    """Convert tensor images to PIL images."""
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    if images.ndim == 4:
        pil_images = []
        for img in images:
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img))
        return pil_images
    if images.shape[0] == 3:
        images = images.transpose(1, 2, 0)
    images = (images * 255).astype(np.uint8)
    return Image.fromarray(images)


def pil_to_tensor(images: list[Image.Image] | Image.Image) -> torch.Tensor:
    """Convert PIL image(s) to ``[B, C, H, W]`` (or ``[C, H, W]``) float tensor."""
    if isinstance(images, list):
        tensors = []
        for img in images:
            arr = np.array(img).astype(np.float32) / 255.0
            if arr.ndim == 3:
                tensors.append(torch.from_numpy(arr.transpose(2, 0, 1)))
            else:
                tensors.append(torch.from_numpy(arr))
        return torch.stack(tensors)
    arr = np.array(images).astype(np.float32) / 255.0
    if arr.ndim == 3:
        return torch.from_numpy(arr.transpose(2, 0, 1))
    return torch.from_numpy(arr)


# ---------- Inference runner -----------------------------------------------


class TunaInference:
    """High-level wrapper that loads a model, builds a pipeline, and dispatches.

    Args:
        inference_mode: One of ``"t2i"``, ``"edit"``, ``"mmu"``. The mode
            decides which pipeline method ``__call__`` dispatches to.
        model: A Tuna training-wrapper module (``TunaModel`` /
            ``Tuna2RPixelModel`` / ``Tuna2PixelModel``).
        use_ckpt: If ``True``, load weights from ``ckpt_path`` (a local
            ``.pt``/``.pth``/``.safetensors`` file or HuggingFace repo id).
        ckpt_path: Path or repo id for ``use_ckpt=True``.
        save_ckpt_path: Optional local path to dump the loaded model state to.
        device: Optional explicit device.
        config: Optional model config (passed through to the pipeline).
        weight_dtype: ``"float32"`` or ``"bf16"``; the model is moved to this
            dtype during init.
        pipe: Pipeline-class hint:
            * ``"TunaPipeline"`` for variant A (latent + WAN 2.2 VAE).
            * ``"Tuna2RPixelPipeline"`` for variant B (SigLIP-only pixel).
            * ``"Tuna2PixelPipeline"`` for variant C (pure patchify).
        Other args are pass-through to the underlying pipeline / generation
        loop. See the tuna ``SimpleInference`` for documentation; semantics
        are unchanged.
    """

    def __init__(
        self,
        inference_mode: str,
        model: nn.Module,
        use_ckpt: bool = False,
        ckpt_path: str | None = None,
        save_ckpt_path: str | None = None,
        device: TorchDevice | int | None = None,
        config: Any = None,
        weight_dtype: str = "float32",
        pipe: str | None = None,
        num_images_per_prompt: int = 1,
        use_chat_template: bool = False,
        add_aspect_ratio_embeds: bool = True,
        second_time: bool = False,
        height: int | str = 512,
        width: int | str = 512,
        centercrop: bool = True,
        # mmu utils
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        max_new_tokens: int = 512,
        generation_mode: str = "t2i",
        # video utils (kept for API-compat with the tuna pipeline; image-only)
        latent_frames: int = 1,
        shift: float = 3.0,
        # prompt utils
        prompt_file_path: str | None = None,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 50,
        reca_mode: bool = False,
        negative_prompt: str | None = None,
        noise_scale: float = 1.0,
        sampling_method: str = "euler",
        cfg_interval: tuple[float, float] | list[float] | None = None,
    ) -> None:
        self.inference_mode = inference_mode
        self.model = model
        self.config = config
        self.weight_dtype = torch.float32 if weight_dtype == "float32" else torch.bfloat16
        self.use_chat_template = use_chat_template
        self.add_aspect_ratio_embeds = add_aspect_ratio_embeds
        self.centercrop = centercrop
        self.second_time = second_time
        self.shift = shift

        t = 1
        a = int(add_aspect_ratio_embeds) * 2
        self.supported_resolutions = [
            ([512, 512], 1024 + t + a),
            ([448, 576], 1008 + t + a),
            ([576, 448], 1008 + t + a),
            ([384, 672], 1008 + t + a),
            ([672, 384], 1008 + t + a),
        ]

        if device is not None:
            if str(device).startswith("cuda") and str(device) != "cuda":
                torch.cuda.set_device(device)
            self.device: TorchDevice | int = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if use_ckpt:
            if ckpt_path is None:
                raise ValueError("use_ckpt=True but no ckpt_path provided.")
            self._load_local_ckpt(ckpt_path)
        if save_ckpt_path is not None:
            self.save_ckpt(save_ckpt_path)

        self.model = self.model.to(device=self.device, dtype=self.weight_dtype)
        # Keep the Wan VAE in fp32 — tuna decodes through a fp32 VAE
        # (frozen at train time, rebuilt fp32 at inference). bf16 here
        # would visibly drop decode quality.
        tuna_model = getattr(self.model, "tuna_model", self.model)
        vae_mod = getattr(tuna_model, "vae", None)
        if vae_mod is not None:
            vae_mod.to(torch.float32)
        self.model.eval()

        self.num_images_per_prompt = num_images_per_prompt
        self.generation_mode = generation_mode
        self.guidance_scale = guidance_scale
        self.latent_frames = latent_frames
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps

        self.reca_mode = reca_mode
        self.negative_prompt = negative_prompt
        self.noise_scale = noise_scale
        self.sampling_method = sampling_method
        self.cfg_interval = (
            tuple(cfg_interval) if cfg_interval is not None else (0.0, 1.0)
        )
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        self.prompt_file_path = prompt_file_path
        self._prompt_index = 0
        self._prompt_file_lines: list[str] | None = None
        if prompt_file_path:
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                self._prompt_file_lines = [line.strip() for line in f if line.strip()]

        self._init_pipeline(pipe=pipe)

    # ---- Checkpoint IO -----------------------------------------------------
    def save_ckpt(self, save_ckpt_path: str) -> None:
        """Dump the current model weights to a local ``.pt`` file."""
        os.makedirs(os.path.dirname(save_ckpt_path) or ".", exist_ok=True)
        inner = getattr(self.model, "tuna_model", self.model)
        torch.save(inner.state_dict(), save_ckpt_path)
        logger.info(f"Saved checkpoint to {save_ckpt_path}")

    def _load_local_ckpt(self, ckpt_path: str) -> None:
        """Load weights from a local file or HuggingFace repo."""
        target = getattr(self.model, "tuna_model", self.model)
        load_checkpoint(target, ckpt_path)
        logger.info(f"Loaded model from {ckpt_path}")

    # ---- Pipeline construction --------------------------------------------
    def _build_vae(self) -> nn.Module | None:
        """Build (or grab) the WAN 2.2 VAE for variant A.

        Tries ``self.model.tuna_model.vision_model.vae`` first (matches the
        original tuna attribute layout); if the decoder is missing, builds a
        fresh ``Wan2_2_VAE.from_pretrained(...)`` using the model's
        ``vae_model_id``.
        """
        tuna_model = getattr(self.model, "tuna_model", self.model)
        vae_model = getattr(tuna_model, "vae", None)
        if vae_model is None:
            vision_model = getattr(tuna_model, "vision_model", None)
            if vision_model is not None:
                vae_model = getattr(vision_model, "vae", None)

        if vae_model is None or isinstance(getattr(vae_model, "decoder", None), nn.Identity):
            vae_model_id = getattr(tuna_model, "vae_model_id", None) or "Wan-AI/Wan2.2-VAE"
            logger.info(f"Building Wan2_2_VAE from pretrained {vae_model_id}")
            vae_model = Wan2_2_VAE.from_pretrained(vae_model_id)
            vae_model.requires_grad_(False)
            vae_model.eval().to(self.device)
        return vae_model

    def _init_pipeline(self, pipe: str | None) -> None:
        """Initialise the pipeline for the chosen Tuna variant."""
        config = getattr(self.model, "config", self.config)
        tuna_model = getattr(self.model, "tuna_model", self.model)

        common_kwargs = {
            "model": self.model,
            "text_tokenizer": getattr(self.model, "text_tokenizer", None),
            "tuna_token_ids": getattr(self.model, "tuna_token_ids", None),
            "config": config,
            "weight_dtype": self.weight_dtype,
            "device": self.device,
            "use_tf32": True,
            "use_chat_template": self.use_chat_template,
            "add_aspect_ratio_embeds": self.add_aspect_ratio_embeds,
            "height": self.height,
            "width": self.width,
            "generation_mode": self.generation_mode,
            "latent_frames": self.latent_frames,
        }

        if pipe == "TunaPipeline":
            vae_model = self._build_vae()
            self.pipe = TunaPipeline(vae_model=vae_model, **common_kwargs)
        elif pipe == "Tuna2RPixelPipeline":
            # Tuna2R pixel still accepts a vae_model arg for API symmetry; we
            # pass through whatever the model exposes (may be None).
            vae_model = getattr(tuna_model, "vae", None)
            self.pipe = Tuna2RPixelPipeline(vae_model=vae_model, **common_kwargs)
        elif pipe == "Tuna2PixelPipeline" or pipe is None:
            self.pipe = Tuna2PixelPipeline(**common_kwargs)
        else:
            raise ValueError(f"Unknown pipeline name: {pipe!r}")

        logger.info(f"Initialised pipeline: {type(self.pipe).__name__}")

    def to_gpu(self, data: dict[str, Any]) -> dict[str, Any]:
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
        return data

    def _data_to_pixel_values(self, data: dict[str, Any]) -> torch.Tensor | None:
        """Coerce the data dict's image entry to a 5D pixel_values tensor.

        Accepts (in priority order):
          * ``data["images"]`` — pre-built tensor (returned as-is);
          * ``data["images_low"]`` — same;
          * ``data["image"]`` — raw PIL.Image (or list of) from predict.py;
            transformed via :func:`image_transform` to ``[B, 1, C, H, W]``.
        Returns ``None`` if no image entry is present.
        """
        if data.get("images") is not None:
            return data["images"]
        if data.get("images_low") is not None:
            return data["images_low"]
        if data.get("image") is not None:
            pil_images = data["image"]
            if not isinstance(pil_images, list):
                pil_images = [pil_images]
            resolution = (int(self.height), int(self.width))
            tensors = [image_transform(img, resolution=resolution) for img in pil_images]
            return (
                torch.stack(tensors).unsqueeze(1).to(self.device, self.weight_dtype)
            )
        return None

    # ---- Dispatch ----------------------------------------------------------
    def __call__(self, data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if self.pipe is None:
            raise RuntimeError("Pipeline not initialised. Cannot perform inference.")

        data = self.to_gpu(data)
        if self.inference_mode == "t2i":
            return self.t2i(data, **kwargs)
        if self.inference_mode == "edit":
            return self.edit(data, **kwargs)
        if self.inference_mode == "mmu":
            return self.mmu(data, **kwargs)
        raise ValueError(
            f"Unknown inference mode: {self.inference_mode!r}. "
            "Supported modes are 't2i', 'edit', 'mmu'."
        )

    # ---- t2i ---------------------------------------------------------------
    def t2i(self, data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if self.prompt_file_path:
            with open(self.prompt_file_path, "r", encoding="utf-8") as f:
                current_lines = [line.strip() for line in f if line.strip()]
            current_index = min(self._prompt_index, len(current_lines) - 1)
            prompts = [current_lines[current_index]]
            self._prompt_index = current_index + 1
            logger.info(f"Using prompt at index {current_index}: {prompts[0]}")
        else:
            prompts = data.get("sentence", data.get("text", []))
            if isinstance(prompts, str):
                prompts = [prompts]

        # Pixel values are only used in reconstruction mode (reca path —
        # not exposed via demo scripts; tensor inputs only).
        pixel_values = (
            data.get("images", data.get("images_low")) if self.reca_mode else None
        )

        num_inference_steps = kwargs.get("num_inference_steps", self.num_inference_steps)
        guidance_scale = kwargs.get("guidance_scale", self.guidance_scale)
        if self.prompt_file_path:
            seed = kwargs.get("seed", random.randint(0, 2**32 - 1))
        else:
            seed = kwargs.get("seed", 42)
        sampling_method = kwargs.get("sampling_method", self.sampling_method)
        atol = kwargs.get("atol", 1e-6)
        rtol = kwargs.get("rtol", 1e-3)
        num_images_per_prompt = self.num_images_per_prompt

        save_name_prefix = None
        save_path_list = data.get("save_path")
        if save_path_list and len(save_path_list) > 0:
            save_name_prefix = save_path_list[0]

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        transport = kwargs.get("transport", None)
        sampler = kwargs.get("sampler", None)
        if transport is None and hasattr(self.model, "transport"):
            transport = self.model.transport
        if sampler is None and hasattr(self.model, "sampler"):
            sampler = self.model.sampler

        all_images: list[torch.Tensor] = []
        save_paths: list[str] = []

        for i in range(num_images_per_prompt):
            current_seed = seed + i if seed is not None else None
            if current_seed is not None:
                torch.manual_seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(current_seed)

            with torch.inference_mode():
                pil_images = self.pipe.t2i(
                    prompts=prompts,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    transport=transport,
                    sampler=sampler,
                    sampling_method=sampling_method,
                    atol=atol,
                    rtol=rtol,
                    time_shifting_factor=self.shift,
                    pixel_values=pixel_values,
                    second_time=self.second_time,
                    negative_prompt=self.negative_prompt,
                    noise_scale=self.noise_scale,
                    cfg_interval=self.cfg_interval,
                )

            images_tensor = pil_to_tensor(pil_images)
            if images_tensor.dim() == 4:
                for img in images_tensor:
                    all_images.append(img)
            else:
                all_images.append(images_tensor)

            if save_name_prefix:
                save_paths.append(f"{save_name_prefix}-{i}.png")

        outputs: dict[str, Any] = {
            "generated_image": all_images,
            "prompts": prompts,
            "task_type": "generation",
            "save_path": save_paths if save_paths else None,
        }
        if pixel_values is not None:
            # Same 5D-aware shape handling as `edit` so SaveImageCallback can
            # write the source image alongside the reconstruction.
            if pixel_values.dim() == 5 and pixel_values.shape[1] > 1:
                outputs["original_image"] = pixel_values[:, 0]
            elif pixel_values.dim() == 5:
                outputs["original_image"] = pixel_values.squeeze(1)
            else:
                outputs["original_image"] = pixel_values
        return outputs

    # ---- edit --------------------------------------------------------------
    def edit(self, data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        prompts: list[str] = []
        pixel_values: torch.Tensor | None = None

        if self.prompt_file_path:
            with open(self.prompt_file_path, "r", encoding="utf-8") as f:
                current_lines = [line.strip() for line in f if line.strip()]
            current_index = min(self._prompt_index, len(current_lines) - 1)
            line_parts = current_lines[current_index].split(None, 1)
            image_path = line_parts[0]
            prompt_text = line_parts[1] if len(line_parts) > 1 else ""
            prompts = [prompt_text]
            self._prompt_index = current_index + 1
            logger.info(
                f"Using edit at index {current_index}: image={image_path}, prompt={prompt_text}"
            )

            pil_image = Image.open(image_path).convert("RGB")
            transformed_img = image_transform(pil_image, resolution=(int(self.height), int(self.width)))

            base_path, ext = os.path.splitext(image_path)
            resize_path = f"{base_path}_resize{ext}"
            save_tensor = torch.clamp((transformed_img + 1) / 2, 0, 1)
            tvF.to_pil_image(save_tensor).save(resize_path)
            logger.info(f"Saved resized image to {resize_path}")

            pixel_values = (
                transformed_img.unsqueeze(0).unsqueeze(0).to(self.device, self.weight_dtype)
            )
        else:
            try:
                prompts = data["sentence"]
            except KeyError:
                prompts = data["text"]
            if isinstance(prompts, str):
                prompts = [prompts]
            pixel_values = self._data_to_pixel_values(data)

        save_path_list = data.get("save_path")
        save_path = save_path_list[0] if save_path_list and len(save_path_list) > 0 else None
        image_masks = data.get("image_masks")

        num_inference_steps = kwargs.get("num_inference_steps", self.num_inference_steps)
        guidance_scale = kwargs.get("guidance_scale", self.guidance_scale)
        seed = kwargs.get("seed", random.randint(0, 2**32 - 1))
        sampling_method = kwargs.get("sampling_method", self.sampling_method)
        atol = kwargs.get("atol", 1e-6)
        rtol = kwargs.get("rtol", 1e-3)
        num_images_per_prompt = self.num_images_per_prompt

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        transport = kwargs.get("transport", None)
        sampler = kwargs.get("sampler", None)
        if transport is None and hasattr(self.model, "transport"):
            transport = self.model.transport
        if sampler is None and hasattr(self.model, "sampler"):
            sampler = self.model.sampler

        if transport is None or sampler is None:
            raise ValueError(
                "Transport and sampler are required for Tuna inference."
            )

        all_images: list[torch.Tensor] = []
        for i in range(num_images_per_prompt):
            current_seed = seed + i if seed is not None else None
            if current_seed is not None:
                torch.manual_seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(current_seed)

            with torch.inference_mode():
                pil_images = self.pipe.t2i_edit(
                    prompts=prompts,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    transport=transport,
                    sampler=sampler,
                    sampling_method=sampling_method,
                    atol=atol,
                    rtol=rtol,
                    time_shifting_factor=self.shift,
                    pixel_values=pixel_values,
                    image_masks=image_masks,
                    noise_scale=self.noise_scale,
                    negative_prompt=self.negative_prompt,
                )

            images_tensor = pil_to_tensor(pil_images)
            if images_tensor.dim() == 4:
                for img in images_tensor:
                    all_images.append(img)
            else:
                all_images.append(images_tensor)

        outputs: dict[str, Any] = {
            "generated_image": all_images,
            "prompts": prompts,
            "task_type": "generation",
            "save_path": save_path,
        }
        if pixel_values is not None:
            if pixel_values.dim() == 5 and pixel_values.shape[1] > 1:
                outputs["original_image"] = pixel_values[:, 0]
            elif pixel_values.dim() == 5:
                outputs["original_image"] = pixel_values.squeeze(1)
            else:
                outputs["original_image"] = pixel_values
        return outputs

    # ---- mmu ---------------------------------------------------------------
    def mmu(self, data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        prompt = "Describe the image in detail."
        image = data["image"]

        if self.height == "auto" or self.width == "auto":
            in_w, in_h = image[0].size
            in_ar = in_w / max(in_h, 1e-6)
            best_res = best_tokens = None
            min_diff = float("inf")
            for resolution, num_tokens in self.supported_resolutions:
                h, w = resolution
                res_ar = w / h
                diff = abs(in_ar - res_ar)
                if diff < min_diff:
                    min_diff = diff
                    best_res = resolution
                    best_tokens = num_tokens
            assert best_res is not None and best_tokens is not None
            best_h, best_w = best_res
            pixel_values = [
                image_transform(img, resolution=(best_h, best_w), centercrop=self.centercrop)
                for img in image
            ]
            input_height, input_width = best_h, best_w
        else:
            input_height, input_width = int(self.height), int(self.width)
            pixel_values = [
                image_transform(
                    img,
                    resolution=(input_height, input_width),
                    centercrop=self.centercrop,
                )
                for img in image
            ]
        pixel_values = torch.stack(pixel_values).to(self.device, self.weight_dtype)

        result = self.pipe.mmu(
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            prompt=prompt,
            pixel_values=pixel_values,
            height=input_height,
            width=input_width,
        )

        return {"generated_text": result, "task_type": "understanding"}
