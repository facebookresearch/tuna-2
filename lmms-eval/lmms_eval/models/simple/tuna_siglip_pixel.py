# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
import sys
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torchvision.transforms.functional as F
from accelerate import Accelerator, DistributedType
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

_TUNA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _TUNA_ROOT not in sys.path:
    sys.path.insert(0, _TUNA_ROOT)

from tuna.models.tuna_2r_pixel import Tuna2RPixelModel
from tuna.pipelines.tuna_2r_pixel_pipeline import Tuna2RPixelPipeline

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": None,
}

eval_logger = logging.getLogger("lmms-eval")


def resize_center_crop(
    img: Image.Image,
    size: Tuple[int, int],
    *,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> Image.Image:
    target_h, target_w = int(size[0]), int(size[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError("size must be positive integers (h, w)")
    orig_w, orig_h = img.size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError("Input image has non-positive dimensions")
    target_ratio = target_h / target_w
    orig_ratio = orig_h / orig_w
    if math.isclose(target_ratio, orig_ratio, rel_tol=0.0, abs_tol=1e-12):
        return F.resize(img, [target_h, target_w], interpolation=interpolation, antialias=True)
    scale = target_h / orig_h if target_ratio > orig_ratio else target_w / orig_w
    new_w = max(int(math.ceil(orig_w * scale)), target_w)
    new_h = max(int(math.ceil(orig_h * scale)), target_h)
    resized = F.resize(img, [new_h, new_w], interpolation=interpolation, antialias=True)
    left = min(max(0, (new_w - target_w) // 2), new_w - target_w)
    top = min(max(0, (new_h - target_h) // 2), new_h - target_h)
    return F.crop(resized, top=top, left=left, height=target_h, width=target_w)


def image_transform(
    image,
    resolution=(256, 256),
    normalize=True,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    centercrop=True,
):
    image = image.convert("RGB")
    if centercrop:
        image = resize_center_crop(image, resolution)
    else:
        image = F.resize(
            image, [resolution[0], resolution[1]],
            interpolation=InterpolationMode.BICUBIC, antialias=True,
        )
    tensor = transforms.ToTensor()(image)
    if normalize:
        tensor = transforms.Normalize(mean=mean, std=std, inplace=True)(tensor)
    return tensor


@register_model("tuna_siglip_pixel")
class Tuna_SiglipPixel(lmms):
    """Tuna-R (SigLIP pixel) evaluation adapter for lmms-eval."""

    def __init__(
        self,
        config_file: str = "lmms_eval/models/configs/tuna_siglip_pixel_7b.yaml",
        ckpt_path: str = "/path/to/checkpoint.pt",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        single_image=True,
        use_cache=True,
        height="auto",
        width="auto",
        centercrop=False,
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
        max_new_tokens=512,
        precision="fp32",
        no_image_mode="off",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self._device = torch.device(device) if isinstance(device, str) else device
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")

        yaml_conf = OmegaConf.load(config_file)
        yaml_conf.load_stage1_model = ckpt_path
        self._model = Tuna2RPixelModel(**yaml_conf)

        self._tokenizer = self._model.text_tokenizer
        self._max_length = 2048

        print(f"Config: {config_file} | Checkpoint: {ckpt_path}")

        t = 1
        self.supported_resolutions = [
            ([512, 512], 1024 + t),
            ([448, 576], 1008 + t),
            ([576, 448], 1008 + t),
            ([384, 672], 1008 + t),
            ([672, 384], 1008 + t),
        ]

        self.height = height
        self.width = width
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.centercrop = centercrop
        self.single_image = single_image
        if isinstance(no_image_mode, bool):
            self.no_image_mode = "keep_pad" if no_image_mode else "off"
        else:
            self.no_image_mode = str(no_image_mode).lower()

        self.precision = _DTYPE_MAP[precision]
        weight_dtype = self.precision if self.precision is not None else torch.float32

        if self.precision is not None:
            self.model.to(dtype=weight_dtype)

        self.pipe = Tuna2RPixelPipeline(
            model=self.model,
            vae_model=None,
            text_tokenizer=self.model.text_tokenizer,
            tuna_token_ids=self.model.tuna_token_ids,
            config=None,
            weight_dtype=weight_dtype,
            device=self._device,
            use_tf32=True,
            use_chat_template=False,
            add_aspect_ratio_embeds=False,
            height=self.height,
            width=self.width,
            generation_mode="mmu",
            latent_frames=1,
        )

        self.maybe_autocast_precision = torch.autocast(
            device_type=self._device.type,
            dtype=self.precision,
            enabled=self.precision is not None,
        )

        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self._rank = 0
        self._world_size = 1

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP, DistributedType.MULTI_GPU,
            ]
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def flatten(self, input):
        return [item for sublist in input for item in sublist]

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return -len(self.tokenizer.encode(x[0])), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task, split = task[0], split[0]
            visuals = self.flatten(
                [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            )

            best_res = best_tokens = None
            v_tsr_list, input_height_list, input_width_list = [], [], []
            for vis in visuals:
                if self.height == "auto" or self.width == "auto":
                    if best_res is None:
                        in_w, in_h = vis.size
                        in_ar = in_w / max(in_h, 1e-6)
                        best_res, best_tokens = min(
                            self.supported_resolutions,
                            key=lambda r: abs(r[0][1] / r[0][0] - in_ar),
                        )
                    best_h, best_w = best_res
                    pixel_values = image_transform(vis, resolution=[best_h, best_w], centercrop=self.centercrop)
                    input_height, input_width = best_h, best_w
                else:
                    input_height, input_width = self.height, self.width
                    pixel_values = image_transform(vis, resolution=[self.height, self.width], centercrop=self.centercrop)
                v_tsr_list.append(pixel_values)
                input_height_list.append(input_height)
                input_width_list.append(input_width)
            v_tsr = torch.stack(v_tsr_list, dim=0).to(self.device, non_blocking=True)
            v_tsr = v_tsr[:1] if self.single_image else v_tsr

            if self.no_image_mode != "off":
                v_tsr = None

            contexts = list(contexts) if isinstance(contexts, tuple) else contexts
            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "").strip()

            prompt = contexts[0]
            gen_kwargs = all_gen_kwargs[0]

            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
            if "image_sizes" not in gen_kwargs:
                try:
                    gen_kwargs["image_sizes"] = [visuals[0].size]
                except Exception:
                    gen_kwargs["image_sizes"] = None
            gen_kwargs.setdefault("max_new_tokens", self.max_new_tokens)
            gen_kwargs.setdefault("temperature", self.temperature)
            gen_kwargs.setdefault("do_sample", self.do_sample)
            gen_kwargs.setdefault("top_k", self.top_k)
            gen_kwargs.setdefault("top_p", self.top_p)
            gen_kwargs.setdefault("num_beams", 1)

            with self.maybe_autocast_precision:
                result = self.pipe.mmu(
                    do_sample=gen_kwargs["do_sample"],
                    temperature=gen_kwargs["temperature"],
                    top_k=gen_kwargs["top_k"],
                    top_p=gen_kwargs["top_p"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    prompt=prompt,
                    pixel_values=v_tsr,
                    height=input_height_list[0],
                    width=input_width_list[0],
                )

            result = result.strip()
            res.append(result)
            self.cache_hook.add_partial("generate_until", (contexts[0], gen_kwargs), result)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError
