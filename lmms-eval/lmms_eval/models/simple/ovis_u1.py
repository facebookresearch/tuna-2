# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Ovis-U1 wrapper for lmms_eval.

Adapted from Ovis-U1/test_img_to_txt.py and test_multi_img_to_txt.py.
"""

import warnings
from typing import List, Optional, Tuple, Union

import PIL
import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")


@register_model("ovis_u1")
class OvisU1(lmms):
    """Ovis-U1 unified multimodal model (3B).

    HuggingFace: AIDC-AI/Ovis-U1-3B
    Paper: https://arxiv.org/abs/2506.23044
    """

    def __init__(
        self,
        pretrained: str = "AIDC-AI/Ovis-U1-3B",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: Union[int, str] = 1,
        max_new_tokens: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval().to(self._device).to(torch.bfloat16)

        self._text_tokenizer = self._model.get_text_tokenizer()
        self._visual_tokenizer = self._model.get_visual_tokenizer()
        self.batch_size_per_gpu = int(batch_size)
        self.max_new_tokens = max_new_tokens
        self.accelerator = accelerator
        self._rank = 0
        self._world_size = 1

    @property
    def tokenizer(self):
        return self._text_tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self._text_tokenizer.eos_token_id

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
        raise NotImplementedError("Loglikelihood is not implemented for Ovis-U1")

    def flatten(self, input):
        return [j for i in input for j in i]

    def _build_inputs(self, prompt, pil_images):
        """Tokenize the prompt + images using Ovis-U1's preprocess_inputs.

        Mirrors test_img_to_txt.py / test_multi_img_to_txt.py.
        """
        if len(pil_images) == 1:
            multimodal_type = "single_image"
            images_arg = pil_images
        else:
            multimodal_type = "multiple_image"
            images_arg = pil_images

        prompt, input_ids, pixel_values, grid_thws = self._model.preprocess_inputs(
            prompt,
            images_arg,
            generation_preface="",
            return_labels=False,
            propagate_exception=False,
            multimodal_type=multimodal_type,
            fix_sample_overall_length_navit=False,
        )
        attention_mask = torch.ne(input_ids, self._text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self._model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self._model.device)
        if pixel_values is not None:
            pixel_values = torch.cat(
                [pixel_values.to(device=self._visual_tokenizer.device, dtype=torch.bfloat16)],
                dim=0,
            )
        if grid_thws is not None:
            grid_thws = torch.cat(
                [grid_thws.to(device=self._visual_tokenizer.device)],
                dim=0,
            )
        return input_ids, pixel_values, attention_mask, grid_thws

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self._text_tokenizer.encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (len(requests) + self.batch_size - 1) // self.batch_size
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            assert self.batch_size_per_gpu == 1, "Batch size must be 1 for Ovis-U1"

            context = contexts[0]
            gen_kwargs = all_gen_kwargs[0]
            until = gen_kwargs.get("until", [self._text_tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)

            # Build prompt: insert <image> placeholders if missing
            if "<image>" not in context:
                if len(visuals) == 1:
                    context = "<image>\n" + context
                else:
                    context = (
                        "\n".join([f"Image {i+1}: <image>" for i in range(len(visuals))])
                        + "\n"
                        + context
                    )

            valid_visuals = [v for v in visuals if isinstance(v, PIL.Image.Image)]

            try:
                input_ids, pixel_values, attention_mask, grid_thws = self._build_inputs(
                    context, valid_visuals
                )
                with torch.inference_mode():
                    output_ids = self._model.generate(
                        input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        grid_thws=grid_thws,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        top_p=None,
                        top_k=None,
                        temperature=None,
                        repetition_penalty=None,
                        eos_token_id=self._text_tokenizer.eos_token_id,
                        pad_token_id=self._text_tokenizer.pad_token_id,
                        use_cache=True,
                    )[0]
                gen_text = self._text_tokenizer.decode(output_ids, skip_special_tokens=True)
            except Exception as e:
                eval_logger.error(f"Error generating for doc_id {doc_id[0]}: {e}")
                gen_text = ""

            # Apply until filter
            for term in until:
                if term and term in gen_text:
                    gen_text = gen_text.split(term)[0]

            res.append(gen_text)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), gen_text)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round not implemented for Ovis-U1")
