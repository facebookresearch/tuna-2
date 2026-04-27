"""OmniGen2 wrapper for lmms_eval (image-to-text understanding only).

OmniGen2 uses Qwen2.5-VL as its multimodal LLM. We load only the mllm + processor
sub-modules to keep memory usage low; the diffusion transformer and VAE are not
needed for image understanding.

Adapted from OmniGen2/inference_chat.py and OmniGen2ChatPipeline.generate_text.
"""

import os
import sys
import warnings
from typing import List, Optional, Tuple, Union

import PIL
import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen2VLImageProcessor,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")

_DEFAULT_OMNIGEN2_REPO = "/data/repos/fbsource/fbcode/genads/showme-lmms-eval-main/OmniGen2"


def _ensure_repo_on_path(repo_path: str) -> None:
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


@register_model("omnigen2")
class OmniGen2(lmms):
    """OmniGen2 unified multimodal model (3B+4B; understanding via Qwen2.5-VL).

    HuggingFace: OmniGen2/OmniGen2
    Paper: https://arxiv.org/abs/2506.18871
    """

    def __init__(
        self,
        pretrained: str = "OmniGen2/OmniGen2",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: Union[int, str] = 1,
        max_new_tokens: int = 512,
        repo_path: str = _DEFAULT_OMNIGEN2_REPO,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Make sure local OmniGen2 source is importable (for any custom code refs)
        _ensure_repo_on_path(repo_path)

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        # Load only the mllm (Qwen2.5-VL) and processor sub-modules
        self._mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained,
            subfolder="mllm",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self._device).eval()

        # Qwen2_5_VLProcessor.from_pretrained needs to resolve a video
        # processor, which requires a config.json with `model_type`. The
        # bundled processor/ snapshot lacks one, so symlink the mllm config
        # into it on first run.
        snapshot_root = snapshot_download(
            pretrained, allow_patterns=["processor/*", "mllm/config.json"]
        )
        processor_dir = os.path.join(snapshot_root, "processor")
        config_link = os.path.join(processor_dir, "config.json")
        if not os.path.exists(config_link):
            try:
                os.symlink(
                    os.path.join(snapshot_root, "mllm", "config.json"),
                    config_link,
                )
            except OSError:
                import shutil
                shutil.copy(
                    os.path.join(snapshot_root, "mllm", "config.json"),
                    config_link,
                )
        self._processor = Qwen2_5_VLProcessor.from_pretrained(processor_dir)
        self._tokenizer = self._processor.tokenizer

        self.batch_size_per_gpu = int(batch_size)
        self.max_new_tokens = max_new_tokens
        self.accelerator = accelerator
        self._rank = 0
        self._world_size = 1

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._mllm

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id

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
        raise NotImplementedError("Loglikelihood is not implemented for OmniGen2")

    def flatten(self, input):
        return [j for i in input for j in i]

    @staticmethod
    def _apply_chat_template(prompt: str, num_images: int) -> str:
        """Mirror OmniGen2ChatPipeline._apply_chat_template (text-only path).

        The original prepends `<imgN>: <|vision_start|><|image_pad|><|vision_end|>`
        for each input image, then wraps the user message with the chat template.
        """
        if num_images:
            image_prefix = "".join(
                f"<img{i}>: <|vision_start|><|image_pad|><|vision_end|>"
                for i in range(1, num_images + 1)
            )
        else:
            image_prefix = ""
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant that generates high-quality images based on user instructions.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{image_prefix}{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self._tokenizer.encode(x[0])
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
            assert self.batch_size_per_gpu == 1, "Batch size must be 1 for OmniGen2"

            context = contexts[0]
            gen_kwargs = all_gen_kwargs[0]
            until = gen_kwargs.get("until", [self._tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)

            # Strip <image> placeholder from upstream task; the OmniGen2 chat
            # template inserts its own image markers based on `num_images`.
            context_no_placeholder = context.replace("<image>", "").strip()
            valid_visuals = [v for v in visuals if isinstance(v, PIL.Image.Image)]

            prompt = self._apply_chat_template(context_no_placeholder, len(valid_visuals))

            ori_padding_side = self._processor.tokenizer.padding_side
            self._processor.tokenizer.padding_side = "left"
            try:
                inputs = self._processor(
                    text=[prompt],
                    images=valid_visuals if valid_visuals else None,
                    videos=None,
                    padding=True,
                    return_tensors="pt",
                ).to(self._device)
            finally:
                self._processor.tokenizer.padding_side = ori_padding_side

            try:
                with torch.inference_mode():
                    generated_ids = self._mllm.generate(
                        **inputs,
                        tokenizer=self._tokenizer,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        stop_strings=["<|im_end|>", "<|img|>", "<|endoftext|>"],
                    )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                gen_text = self._processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )[0]
            except Exception as e:
                eval_logger.error(f"Error generating for doc_id {doc_id[0]}: {e}")
                gen_text = ""

            # Drop OmniGen2-specific terminators
            for special in ("<|im_end|>", "<|img|>", "<|endoftext|>"):
                gen_text = gen_text.replace(special, "")

            # Apply task-specified until filter
            for term in until:
                if term and term in gen_text:
                    gen_text = gen_text.split(term)[0]

            res.append(gen_text.strip())
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), gen_text)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round not implemented for OmniGen2")
