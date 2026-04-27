"""Tar wrapper for lmms_eval (NeurIPS 2025).

Adapted from Tar/i2t_inference.py. Uses TextAlignedTokenizer (TA-Tok) to
encode the image into discrete <I*> tokens that get fed to a Qwen2 backbone.

Adds the local Tar repo's `tok` package to sys.path automatically.
"""

import os
import sys
import warnings
from typing import List, Optional, Tuple, Union

import PIL
import torch
from accelerate import Accelerator
from huggingface_hub import hf_hub_download
from loguru import logger as eval_logger
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2ForCausalLM

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")

# Path where the user cloned the Tar repo
_DEFAULT_TAR_REPO = "/data/repos/fbsource/fbcode/genads/showme-lmms-eval-main/Tar"


def _ensure_tar_on_path(tar_repo_path: str) -> None:
    if tar_repo_path not in sys.path:
        sys.path.insert(0, tar_repo_path)


@register_model("tar")
class Tar(lmms):
    """Tar 7B unified multimodal model.

    HuggingFace: csuhan/Tar-7B (LLM) + csuhan/TA-Tok (visual tokenizer)
    Paper: https://arxiv.org/abs/2506.18898
    """

    def __init__(
        self,
        pretrained: str = "csuhan/Tar-7B",
        ta_tok_repo: str = "csuhan/TA-Tok",
        ta_tok_filename: str = "ta_tok.pth",
        ta_tok_path: Optional[str] = None,  # local override
        tar_repo_path: str = _DEFAULT_TAR_REPO,
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: Union[int, str] = 1,
        max_new_tokens: int = 256,
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

        # Make sure Tar's `tok` package is importable
        _ensure_tar_on_path(tar_repo_path)
        from tok.ta_tok import TextAlignedTokenizer  # noqa: WPS433

        # Load LLM + tokenizer
        self._model = Qwen2ForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
        ).to(self._device).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

        # Load visual tokenizer (TA-Tok)
        if ta_tok_path is None:
            ta_tok_path = hf_hub_download(ta_tok_repo, ta_tok_filename)
        self._visual_tokenizer = TextAlignedTokenizer.from_checkpoint(
            ta_tok_path,
            load_teacher=False,
            input_type="indices",
        ).to(self._device)

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
        return self._model

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
        raise NotImplementedError("Loglikelihood is not implemented for Tar")

    def flatten(self, input):
        return [j for i in input for j in i]

    def _encode_image(self, pil_image: PIL.Image.Image) -> str:
        """Encode an image with TA-Tok and return its <I*> token string."""
        img = pil_image.convert("RGB")
        img_tensor = to_tensor(img).unsqueeze(0).to(self._device)
        image_code = self._visual_tokenizer(img_tensor)["encoded"]
        image_text = "".join([f"<I{x}>" for x in image_code[0].cpu().tolist()])
        return image_text

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
            assert self.batch_size_per_gpu == 1, "Batch size must be 1 for Tar"

            context = contexts[0]
            gen_kwargs = all_gen_kwargs[0]
            until = gen_kwargs.get("until", [self._tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)

            # Strip <image> placeholders, encode all images, then prepend
            context_no_placeholder = context.replace("<image>", "").strip()
            image_chunks = []
            for v in visuals:
                if isinstance(v, PIL.Image.Image):
                    try:
                        image_chunks.append(self._encode_image(v))
                    except Exception as e:  # pragma: no cover
                        eval_logger.error(f"TA-Tok encode failed for doc {doc_id[0]}: {e}")
            image_prefix = "\n".join(image_chunks)
            user_content = (
                f"{image_prefix}\n{context_no_placeholder}".strip()
                if image_prefix
                else context_no_placeholder
            )

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ]
            input_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(input_text, return_tensors="pt").to(self._device)

            try:
                with torch.inference_mode():
                    gen_ids = self._model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        eos_token_id=self._tokenizer.eos_token_id,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
                gen_text = self._tokenizer.batch_decode(
                    gen_ids[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )[0]
            except Exception as e:
                eval_logger.error(f"Error generating for doc_id {doc_id[0]}: {e}")
                gen_text = ""

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
        raise NotImplementedError("Multi-round not implemented for Tar")
