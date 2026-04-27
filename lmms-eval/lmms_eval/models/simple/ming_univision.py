"""Ming-UniVision wrapper for lmms_eval.

Adapted from Ming-UniVision/mingunivision/test_infer_unified.py and
Ming-UniVision/mingunivision/mingunivisioninfer.py. We instantiate the upstream
``MingUniVisionInfer`` helper, which is the cleanest entry point.

The local clone path must contain the ``mingunivision/`` package; both the model
code and the bundled tokenizer/processor configs live there.
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

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")

_DEFAULT_MING_REPO = (
    "/data/repos/fbsource/fbcode/genads/showme-lmms-eval-main/Ming-UniVision"
)


def _ensure_ming_on_path(repo_path: str) -> None:
    """Put both the repo root and the ``mingunivision/`` subdir on sys.path.

    The MingUniVisionInfer source uses ``sys.path.insert(0, ...)`` with the
    file's own directory and then imports e.g. ``modeling_bailingmm`` directly,
    so the inner package directory must be importable as a top-level path.
    """
    inner = os.path.join(repo_path, "mingunivision")
    for p in (inner, repo_path):
        if p not in sys.path:
            sys.path.insert(0, p)


@register_model("ming_univision")
class MingUniVision(lmms):
    """Ming-UniVision unified multimodal model (16B-A3B MoE).

    HuggingFace: inclusionAI/Ming-UniVision-16B-A3B (+ inclusionAI/MingTok-Vision)
    Paper: https://arxiv.org/abs/2510.06590
    """

    def __init__(
        self,
        pretrained: str = "inclusionAI/Ming-UniVision-16B-A3B",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: Union[int, str] = 1,
        max_new_tokens: int = 512,
        dtype: str = "bf16",
        repo_path: str = _DEFAULT_MING_REPO,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        _ensure_ming_on_path(repo_path)

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        # MingUniVisionInfer's ``load_model_processor`` calls
        # ``AutoTokenizer.from_pretrained("./mingunivision", ...)``, which
        # requires the working directory to contain ``mingunivision/``.
        prev_cwd = os.getcwd()
        try:
            os.chdir(repo_path)
            from mingunivision.mingunivisioninfer import MingUniVisionInfer  # noqa: WPS433

            self._infer = MingUniVisionInfer(pretrained, dtype=dtype)
        finally:
            os.chdir(prev_cwd)

        self._tokenizer = self._infer.tokenizer
        self._processor = self._infer.processor

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
        return self._infer.model

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
        raise NotImplementedError("Loglikelihood is not implemented for Ming-UniVision")

    def flatten(self, input):
        return [j for i in input for j in i]

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
            assert self.batch_size_per_gpu == 1, "Batch size must be 1 for Ming-UniVision"

            context = contexts[0]
            gen_kwargs = all_gen_kwargs[0]
            until = gen_kwargs.get("until", [self._tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)

            valid_visuals = [v for v in visuals if isinstance(v, PIL.Image.Image)]

            # Build messages following MingUniVision's chat format.
            content = []
            for img in valid_visuals:
                content.append({"type": "image", "image": img})
            text_part = context.replace("<image>", "").strip()
            content.append({"type": "text", "text": text_part})
            messages = [{"role": "HUMAN", "content": content}]

            try:
                # Reset multi-turn state between examples (the model otherwise
                # accumulates KV cache across calls).
                self._infer.reset_inner_state()
                gen_text = self._infer.generate(
                    messages,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as e:
                eval_logger.error(f"Error generating for doc_id {doc_id[0]}: {e}")
                gen_text = ""

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
        raise NotImplementedError("Multi-round not implemented for Ming-UniVision")
