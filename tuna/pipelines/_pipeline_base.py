"""Shared base for the three Tuna pipeline classes.

Owns the methods that are byte-for-byte identical across `TunaPipeline`,
`Tuna2PixelPipeline`, and `Tuna2RPixelPipeline`. Each subclass keeps its own
``__init__``, ``t2i``, ``t2i_edit``, ``_decode_latents``, and
``_init_hyperparams`` (which calls a module-local ``get_hyper_params`` whose
seq-len bucket choices differ between variants).
"""

# (c) Meta Platforms, Inc. and affiliates. Apache-2.0.

from __future__ import annotations

# pyre-unsafe

from typing import List, Optional

import torch


class TunaPipelineBase:
    """Base class shared by all three Tuna pipelines.

    Subclasses must populate the attributes used by ``mmu`` via their own
    ``__init__`` and ``_init_hyperparams``: ``self.text_tokenizer``,
    ``self.conversation``, ``self.device``, ``self.weight_dtype``,
    ``self.boi_id``, ``self.eoi_id``, ``self.img_pad_id``, ``self.pad_id``,
    ``self.num_image_tokens``, ``self.model``.
    """

    def mmu(
        self,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_new_tokens: int = 512,
        prompt: str = "",
        pixel_values: Optional[torch.Tensor] = None,
        height: int = 512,
        width: int = 512,
    ) -> List[str]:
        self._init_hyperparams(height, width)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
            prompt = "<image>\n" + prompt
        conversation = self.conversation.copy()
        conversation.append({"role": "user", "content": prompt})
        conv_prompt = self.text_tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        text_tokens = self.text_tokenizer(
            conv_prompt, add_special_tokens=False
        ).input_ids
        img_id = self.text_tokenizer("<image>", add_special_tokens=False).input_ids[0]
        img_idx = text_tokens.index(img_id)
        text_tokens = (
            text_tokens[:img_idx]
            + [self.boi_id]
            + [self.img_pad_id] * self.num_image_tokens
            + [self.eoi_id]
            + text_tokens[img_idx + 1 :]
        )
        text_tokens = torch.tensor(text_tokens).unsqueeze(0).to(self.device)
        modality_positions = (
            torch.tensor([[img_idx + 1, self.num_image_tokens]])
            .unsqueeze(0)
            .to(self.device)
        )
        text_masks = torch.where(
            (text_tokens != self.img_pad_id) & (text_tokens != self.pad_id),
            torch.ones_like(text_tokens),
            torch.zeros_like(text_tokens),
        )
        image_masks = torch.where(
            text_tokens == self.img_pad_id,
            torch.ones_like(text_tokens),
            torch.zeros_like(text_tokens),
        ).to(self.device)
        data_type = ["mmu"]

        image_latents, t, image_labels, image_masks, image_latents_clean = (
            self.model.prepare_latents_and_labels(pixel_values, data_type, image_masks)
        )
        attention_mask, attention_mask_diffhead = self.model.create_attention_mask(
            text_tokens.size(0),
            text_tokens.size(1),
            modality_positions,
            self.device,
            self.weight_dtype,
        )

        model_output = self.model.tuna_model(
            text_tokens=text_tokens,
            image_latents=image_latents,
            t=t.to(self.weight_dtype),
            attention_mask=attention_mask,
            text_masks=text_masks,
            image_masks=image_masks,
            text_labels=None,
            image_labels=image_labels,
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=text_tokens.size(1),
            device=text_tokens.device,
            return_input_embeds=True,
        )

        if self.model.mrope_type == "none" or self.model.mrope_type == "dit_3drope_mm":
            input_embeds = model_output
            output_tokens = self.model.tuna_model.mmu_generate(
                input_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token=self.text_tokenizer.eos_token_id,
            )
        else:
            input_embeds, position_ids = model_output
            output_tokens = self.model.tuna_model.mmu_generate(
                input_embeds=input_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token=self.text_tokenizer.eos_token_id,
            )

        text = self.text_tokenizer.decode(output_tokens, skip_special_tokens=True)

        return text
