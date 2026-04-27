# pyre-unsafe
from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import logging

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)
logger = logging.get_logger(__name__)


def apply_rope_image3d_text1d(
    query_states: torch.Tensor,  # [B, H, L, Dh]
    key_states: torch.Tensor,  # [B, H, L, Dh]
    modality_positions: Optional[
        Union[torch.Tensor, list]
    ],  # [B, num_imgs, 2] -> (offset, length)
    rope_3d: torch.Tensor,  # [S_img, Dh, 2, 2] from build_rope(T,h,w, patch_size=self.config.patch_size)
    cos_1d: Optional[torch.Tensor] = None,
    sin_1d: Optional[torch.Tensor] = None,
    add_time_embeds: bool = True,
    add_ar_embeds: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if modality_positions is None:
        return query_states, key_states

    B, H, L, Dh = query_states.shape
    device = query_states.device
    dtype = query_states.dtype

    def _apply_rope3d_slice(
        q_slice: torch.Tensor, k_slice: torch.Tensor, rope_slice: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert q_slice.dim() == 4 and k_slice.dim() == 4
        _, Hq, Sq, Dh = q_slice.shape
        _, Hk, Sk, Dhk = k_slice.shape
        S = min(Sq, Sk, rope_slice.size(0))

        # if S == 0:
        #     return q_slice, k_slice

        q = q_slice.narrow(2, 0, S).contiguous().reshape(1, Hq, S, Dh // 2, 2)
        k = k_slice.narrow(2, 0, S).contiguous().reshape(1, Hk, S, Dh // 2, 2)

        R = rope_slice.narrow(0, 0, S).unsqueeze(0).unsqueeze(0)

        # (...,2) x (...,2,2) -> (...,2)
        q_new = torch.matmul(q.unsqueeze(-2), R).squeeze(-2)
        k_new = torch.matmul(k.unsqueeze(-2), R).squeeze(-2)

        q_rot = q_new.reshape(1, Hq, S, Dh)
        k_rot = k_new.reshape(1, Hk, S, Dh)

        out_q = q_slice.clone()
        out_k = k_slice.clone()
        out_q[:, :, :S, :] = q_rot
        out_k[:, :, :S, :] = k_rot
        return out_q, out_k

    text_masks = []
    for b in range(B):
        text_mask = torch.ones(L, dtype=torch.bool, device=device)

        mpos_b = modality_positions[b]
        mpos_iter = mpos_b.tolist() if torch.is_tensor(mpos_b) else mpos_b

        meta = (1 if add_time_embeds else 0) + (2 if add_ar_embeds else 0)

        for offset, length in mpos_iter:
            # if length is None:
            #     continue
            seg_start = max(0, int(offset))
            seg_end = max(seg_start, min(L, int(offset + max(length, meta))))
            if seg_start < seg_end:
                text_mask[seg_start:seg_end] = False

        text_masks.append(text_mask)
    text_mask = torch.stack(text_masks, dim=0)  # [B, L]
    if cos_1d is not None and sin_1d is not None:
        if cos_1d.dim() == 2:  # [L, Dh] -> [1, L, Dh]
            cos_1d = cos_1d.unsqueeze(0)
            sin_1d = sin_1d.unsqueeze(0)

        if cos_1d.size(0) == 1 and query_states.size(0) > 1:
            cos_1d = cos_1d.expand(query_states.size(0), -1, -1)
            sin_1d = sin_1d.expand(query_states.size(0), -1, -1)

        B, H, L, Dh = query_states.shape

        for b in range(B):
            idx = torch.nonzero(text_mask[b], as_tuple=False).flatten()  # [S_txt]
            S_txt = idx.numel()
            # if S_txt == 0:
            #     continue

            q_txt = query_states.narrow(0, b, 1).index_select(
                2, idx
            )  # [1, H, S_txt, Dh]
            k_txt = key_states.narrow(0, b, 1).index_select(2, idx)  # [1, H, S_txt, Dh]

            cos_b = cos_1d.narrow(0, b, 1).index_select(1, idx)  # [1, S_txt, Dh]
            sin_b = sin_1d.narrow(0, b, 1).index_select(1, idx)  # [1, S_txt, Dh]

            q_rot, k_rot = apply_rotary_pos_emb(
                q_txt, k_txt, cos_b, sin_b
            )  # -> [1,H,S_txt,Dh]

            query_states[b, :, idx, :] = q_rot.squeeze(0)  # [H, S_txt, Dh]
            key_states[b, :, idx, :] = k_rot.squeeze(0)
    for b in range(B):
        mpos_b = modality_positions[b]
        mpos_iter = mpos_b.tolist() if torch.is_tensor(mpos_b) else mpos_b

        for offset, length in mpos_iter:
            # if length is None or length <= 0:
            #     continue

            meta = (1 if add_time_embeds else 0) + (2 if add_ar_embeds else 0)
            img_start = int(offset + meta)
            img_end = int(offset + length)

            # if img_start >= L:
            #     continue
            img_end = min(L, img_end)
            S_img = img_end - img_start
            # if S_img <= 0:
            #     continue

            S_rope = rope_3d.size(0)
            use_len = min(S_img, S_rope)
            if use_len < S_img:
                pass

            q_img = query_states[
                b : b + 1, :, img_start : img_start + use_len, :
            ].contiguous()
            k_img = key_states[
                b : b + 1, :, img_start : img_start + use_len, :
            ].contiguous()

            rot2x2 = rope_3d[:use_len].to(device)  # [use_len, Dh, 2, 2] (Dh consistent)
            q_img_rot, k_img_rot = _apply_rope3d_slice(q_img, k_img, rot2x2)

            query_states[b, :, img_start : img_start + use_len, :] = q_img_rot.squeeze(
                0
            )
            key_states[b, :, img_start : img_start + use_len, :] = k_img_rot.squeeze(0)

    return query_states, key_states


def apply_rope_only_on_image_tokens(
    query_states: torch.Tensor,  # [B, H, L, Dh]
    key_states: torch.Tensor,  # [B, H, L, Dh]
    modality_positions: Optional[
        torch.Tensor
    ],  # Tensor/List: [B, num_imgs, 2] -> (offset, length)
    rope_3d: torch.Tensor,  # [img_len, ...] from build_rope(...)
    add_time_embeds: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if modality_positions is None or rope_3d is None:
        return query_states, key_states

    B, H, L, Dh = query_states.shape
    device = query_states.device

    # Process per-sample per-segment to avoid mistakenly chaining different image segments
    for b in range(B):
        # modality_positions[b]: [num_imgs, 2] -> (offset, length)
        mpos_b = modality_positions[b]
        # Supports both tensor and list
        if torch.is_tensor(mpos_b):
            mpos_iter = mpos_b.tolist()
        else:
            mpos_iter = mpos_b

        for offset, length in mpos_iter:
            # length<=1 means only time, no image token
            if length is None or length <= 1:
                continue

            # Apply only to image tokens (skip time)
            img_start = offset + 1 if add_time_embeds else offset
            img_end = offset + length
            if img_start >= img_end or img_start >= L:
                continue
            img_end = min(img_end, L)
            img_len = img_end - img_start

            # Take corresponding 3D RoPE in segment order 0..img_len-1
            idx = torch.arange(img_len, device=device)

            # Slice: [H, img_len, Dh]; add a batch dim to match [B,H,L,D] interface of common apply_rotary_emb
            q_slice = query_states[b, :, img_start:img_end, :].unsqueeze(
                0
            )  # [1,H,img_len,Dh]
            k_slice = key_states[b, :, img_start:img_end, :].unsqueeze(
                0
            )  # [1,H,img_len,Dh]

            # Assumes apply_rotary_emb supports (q, rope[idx]) broadcasting;
            # if your implementation requires stricter shapes between [B,H,L,D] and rope, do reshape/broadcast here.
            q_rot = apply_rotary_emb(q_slice, rope_3d[idx])  # -> [1,H,img_len,Dh]
            k_rot = apply_rotary_emb(k_slice, rope_3d[idx])

            # Write back
            query_states[b, :, img_start:img_end, :] = q_rot.squeeze(0)
            key_states[b, :, img_start:img_end, :] = k_rot.squeeze(0)

    return query_states, key_states


class UndTransConfig(PretrainedConfig):
    def __init__(self):
        self.attn_implementation = "sdpa"
        self.max_position_embeddings = 131072
        self.hidden_size = 1024
        self.intermediate_size = 4096
        # self.hidden_size = 1152
        # self.intermediate_size = 4608
        self.num_attention_heads = 32
        self.num_key_value_heads = 8
        self.hidden_act = "silu"
        self.rms_norm_eps = 1e-05
        self.mlp_bias = False
        self.head_dim = 64
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.rope_theta = 500000.0
        self.rope_scaling = {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }
        super().__init__()


class DiffusionHeadConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=2048,
        num_attention_heads=12,
        num_key_value_heads=8,
        intermediate_size=8192,
        max_position_embeddings=131072,
        use_mrope=False,
        head_dim=None,
    ):
        self.attn_implementation = "sdpa"
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = "silu"
        self.rms_norm_eps = 1e-05
        self.mlp_bias = False
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.rope_theta = 500000.0
        self.use_mrope = use_mrope
        if use_mrope:
            self.rope_scaling = {
                "type": "default",
                "rope_type": "default",
                "mrope_section": [16, 24, 24],
            }
        else:
            self.rope_scaling = {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            }
        self.qk_norm = True
        super().__init__()


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    shift = shift.to(torch.float32)
    scale = scale.to(torch.float32)
    if len(x.shape) != len(shift.shape):
        return (x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)).to(input_dtype)
    else:
        return (x * (1 + scale) + shift).to(input_dtype)


class ModulatedAttentionBlock(nn.Module):
    def __init__(
        self,
        config: DiffusionHeadConfig,
        layer_idx: int,
        model_args: Optional[Any] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.model_args = model_args

        self.self_attn = ATTENTION_CLASSES[config.attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.interleave_degree = getattr(model_args, "interleave_degree", 2)
        if self.interleave_degree > 1:
            self.ec_dit = (
                getattr(model_args, "ec_dit", False)
                and layer_idx % self.interleave_degree == 1
            )
            self.diffmoe = (
                getattr(model_args, "diffmoe", False)
                and layer_idx % self.interleave_degree == 1
            )
        else:
            self.ec_dit = getattr(model_args, "ec_dit", False)
            self.diffmoe = getattr(model_args, "diffmoe", False)
        self.optimized_moe = getattr(model_args, "optimized_moe", False)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        tuna_cfg = (
            getattr(model_args, "tuna_config", model_args)
            if model_args is not None
            else None
        )
        self.share_adaln = (
            getattr(tuna_cfg, "share_adaln", False) if tuna_cfg is not None else False
        )
        if self.share_adaln:
            self.scale_shift_table = nn.Parameter(
                torch.randn(6, config.hidden_size) / config.hidden_size**0.5
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    config.hidden_size,
                    6 * config.hidden_size,
                    bias=True,
                ),
            )
            nn.init.zeros_(self.adaLN_modulation[1].weight)
            nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        adaln_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        rope_3d: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        modality_positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        if self.share_adaln:
            # adaln_input is pre-computed (B*num_imgs, 6*D) from shared projection
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None].reshape(1, -1) + adaln_input
            ).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)
            )

        # We only modulate the image embeddings
        (
            shift_msa_new,
            scale_msa_new,
            gate_msa_new,
            shift_mlp_new,
            scale_mlp_new,
            gate_mlp_new,
        ) = (
            torch.zeros_like(hidden_states),
            torch.zeros_like(hidden_states),
            torch.ones_like(hidden_states),
            torch.zeros_like(hidden_states),
            torch.zeros_like(hidden_states),
            torch.ones_like(hidden_states),
        )

        for i, modality_batch in enumerate(modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                idx = i * modality_positions.size(1) + j
                shift_msa_new[i, offset : offset + length] = shift_msa[idx]
                scale_msa_new[i, offset : offset + length] = scale_msa[idx]
                gate_msa_new[i, offset : offset + length] = gate_msa[idx]
                shift_mlp_new[i, offset : offset + length] = shift_mlp[idx]
                scale_mlp_new[i, offset : offset + length] = scale_mlp[idx]
                gate_mlp_new[i, offset : offset + length] = gate_mlp[idx]
        # We only modulate the image embeddings

        residual = hidden_states
        hidden_states = modulate(
            self.input_layernorm(hidden_states), shift_msa_new, scale_msa_new
        )

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            rope_3d=rope_3d,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            modality_positions=modality_positions,
            **kwargs,
        )

        hidden_states = residual + gate_msa_new * hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = modulate(
            self.post_attention_layernorm(hidden_states), shift_mlp_new, scale_mlp_new
        )
        if self.diffmoe:
            bs = hidden_states.shape[0]
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            hidden_states = self.mlp(hidden_states, bs)
            hidden_states = hidden_states.view(bs, -1, hidden_states.shape[-1])
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + gate_mlp_new * hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
) -> torch.Tensor:
    x_ = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    x_out = (
        freqs_cis[..., 0] * x_[..., 0] + freqs_cis[..., 1] * x_[..., 1]
    )
    return x_out.reshape(*x.shape).type_as(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[DiffusionHeadConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`RotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get(
                    "rope_type", config.rope_scaling.get("type")
                )
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device, **self.rope_kwargs
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(
        self, position_ids: torch.Tensor, device: torch.device
    ) -> None:
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            )  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DiffusionHeadConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        self.qk_norm = config.qk_norm if "qk_norm" in config else False
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.rope_scaling = config.rope_scaling
        # Store rope parameters for build_rope function
        self.rope_3d = None  # Will be built when needed
        self.rotary_emb = RotaryEmbedding(config=self.config)
        self.mm_3drope = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        rope_3d: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        modality_positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # Apply 3D RoPE using shoufa.py style
        # Apply 3D RoPE only to image positions, excluding time_embedding
        if modality_positions is not None:
            # Reshape for RoPE application
            query_states_reshaped = (
                query_states.permute(2, 0, 1, 3)
                .contiguous()
                .view(q_len, bsz * self.num_heads, self.head_dim)
            )
            key_states_reshaped = (
                key_states.permute(2, 0, 1, 3)
                .contiguous()
                .view(q_len, bsz * self.num_key_value_heads, self.head_dim)
            )

            # Clone states and apply RoPE selectively
            query_states_rotated = query_states_reshaped.clone()
            key_states_rotated = key_states_reshaped.clone()

            # Create position mask for 3D rope positions
            rope_position_indices = []

            for i, modality_batch in enumerate(modality_positions):
                for _j, (offset, length) in enumerate(modality_batch):
                    if length > 0:
                        # Skip time_embedding: image starts at offset+1
                        image_start = offset + 1
                        image_end = offset + length
                        for pos in range(image_start, image_end):
                            rope_pos_idx = (
                                pos * bsz + i
                            )  # Position in flattened rope_3d
                            rope_position_indices.append(rope_pos_idx)

            if rope_position_indices:
                # Get valid rope_3d positions
                rope_position_indices = torch.tensor(
                    rope_position_indices, device=hidden_states.device
                )
                valid_rope_positions = rope_3d[rope_position_indices]

                # Apply rope to query states
                query_image_mask = torch.zeros(
                    q_len * bsz * self.num_heads,
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
                for rope_idx in rope_position_indices:
                    for h in range(self.num_heads):
                        query_image_mask[rope_idx * self.num_heads + h] = True

                if query_image_mask.any():
                    query_states_to_rotate = query_states_reshaped.view(
                        -1, self.head_dim
                    )[query_image_mask]
                    rotated_queries = apply_rotary_emb(
                        query_states_to_rotate,
                        valid_rope_positions.repeat_interleave(self.num_heads, dim=0),
                    )
                    query_states_rotated.view(-1, self.head_dim)[query_image_mask] = (
                        rotated_queries
                    )

                # Apply rope to key states
                key_image_mask = torch.zeros(
                    q_len * bsz * self.num_key_value_heads,
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
                for rope_idx in rope_position_indices:
                    for h in range(self.num_key_value_heads):
                        key_image_mask[rope_idx * self.num_key_value_heads + h] = True

                if key_image_mask.any():
                    key_states_to_rotate = key_states_reshaped.view(-1, self.head_dim)[
                        key_image_mask
                    ]
                    rotated_keys = apply_rotary_emb(
                        key_states_to_rotate,
                        valid_rope_positions.repeat_interleave(
                            self.num_key_value_heads, dim=0
                        ),
                    )
                    key_states_rotated.view(-1, self.head_dim)[key_image_mask] = (
                        rotated_keys
                    )

            # Reshape back
            query_states = query_states_rotated.view(
                q_len, bsz, self.num_heads, self.head_dim
            ).permute(1, 2, 0, 3)
            key_states = key_states_rotated.view(
                q_len, bsz, self.num_key_value_heads, self.head_dim
            ).permute(1, 2, 0, 3)

        if past_key_value is not None:
            # cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if isinstance(attention_mask, BlockMask):
            raise NotImplementedError

        causal_mask = attention_mask
        attn_weights = attn_weights + causal_mask
        # if attention_mask is not None:  # no matter the length, we just slice it
        #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        #     attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class SdpaAttention(Attention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        rope_3d: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        modality_positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                rope_3d=rope_3d,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        if self.mm_3drope:
            if rope_3d is not None:
                query_states, key_states = apply_rope_image3d_text1d(
                    query_states,
                    key_states,
                    modality_positions,
                    rope_3d,
                    cos_1d=cos,
                    sin_1d=sin,
                    add_time_embeds=True,  # default only adds time, not aspect ratio
                )
        else:
            if rope_3d is not None:
                query_states, key_states = apply_rope_only_on_image_tokens(
                    query_states=query_states,
                    key_states=key_states,
                    modality_positions=modality_positions,  # [B, num_imgs, 2] of (offset, length)
                    rope_3d=rope_3d,
                    add_time_embeds=True,  # time enabled by default
                )

        if past_key_value is not None:
            # cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        # if attention_mask is not None:
        #     causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)

        if isinstance(attention_mask, BlockMask):
            attn_output = flex_attention(
                query_states, key_states, value_states, block_mask=attention_mask
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(RMSNorm)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


ATTENTION_CLASSES = {
    "eager": Attention,
    "sdpa": SdpaAttention,
}


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = ATTENTION_CLASSES[config.attn_implementation](
            config=config, layer_idx=layer_idx
        )
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


from timm.layers.helpers import to_2tuple


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        kernel_size: Optional[int] = None,
        padding: int = 0,
        norm_layer: Optional[Callable[[int], Any]] = None,
        flatten: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = kernel_size or patch_size
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        adaln_input: torch.Tensor,
        modality_positions: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(adaln_input).chunk(2, dim=1)

        # We only modulate the image embeddings
        shift_new, scale_new = torch.zeros_like(x), torch.zeros_like(x)
        for i, modality_batch in enumerate(modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                idx = i * modality_positions.size(1) + j
                shift_new[i, offset : offset + length] = shift[idx]
                scale_new[i, offset : offset + length] = scale[idx]
        # We only modulate the image embeddings

        x = modulate(self.norm_final(x), shift_new, scale_new)
        x = self.linear(x)
        return x


class UpdatedVisionTransformer(nn.Module):
    def __init__(self, model, del_last_layer=True):
        super().__init__()
        self.model = model
        if del_last_layer:
            del self.model.transformer.resblocks[-1]

    def forward(self, x: torch.Tensor):
        x = self.model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.model.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.positional_embedding.to(x.dtype)
        x = self.model.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)[:, 1:]  # LND -> NLD

        return x


class CLIPVisionEncoder(nn.Module):
    def __init__(self, model, del_last_layer=False):
        super().__init__()
        self.model = model
        if del_last_layer:
            del self.model.transformer.resblocks[-1]

    def forward(self, x: torch.Tensor):
        x = self.model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.model.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.positional_embedding.to(x.dtype)
        x = self.model.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)[:, 1:]  # LND -> NLD

        return x


class SigLipVisionEncoder(nn.Module):
    def __init__(self, model, del_last_layer=True):
        """
        A wrapper for extracting features from the penultimate layer of a vision transformer model.

        Args:
            model: The pre-trained model (e.g., CLIP or SigLIP).
            del_last_layer (bool): Whether to delete the last layer of the vision transformer.
        """
        super().__init__()
        self.model = model

        # Remove the text model (if not needed)
        if hasattr(self.model, "text_model"):
            del self.model.text_model

        # Remove the last layer of the vision transformer
        if del_last_layer and hasattr(self.model.vision_model, "encoder"):
            del self.model.vision_model.encoder.layers[-1]

        # Replace the classification head (if it exists) with an identity layer
        if hasattr(self.model.vision_model, "head"):
            self.model.vision_model.head = nn.Identity()
        if hasattr(self.model.vision_model, "post_layernorm"):
            self.model.vision_model.post_layernorm = nn.Identity()

    def forward(self, x):
        """
        Forward pass to extract features from the penultimate layer.

        Args:
            x: Input image tensor (pixel values).

        Returns:
            Tensor: Features from the penultimate layer.
        """
        return self.model.get_image_features(pixel_values=x)
