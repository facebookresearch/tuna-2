# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# coding=utf-8
from __future__ import annotations

# pyre-unsafe
from typing import Callable

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def full(
    b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
) -> torch.Tensor:
    return q_idx >= 0


def modality(
    offset: int, length: int
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    def mask_fn(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        return (q_idx >= offset) & (kv_idx < (offset + length))

    return mask_fn


# code is borrowed from https://github.com/lucidrains/transfusion-pytorch
def omni_attn_mask(modalities):
    modalities = modalities.long()

    def mask_mod(b, h, q_idx, kv_idx):
        mask = causal(b, h, q_idx, kv_idx)

        modality_batch = modalities[b]

        for offset, length in modality_batch:
            mask = mask | modality(offset, length)(b, h, q_idx, kv_idx)

        return mask

    return mask_mod


def omni_attn_mask_naive(
    B: int,
    LEN: int,
    modalities: torch.Tensor,
    device: torch.device,
    inverted: bool = True,
) -> torch.Tensor:
    attention_mask = torch.tril(torch.ones((B, 1, LEN, LEN), dtype=torch.long)).to(
        device
    )
    for b in range(B):
        modality_batch = modalities[b]
        for offset, length in modality_batch:
            if offset < 0 or length <= 0:
                continue
            attention_mask[b, :, offset : offset + length, offset : offset + length] = 1

    if inverted:
        inverted_attention_mask = 1 - attention_mask
        inverted_attention_mask = inverted_attention_mask.masked_fill(
            inverted_attention_mask.to(torch.bool), torch.iinfo(torch.long).min
        )
        return inverted_attention_mask
    else:
        return attention_mask


def full_attn_mask_naive(
    B: int, LEN: int, device: torch.device, inverted: bool = True
) -> torch.Tensor:
    attention_mask = torch.ones((B, 1, LEN, LEN), dtype=torch.long).to(device)
    if inverted:
        inverted_attention_mask = 1 - attention_mask
        inverted_attention_mask = inverted_attention_mask.masked_fill(
            inverted_attention_mask.to(torch.bool), torch.iinfo(torch.long).min
        )
        return inverted_attention_mask
    else:
        return attention_mask


def causal_attn_mask_naive(B, LEN, device, inverted=True):
    attention_mask = torch.tril(torch.ones((B, 1, LEN, LEN), dtype=torch.long)).to(
        device
    )
    if inverted:
        inverted_attention_mask = 1 - attention_mask
        inverted_attention_mask = inverted_attention_mask.masked_fill(
            inverted_attention_mask.to(torch.bool), torch.iinfo(torch.long).min
        )
        return inverted_attention_mask
    else:
        return attention_mask


def omni_attn_mask_flexattention(
    modality_positions: torch.Tensor,  # [B, N, 2], -1 padded, no overlaps, in-bounds
    seq_len: int,
    num_heads: int = 1,
    block_size: int = 128,
    device: torch.device | str | None = None,
    compile_mask: bool = False,
):
    if device is None:
        device = modality_positions.device
    B, N, _ = modality_positions.shape
    L = seq_len

    offs = modality_positions[..., 0].to(device, torch.int64)  # [B,N]
    lens = modality_positions[..., 1].to(device, torch.int64)  # [B,N]
    valid = offs >= 0  # lengths >0 assumed when valid

    t = torch.arange(L, device=device, dtype=torch.int64)[None, None, :]  # [1,1,L]
    spans = (
        (t >= offs[..., None]) & (t < (offs + lens)[..., None]) & valid[..., None]
    )  # [B,N,L]

    # Winner-takes-all labeling across spans (no overlaps => irrelevant)
    labels = torch.arange(1, N + 1, device=device, dtype=torch.int64)[
        None, :, None
    ]  # 1..N
    span_id = (
        (spans.to(torch.int64) * labels).max(dim=1).values.to(torch.int32)
    )  # [B,L]

    def mask_mod(b, h, q, k):
        ids = span_id[b]  # [L]
        iq, ik = ids[q.to(torch.long)], ids[k.to(torch.long)]
        return (q >= k) | ((iq != 0) & (iq == ik))  # causal OR same (nonzero) span

    return create_block_mask(
        mask_mod,
        B=B,
        H=num_heads,
        Q_LEN=L,
        KV_LEN=L,
        device=device,
        BLOCK_SIZE=block_size,
        _compile=compile_mask,
    )


@torch.no_grad()
def extend_block_mask_causal_append(old_bm: BlockMask, new_len: int) -> BlockMask:
    """
    Extend an existing FlexAttention BlockMask from length L to `new_len` (>= L+1)
    for *autoregressive decoding*, where the *new last token* can attend to all
    previous tokens. Works for arbitrary B and H (vectorized over their dims).

    Notes
    -----
    - We modify only the *last Q row-tile* to include all KV block-columns
      up to `new_len`. Earlier rows are left unchanged, so past tokens do NOT
      see the future token.
    - We keep `mask_mod` from the old BlockMask so partial blocks still get
      elementwise masking (e.g., mixed causal/full spans).
    - We skip `full_kv_*` (optional fast-path); correctness is unaffected.

    Returns
    -------
    BlockMask for (Q_LEN=new_len, KV_LEN=new_len)
    """
    # --- read old metadata ---
    old_q_len, old_kv_len = old_bm.seq_lengths
    assert old_q_len == old_kv_len, "This helper expects self-attention masks."
    assert new_len >= old_q_len + 1, "Use a larger new_len (must append at least 1)."

    KV_BS, Q_BS = old_bm.BLOCK_SIZE
    device = old_bm.kv_indices.device
    dtype_nb = old_bm.kv_num_blocks.dtype
    dtype_idx = old_bm.kv_indices.dtype

    # tile counts (rows = Q tiles, cols = KV tiles)
    rows_old = (old_q_len + Q_BS - 1) // Q_BS
    rows_new = (new_len + Q_BS - 1) // Q_BS
    cols_new = (new_len + KV_BS - 1) // KV_BS

    # kv_num_blocks shape = [..., rows]; kv_indices shape = [..., rows, max_blocks]
    leading = old_bm.kv_num_blocks.shape[:-1]
    old_max_blocks = old_bm.kv_indices.shape[-1]
    new_max_blocks = max(old_max_blocks, cols_new)

    # allocate new containers
    new_kv_num_blocks = torch.zeros(*leading, rows_new, dtype=dtype_nb, device=device)
    new_kv_indices = torch.zeros(
        *leading, rows_new, new_max_blocks, dtype=dtype_idx, device=device
    )

    # copy old rows verbatim
    new_kv_num_blocks[..., :rows_old] = old_bm.kv_num_blocks
    new_kv_indices[..., :rows_old, :old_max_blocks] = old_bm.kv_indices

    # identify the (single) new last row-tile that contains the appended token
    last_row = rows_new - 1

    # ensure the last row-tile includes *all* KV block-columns up to cols_new
    arange_cols = torch.arange(cols_new, device=device, dtype=dtype_idx)
    # broadcast to any leading (B,H,...) dims automatically
    new_kv_indices[..., last_row, :cols_new] = arange_cols
    new_kv_num_blocks[..., last_row] = cols_new

    # IMPORTANT:
    #  - We did NOT touch earlier rows, so they won't see the new kv column if cols_new>cols_old.
    #  - Within the last row-tile, only the true last query position should see all columns.
    #    Earlier queries in that tile stay correct because we keep the original `mask_mod`,
    #    which applies elementwise masking inside partially-filled tiles.

    # rebuild a BlockMask; carry over mask_mod for partial-block semantics
    new_bm = BlockMask.from_kv_blocks(
        new_kv_num_blocks,
        new_kv_indices,
        BLOCK_SIZE=(KV_BS, Q_BS),
        mask_mod=old_bm.mask_mod,
        seq_lengths=(new_len, new_len),
    )
    return new_bm


@torch.no_grad()
def extend_block_mask_by_one(old_bm: BlockMask) -> BlockMask:
    """Convenience wrapper for the common '+1 token' case."""
    L, _ = old_bm.seq_lengths
    return extend_block_mask_causal_append(old_bm, L + 1)


@torch.no_grad()
def step_block_mask_from_old(
    old_bm: BlockMask,
    new_kv_len: int,  # KV length after appending the new token
    keep_elementwise: bool = False,  # reuse old_bm.mask_mod (if any) for partial-block rules
) -> BlockMask:
    """
    Build a BlockMask for step decoding (Q_LEN=1). The single query attends to
    *all* KV positions [0..new_kv_len-1]. Uses the old mask's BLOCK_SIZE/leading dims.
    """
    # --- read metadata ---
    _, old_kv_len = old_bm.seq_lengths
    assert new_kv_len >= old_kv_len, "KV length should be non-decreasing."
    KV_BS, Q_BS = old_bm.BLOCK_SIZE
    device = old_bm.kv_indices.device
    dtype_nb = old_bm.kv_num_blocks.dtype
    dtype_idx = old_bm.kv_indices.dtype

    # one query row; how many KV block-columns?
    cols_new = (new_kv_len + KV_BS - 1) // KV_BS
    leading = old_bm.kv_num_blocks.shape[:-1]  # e.g., [B, H]
    old_max_blocks = old_bm.kv_indices.shape[-1]
    new_max_blocks = max(old_max_blocks, cols_new)

    # allocate block lists
    kv_num_blocks = torch.zeros(*leading, 1, dtype=dtype_nb, device=device)
    kv_indices = torch.zeros(
        *leading, 1, new_max_blocks, dtype=dtype_idx, device=device
    )

    # single query attends all KV blocks
    kv_indices[..., 0, :cols_new] = torch.arange(
        cols_new, device=device, dtype=dtype_idx
    )
    kv_num_blocks[..., 0] = cols_new

    # mask_mod must return a 0-D torch.bool tensor (NOT a Python bool)
    if keep_elementwise and getattr(old_bm, "mask_mod", None) is not None:
        mask_mod = old_bm.mask_mod
    else:

        def mask_mod(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            # Always-true *tensor*; safe for Inductor
            return kv_idx >= kv_idx  # yields a 0-D torch.bool tensor == True

    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        BLOCK_SIZE=(KV_BS, Q_BS),
        mask_mod=mask_mod,
        seq_lengths=(1, new_kv_len),
    )
