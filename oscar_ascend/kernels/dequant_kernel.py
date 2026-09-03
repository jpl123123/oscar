"""oscar_ascend.kernels.dequant — 前缀反量化（Triton `_oscar_full_dequant_kv` port + 分发）。"""
from __future__ import annotations

import math

import torch

from ..format import K_IDX_OFF, META_BYTES, VALUES_PER_BYTE
from .decode_kernel import oscar_full_dequant_ref

try:
    from vllm.triton_utils import triton, tl  # type: ignore
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _oscar_full_dequant_kv(
        KCache8_ptr, VCache8_ptr,
        BlockTable_ptr,      # [T] kernel 粒度
        K_out_ptr, V_out_ptr,  # [T, Hk, D] fp16 (rotated space)
        stride_ko_t, stride_ko_h, stride_ko_d,
        stride_vo_t, stride_vo_h, stride_vo_d,
        stride_kb, stride_kp, stride_kh,
        stride_vb, stride_vp, stride_vh,
        HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr, NUM_KV_HEADS: tl.constexpr,
        DATA_BYTES: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pos = tl.program_id(0)
        bh = tl.program_id(1)
        bid = bh // NUM_KV_HEADS
        hid = bh % NUM_KV_HEADS
        page_idx = pos // BLOCK_SIZE
        page_off = pos % BLOCK_SIZE
        block_num = tl.load(BlockTable_ptr + bid * 1 + page_idx).to(tl.int64)
        k_off = block_num * stride_kb + tl.cast(page_off, tl.int64) * stride_kp \
            + tl.cast(hid, tl.int64) * stride_kh
        v_off = block_num * stride_vb + tl.cast(page_off, tl.int64) * stride_vp \
            + tl.cast(hid, tl.int64) * stride_vh

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < HEAD_DIM
        byte_idx = d_offs // 4
        bit_shift = (d_offs % 4) * 2

        k_byte = tl.load(KCache8_ptr + k_off + (K_IDX_OFF + byte_idx), mask=d_mask, other=0).to(tl.int32)
        q_k = ((k_byte >> bit_shift) & 3).to(tl.float32)
        ksc_lo = tl.load(KCache8_ptr + k_off).to(tl.uint16)
        ksc_hi = tl.load(KCache8_ptr + k_off + 1).to(tl.uint16)
        k_scale = (ksc_lo | (ksc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        kzr_lo = tl.load(KCache8_ptr + k_off + 2).to(tl.uint16)
        kzr_hi = tl.load(KCache8_ptr + k_off + 3).to(tl.uint16)
        k_zero = (kzr_lo | (kzr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        k_recon = q_k * k_scale + k_zero
        tl.store(
            K_out_ptr + pos * stride_ko_t + hid * stride_ko_h + d_offs,
            k_recon.to(tl.float16), mask=d_mask,
        )

        v_byte = tl.load(VCache8_ptr + v_off + byte_idx, mask=d_mask, other=0).to(tl.int32)
        q_v = ((v_byte >> bit_shift) & 3).to(tl.float32)
        vsc_lo = tl.load(KCache8_ptr + k_off + 4).to(tl.uint16)
        vsc_hi = tl.load(KCache8_ptr + k_off + 5).to(tl.uint16)
        v_scale = (vsc_lo | (vsc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        vzr_lo = tl.load(KCache8_ptr + k_off + 6).to(tl.uint16)
        vzr_hi = tl.load(KCache8_ptr + k_off + 7).to(tl.uint16)
        v_zero = (vzr_lo | (vzr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        v_recon = q_v * v_scale + v_zero
        tl.store(
            V_out_ptr + pos * stride_vo_t + hid * stride_vo_h + d_offs,
            v_recon.to(tl.float16), mask=d_mask,
        )


def oscar_full_dequant_triton(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    cached_len: int,
    hk: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    k8, v8 = k_cache.view(torch.uint8), v_cache.view(torch.uint8)
    bs = k8.shape[1]
    alloc_len = max(1, math.ceil(cached_len / bs) * bs)
    BLOCK_D = triton.next_power_of_2(D)
    k_buf = torch.empty(alloc_len, hk, D, dtype=torch.float16, device=k_cache.device)
    v_buf = torch.empty_like(k_buf)
    _oscar_full_dequant_kv[(alloc_len, hk)](
        k8, v8, block_table_row, k_buf, v_buf,
        k_buf.stride(0), k_buf.stride(1), k_buf.stride(2),
        v_buf.stride(0), v_buf.stride(1), v_buf.stride(2),
        k8.stride(0), k8.stride(1), k8.stride(2),
        v8.stride(0), v8.stride(1), v8.stride(2),
        HEAD_DIM=D, BLOCK_SIZE=bs, NUM_KV_HEADS=hk,
        DATA_BYTES=D // VALUES_PER_BYTE, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=1,
    )
    return k_buf[:cached_len], v_buf[:cached_len]


def oscar_full_dequant(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    cached_len: int,
    hk: int,
    D: int,
    use_triton: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_triton and triton is not None:
        return oscar_full_dequant_triton(k_cache, v_cache, block_table_row, cached_len, hk, D)
    return oscar_full_dequant_ref(k_cache, v_cache, block_table_row, cached_len, hk, D)
