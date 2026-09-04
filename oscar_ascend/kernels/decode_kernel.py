"""oscar_ascend.kernels.decode — INT2 fused decode + 前缀反量化（参考实现 + Triton）。

参考实现（torch，CPU/NPU 通用）：
  * oscar_decode_ref         —— decode：INT2 解包 → SDPA 打分（B×Hq×L 循环，匹配语义）
  * oscar_prefill_ref        —— prefill continuation：反量化前缀 + concat 当前 chunk → 因果 SDPA
  * oscar_full_dequant_ref   —— [cached_len, Hk, D] 前缀反量化（rotated space）

Triton（triton-ascend）：port PR #46774 `triton_oscar_decode.py` stage1/stage2，
按本插件槽偏移（K 槽 meta@0-7 + Kidx@32，V 槽 Vidx@0-63）。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..format import (
    K_IDX_OFF,
    META_BYTES,
    VALUES_PER_BYTE,
    check_d,
    f16_be_from_le,
)
from .store_kernel import dequant_split_ref, gather_kv_ref

try:
    from vllm.triton_utils import triton, tl  # type: ignore
except Exception:  # pragma: no cover
    triton = None
    tl = None


# ---------------------------------------------------------------------------
# 参考实现
# ---------------------------------------------------------------------------
def oscar_decode_ref(
    q: torch.Tensor,              # [B, Hq, D] 已旋转 Q@R_k
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,    # [B, T] kernel 粒度
    seq_lens: torch.Tensor,       # [B]
    scale: float,
    hk: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (out [B,Hq,D] fp32 rotated-V 空间, lse [B,Hq] fp32)。"""
    ks, vs = gather_kv_ref(k_cache, v_cache, block_table, seq_lens, hk, D)
    B, Hq = q.shape[0], q.shape[1]
    g = Hq // hk
    qf = q.float()
    out = torch.empty(B, Hq, D, dtype=torch.float32, device=q.device)
    lse = torch.empty(B, Hq, dtype=torch.float32, device=q.device)
    for b in range(B):
        k = ks[b].float()          # [L, Hk, D]
        v = vs[b].float()
        k_rep = k.repeat_interleave(g, dim=1)     # [L, Hq, D]
        v_rep = v.repeat_interleave(g, dim=1)
        scores = torch.einsum("hd,lhd->hl", qf[b], k_rep) * scale   # [Hq, L]
        p = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hl,lhd->hd", p, v_rep)
        lse[b] = torch.logsumexp(scores, dim=-1)
    return out, lse


def oscar_prefill_ref(
    q_chunk: torch.Tensor,        # [N, Hq, D]
    k_chunk: torch.Tensor,        # [N, Hk, D]
    v_chunk: torch.Tensor,        # [N, Hk, D]
    k_cached: torch.Tensor,       # [C, Hk, D]（rotated space，尚未逆旋转——由调用方先逆旋转）
    v_cached: torch.Tensor,
    scale: float,
    hk: int,
    D: int,
) -> torch.Tensor:
    """q 与 k/v 均为原空间调用方传入；缓存部分先由调用方逆旋转成原空间。"""
    N, Hq = q_chunk.shape[0], q_chunk.shape[1]
    C = k_cached.shape[0]
    g = Hq // hk
    k_full = torch.cat([k_cached, k_chunk], dim=0).float().transpose(0, 1).unsqueeze(0)
    v_full = torch.cat([v_cached, v_chunk], dim=0).float().transpose(0, 1).unsqueeze(0)
    q_t = q_chunk.float().transpose(0, 1).unsqueeze(0)
    if C <= 0:
        out = F.scaled_dot_product_attention(
            q_t, k_full, v_full, is_causal=True, scale=scale, enable_gqa=(g > 1)
        )
    else:
        q_pos = torch.arange(N, device=q_chunk.device).unsqueeze(1) + C   # [N,1]
        k_pos = torch.arange(C + N, device=q_chunk.device).unsqueeze(0)   # [1,C+N]
        mask = k_pos <= q_pos
        out = F.scaled_dot_product_attention(
            q_t, k_full, v_full, attn_mask=mask, scale=scale, enable_gqa=(g > 1)
        )
    return out[0].transpose(0, 1)[:, :Hq] if out.shape[1] != Hq else out[0].transpose(0, 1)


def oscar_full_dequant_ref(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table_row: torch.Tensor,   # [T] kernel 粒度块号（单序列）
    cached_len: int,
    hk: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    k8, v8 = k_cache.view(torch.uint8), v_cache.view(torch.uint8)
    bs = k8.shape[1]
    blk_idx = torch.arange(cached_len, device=k8.device) // bs
    pos = torch.arange(cached_len, device=k8.device) % bs
    bnums = block_table_row[blk_idx]
    k, v = dequant_split_ref(k8, v8, bnums, pos, hk, D)   # [C, Hk, D]
    return k, v


# ---------------------------------------------------------------------------
# Triton kernels（port PR; 槽偏移见模块 docstring）
# ---------------------------------------------------------------------------
def _triton_required(fn_name: str):
    if triton is None or tl is None:
        raise RuntimeError(
            f"triton 不可用（HAS_TRITON=False），无法执行 {fn_name}；"
            "请安装 triton-ascend 或设置 OSCAR_ASCEND_FORCE_TORCH=1 走 torch 参考路径"
        )


if triton is not None:

    @triton.jit
    def _oscar_decode_stage1(
        Q_rot_ptr,          # [B, Hq, D] fp32
        KCache8_ptr, VCache8_ptr,   # flat uint8
        BlockTable_ptr,     # [B, T] int32
        SeqLens_ptr,        # [B] int32
        Mid_o_ptr,          # [B, Hq, NUM_SPLITS, D+1] fp32
        stride_qb, stride_qh,
        stride_kb, stride_kp, stride_kh,
        stride_vb, stride_vp, stride_vh,
        stride_bt_b,
        stride_mid_b, stride_mid_h, stride_mid_s,
        NUM_KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
        NUM_KV_SPLITS: tl.constexpr, KV_GROUP_SIZE: tl.constexpr,
        DATA_BYTES: tl.constexpr, ATTN_SCALE: tl.constexpr,
        K_IDX_OFF: tl.constexpr,
        BLOCK_D: tl.constexpr, BLOCK_KV: tl.constexpr,
    ):
        bid = tl.program_id(0)
        hid = tl.program_id(1)
        sid = tl.program_id(2)
        kv_head = hid // KV_GROUP_SIZE
        seq_len = tl.load(SeqLens_ptr + bid)
        split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
        split_start = split_len * sid
        split_end = tl.minimum(split_start + split_len, seq_len)
        if split_start >= split_end:
            return
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < HEAD_DIM
        kv_range = tl.arange(0, BLOCK_KV)
        byte_idx = d_offs // 4
        bit_shift = (d_offs % 4) * 2
        q_base = bid * stride_qb + hid * stride_qh
        q_rot = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
        m_prev = -float("inf")
        l_prev = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        bt_base = bid * stride_bt_b
        for start_n in range(split_start, split_end, BLOCK_KV):
            kv_offs = start_n + kv_range
            kv_mask = kv_offs < split_end
            page_idx = kv_offs // BLOCK_SIZE
            page_off = kv_offs % BLOCK_SIZE
            block_nums = tl.load(
                BlockTable_ptr + bt_base + page_idx, mask=kv_mask, other=0
            ).to(tl.int64)
            k_slot = (
                block_nums * stride_kb + page_off.to(tl.int64) * stride_kp
                + tl.cast(kv_head, tl.int64) * stride_kh
            )
            v_slot = (
                block_nums * stride_vb + page_off.to(tl.int64) * stride_vp
                + tl.cast(kv_head, tl.int64) * stride_vh
            )
            # ---- K: meta[0..7] + idx[32..32+D/4] ----
            k_byte = tl.load(
                KCache8_ptr + k_slot[:, None] + (K_IDX_OFF + byte_idx[None, :]),
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            q_k = ((k_byte >> bit_shift[None, :]) & 3).to(tl.float32)
            k_meta_base = k_slot
            ksc_lo = tl.load(KCache8_ptr + k_meta_base, mask=kv_mask, other=0).to(tl.uint16)
            ksc_hi = tl.load(KCache8_ptr + k_meta_base + 1, mask=kv_mask, other=0).to(tl.uint16)
            k_scale = (ksc_lo | (ksc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            kzr_lo = tl.load(KCache8_ptr + k_meta_base + 2, mask=kv_mask, other=0).to(tl.uint16)
            kzr_hi = tl.load(KCache8_ptr + k_meta_base + 3, mask=kv_mask, other=0).to(tl.uint16)
            k_zero = (kzr_lo | (kzr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            k_deq = q_k * k_scale[:, None] + k_zero[:, None]
            scores = (
                tl.sum(tl.where(d_mask[None, :], q_rot[None, :] * k_deq, 0.0), axis=1)
                * ATTN_SCALE
            )
            scores = tl.where(kv_mask, scores, -float("inf"))
            n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
            re_scale = tl.exp(m_prev - n_e_max)
            p = tl.exp(scores - n_e_max)
            # ---- V: idx[0..D/4]（meta 在 K 槽 +4/+6）----
            v_byte = tl.load(
                VCache8_ptr + v_slot[:, None] + byte_idx[None, :],
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            q_v = ((v_byte >> bit_shift[None, :]) & 3).to(tl.float32)
            vsc_lo = tl.load(KCache8_ptr + k_meta_base + 4, mask=kv_mask, other=0).to(tl.uint16)
            vsc_hi = tl.load(KCache8_ptr + k_meta_base + 5, mask=kv_mask, other=0).to(tl.uint16)
            v_scale = (vsc_lo | (vsc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            vzr_lo = tl.load(KCache8_ptr + k_meta_base + 6, mask=kv_mask, other=0).to(tl.uint16)
            vzr_hi = tl.load(KCache8_ptr + k_meta_base + 7, mask=kv_mask, other=0).to(tl.uint16)
            v_zero = (vzr_lo | (vzr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = q_v * v_scale[:, None] + v_zero[:, None]
            acc = acc * re_scale + tl.sum(p[:, None] * values, 0)
            l_prev = l_prev * re_scale + tl.sum(p, 0)
            m_prev = n_e_max
        out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
        safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
        tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
        tl.store(Mid_o_ptr + out_base + HEAD_DIM, m_prev + tl.log(safe_l))

    @triton.jit
    def _oscar_decode_stage2(
        Mid_o_ptr,      # [B, Hq, S, D+1]
        Out_ptr,        # [B, Hq, D]
        Lse_ptr,        # [B, Hq]
        Seq_lens_ptr,   # [B] —— 空 split 结构守卫用（见循环内）
        stride_mid_b, stride_mid_h, stride_mid_s,
        stride_out_b, stride_out_h,
        stride_lse_b,
        NUM_KV_SPLITS: tl.constexpr, BLOCK_D: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        # 语义对齐 vLLM _fwd_kernel_stage2（vllm/v1/attention/ops/triton_decode_attention.py
        # :549-613）：① e_sum 跟踪 + 最终 term/e_sum 归一化（旧版丢失 → 输出整体差
        # Σexp(lse−M)=L 倍，真机 07:39 probe err=4.39 即此）；② 空 split 结构守卫
        # （seq_len < NUM_SPLITS×split_len 时 stage1 提前 return、mid 为 torch.empty
        # 垃圾，不可读——多请求短序列场景必现）。
        bid = tl.program_id(0)
        hid = tl.program_id(1)
        seq_len = tl.load(Seq_lens_ptr + bid)
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < HEAD_DIM
        base = bid * stride_mid_b + hid * stride_mid_h
        m = -float("inf")
        e_sum = 0.0
        term = tl.zeros([BLOCK_D], dtype=tl.float32)
        for s in range(NUM_KV_SPLITS):
            split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
            split_start = split_len * s
            split_end = tl.minimum(split_start + split_len, seq_len)
            if split_end > split_start:
                c = tl.load(Mid_o_ptr + base + s * stride_mid_s + HEAD_DIM)
                m_new = tl.maximum(c, m)
                old_scale = tl.exp(m - m_new)
                exp_logic = tl.exp(c - m_new)
                o = tl.load(Mid_o_ptr + base + s * stride_mid_s + d_offs, mask=d_mask, other=0.0)
                term = term * old_scale + exp_logic * o
                e_sum = e_sum * old_scale + exp_logic
                m = m_new
        tl.store(Out_ptr + bid * stride_out_b + hid * stride_out_h + d_offs, term / e_sum, mask=d_mask)
        tl.store(Lse_ptr + bid * stride_lse_b + hid, m + tl.log(e_sum))


if triton is not None:  # noqa: E305

    def oscar_decode_triton(
        q_rot: torch.Tensor,        # [B, Hq, D]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        scale: float,
        hk: int,
        D: int,
        max_num_kv_splits: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (out_rot [B,Hq,D] fp32, lse [B,Hq] fp32)。"""
        B, Hq = q_rot.shape[0], q_rot.shape[1]
        k8, v8 = k_cache.view(torch.uint8), v_cache.view(torch.uint8)
        bs = k8.shape[1]
        BLOCK_D = triton.next_power_of_2(D)
        NUM_SPLITS = max(1, min(max_num_kv_splits, max(1, int(seq_lens.max().item())))) \
            if seq_lens.numel() > 0 else 1
        mid_o = torch.empty(B, Hq, NUM_SPLITS, D + 1, dtype=torch.float32, device=q_rot.device)
        grid = (B, Hq, NUM_SPLITS)
        _oscar_decode_stage1[grid](
            q_rot.contiguous().float(),
            k8, v8, block_table, seq_lens, mid_o,
            q_rot.stride(0), q_rot.stride(1),
            k8.stride(0), k8.stride(1), k8.stride(2),
            v8.stride(0), v8.stride(1), v8.stride(2),
            block_table.stride(0),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            NUM_KV_HEADS=hk, HEAD_DIM=D, BLOCK_SIZE=bs,
            NUM_KV_SPLITS=NUM_SPLITS, KV_GROUP_SIZE=Hq // hk,
            DATA_BYTES=D // VALUES_PER_BYTE, ATTN_SCALE=scale,
            K_IDX_OFF=K_IDX_OFF,
            BLOCK_D=BLOCK_D, BLOCK_KV=4,
            num_warps=1, num_stages=1,
        )
        out = torch.empty(B, Hq, D, dtype=torch.float32, device=q_rot.device)
        lse = torch.empty(B, Hq, dtype=torch.float32, device=q_rot.device)
        _oscar_decode_stage2[(B, Hq)](
            mid_o, out, lse, seq_lens,
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            out.stride(0), out.stride(1), lse.stride(0),
            NUM_KV_SPLITS=NUM_SPLITS, BLOCK_D=BLOCK_D, HEAD_DIM=D,
            num_warps=4, num_stages=1,
        )
        return out, lse
