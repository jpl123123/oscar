"""oscar_ascend.kernels — Triton / torch 双路径算子（store / decode / dequant）。

约定：
  * 缓存视图 k8/v8 为 (nb, bs, hk, 512) uint8（D=256 时每 token 每头 512B 原生槽），
    STRIDES 均以字节为单位传入（uint8 元素 = 1 字节）。
  * 槽内偏移（逻辑 160B = K 槽 96B ⊕ V 槽 64B，N-01）：
      K 槽: [0:2] K scale LE | [2:4] K zero LE | [4:6] V scale LE | [6:8] V zero LE
            | [8:32] pad | [32:32+D/4] K idx
      V 槽: [0:D/4] V idx
  * 参考实现（torch，CPU/NPU 通用）= format.py 的同一量化/打包函数；
    Triton 内核必须与参考实现逐字节一致（probe 判据：字节差 == 0）。
"""
from __future__ import annotations

import torch

from ..format import (
    K_IDX_OFF,
    LEVELS,
    META_BYTES,
    SCALE_FLOOR,
    VALUES_PER_BYTE,
    check_d,
    f16_le,
    quantize,
)

try:  # triton 可选；无 triton 时 (store/decode/dequant)_triton 调用即报错
    from vllm.triton_utils import triton, tl  # type: ignore
except Exception:  # pragma: no cover
    triton = None
    tl = None


def _slot_bases(
    slot_mapping: torch.Tensor, bs: int, hk: int, k8: torch.Tensor, v8: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """slot_mapping [N]（kernel 粒度 token 单位，L-20260831-01）→ (k_off[N,H], v_off[N,H]) 字节偏移。"""
    blk = slot_mapping // bs
    off = slot_mapping % bs
    h = torch.arange(hk, device=slot_mapping.device)
    kb = k8.stride(0)  # 字节
    kp = k8.stride(1)
    kh = k8.stride(2)
    vb, vp, vh = v8.stride(0), v8.stride(1), v8.stride(2)
    k_off = (blk * kb + off * kp).unsqueeze(-1) + h * kh     # [N, H] int64
    v_off = (blk * vb + off * vp).unsqueeze(-1) + h * vh
    return k_off, v_off


# ---------------------------------------------------------------------------
# 参考实现（torch；CPU/NPU 通用；被 probe/测试当作基准）
# ---------------------------------------------------------------------------
def oscar_store_ref(
    k_rot: torch.Tensor,          # [N, H, D] fp32/fp16 已旋转（+裁剪）
    v_rot: torch.Tensor,          # [N, H, D]
    k_cache: torch.Tensor,        # (nb, bs, hk, D) bf16/fp16 原生视图（或其 uint8 视图）
    v_cache: torch.Tensor,        # (nb, bs, hk, D)
    slot_mapping: torch.Tensor,   # [N] int64（kernel 粒度 token 单位）
) -> None:
    k8 = k_cache.view(torch.uint8)
    v8 = v_cache.view(torch.uint8)
    assert k8.is_contiguous() and v8.is_contiguous(), "k8/v8 必须连续（组张量内切片）"
    N, H, D = k_rot.shape
    check_d(D)
    data_bytes = D // VALUES_PER_BYTE
    bs = k8.shape[1]
    k_off, v_off = _slot_bases(slot_mapping, bs, H, k8, v8)     # [N, H]

    k_packed, k_scale, k_zero = quantize(k_rot.float().reshape(N, H, D))
    v_packed, v_scale, v_zero = quantize(v_rot.float().reshape(N, H, D))
    ks_lo, ks_hi = f16_le(k_scale)   # [N,H,1]
    kz_lo, kz_hi = f16_le(k_zero)
    vs_lo, vs_hi = f16_le(v_scale)
    vz_lo, vz_hi = f16_le(v_zero)

    flat_k = k8.view(-1)
    flat_v = v8.view(-1)
    io = torch.arange(data_bytes, device=k8.device)

    # K 槽 meta：K scale(0-1) K zero(2-3) V scale(4-5) V zero(6-7)
    flat_k.index_put_((k_off + 0,), ks_lo.squeeze(-1))
    flat_k.index_put_((k_off + 1,), ks_hi.squeeze(-1))
    flat_k.index_put_((k_off + 2,), kz_lo.squeeze(-1))
    flat_k.index_put_((k_off + 3,), kz_hi.squeeze(-1))
    flat_k.index_put_((k_off + 4,), vs_lo.squeeze(-1))
    flat_k.index_put_((k_off + 5,), vs_hi.squeeze(-1))
    flat_k.index_put_((k_off + 6,), vz_lo.squeeze(-1))
    flat_k.index_put_((k_off + 7,), vz_hi.squeeze(-1))
    # K 槽 idx @ +32
    k_w = (k_off.unsqueeze(-1) + K_IDX_OFF) + io
    flat_k.index_put_((k_w,), k_packed)
    # V 槽 idx @ +0
    v_w = v_off.unsqueeze(-1) + io
    flat_v.index_put_((v_w,), v_packed)


def dequant_split_ref(
    k8: torch.Tensor, v8: torch.Tensor,
    block_idx: torch.Tensor,   # [T] 物理 kernel 块号
    pos: torch.Tensor,         # [T] 块内位置
    hk: int, D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 block_idx/pos（token 单位）反量化 → (k[T,H,D], v[T,H,D]) fp32（rotated space）。"""
    check_d(D)
    data_bytes = D // VALUES_PER_BYTE
    k_off = block_idx * k8.stride(0) + pos * k8.stride(1)
    v_off = block_idx * v8.stride(0) + pos * v8.stride(1)
    h = torch.arange(hk, device=k8.device)
    k_off = (k_off.unsqueeze(-1) + h * k8.stride(2)).unsqueeze(-1)     # [T,H,1]
    v_off = (v_off.unsqueeze(-1) + h * v8.stride(2)).unsqueeze(-1)
    io = torch.arange(data_bytes, device=k8.device)
    flat_k, flat_v = k8.view(-1), v8.view(-1)

    kmeta = flat_k[k_off + torch.arange(META_BYTES, device=k8.device)]   # [T,H,8]
    from ..format import f16_be_from_le
    k_scale = f16_be_from_le(kmeta[..., 0:1], kmeta[..., 1:2])
    k_zero = f16_be_from_le(kmeta[..., 2:3], kmeta[..., 3:4])
    v_scale = f16_be_from_le(kmeta[..., 4:5], kmeta[..., 5:6])
    v_zero = f16_be_from_le(kmeta[..., 6:7], kmeta[..., 7:8])
    k_packed = flat_k[k_off + K_IDX_OFF + io]                            # [T,H,D/4]
    v_packed = flat_v[v_off + io]
    from ..format import dequant
    k = dequant(k_packed, k_scale, k_zero, D)      # [T,H,D]
    v = dequant(v_packed, v_scale, v_zero, D)
    return k, v


def gather_kv_ref(
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    block_table: torch.Tensor,   # [B, T] kernel 粒度块号
    seq_lens: torch.Tensor,      # [B] int32（含当前 token）
    hk: int, D: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """decode 用：按 block_table 收集每组序列 K/V（rotated space fp32）。"""
    k8, v8 = k_cache.view(torch.uint8), v_cache.view(torch.uint8)
    bs = k8.shape[1]
    ks, vs = [], []
    for b in range(block_table.shape[0]):
        L = int(seq_lens[b])
        blk_idx = torch.arange(L, device=k8.device) // bs
        pos = torch.arange(L, device=k8.device) % bs
        bnums = block_table[b][blk_idx]
        k, v = dequant_split_ref(k8, v8, bnums, pos, hk, D)
        ks.append(k)
        vs.append(v)
    return ks, vs


# ---------------------------------------------------------------------------
# Triton 内核（port PR#46774 triton_oscar_store.py，按本插件拆槽偏移；字节=ref）
# ---------------------------------------------------------------------------
if triton is not None:

    @triton.jit
    def _quant_pack_vec(
        Src_ptr, base, d_offs, d_mask,
        D: tl.constexpr, LEVELS: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        vec = tl.load(Src_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
        vmin = tl.min(tl.where(d_mask, vec, float("inf")), axis=0)
        vmax = tl.max(tl.where(d_mask, vec, -float("inf")), axis=0)
        scale = (vmax - vmin) / (LEVELS - 1)
        scale = tl.where(scale > 1e-8, scale, 1e-8)
        scale_f16 = scale.to(tl.float16)
        zero_f16 = vmin.to(tl.float16)
        scale = scale_f16.to(tl.float32)
        zero = zero_f16.to(tl.float32)
        q = tl.minimum(tl.maximum(((vec - zero) / scale + 0.5).to(tl.int32), 0), LEVELS - 1)
        q_grp = tl.reshape(q, [BLOCK_D // 4, 4])
        shifts = tl.arange(0, 4) * 2
        packed = tl.sum((q_grp & 0x3) << shifts[None, :], axis=1).to(tl.uint8)
        return packed, scale_f16, zero_f16

    @triton.jit
    def _oscar_store_kernel(
        Key_ptr, Value_ptr,       # [NH, D] fp32 已旋转（+裁剪）
        KCache8_ptr, VCache8_ptr,
        Slot_mapping_ptr,         # [N]
        stride_kb, stride_kp, stride_kh,
        stride_vb, stride_vp, stride_vh,
        D: tl.constexpr, H: tl.constexpr, BLOCK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr, BLOCK_PACK: tl.constexpr, K_IDX_OFF: tl.constexpr,
    ):
        pid = tl.program_id(0)
        token_idx = pid // H
        head_idx = pid % H
        slot = tl.load(Slot_mapping_ptr + token_idx)
        if slot < 0:
            return
        blk = (slot // BLOCK_SIZE).to(tl.int64)
        off = (slot % BLOCK_SIZE).to(tl.int64)
        k_slot_base = (
            blk * stride_kb + off * stride_kp + tl.cast(head_idx, tl.int64) * stride_kh
        )
        v_slot_base = (
            blk * stride_vb + off * stride_vp + tl.cast(head_idx, tl.int64) * stride_vh
        )
        base = pid * D
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        packed_offs = tl.arange(0, BLOCK_PACK)
        pack_mask = packed_offs < (D // 4)

        k_packed, k_scale, k_zero = _quant_pack_vec(
            Key_ptr, base, d_offs, d_mask, D=D, LEVELS=4, BLOCK_D=BLOCK_D
        )
        v_packed, v_scale, v_zero = _quant_pack_vec(
            Value_ptr, base, d_offs, d_mask, D=D, LEVELS=4, BLOCK_D=BLOCK_D
        )
        # K 槽 meta[0..7]：K scale/zero @0..3，V scale/zero @4..7（拼接后=N-01）
        ks_u16 = k_scale.to(tl.uint16, bitcast=True)
        kz_u16 = k_zero.to(tl.uint16, bitcast=True)
        vs_u16 = v_scale.to(tl.uint16, bitcast=True)
        vz_u16 = v_zero.to(tl.uint16, bitcast=True)
        tl.store(KCache8_ptr + k_slot_base + 0, (ks_u16 & 0xFF).to(tl.uint8))
        tl.store(KCache8_ptr + k_slot_base + 1, ((ks_u16 >> 8) & 0xFF).to(tl.uint8))
        tl.store(KCache8_ptr + k_slot_base + 2, (kz_u16 & 0xFF).to(tl.uint8))
        tl.store(KCache8_ptr + k_slot_base + 3, ((kz_u16 >> 8) & 0xFF).to(tl.uint8))
        tl.store(KCache8_ptr + k_slot_base + 4, (vs_u16 & 0xFF).to(tl.uint8))
        tl.store(KCache8_ptr + k_slot_base + 5, ((vs_u16 >> 8) & 0xFF).to(tl.uint8))
        tl.store(KCache8_ptr + k_slot_base + 6, (vz_u16 & 0xFF).to(tl.uint8))
        tl.store(KCache8_ptr + k_slot_base + 7, ((vz_u16 >> 8) & 0xFF).to(tl.uint8))
        # 索引区
        tl.store(
            KCache8_ptr + k_slot_base + (K_IDX_OFF + packed_offs),
            k_packed, mask=pack_mask,
        )
        tl.store(VCache8_ptr + v_slot_base + packed_offs, v_packed, mask=pack_mask)


def oscar_store_triton(
    k_rot: torch.Tensor, v_rot: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Triton INT2 量化打包散写（与 oscar_store_ref 逐字节一致）。"""
    if triton is None:
        raise RuntimeError("triton 不可用")
    N, H, D = k_rot.shape
    if N == 0:
        return
    k8, v8 = k_cache.view(torch.uint8), v_cache.view(torch.uint8)
    bs = k8.shape[1]
    BLOCK_PACK = triton.next_power_of_2(D // VALUES_PER_BYTE)
    k_flat = k_rot.reshape(N * H, D).contiguous().float()
    v_flat = v_rot.reshape(N * H, D).contiguous().float()
    _oscar_store_kernel[(N * H,)](
        k_flat, v_flat, k8, v8, slot_mapping,
        k8.stride(0), k8.stride(1), k8.stride(2),
        v8.stride(0), v8.stride(1), v8.stride(2),
        D=D, H=H, BLOCK_SIZE=bs,
        BLOCK_D=triton.next_power_of_2(D), BLOCK_PACK=BLOCK_PACK,
        K_IDX_OFF=K_IDX_OFF,
        num_warps=4, num_stages=1,
    )
