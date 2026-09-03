"""oscar_ascend.format — OSCAR INT2 数值格式唯一权威（160B 逻辑槽，N-01 契约）。

与 skill 沙盒 `sandbox/l3_numeric/format.py` 同判据：
  N-01  槽 160B meta-first；N-02 per-vector 非对称量化（scale/zero 先 fp16 舍入）；
  N-03  打包位序 q[4b+k] << 2k（低索引低 2 位）；N-04 fp16 meta 小端 LE。

本文件为纯 torch（CPU/NPU 通用）参考实现；Triton 内核必须与本文件字节一致
（store 判据：字节差 == 0）。禁止从 output shape 反推 D；禁止静默 cast。
"""
from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# 槽几何（D 为入参；默认 256 = Qwen3.5-27B head_dim）
# ---------------------------------------------------------------------------
BITS = 2
LEVELS = 4                 # 2 ** BITS
VALUES_PER_BYTE = 4        # 2 ** BITS
D_DEFAULT = 256
DATA_BYTES = D_DEFAULT // VALUES_PER_BYTE          # 64B / 向量（D=256）

# 逻辑 160B 槽（meta-first，N-01）：拼接 K 槽前 96B ⊕ V 槽前 64B
SLOT_SIZE = 160
K_META_OFF = 0             # [0:2] K scale, [2:4] K zero, [4:6] V scale, [6:8] V zero
META_BYTES = 8
PAD_BYTES = 24             # [8:32]
K_IDX_OFF = 32             # [32:96]  K 索引 64B
V_IDX_OFF = 96             # [96:160] V 索引 64B（= V 槽偏移 0）
K_SLOT_BYTES = V_IDX_OFF   # 96B
V_SLOT_BYTES = SLOT_SIZE - V_IDX_OFF  # 64B

# 数值判据
DEQUANT_TOL = 1e-5
DECODE_TOL = 1e-4
SCALE_FLOOR = 1e-8


def check_d(D: int) -> None:
    if D % VALUES_PER_BYTE != 0:
        raise ValueError(f"D={D} 必须能被 {VALUES_PER_BYTE} 整除（INT2 打包要求）")
    if D // VALUES_PER_BYTE > K_SLOT_BYTES - META_BYTES - PAD_BYTES:
        raise ValueError(f"D={D} 超出单槽容量")


# ---------------------------------------------------------------------------
# 量化 / 反量化（per-vector 非对称 INT2；N-02 舍入顺序）
# ---------------------------------------------------------------------------
def quantize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x: [.., D] fp32（已旋转）→ (packed[.., D/4] uint8, scale[..,1] fp32, zero[..,1] fp32)。

    scale = max((max-min)/3, 1e-8) 先 fp16 舍入；zero = min 先 fp16 舍入；
    q = clamp(round((x - zero)/scale), 0, 3)。全部 torch 算子（CPU/NPU 通用）。
    """
    if x.dtype != torch.float32:
        x = x.float()
    vmin = x.amin(dim=-1, keepdim=True)
    vmax = x.amax(dim=-1, keepdim=True)
    scale = (vmax - vmin) / (LEVELS - 1)
    scale = torch.where(scale > SCALE_FLOOR, scale, torch.full_like(scale, SCALE_FLOOR))
    # N-02：先 fp16 舍入，再按舍入后的 scale/zero 量化（避免 0.5 边界 bin 翻转）
    scale_f16 = scale.half().float()
    zero_f16 = vmin.half().float()

    q = torch.clamp(torch.round((x - zero_f16) / scale_f16), 0, LEVELS - 1)
    q = q.to(torch.int32)  # 用 int32 位运算，避免 uint8 位移差异
    q4 = q.reshape(*q.shape[:-1], q.shape[-1] // VALUES_PER_BYTE, VALUES_PER_BYTE)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=x.device)
    packed = (q4 & (LEVELS - 1)) << shifts
    packed = packed.sum(dim=-1).to(torch.uint8)  # [.., D/4]
    return packed, scale_f16, zero_f16


def dequant(
    packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, D: int
) -> torch.Tensor:
    """packed [.., D/4] uint8 + scale/zero [..,1] fp32 → [.., D] fp32。"""
    q = _unpack(packed, D).to(torch.float32)
    return q * scale + zero


def _unpack(packed: torch.Tensor, D: int) -> torch.Tensor:
    """一字节 4×INT2 → [.., D] int32（N-03 位序）。"""
    p = packed.to(torch.int32).unsqueeze(-1)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=packed.device)
    q = (p >> shifts) & (LEVELS - 1)  # [.., D/4, 4]
    return q.reshape(*q.shape[:-2], D)


def f16_le(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp16 标量（[..,1] fp32）→ (lo, hi) uint8（N-04 小端）。"""
    h = x.contiguous().half()
    bits = h.view(torch.int16) & 0xFFFF  # 负 fp16 用模 2^16 窄化
    lo = (bits & 0xFF).to(torch.uint8)
    hi = ((bits >> 8) & 0xFF).to(torch.uint8)
    return lo, hi


def f16_be_from_le(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """(lo, hi) uint8 → fp16 标量（[..,1] fp32）。"""
    bits = lo.to(torch.int16) | (hi.to(torch.int16) << 8)
    return bits.view(torch.float16).float()


# ---------------------------------------------------------------------------
# 逻辑 160B 槽（供测试/探针对账；实际物理落位见 kernels 拆分槽）
# ---------------------------------------------------------------------------
def make_slot_bytes(k_rot: torch.Tensor, v_rot: torch.Tensor) -> torch.Tensor:
    """k/v [N,H,D]（已旋转）→ [N,H,160] uint8（N-01 字节序）。"""
    N, H, D = k_rot.shape
    check_d(D)
    data_bytes = D // VALUES_PER_BYTE
    k_packed, k_scale, k_zero = quantize(k_rot)
    v_packed, v_scale, v_zero = quantize(v_rot)

    slot = torch.zeros(N, H, SLOT_SIZE, dtype=torch.uint8, device=k_rot.device)
    ks_lo, ks_hi = f16_le(k_scale)     # [N,H,1]
    kz_lo, kz_hi = f16_le(k_zero)
    vs_lo, vs_hi = f16_le(v_scale)
    vz_lo, vz_hi = f16_le(v_zero)
    slot[..., 0:1] = ks_lo
    slot[..., 1:2] = ks_hi
    slot[..., 2:3] = kz_lo
    slot[..., 3:4] = kz_hi
    slot[..., 4:5] = vs_lo
    slot[..., 5:6] = vs_hi
    slot[..., 6:7] = vz_lo
    slot[..., 7:8] = vz_hi
    slot[..., K_IDX_OFF : K_IDX_OFF + data_bytes] = k_packed
    slot[..., V_IDX_OFF : V_IDX_OFF + data_bytes] = v_packed
    return slot


def parse_slot_bytes(slot: torch.Tensor, D: int) -> tuple[torch.Tensor, torch.Tensor]:
    """slot [..,160] uint8 → (k[..,D], v[..,D]) fp32。"""
    check_d(D)
    data_bytes = D // VALUES_PER_BYTE
    k_scale = f16_be_from_le(slot[..., 0:1], slot[..., 1:2])
    k_zero = f16_be_from_le(slot[..., 2:3], slot[..., 3:4])
    v_scale = f16_be_from_le(slot[..., 4:5], slot[..., 5:6])
    v_zero = f16_be_from_le(slot[..., 6:7], slot[..., 7:8])
    k = dequant(slot[..., K_IDX_OFF : K_IDX_OFF + data_bytes], k_scale, k_zero, D)
    v = dequant(slot[..., V_IDX_OFF : V_IDX_OFF + data_bytes], v_scale, v_zero, D)
    return k, v


# ---------------------------------------------------------------------------
# 旋转（正交不变性；仅数学约定，加载见 rotation.py）
# ---------------------------------------------------------------------------
def rotate(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """x [..., D] @ R [D, D]。"""
    return torch.matmul(x.float(), R)


def rotate_query_out(q: torch.Tensor, R_k: torch.Tensor, R_vT: torch.Tensor):
    return torch.matmul(q.float(), R_k), None  # out 由调用方乘 R_vT

__all__ = [
    "BITS", "LEVELS", "VALUES_PER_BYTE", "D_DEFAULT", "DATA_BYTES",
    "SLOT_SIZE", "K_META_OFF", "META_BYTES", "PAD_BYTES", "K_IDX_OFF", "V_IDX_OFF",
    "K_SLOT_BYTES", "V_SLOT_BYTES",
    "DEQUANT_TOL", "DECODE_TOL", "SCALE_FLOOR",
    "check_d", "quantize", "dequant", "f16_le", "f16_be_from_le",
    "make_slot_bytes", "parse_slot_bytes", "rotate",
]
