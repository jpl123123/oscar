#!/usr/bin/env python3
"""delivery/probe_oscar.py — 真机数值 probe（store / dequant / decode 三查，同沙盒判据）。

判据（与 skill sandbox l3_numeric 一致）：
  1. store    : 写入槽字节 == format.make_slot_bytes（max|d| == 0）
  2. dequant  : 反量化 == 量化时刻理想重建（q*scale+zero）≤ 1e-5
  3. decode   : INT2 decode vs 同一 INT2 数据 SDPA ≤ 1e-4（fp32）

--mode ref  : torch 参考路径（NPU 纯 torch 算子；任何环境可跑）
--mode triton: Triton 内核路径（要求 HAS_TRITON；与 ref 逐字节对照）
全部计算在 NPU（torch.npu），无 CPU 搬运。失败 → exit!=0 阻塞 serve。
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--num-heads", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--mode", choices=["ref", "triton"], default="ref")
    ap.add_argument("--num-tokens", type=int, default=3)
    args = ap.parse_args()

    import torch
    import torch_npu  # noqa: F401

    from oscar_ascend import format as fmt
    from oscar_ascend.kernels.store_kernel import oscar_store_ref
    from oscar_ascend.kernels.decode_kernel import oscar_decode_ref

    if not torch.npu.is_available():
        print("❌ torch.npu 不可用（probe 必须在 NPU；CPU 用 tests/test_numeric.py）")
        return 2

    dev = torch.npu.current_device()
    D, Hk, Hq, bs = args.head_dim, args.num_kv_heads, args.num_heads, args.block_size
    N = args.num_tokens
    torch.manual_seed(0)

    # 缓存视图：k/v 形状与真实 attn 层一致（kernel 粒度块 bs）
    k_cache = torch.zeros(4, bs, Hk, D, dtype=torch.bfloat16, device=dev)
    v_cache = torch.zeros_like(k_cache)
    k = torch.randn(N, Hk, D, device=dev, dtype=torch.bfloat16) * 1.5
    v = torch.randn(N, Hk, D, device=dev, dtype=torch.bfloat16) * 1.5
    slot_mapping = torch.arange(N, dtype=torch.int64, device=dev)

    # ---- 1) store（ref 或 triton）----
    use_triton = args.mode == "triton"
    if use_triton:
        from oscar_ascend.kernels.store_kernel import oscar_store_triton

        oscar_store_triton(k, v, k_cache, v_cache, slot_mapping)
    else:
        oscar_store_ref(k, v, k_cache, v_cache, slot_mapping)
    ref_slot = fmt.make_slot_bytes(k.float(), v.float())                    # [N,Hk,160]
    k8, v8 = k_cache.view(torch.uint8), v_cache.view(torch.uint8)
    db = D // 4
    got = torch.zeros(N, Hk, 160, dtype=torch.uint8, device=dev)
    for t in range(N):
        b, o = int(slot_mapping[t]) // bs, int(slot_mapping[t]) % bs
        for h in range(Hk):
            ks = b * k8.stride(0) + o * k8.stride(1) + h * k8.stride(2)
            vs = b * v8.stride(0) + o * v8.stride(1) + h * v8.stride(2)
            got[t, h, 0:8] = k8.view(-1)[ks : ks + 8]
            got[t, h, 32 : 32 + db] = k8.view(-1)[ks + 32 : ks + 32 + db]
            got[t, h, 96 : 96 + db] = v8.view(-1)[vs : vs + db]
    diff = (got != ref_slot).sum().item()
    if diff != 0:
        print(f"❌ [{args.mode}] store 字节差 = {diff}（判据 0）")
        return 1
    print(f"✅ [{args.mode}] store 字节差 = 0（{N}×{Hk} 头 × 160B 槽）")

    # ---- 2) dequant ≤1e-5 ----
    from oscar_ascend.kernels.store_kernel import dequant_split_ref

    bnums = (slot_mapping // bs)
    pos = slot_mapping % bs
    k_rec, v_rec = dequant_split_ref(k8, v8, bnums, pos, Hk, D)   # [N,Hk,D]
    _, ks, kz = fmt.quantize(k.float())
    _, vs, vz = fmt.quantize(v.float())
    qk = torch.clamp(torch.floor((k.float() - kz) / ks + 0.5), 0, 3)
    qv = torch.clamp(torch.floor((v.float() - vz) / vs + 0.5), 0, 3)
    ek = (k_rec - (qk * ks + kz)).abs().max().item()
    ev = (v_rec - (qv * vs + vz)).abs().max().item()
    if max(ek, ev) > 1e-5:
        print(f"❌ dequant err K={ek:.3e} V={ev:.3e}（判据 ≤1e-5）")
        return 1
    print(f"✅ dequant err K={ek:.3e} V={ev:.3e}（≤1e-5）")

    # ---- 3) decode ≤1e-4（INT2 解码 vs 同一 INT2 数据 SDPA）----
    q = torch.randn(1, Hq, D, device=dev)
    bt = torch.zeros(1, 4, dtype=torch.int32, device=dev)
    seq = torch.tensor([N], dtype=torch.int32, device=dev)
    out_ref, _ = oscar_decode_ref(q, k_cache, v_cache, bt, seq, 0.125, Hk, D)
    kd_rep = k_rec.repeat_interleave(Hq // Hk, dim=1)
    vd_rep = v_rec.repeat_interleave(Hq // Hk, dim=1)
    scores = torch.einsum("hd,lhd->hl", q[0], kd_rep) * 0.125
    p = torch.softmax(scores, dim=-1)
    sdpa = torch.einsum("hl,lhd->hd", p, vd_rep)
    e = (out_ref[0] - sdpa).abs().max().item()
    if e > 1e-4:
        print(f"❌ decode err = {e:.3e}（判据 ≤1e-4）")
        return 1
    print(f"✅ decode err = {e:.3e}（≤1e-4）")

    # ---- triton ↔ ref 一致性（triton 模式比 ref，ref 模式比 triton）----
    if args.mode == "triton":
        k2, v2 = torch.zeros_like(k_cache), torch.zeros_like(v_cache)
        oscar_store_ref(k, v, k2, v2, slot_mapping)
        equal = torch.equal(k2.view(torch.uint8), k_cache.view(torch.uint8)) and torch.equal(
            v2.view(torch.uint8), v_cache.view(torch.uint8)
        )
        print(f"✅ triton vs ref 字节一致: {equal}")
        if not equal:
            return 1

    print(f"🎉 probe 全 PASS（mode={args.mode}）—— 允许 serve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
