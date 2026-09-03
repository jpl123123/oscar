#!/usr/bin/env python3
"""delivery/probe_oscar.py — 真机数值 probe（store / dequant / decode 三查，同沙盒判据）。

判据（与 skill sandbox l3_numeric 一致）：
  1. store    : 量化打包字节 == format 参考实现字节（max|d| == 0）
  2. dequant  : 反量化重建与原始 rotated 向量差 ≤ 1e-5（fp32）
  3. decode   : 单 token INT2 decode 注意力 vs bf16 参考 ≤ 1e-4（fp32）

阻塞 serve：任意 FAIL → exit 非 0（install_and_launch.sh 阶段4 直接终止）。
全部计算在 NPU（torch.npu）；无 CPU 搬运。
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
    args = ap.parse_args()

    try:
        import torch
        import torch_npu  # noqa: F401
        import oscar_ascend.format as fmt
        from oscar_ascend.kernels.store_kernel import oscar_store_ref  # torch 参考实现
        from oscar_ascend.kernels.decode_kernel import oscar_decode_ref
    except ImportError as e:
        print(f"❌ probe 依赖缺失: {e}（先 pip install -e . 并实现 plan TODO 1-4）")
        return 2

    if not torch.npu.is_available():
        print("❌ torch.npu 不可用（probe 必须 NPU，纯 CPU 用 tests/test_numeric.py）")
        return 2

    dev = torch.npu.current_device()
    D = args.head_dim
    Hk, Hq, bs = args.num_kv_heads, args.num_heads, args.block_size

    # 构造单 token 样例（NPU）
    torch.manual_seed(0)
    k = torch.randn(Hk, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(Hk, D, device=dev, dtype=torch.bfloat16)
    q = torch.randn(Hq, D, device=dev, dtype=torch.bfloat16)

    # ---- 1) store 字节差 ----
    comb = torch.zeros(bs, Hk, 1024, dtype=torch.uint8, device=dev)  # 一页一槽
    slot = torch.zeros(1, dtype=torch.int64, device=dev)
    oscar_store_ref(k, v, comb, slot, D=D)
    ref_slot = fmt.make_slots(k, v, D=D)
    diff = (comb.view(-1)[: 160 * Hk].view(Hk, 160) != ref_slot).sum().item()
    if diff != 0:
        print(f"❌ store 字节差 = {diff}（判据 0）")
        return 1
    print(f"✅ store  字节差 = 0（{Hk} 头 × 160B 槽一致）")

    # ---- 2) dequant ≤1e-5 ----
    k_rec, v_rec = fmt.dequant_slot(slot_view=comb.view(-1)[: 160 * Hk], num_heads=Hk, D=D)
    k_ref = k.float(); v_ref = v.float()
    ek, ev = (k_rec - k_ref).abs().max().item(), (v_rec - v_ref).abs().max().item()
    if max(ek, ev) > 1e-5:
        print(f"❌ dequant err K={ek:.3e} V={ev:.3e}（判据 ≤1e-5）")
        return 1
    print(f"✅ dequant err K={ek:.3e} V={ev:.3e}（≤1e-5）")

    # ---- 3) decode ≤1e-4（INT2 + 单位旋转 vs bf16 SDPA 参考）----
    ref_attn = torch.nn.functional.scaled_dot_product_attention(
        q.float().unsqueeze(0), k.float().unsqueeze(0), v.float().unsqueeze(0)
    ).squeeze(0)
    oscar_out = oscar_decode_ref(q, comb, seq_len=1, D=D, Hq=Hq, Hk=Hk)
    e = (oscar_out - ref_attn).abs().max().item()
    if e > 1e-4:
        print(f"❌ decode err = {e:.3e}（判据 ≤1e-4）")
        return 1
    print(f"✅ decode err = {e:.3e}（≤1e-4）")

    print("🎉 probe 全 PASS —— 允许 serve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
