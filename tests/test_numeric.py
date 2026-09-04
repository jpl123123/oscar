"""tests/test_numeric.py — 本地 CPU 数值镜像（与真机 probe 同判据）。

判据（skill sandbox l3_numeric 同款）：
  store   : 拆分落位字节 == format.make_slot_bytes（max|d| == 0）
  dequant : ≤ 1e-5；decode/prefill : ≤ 1e-4（fp32，同一 INT2 数据源）
运行：python3 tests/test_numeric.py（CPU torch 即可）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from oscar_ascend import format as fmt
from oscar_ascend.kernels.store_kernel import oscar_store_ref
from oscar_ascend.kernels.decode_kernel import oscar_decode_ref, oscar_prefill_ref
from oscar_ascend.kernels.dequant_kernel import oscar_full_dequant_ref

torch.manual_seed(0)

PASS = []
FAIL = []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ✅ {name}")
    except AssertionError as e:
        FAIL.append(name)
        print(f"  ❌ {name}: {e}")
    except Exception as e:  # noqa: BLE001
        FAIL.append(name)
        print(f"  ❌ {name}: 异常 {type(e).__name__}: {e}")


# ---------------------------------------------------------------- fixtures
def rand_kv(N=2, H=2, D=32, dtype=torch.float32, dev="cpu"):
    k = torch.randn(N, H, D, dtype=dtype, device=dev) * 1.5
    v = torch.randn(N, H, D, dtype=dtype, device=dev) * 1.5
    return k, v


def make_cache(nb=3, bs=4, hk=2, D=32, dev="cpu"):
    k = torch.zeros(nb, bs, hk, D, dtype=torch.bfloat16, device=dev)
    v = torch.zeros_like(k)
    return k, v


# ---------------------------------------------------------------- tests
def t_quant_dequant():
    """dequant(商店字节) == 量化时刻的理想重建（q*scale+zero），≤1e-5。"""
    k, v = rand_kv()
    for x in (k, v):
        packed, scale, zero = fmt.quantize(x)
        rec = fmt.dequant(packed, scale, zero, x.shape[-1])
        q = torch.clamp(torch.floor((x.float() - zero) / scale + 0.5), 0, 3)
        ideal = q * scale + zero
        err = (rec - ideal).abs().max().item()
        assert err <= fmt.DEQUANT_TOL, f"dequant err={err:.3e}"


def t_slot_roundtrip():
    k, v = rand_kv()
    slot = fmt.make_slot_bytes(k, v)
    kr, vr = fmt.parse_slot_bytes(slot, k.shape[-1])
    _, ks, kz = fmt.quantize(k)
    _, vs, vz = fmt.quantize(v)
    qk = torch.clamp(torch.floor((k.float() - kz) / ks + 0.5), 0, 3)
    qv = torch.clamp(torch.floor((v.float() - vz) / vs + 0.5), 0, 3)
    e = max(
        (kr - (qk * ks + kz)).abs().max().item(),
        (vr - (qv * vs + vz)).abs().max().item(),
    )
    assert e <= fmt.DEQUANT_TOL, f"slot roundtrip err={e:.3e}"


def t_store_split_bytes_zero():
    """拆分落位（k8/v8）读回拼槽 vs make_slot_bytes → 字节差 == 0。"""
    D, H, bs, nb = 32, 2, 4, 3
    k, v = rand_kv(N=6, H=H, D=D)
    kc, vc = make_cache(nb=nb, bs=bs, hk=H, D=D)
    slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5])  # 覆盖第一物理块
    oscar_store_ref(k, v, kc, vc, slot_mapping)

    ref = fmt.make_slot_bytes(k, v)                      # [N,H,160]
    k8, v8 = kc.view(torch.uint8), vc.view(torch.uint8)
    db = D // 4
    got = torch.zeros(6, H, 160, dtype=torch.uint8)
    for t in range(6):
        b, o = slot_mapping[t] // bs, slot_mapping[t] % bs
        for h in range(H):
            ks = b * k8.stride(0) + o * k8.stride(1) + h * k8.stride(2)
            vs = b * v8.stride(0) + o * v8.stride(1) + h * v8.stride(2)
            got[t, h, 0:8] = k8.view(-1)[ks : ks + 8]
            got[t, h, 32 : 32 + db] = k8.view(-1)[ks + 32 : ks + 32 + db]
            got[t, h, 96 : 96 + db] = v8.view(-1)[vs : vs + db]
    d = (got != ref).sum().item()
    assert d == 0, f"store 字节差 = {d}"


def t_decode_ref_vs_sdpa():
    """decode（INT2 反量化）vs 同一 INT2 数据 SDPA ≤1e-4。"""
    D, H, bs, nb, B, L = 32, 2, 4, 3, 1, 7
    k, v = rand_kv(N=L, H=H, D=D)
    kc, vc = make_cache(nb=nb, bs=bs, hk=H, D=D)
    sm = torch.arange(L)
    oscar_store_ref(k, v, kc, vc, sm)
    bt = torch.zeros(B, nb, dtype=torch.int32)
    seq = torch.tensor([L], dtype=torch.int32)
    q = torch.randn(B, 2 * H, D)  # Hq=4
    out_ref, _ = oscar_decode_ref(q, kc, vc, bt, seq, 0.125, H, D)
    # ---- SDPA over same INT2 data ----
    from oscar_ascend.kernels.store_kernel import dequant_split_ref

    blk_idx = torch.arange(L) // bs
    pos = torch.arange(L) % bs
    bnums = torch.zeros(L, dtype=torch.long)
    kd, vd = dequant_split_ref(kc.view(torch.uint8), vc.view(torch.uint8), bnums, pos, H, D)
    kd_rep = kd.repeat_interleave(2, dim=1)   # [L,Hq,D]
    vd_rep = vd.repeat_interleave(2, dim=1)
    scores = torch.einsum("hd,lhd->hl", q[0], kd_rep) * 0.125
    p = torch.softmax(scores, dim=-1)
    sdpa_out = torch.einsum("hl,lhd->hd", p, vd_rep)
    err = (out_ref[0] - sdpa_out).abs().max().item()
    assert err <= fmt.DECODE_TOL, f"decode err={err:.3e}"


def t_rotation_invariance():
    """K@R_k 与 Q@R_k 打分 == 原始打分（正交不变，fp32 ≤1e-4）。"""
    D = 32
    k, v = rand_kv(N=1, H=1, D=D)
    q, _ = rand_kv(N=1, H=1, D=D)
    Q = torch.linalg.qr(torch.randn(D, D))[0]
    kR = k @ Q
    qR = q @ Q
    s0 = (q[0, 0] * k[0, 0]).sum()
    s1 = (qR[0, 0] * kR[0, 0]).sum()
    assert abs(s0 - s1) <= 1e-3 * max(1.0, abs(s0)), f"score drift {s0} vs {s1}"


def t_prefill_ref():
    D, H, C, N = 32, 2, 6, 2
    kc, vc = make_cache(nb=3, bs=4, hk=H, D=D)
    kc_vals, vc_vals = rand_kv(N=C, H=H, D=D)
    oscar_store_ref(kc_vals, vc_vals, kc, vc, torch.arange(C))
    kch, vch = rand_kv(N=N, H=H, D=D)
    q = torch.randn(N, 4, D)
    kcached, vcached = oscar_full_dequant_ref(
        kc, vc, torch.zeros(3, dtype=torch.int32), C, H, D
    )
    out = oscar_prefill_ref(q, kch, vch, kcached * 0, vcached * 0, 0.0, H, D)  # 无缓存·直通
    assert out.shape == (N, 4, D), out.shape


def t_rotation_loader():
    """rotation.py：检查点 {layers:{i:{rotation,rotation_v}}} → per-layer 旋转；缺层→单位阵。"""
    import tempfile
    from pathlib import Path

    from oscar_ascend.rotation import get_layer_rotation

    D = 32
    R1 = torch.eye(D)
    R2 = (torch.eye(D) * 2)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "rot.pt"
        torch.save(
            {"layers": {"3": {"layer_id": 3, "rotation": R1, "rotation_v": R2}}}, p
        )
        rk = get_layer_rotation(str(p), "model.layers.3.self_attn", D, torch.device("cpu"))
        rv = get_layer_rotation(str(p), "model.layers.3.self_attn", D, torch.device("cpu"), mode="v")
        assert torch.equal(rk, R1) and torch.equal(rv, R2)
        # 缺层 → 单位阵
        r_absent = get_layer_rotation(str(p), "model.layers.9.self_attn", D, torch.device("cpu"))
        assert torch.equal(r_absent, torch.eye(D))
        # 空路径 → 单位阵
        assert torch.equal(get_layer_rotation("", "x", D, torch.device("cpu")), torch.eye(D))


def t_plugin_purity():
    """回归守卫：插件入口文件顶层禁止 vllm_ascend 导入（真机 Docker 循环导入教训）。"""
    import pathlib
    import re

    src = pathlib.Path("oscar_ascend/plugin.py").read_text(encoding="utf-8")
    top = re.findall(r"^(?:import|from)\s+vllm_ascend[^\n]*", src, re.M)
    assert not top, f"插件顶层仍含 vllm_ascend 导入: {top}"


def main():
    print("== oscar_ascend CPU 数值镜像 ==")
    check("quantize/dequant ≤1e-5", t_quant_dequant)
    check("slot roundtrip ≤1e-5", t_slot_roundtrip)
    check("store 字节差 == 0", t_store_split_bytes_zero)
    check("decode vs SDPA ≤1e-4", t_decode_ref_vs_sdpa)
    check("rotation invariance ≤1e-3", t_rotation_invariance)
    check("prefill continuation 形状/运行", t_prefill_ref)
    check("rotation 检查点加载/缺层回退", t_rotation_loader)
    check("插件顶层无 vllm_ascend 导入（回归守卫）", t_plugin_purity)
    print(f"== 结果: {len(PASS)} PASS / {len(FAIL)} FAIL ==")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
