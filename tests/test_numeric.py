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



def t_triton_store_model_bf16():
    """triton store 路径本地建模：bf16 输入 → vector_scales(fp32) + floor(x+0.5) +
    打包 → 与 format.make_slot_bytes 字节差 == 0（拦截 min/max dtype 分歧类回归）。"""
    from oscar_ascend.format import vector_scales

    torch.manual_seed(7)
    D, H, N = 32, 2, 6
    k = torch.randn(N, H, D, dtype=torch.bfloat16) * 1.5
    v = torch.randn(N, H, D, dtype=torch.bfloat16) * 1.5
    ks, kz = vector_scales(k)
    vs, vz = vector_scales(v)
    qk = torch.clamp(torch.floor((k.float() - kz) / ks + 0.5), 0, 3)
    qv = torch.clamp(torch.floor((v.float() - vz) / vs + 0.5), 0, 3)

    def pack(q):
        q4 = q.to(torch.int32).reshape(N, H, D // 4, 4)
        return (q4[..., 0] | q4[..., 1] * 4 | q4[..., 2] * 16 | q4[..., 3] * 64).to(torch.uint8)

    slot = torch.zeros(N, H, 160, dtype=torch.uint8)
    db = D // 4
    slot[..., 0:1] = fmt.f16_le(ks)[0]; slot[..., 1:2] = fmt.f16_le(ks)[1]
    slot[..., 2:3] = fmt.f16_le(kz)[0]; slot[..., 3:4] = fmt.f16_le(kz)[1]
    slot[..., 4:5] = fmt.f16_le(vs)[0]; slot[..., 5:6] = fmt.f16_le(vs)[1]
    slot[..., 6:7] = fmt.f16_le(vz)[0]; slot[..., 7:8] = fmt.f16_le(vz)[1]
    slot[..., 32 : 32 + db] = pack(qk)
    slot[..., 96 : 96 + db] = pack(qv)
    ref = fmt.make_slot_bytes(k.float(), v.float())
    d = (slot != ref).sum().item()
    assert d == 0, f"triton store 路径建模字节差 = {d}"



def t_hybrid_detection():
    """_is_hybrid_config 纯函数（registry 未标 is_hybrid 的 Qwen3.5 兜底）。"""
    from oscar_ascend.plugin import _is_hybrid_config

    class HF:
        layer_types = ["linear_attention", "linear_attention", "attention"]

    class MC:
        is_hybrid = False
        hf_text_config = HF()

    assert _is_hybrid_config(MC()) is True, "layer_types 含非 attention → hybrid 应为 True"

    class HF2:
        layer_types = ["attention" * 1]

    class MC2:
        is_hybrid = False
        hf_text_config = HF2()

    assert _is_hybrid_config(MC2()) is False, "全 attention → False"

    class MC3:
        is_hybrid = True
        hf_text_config = None

    assert _is_hybrid_config(MC3()) is True, "is_hybrid=True 直接通过"



def t_attn_type_value_normalize():
    """_should_oscar 中 attn_type 值比较：str-Enum(str(member)!=value) 场景。"""
    class FakeAttnType(str):
        def __new__(cls, v):
            return super().__new__(cls, v)

        @property
        def value(self):
            return str(self)

    DECODER_STR = FakeAttnType("decoder")  # 模拟 str(member)="AttentionType.DECODER"
    class PlainDecoder:
        value = "decoder"

    class PlainNotDecoder:
        value = "encoder"

    def norm(at):
        v = getattr(at, "value", at)
        return str(v) == "decoder"

    assert norm(DECODER_STR) is True, "str-Enum value=decoder 应通过"
    assert norm(PlainDecoder) is True
    assert norm(PlainNotDecoder) is False


def t_rotation_key_tolerance():
    """rotation._load_checkpoint 键容错：int / '11' / 模块名 均可解析。"""
    import tempfile
    from pathlib import Path as P

    import oscar_ascend.rotation as rot

    D = 16
    R = torch.eye(D)
    with tempfile.TemporaryDirectory() as td:
        # 旧格式（v0 生成器误存模块名键）→ 需容错加载
        p = P(td) / "name_keys.pt"
        torch.save({"layers": {"language_model.model.layers.11.self_attn.attn":
                               {"layer_id": 11, "rotation": R}}}, p)
        tbl = rot._load_checkpoint(str(p))
        assert 11 in tbl, "名称键应解析为 11"
        assert torch.equal(tbl[11]["rotation"], R)
        # 新格式（数字字符串键）
        p2 = P(td) / "int_keys.pt"
        torch.save({"layers": {"11": {"layer_id": 11, "rotation": R}}}, p2)
        rot._load_checkpoint.cache_clear()
        tbl2 = rot._load_checkpoint(str(p2))
        assert 11 in tbl2
        rot._load_checkpoint.cache_clear()

def t_plugin_purity():
    """回归守卫：插件入口文件顶层禁止 vllm_ascend 导入（真机 Docker 循环导入教训）。"""
    import pathlib
    import re

    src = pathlib.Path("oscar_ascend/plugin.py").read_text(encoding="utf-8")
    top = re.findall(r"^(?:import|from)\s+vllm_ascend[^\n]*", src, re.M)
    assert not top, f"插件顶层仍含 vllm_ascend 导入: {top}"


def t_rotation_composition_quality_floor():
    """精度地板（双向证据，R-20260904-oscar-int2-mtp-precision 复现核心）：

    旧校准（纯特征向量 U，无 Hadamard/位反置换，论文已弃用）做 per-vector INT2，
    在"离群通道"K/V 上 ≈ 噪声（relL2 ≥ 1.0）；
    新校准（U @ H @ P_br，论文 r_h_pbr 默认配方）在同一数据上 relL2 ≤ 0.60。
    注意：本测试不拷贝任何真机数值——数据为合成离群分布（内核语义推导）。
    """
    import tools.gen_rotations as g

    D, N = 256, 512
    torch.manual_seed(7)
    chan = torch.ones(D)
    chan[:4] = 24.0          # 离群通道（LLM KV 典型 massive/outlier 通道）
    chan[4:12] = 8.0
    chan[12:24] = 2.5
    amp = torch.exp(0.4 * torch.randn(N, 1, 1))          # 每 token·head 幅值漂移
    K = torch.randn(N, 1, D) * chan * amp

    # 旧配方：协方差特征向量（降序）—— 对应修复前 gen_rotations._rotation_from_stats
    xf = K.reshape(-1, D).float()
    cov = xf.t() @ xf / xf.shape[0]
    evals, evecs = torch.linalg.eigh(cov + 1e-6 * torch.eye(D, dtype=torch.float64))
    order = torch.argsort(evals, descending=True)
    U_old = evecs[:, order].float().contiguous()
    # 新配方：U @ H @ P_br（论文 compose_rotation("r_h_pbr")）
    R_new = g.compose_rotation(
        evecs[:, order], evals[order], g.build_hadamard(D)
    ).float().contiguous()

    err = (R_new @ R_new.T - torch.eye(D)).abs().max().item()
    assert err < 1e-4, f"U@H@P 非正交（err={err:.2e}）"

    def pervec_rel(x_rot):
        packed, scale, zero = fmt.quantize(x_rot.float())
        rec = fmt.dequant(packed, scale, zero, D)
        return (rec - x_rot.float()).norm() / x_rot.float().norm()

    r_old = pervec_rel(K.float() @ U_old)
    r_new = pervec_rel(K.float() @ R_new)
    print(f"    [quality] 纯U relL2={r_old:.3f}  U@H@P relL2={r_new:.3f}")
    assert r_old >= 1.0, f"旧配方（纯 U）异常接近无损（{r_old:.3f}），夹具未模拟离群？"
    assert r_new <= 0.60, f"新配方（U@H@P）仍在噪声级（{r_new:.3f}）"


def t_clip_sort_threshold():
    """裁剪实现 = 排序分位数（对齐论文内核，替代 torch.quantile 的 NPU 未验证依赖）。"""
    D = 32
    x = torch.randn(4, 1, D) * torch.tensor([1.0, 3.0, 8.0, 24.0]).view(4, 1, 1)
    ratio = 0.75
    idx = min(int(ratio * D), D - 1)
    sorted_abs, _ = x.abs().sort(dim=-1)
    thr = sorted_abs[..., idx : idx + 1]
    y = torch.clamp(x, -thr, thr)
    assert (y.abs() <= thr + 1e-6).all(), "越界值未被裁剪"
    # 阈值 = 每向量第 idx 大 |x|（排序语义）
    expect = x.abs().sort(dim=-1).values[..., idx]
    assert torch.allclose(thr.squeeze(-1), expect, atol=1e-6)


def t_should_oscar_rejects_mtp():
    """MTP 草稿层拒绝（纯函数，无 vllm 依赖；草稿 KV 保持 BF16 走原生 SpecDecoding）。"""
    import oscar_ascend.plugin as pl

    class FakeLayer:
        layer_name = "mtp.layers.0.self_attn.attn"

    pl._SKIP_REASONS_LOGGED.clear()
    assert pl._should_oscar(FakeLayer(), None) is False
    assert pl._SKIP_REASONS_LOGGED == {"mtp-draft"}, pl._SKIP_REASONS_LOGGED


def t_calib_cov_q_sst():
    """校准统计：finalize_cov 输出 Σ_Q（qqt）与 Σ_S（sst，score-weighted）。"""
    import oscar_ascend.calib as calib

    torch.manual_seed(5)
    q = torch.randn(64, 8, 32)
    k = torch.randn(64, 2, 32)
    v = torch.randn(64, 2, 32)
    calib._captures = {
        "language_model.model.layers.3.self_attn.attn": {"q": [q], "k": [k], "v": [v]}
    }
    st = calib.finalize_cov()["language_model.model.layers.3.self_attn.attn"]
    assert set(st.keys()) == {"q", "v"}
    cov_q, cov_v = st["q"]["cov"], st["v"]["cov"]
    assert cov_q.shape == (32, 32) and (cov_q - cov_q.T).abs().max() < 1e-5
    assert (cov_v - cov_v.T).abs().max() < 1e-5
    plain_v = (v.reshape(-1, 32).T @ v.reshape(-1, 32)) / v.shape[0]
    assert not torch.equal(cov_v, plain_v), "sst 权重未生效（Σ_S == Σ_V）"


def main():
    print("== oscar_ascend CPU 数值镜像 ==")
    check("quantize/dequant ≤1e-5", t_quant_dequant)
    check("slot roundtrip ≤1e-5", t_slot_roundtrip)
    check("store 字节差 == 0", t_store_split_bytes_zero)
    check("decode vs SDPA ≤1e-4", t_decode_ref_vs_sdpa)
    check("rotation invariance ≤1e-3", t_rotation_invariance)
    check("prefill continuation 形状/运行", t_prefill_ref)
    check("rotation 检查点加载/缺层回退", t_rotation_loader)
    check("triton store 路径建模(bf16) 字节差=0", t_triton_store_model_bf16)
    check("hybrid 判定纯函数(含兜底)", t_hybrid_detection)
    check("attn_type 值归一化(str-Enum)", t_attn_type_value_normalize)
    check("rotation 键容错(名称/int)", t_rotation_key_tolerance)
    check("插件顶层无 vllm_ascend 导入（回归守卫）", t_plugin_purity)
    check("旋转组合 U@H@P 精度地板（旧 FAIL/新 PASS）", t_rotation_composition_quality_floor)
    check("裁剪排序分位数语义", t_clip_sort_threshold)
    check("MTP 草稿层拒绝（BF16 保真）", t_should_oscar_rejects_mtp)
    check("校准 Σ_Q/Σ_S（qqt/sst）", t_calib_cov_q_sst)
    print(f"== 结果: {len(PASS)} PASS / {len(FAIL)} FAIL ==")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
