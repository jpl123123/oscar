#!/usr/bin/env python3
"""tools/gen_rotations.py — 真机离线生成 OSCAR per-layer 旋转检查点（.pt）。

方法（vLLM 0.23 API，端到端；对齐 OSCAR 论文/PR 已验证配方）：
  1. 构建 LLM（TP4，0.9 显存；模型装载即完成——不跑 generate）；
  2. `llm.llm_engine.apply_model(oscar_ascend.calib.capture_cov)`：每个 worker 对
     已加载模型执行一次纯文本前向，捕获 full-attention 层**未量化** Q/K/V，
     统计 attention-aware 协方差（worker 内完成 qqt/sst，跨 rank 加权合并）；
  3. 合并 → `torch.linalg.eigh` → 逐层组合旋转 **R = U @ H @ P_br**
     （论文 compute_kv_rotation.py:234-265 的 `r_h_pbr` 默认验证配方：
      U=hessian 特征向量、H=Hadamard、P_br=按特征值位反转置换，
      把高方差方向均匀摊开，避免 per-vector INT2 min/max 被离群方向主导）；
  4. 保存 {layers: {i: {layer_id, rotation(=R_k), rotation_v(=R_v), eigenvalues…}}}。

校准目标（论文 README:312-318）：
  K 旋转 hessian = Σ_Q = (1/H_kv)·Σ_h Q_h^T Q_h / n   （qqt）
  V 旋转 hessian = Σ_S = (1/H_kv)·Σ_h (V_h√w)^T(V_h√w)/n，w_i = k_i^T Σ_Qh k_i （sst，score-weighted）

诚实边界：优先 NPU eigh；NPU 不支持时 CPU 回退并醒目警告（一次性离线校准，非热路径）。
用法（真机）：
  bash delivery/install_and_launch.sh           # 一键内嵌（推荐）
  OSCAR_ASCEND_GEN_LLM_ARGS='{"tensor_parallel_size":4}' \\
    python3 tools/gen_rotations.py --model /path/... --save oscar_rotations.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

# 真机实测（2026-09-03 12:58）：多线程父进程 fork 出 EngineCore 后，autograd 线程
# set_num_threads 触发 "ParallelOpenMP.cpp:64 Invalid thread pool" 硬崩溃。
# 必须在任何 vllm 导入前设置（vllm.envs 于 import 时读取该值）。
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# vendor vllm serial_utils 默认不允许跨进程传函数（collective_rpc 需要）：
# 错误提示官方出口 = VLLM_ALLOW_INSECURE_SERIALIZATION=1（回退 pickle；模块级函数可 pickle）
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _npu() -> bool:
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except Exception:
        return False


def _force_ascend_platform() -> bool:
    """兜底：platform 未自动激活时强制 NPU（vendor 白名单修复后通常为 no-op）。"""
    try:
        import vllm.platforms as vp

        if getattr(vp.current_platform, "device_type", "") == "npu":
            return True
        from vllm_ascend.platform import NPUPlatform

        vp.current_platform = NPUPlatform()
        ok = getattr(vp.current_platform, "device_type", "") == "npu"
        print(f"[gen-rotations] 强制激活 NPUPlatform: {'OK' if ok else 'FAIL'} "
              f"(device_type={getattr(vp.current_platform, 'device_type', '?')!r})")
        return ok
    except Exception as e:
        print(f"[gen-rotations] 强制激活 NPUPlatform 异常: {type(e).__name__}: {e}")
        return False


def _merge_rank_stats(per_rank: list[dict], D: int) -> dict[str, dict[str, dict]]:
    """跨 rank 加权合并协方差（每 rank 协方差已按 token 平均；按 count 加权）。"""
    merged: dict[str, dict[str, dict]] = {}
    for rank_stats in per_rank:
        for layer, st in rank_stats.items():
            for kind in ("q", "v"):
                agg = merged.setdefault(layer, {}).setdefault(
                    kind,
                    {"cov_sum": torch.zeros(D, D), "count": 0},
                )
                cov = st[kind]["cov"]
                c = st[kind]["count"]
                agg["cov_sum"] = agg["cov_sum"] + cov * c
                agg["count"] += c
    return merged


# ---------------------------------------------------------------------------
# 旋转组合（论文 compute_kv_rotation.py:23-46, 234-265 移植）
# ---------------------------------------------------------------------------
def build_hadamard(n: int) -> torch.Tensor:
    """[n,n] 归一化 Hadamard（n 必须 2 的幂）。"""
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    if n == 1:
        return torch.ones(1, 1, dtype=torch.float64)
    h = build_hadamard(n // 2)
    return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0) / math.sqrt(2)


def bit_reversal_perm(d: int) -> torch.Tensor:
    if d < 1 or d & (d - 1):
        raise ValueError(f"Bit-reversal size must be a power of two, got {d}")
    bits = int(math.log2(d))
    return torch.tensor([int(bin(i)[2:].zfill(bits)[::-1], 2) for i in range(d)])


def make_br_perm_matrix(eigenvalues: torch.Tensor) -> torch.Tensor:
    """P_br：按特征值（降序）排序后位反转放置——高方差方向均匀交错。"""
    d = len(eigenvalues)
    sorted_idx = torch.argsort(eigenvalues, descending=True)
    br = bit_reversal_perm(d)
    perm = torch.zeros(d, dtype=torch.long)
    for i in range(d):
        perm[br[i]] = sorted_idx[i]
    return torch.eye(d, dtype=torch.float64)[:, perm]


def compose_rotation(rotation: torch.Tensor, eigvals: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    """R · H · P_br（论文默认验证配方 r_h_pbr）。"""
    pbr = make_br_perm_matrix(eigvals)
    return rotation @ hadamard @ pbr


def _rotation_from_stats(stats: dict, D: int, dev: str) -> tuple[torch.Tensor, torch.Tensor]:
    """hessian 协方差 → (R = U·H·P_br, eigenvalues)（fp64 计算后转 fp32）。"""
    c = max(1, int(stats["count"]))
    cov = stats["cov_sum"] / c
    eps = 1e-6 * torch.eye(D)
    cov64 = (cov.double() + cov.double().T) / 2
    try:
        evals, evecs = torch.linalg.eigh((cov64 + eps.double()).to(dev))
    except Exception as e:
        print(
            f"[gen-rotations] ⚠️ NPU eigh 不可用，CPU 回退（离线校准，非热路径）: {e}"
        )
        try:
            evals, evecs = torch.linalg.eigh(cov64 + eps.double())
        except Exception as e2:
            raise RuntimeError(f"eigh 失败（cov 含 NaN?）: {e2}") from e2
    H = build_hadamard(D)
    rot = compose_rotation(evecs, evals, H)
    err = (rot @ rot.T - torch.eye(D, dtype=torch.float64)).abs().max().item()
    if err > 1e-6:
        print(f"[gen-rotations] ⚠️ 组合旋转正交误差 {err:.2e}（eigh 精度；继续）")
    return rot.float().contiguous(), evals.float().contiguous()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--save", default="oscar_rotations.pt")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--head-dim", type=int, default=256)
    args = ap.parse_args()

    on_npu = _npu()
    dev = "npu" if on_npu else "cpu"
    # CPU 压力控制：父进程只做一次性合并+eigh（NPU 不支持 eigh → CPU 回退），限 4 线程
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    print(f"[gen-rotations] device={dev} seq_len={args.max_len} cpu_threads={torch.get_num_threads()}")

    if not _force_ascend_platform():
        print("[gen-rotations] ❌ 平台未激活。请运行 python3 tools/diag_platform.py 并回传输出。")
        return 3

    from vllm import LLM, SamplingParams

    from oscar_ascend import calib

    # 校准模式：插件在 worker 里挂 K/V 捕获钩子（不替换 impl，不做裸 model 前向）
    os.environ["OSCAR_ASCEND_CALIB"] = "1"
    os.environ.pop("OSCAR_ASCEND_ENABLE", None)  # 钩子由 CALIB 分支处理，无需注入开关
    llm_args = json.loads(os.environ.get("OSCAR_ASCEND_GEN_LLM_ARGS", "{}"))
    # 默认 TP4 + 0.9 显存（target serve 同款）：27B W8A8 权重 > 单卡 29.49GiB（实测必 OOM）
    llm_args.setdefault("tensor_parallel_size", 4)
    llm_args.setdefault("gpu_memory_utilization", 0.9)
    # 与目标 serve 命令一致：GDN 状态必须 BF16（aclnnChunkGatedDeltaRule 仅支持
    # DT_BFLOAT16；默认 auto 在真机上会生成 FP32 initialState → EZ1001 参数错误）
    llm_args.setdefault("mamba_cache_dtype", "bfloat16")
    llm_args.setdefault("mamba_ssm_cache_dtype", "bfloat16")
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        max_model_len=max(args.max_len + 16, 128),
        dtype="bfloat16",
        **llm_args,
    )
    print("[gen-rotations] 引擎就绪；清空捕获缓冲 + 跑校准前向（引擎正常路径）…")

    llm.llm_engine.apply_model(calib.reset_captures)
    # 一条长校准文本：prefill 即产生 ~256 tokens 的 K/V（每个 worker 每层 cap=512）
    prompt = (
        "OSCAR 旋转校准用例：量子计算与自然语言处理都是非结构化数据上的双重挑战。"
        "模型应当对长程序列保持稳定的隐藏表示，并在注意力方向上分配较低的量化噪声。" * 4
    )[: max(64, args.max_len - 8)]
    llm.generate(prompt, SamplingParams(max_tokens=4, temperature=0.0))
    print("[gen-rotations] 校准前向完成；worker 内统计协方差…")

    per_rank = llm.llm_engine.apply_model(calib.finalize_cov)
    if not per_rank or not per_rank[0]:
        print("[gen-rotations] ❌ 未捕获到任何 K/V（检查 OSCAR_ASCEND_CALIB 钩子/模型结构）")
        return 4

    import re as _re

    D = args.head_dim
    merged = _merge_rank_stats(per_rank, D)
    layers_out = {}
    for layer, st in sorted(merged.items()):
        m = _re.search(r"\.layers\.(\d+)\.", layer)
        if m is None:
            print(f"  ⏭️ 跳过无法解析层号的键: {layer}")
            continue
        lid = int(m.group(1))
        # K 旋转 ← Σ_Q（qqt）；V 旋转 ← Σ_S（sst）；组合 R = U·H·P_br
        r_k, e_k = _rotation_from_stats(st["q"], D, dev)
        r_v, e_v = _rotation_from_stats(st["v"], D, dev)
        layers_out[str(lid)] = {   # 键 = 层号字符串（rotation._load_checkpoint 按 int/lid 索引）
            "layer_id": lid,
            "rotation": r_k,
            "rotation_v": r_v,
            "eigenvalues": e_k,
            "eigenvalues_v": e_v,
        }
        print(f"  layer {layer}: count={st['q']['count']} R_k/R_v ok (U@H@Pbr)")

    assert layers_out, "无可用于旋转的统计"
    out = {
        "format_version": 2,   # v2 = U@H@P_br 组合配方（install_and_launch 按此强制重校准）
        "objective": "qqt_r_h_pbr_k / sst_r_h_pbr_v",
        "source_grouping": "layer",
        "layers": {
            k: {kk: (vv.cpu() if isinstance(vv, torch.Tensor) else vv)
                for kk, vv in v.items()}
            for k, v in layers_out.items()
        },
    }
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.save)
    print(f"[gen-rotations] saved {args.save}（layers={len(out['layers'])} / D={D}）")
    print(
        "  启动时: OSCAR_ASCEND_K_ROTATION_PATH=该文件 OSCAR_ASCEND_V_ROTATION_PATH=该文件\n"
        "  （K 用 rotation 字段，V 用 rotation_v 字段，见 rotation.py）"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
