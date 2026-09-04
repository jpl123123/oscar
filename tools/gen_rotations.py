#!/usr/bin/env python3
"""tools/gen_rotations.py — 真机离线生成 OSCAR per-layer 旋转检查点（.pt）。

方法（vLLM 0.23 API，端到端）：
  1. 构建 LLM（TP4，0.9 显存；模型装载即完成——不跑 generate）；
  2. `llm.llm_engine.apply_model(oscar_ascend.calib.capture_cov)`：每个 worker 对
     已加载模型执行一次纯文本前向，捕获 full-attention 层**未量化** K/V，
     统计逐层协方差（TP 分片内累计，跨 rank 加权合并）；
  3. 合并 → `torch.linalg.eigh`（列=特征向量，按特征值降序）→ 正交 R_k/R_v；
  4. 保存 {layers: {i: {layer_id, rotation(=R_k), rotation_v(=R_v), eigenvalues…}}}。

诚实边界：优先 NPU eigh；NPU 不支持时 CPU 回退并醒目警告（一次性离线校准，非热路径）。
用法（真机）：
  bash delivery/install_and_launch.sh           # 一键内嵌（推荐）
  OSCAR_ASCEND_GEN_LLM_ARGS='{"tensor_parallel_size":4}' \\
    python3 tools/gen_rotations.py --model /path/... --save oscar_rotations.pt
"""
from __future__ import annotations

import argparse
import json
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
    """跨 rank 加权合并协方差（均值为 rank 内 E[x]；全局 E[x] 需二次展开）。"""
    merged: dict[str, dict[str, dict]] = {}
    for rank_stats in per_rank:
        for layer, st in rank_stats.items():
            for kind in ("k", "v"):
                agg = merged.setdefault(layer, {}).setdefault(
                    kind,
                    {"mean_sum": torch.zeros(D), "cov_sum": torch.zeros(D, D), "count": 0},
                )
                m = st[kind]["mean"]
                cv = st[kind]["cov"]
                c = st[kind]["count"]
                agg["mean_sum"] = agg["mean_sum"] + m * c
                agg["cov_sum"] = agg["cov_sum"] + (cv + torch.outer(m, m)) * c
                agg["count"] += c
    return merged


def _rotation_from_stats(stats: dict, D: int, dev: str) -> tuple[torch.Tensor, torch.Tensor]:
    c = max(1, int(stats["count"]))
    mean = stats["mean_sum"] / c
    cov = stats["cov_sum"] / c - torch.outer(mean, mean)
    eps = 1e-6 * torch.eye(D)
    try:
        evals, evecs = torch.linalg.eigh((cov + eps).to(dev))
    except Exception as e:
        print(
            f"[gen-rotations] ⚠️ NPU eigh 不可用，CPU 回退（离线校准，非热路径）: {e}"
        )
        try:
            evals, evecs = torch.linalg.eigh(cov + eps)
        except Exception as e2:
            raise RuntimeError(f"eigh 失败（cov 含 NaN?）: {e2}") from e2
    idx = torch.argsort(evals, descending=True)
    return evecs[:, idx].contiguous(), evals[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--save", default="oscar_rotations.pt")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--head-dim", type=int, default=256)
    args = ap.parse_args()

    on_npu = _npu()
    dev = "npu" if on_npu else "cpu"
    print(f"[gen-rotations] device={dev} seq_len={args.max_len}")

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

    D = args.head_dim
    merged = _merge_rank_stats(per_rank, D)
    layers_out = {}
    for layer, st in sorted(merged.items()):
        r_k, e_k = _rotation_from_stats(st["k"], D, dev)
        r_v, e_v = _rotation_from_stats(st["v"], D, dev)
        layers_out[layer] = {
            "layer_id": int(layer),
            "rotation": r_k,
            "rotation_v": r_v,
            "eigenvalues": e_k,
            "eigenvalues_v": e_v,
        }
        print(f"  layer {layer}: count={st['k']['count']} R_k/R_v ok")

    assert layers_out, "无可用于旋转的统计"
    out = {
        "format_version": 1,
        "objective": "qqt_r_h_pbr",
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
