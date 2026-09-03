#!/usr/bin/env python3
"""tools/gen_rotations.py — 真机离线生成 OSCAR per-layer 旋转检查点（.pt）。

方法（OSCAR 语义：数据依赖谱旋转）：
  1. vllm.LLM 跑少量 prompt（BF16 权重路径，本工具为独立进程，不加载插件）；
  2. 对每个 full-attention 层的 Attention 模块注册 forward 前 hook，捕获**未量化** key/value；
  3. 每层分别累计 K/V 协方差 → `torch.linalg.eigh`（列=特征向量，按特征值降序）→ 正交 R_k/R_v；
  4. 保存 {format_version, objective, source_grouping, layers: {i: {layer_id, rotation, eigenvalues}}}。

诚实边界（plan §8 R4）：优先 NPU eigh；NPU 不支持时回退 CPU 并醒目警告——这是
一次性离线校准，不属推理热路径 CPU 搬运。

用法（真机）：
  OSCAR_ASCEND_GEN_LLM_ARGS='{"max_model_len":512,"tensor_parallel_size":4}' \
    python3 tools/gen_rotations.py --model /path/Qwen3.5-27B-w8a8-mtp \
    --save oscar_rotations.pt --prompts 4 --max-len 128
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def _npu() -> bool:
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except Exception:
        return False


def _force_ascend_platform() -> bool:
    """自愈：Docker 中 platform 自动激活可能失败（device_type=""），在此强制 NPU。

    成功判定：current_platform.device_type == "npu"。失败不吞异常，交由 diag 定位。
    """
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--save", default="oscar_rotations.pt")
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--head-dim", type=int, default=256)
    args = ap.parse_args()

    on_npu = _npu()
    dev = "npu" if on_npu else "cpu"
    print(f"[gen-rotations] device={dev} prompts={args.prompts} max_len={args.max_len}")

    if not _force_ascend_platform():
        print(
            "[gen-rotations] ❌ 平台未激活（device_type 为空）。请先运行:\n"
            "    python3 tools/diag_platform.py\n"
            " 并把输出（或 /tmp/oscar_ascend_logs/selfcheck_*.log）回传。"
        )
        return 3

    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.attention import Attention

    llm_args = json.loads(os.environ.get("OSCAR_ASCEND_GEN_LLM_ARGS", "{}"))
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        max_model_len=args.max_len,
        dtype="bfloat16",
        **llm_args,
    )

    # ---- 逐层 hook（键：全局 layer idx）----
    capture: dict[int, dict[str, list]] = {}
    hooks = []

    def _make_hook(i):
        def _hook(module, q, k, v, kv_cache, attn_metadata, output):
            if k is None or v is None:
                return None
            d = capture.setdefault(i, {"k": [], "v": []})
            d["k"].append(k.detach().float())
            d["v"].append(v.detach().float())
            return None

        return _hook

    def _find_layers(mod):
        """递归定位 base decoder layers（Qwen3.5 结构可能包一层 language_model/model）。"""
        for attr in ("layers",):
            sub = getattr(mod, attr, None)
            if sub is not None and len(sub) > 0:
                return sub
        for attr in ("language_model", "model", "transformer", "language_model.model"):
            sub = mod
            ok = True
            for part in attr.split("."):
                sub = getattr(sub, part, None)
                if sub is None:
                    ok = False
                    break
            if ok and hasattr(sub, "layers") and len(sub.layers) > 0:
                return sub.layers
        raise SystemExit("未找到模型 decoder layers（结构不符，请检查模型类）")

    model_mod = llm.model.model if hasattr(llm.model, "model") else llm.model
    layers = _find_layers(model_mod)
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        module = getattr(attn, "attn", None)
        if isinstance(module, Attention):
            hooks.append((i, module.register_forward_pre_hook(_make_hook(i))))
            print(f"  hooked model.layers.{i}.self_attn.attn")

    got_layers = sum(1 for i, _ in hooks)
    assert got_layers > 0, "未找到 full-attention Attention 模块（模型结构不符）"

    prompts = [
        f"这是用于 OSCAR 旋转校准的文本序列 #{i}。"
        "量子计算与自然语言处理都是非结构化数据上的双重挑战。" * 2
        for i in range(args.prompts)
    ]
    llm.generate(prompts, SamplingParams(max_tokens=8, temperature=0.0))
    for _, h in hooks:
        h.remove()
    n_tokens = sum(len(v["k"]) * 0 + sum(t.shape[0] for t in v["k"]) for v in capture.values())
    print(f"  captured tokens={n_tokens}")

    # ---- 逐层旋转 ----
    D = args.head_dim
    layers_out = {}
    cpu_warned = False
    for i, d in sorted(capture.items()):
        k = torch.cat([t.reshape(-1, t.shape[-1]) for t in d["k"]], dim=0)[:, :D]
        v = torch.cat([t.reshape(-1, t.shape[-1]) for t in d["v"]], dim=0)[:, :D]
        if k.shape[0] < 8:
            continue

        def _rotation(x: torch.Tensor):
            nonlocal cpu_warned
            xc = x - x.mean(dim=0, keepdim=True)
            cov = (xc.t() @ xc) / max(1, xc.shape[0])
            eps = 1e-6 * torch.eye(D, dtype=cov.dtype, device=cov.device)
            try:
                evals, evecs = torch.linalg.eigh(cov + eps)
            except Exception as e:
                if not cpu_warned:
                    cpu_warned = True
                    print(
                        "[gen-rotations] ⚠️ NPU eigh 不可用，CPU 回退（离线校准，非热路径；"
                        f"原因: {e}）"
                    )
                evals, evecs = torch.linalg.eigh((cov + eps).cpu())
            idx = torch.argsort(evals, descending=True)
            return evecs[:, idx].contiguous(), evals[idx]

        r_k, e_k = _rotation(k)
        r_v, e_v = _rotation(v)
        layers_out[str(i)] = {
            "layer_id": int(i),
            "rotation_k": r_k,
            "rotation_v": r_v,
            "eigenvalues_k": e_k,
            "eigenvalues_v": e_v,
        }
        print(f"  layer {i}: tokens={k.shape[0]} R_k/R_v ok")

    assert layers_out, "无可用于旋转的捕获数据（prompt 过短或 hook 未命中）"

    # rotation.py 兼容格式（PR：layers[i].rotation）；K/V 共用 rotation 字段 → 采用 K 旋转，
    # V 由同文件 rotation_v 提供（backend 从环境变量 V 路径读取同一文件即可）。
    out = {
        "format_version": 1,
        "objective": "qqt_r_h_pbr",
        "source_grouping": "layer",
        "layers": {
            k: {
                "layer_id": v["layer_id"],
                "rotation": v["rotation_k"],
                "rotation_v": v["rotation_v"],
                "eigenvalues": v["eigenvalues_k"],
                "eigenvalues_v": v["eigenvalues_v"],
            }
            for k, v in layers_out.items()
        },
    }
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    # 统一落 CPU（torch.save NPU tensor 需 device 上下文，落盘前移 CPU 更稳）
    for v in out["layers"].values():
        for key in ("rotation", "rotation_v", "eigenvalues", "eigenvalues_v"):
            if isinstance(v.get(key), torch.Tensor):
                v[key] = v[key].cpu()
    torch.save(out, args.save)
    print(f"[gen-rotations] saved {args.save}（layers={len(out['layers'])} / D={D}）")
    print(
        "  启动时: OSCAR_ASCEND_K_ROTATION_PATH=该文件 OSCAR_ASCEND_V_ROTATION_PATH=该文件\n"
        "  （rotation_v 字段被同一文件读取，见 rotation.py/backend.py）"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
