"""oscar_ascend.calib — 旋转校准采样（在 worker 内对已加载模型执行）。

vLLM 0.23 的 `LLM` 不暴露 `.model`（模型对象在 engine-core/worker 进程）。
官方通道：`llm.llm_engine.apply_model(fn)` → 每个 worker 执行 `fn(model)` 并回传结果
（vllm/v1/engine/llm_engine.py:419-420 → worker/worker_base.py:128 `fn(self.get_model())`）。

本模块函数为 **模块级（可 pickle 跨进程）**；`capture_cov` 在每个 worker：
  1. 对 full-attention 层 `self_attn.attn`(Attention) 注册前向钩子捕获**未量化** K/V；
  2. 对模型做一次纯文本前向（无多模态）；
  3. 逐层统计 K/V 协方差（TP 分片内 heads/count 各自累计，父进程按 rank 合并）
     —— 返回小体积 CPU dict，兼容 collective_rpc 回传。
"""
from __future__ import annotations

import torch

try:
    from vllm.model_executor.layers.attention import Attention
except Exception:  # pragma: no cover — 无 vllm 环境仅占位
    Attention = object  # type: ignore

# 采样序列长度（gen_rotations 写后读取；默认 128）
_CALIB_SEQ_LEN = 128


def _find_layers(model):
    mod = getattr(model, "language_model", model)
    for attr in ("layers",):
        sub = getattr(mod, attr, None)
        if sub is not None and len(sub) > 0:
            return sub
    mod = getattr(mod, "model", None) or mod
    return getattr(mod, "layers", [])


def capture_cov(model) -> dict:
    """worker 内执行：采样 K/V 协方差（rotated 之前 = 原始空间）。返回 {layer: {...}}。"""
    layers = _find_layers(model)
    captures: dict[int, dict[str, list]] = {}

    def _make_hook(i):
        def _hook(module, q, k, v, kv_cache, attn_metadata, output):
            if k is None or v is None:
                return None
            d = captures.setdefault(i, {"k": [], "v": []})
            d["k"].append(k.detach().float())
            d["v"].append(v.detach().float())
            return None

        return _hook

    hooks = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        module = getattr(attn, "attn", None)
        if module is not None and isinstance(module, Attention):
            hooks.append((i, module.register_forward_pre_hook(_make_hook(i))))

    # 一次纯文本前向（不依赖 generate/采样路径）
    T = int(max(32, min(512, _CALIB_SEQ_LEN)))
    vocab = 151936
    dev = next(model.parameters()).device
    ids = torch.randint(1, max(2, vocab), (1, T), dtype=torch.long, device=dev)
    pos = torch.arange(T, dtype=torch.long, device=dev).unsqueeze(0)
    model(input_ids=ids, positions=pos)

    for _, h in hooks:
        h.remove()

    out = {}
    for i in sorted(captures):
        k = torch.cat(captures[i]["k"], dim=0).reshape(-1, captures[i]["k"][0].shape[-1])
        v = torch.cat(captures[i]["v"], dim=0).reshape(-1, captures[i]["v"][0].shape[-1])
        if k.shape[0] < 4:
            continue

        def _stats(x: torch.Tensor) -> dict:
            x = x.float()
            mean = x.mean(dim=0)                       # [D]
            xc = x - mean.unsqueeze(0)
            cov = (xc.t() @ xc) / max(1, xc.shape[0])  # [D,D]
            # 回传 CPU 小张量（RPC 帧友好）
            return {"mean": mean.cpu(), "cov": cov.cpu(), "count": int(xc.shape[0])}

        out[str(i)] = {"k": _stats(k), "v": _stats(v)}
    return out
