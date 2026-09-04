"""oscar_ascend.calib — OSCAR 旋转校准采样（worker 侧捕获 + 二次 apply_model 取数据）。

vLLM 0.23 / vllm-ascend 的层前向**只能**在引擎 forward context 内执行
（裸调 model(...) 会触发 `Forward context is not set`，真机 00:31 实测）。
因此校准流程为：
  1. 插件（worker 内，OSCAR_ASCEND_CALIB=1 时）为 full-attention 的 Attention 模块
     注册前向钩子，在**正常运行**（llm.generate → execute_model 带 context）时把
     未量化 K/V 追加到本模块注册表（cap 限制 token 数）；
  2. gen_rotations 跑一次 llm.generate（引擎正常前向）；
  3. 第二次 `llm.llm_engine.apply_model(calib.finalize_cov)`：worker 内**仅读取**
     钩子缓冲、算协方差（无需任何 model 前向），回传小体积 CPU dict；
  4. 父进程跨 rank 合并 → eigh → 旋转。

模块级函数（reset_captures / finalize_cov / register_attention_hook）= pickle 安全。
"""
from __future__ import annotations

import os

import torch

try:
    from vllm.model_executor.layers.attention import Attention
except Exception:  # pragma: no cover — 无 vllm 环境仅占位
    Attention = object  # type: ignore

# worker 侧捕获注册表：{layer_name: {"k": [..], "v": [..]}}（每层最多 _TOKEN_CAP 个 token）
_captures: dict[str, dict[str, list]] = {}
_TOKEN_CAP = int(os.environ.get("OSCAR_ASCEND_CALIB_TOKEN_CAP", "512"))
_REGISTERED_HOOK_IDS: set[int] = set()


def register_attention_hook(attn_module) -> None:
    """worker 内（Attention.__init__ 完成后）注册捕获钩子；只捕获 decoder self_attn。"""
    layer_name = getattr(attn_module, "layer_name", "") or ""
    if ".self_attn" not in layer_name:
        return
    if ".linear_attn" in layer_name:
        return
    key = id(attn_module)
    if key in _REGISTERED_HOOK_IDS:
        return
    _REGISTERED_HOOK_IDS.add(key)

    def _hook(module, q, k, v, kv_cache, attn_metadata, output):
        if k is None or v is None:
            return None
        d = _captures.setdefault(layer_name, {"k": [], "v": []})
        cur = sum(t.shape[0] for t in d["k"])
        if cur >= _TOKEN_CAP:
            return None
        take = k.shape[0]
        if cur + take > _TOKEN_CAP:
            take = _TOKEN_CAP - cur
        d["k"].append(k[:take].detach().float())
        d["v"].append(v[:take].detach().float())
        return None

    attn_module.register_forward_pre_hook(_hook)


def reset_captures(model=None) -> None:
    """清空捕获缓冲（apply_model 调用；签名兼容 fn(model)）。"""
    global _captures
    _captures = {}
    return None


def finalize_cov(model=None) -> dict:
    """worker 内读取钩子缓冲 → 协方差统计（无任何 model 前向）。

    返回 {layer_name: {"k": {"mean","cov","count"}, "v": {...}}}，张量已 CPU。
    """
    out = {}
    for layer_name, d in _captures.items():
        if not d["k"]:
            continue
        k = torch.cat(d["k"], dim=0).reshape(-1, d["k"][0].shape[-1])
        v = torch.cat(d["v"], dim=0).reshape(-1, d["v"][0].shape[-1])
        if k.shape[0] < 4:
            continue

        def _stats(x: torch.Tensor) -> dict:
            x = x.float()
            mean = x.mean(dim=0)
            xc = x - mean.unsqueeze(0)
            cov = (xc.t() @ xc) / max(1, xc.shape[0])
            return {"mean": mean.cpu(), "cov": cov.cpu(), "count": int(xc.shape[0])}

        out[layer_name] = {"k": _stats(k), "v": _stats(v)}
    return out
