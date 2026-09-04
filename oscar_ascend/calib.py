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
    if ".mtp." in layer_name or layer_name.startswith("mtp."):
        # 与 plugin._should_oscar 一致：MTP 草稿层不量化（BF16 原生路径）→ 无需校准
        return
    key = id(attn_module)
    if key in _REGISTERED_HOOK_IDS:
        return
    _REGISTERED_HOOK_IDS.add(key)

    def _hook(module, args, kwargs=None):
        # register_forward_pre_hook 回调签名 = (module, args[, kwargs])；
        # Attention.forward(query, key, value, …)：**入口为 2D** [N, H*D]
        # （vllm attention.py:483-488 才 view 成 3D —— 真机 03:43 崩溃教训），
        # 在这里用模块头参数还原头视图（与 Attention.forward 内部 view 同语义）。
        if not args or len(args) < 3:
            return None
        q, k, v = args[0], args[1], args[2]
        if k is None or v is None or q is None:
            return None
        Hq = int(getattr(module, "num_heads", 0) or 0)
        Hk = int(getattr(module, "num_kv_heads", 0) or 0)
        D = int(getattr(module, "head_size", 0) or 0)
        if Hq <= 0 or Hk <= 0 or D <= 0 or q.shape[-1] % (Hq * D) != 0:
            print(f"[oscar-ascend][calib] ⚠️ 无法还原头视图（module attrs 缺失），跳过 {layer_name}")
            return None
        d = _captures.setdefault(layer_name, {"q": [], "k": [], "v": []})
        cur = sum(t.shape[0] for t in d["k"])
        if cur >= _TOKEN_CAP:
            return None
        take = min(k.shape[0], _TOKEN_CAP - cur)
        d["q"].append(
            q[:take].view(-1, Hq, D).detach().float() if q.dim() == 2
            else q[:take].detach().float()
        )
        d["k"].append(
            k[:take].view(-1, Hk, D).detach().float() if k.dim() == 2
            else k[:take].detach().float()
        )
        d["v"].append(
            v[:take].view(-1, Hk, D).detach().float() if v.dim() == 2
            else v[:take].detach().float()
        )
        return None

    attn_module.register_forward_pre_hook(_hook)


def reset_captures(model=None) -> None:
    """清空捕获缓冲（apply_model 调用；签名兼容 fn(model)）。"""
    global _captures
    _captures = {}
    return None


def finalize_cov(model=None) -> dict:
    """worker 内读取钩子缓冲 → 旋转校准统计（无任何 model 前向）。

    返回 {layer_name: {"q": Σ_Q, "v": Σ_S, "count"}}：
      Σ_Q = (1/H_kv) Σ_h Q_g^T Q_g / n  —— K 旋转的 hessian（qqt 目标，论文
      compute_kv_rotation.py:93-108 / README:312）；
      Σ_S = (1/H_kv) Σ_h (V_h·√w)^T(V_h·√w)/n，w_i=(k_i^T Σ_Qh k_i) —— V 旋转的
      score-weighted（sst 目标，compute_kv_rotation.py:111-136）。
    张量已 CPU。
    """
    out = {}
    for layer_name, d in _captures.items():
        if not d["k"] or not d["q"]:
            continue
        q = torch.cat(d["q"], dim=0)
        k = torch.cat(d["k"], dim=0)
        v = torch.cat(d["v"], dim=0)
        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            # 旧钩子/异常捕获可能仍是 2D（真机 03:43 IndexError 根因）——防御性跳过
            print(f"[oscar-ascend][calib] ⚠️ {layer_name} 捕获非 3D（dim={q.dim()}/{k.dim()}/{v.dim()}），跳过")
            continue
        n_hq = q.shape[1]
        n_hk = k.shape[1]
        if n_hq % n_hk != 0 or n_hq < n_hk:
            print(f"[oscar-ascend][calib] ⚠️ {layer_name} 头数异常 Hq={n_hq}/Hk={n_hk}，跳过")
            continue
        g = n_hq // n_hk
        n = k.shape[0]

        def _sym(x: torch.Tensor) -> torch.Tensor:
            return (x + x.T) / 2

        def _eigh_dims_stats(x: torch.Tensor) -> torch.Tensor:
            return _sym(x.float().reshape(-1, x.shape[-1]).t() @ x.float().reshape(-1, x.shape[-1]) / x.shape[0])

        cov_q = torch.zeros(q.shape[-1], q.shape[-1], dtype=torch.float64)
        cov_s = torch.zeros(v.shape[-1], v.shape[-1], dtype=torch.float64)
        for h in range(n_hk):
            qg = q[:, h * g : (h + 1) * g, :].float().reshape(-1, q.shape[-1])
            kh = k[:, h, :].float()
            vh = v[:, h, :].float()
            qtq = _sym(qg.t() @ qg / qg.shape[0])
            cov_q += qtq
            weights = (kh @ qtq * kh).sum(1)
            weights = weights / weights.sum().clamp(min=1e-12) * n
            vw = vh * weights.unsqueeze(1).sqrt()
            cov_s += _sym(vw.t() @ vw / n)
        cov_q = cov_q / n_hk
        cov_s = cov_s / n_hk
        out[layer_name] = {
            "q": {"cov": cov_q.cpu(), "count": n},
            "v": {"cov": cov_s.cpu(), "count": n},
        }
    return out
