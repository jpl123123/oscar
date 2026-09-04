"""oscar_ascend.plugin — vllm.general_plugins 入口（零侵入运行期接入）。

**重要（真机 Docker 教训 2026-09-03）**：绝不能在 general-plugins 阶段直接
`import vllm_ascend` —— 那会抢在 `vllm_ascend:register`（platform 插件，负责先备好
环 境再导入）之前触发 `vllm_ascend.device.device_op` 的 `DeviceOperator` 循环导入，
既让注入失败，还会连累 LLM 的 platform `device_type` 注册（`Device string must not
be empty`）。因此本插件**不导入 vllm_ascend**，改为在 `Attention.__init__` 完成后的
after-hook 里做 impl 类外科手术（此时 vllm_ascend 已成为平台插件正常加载完毕）。

职责：
  1. 包装 `vllm.model_executor.layers.attention.attention.Attention.__init__`（fail-soft）；
  2. 构造完成后：hybrid 模型 full-attention 层 → `impl.__class__` 换为
     `AscendOscarAttentionBackendImpl`（先例 kv_c8.py:130）+ `_oscar_setup()`；
  3. 心跳计数（一键脚本三防线校验）。

入口点：pyproject [project.entry-points."vllm.general_plugins"]
    oscar_ascend = "oscar_ascend.plugin:load_plugin"
"""
from __future__ import annotations

import os

_HEARTBEAT: dict = {"loaded": 0, "layers": [], "errors": []}
_PATCHED = False


def _bootstrap_platform() -> None:
    """进程级平台引导：强制 current_platform = NPUPlatform（vendor 自动激活失效）。

    真机实测（2026-09-03 13:02，spawn 后）：父进程的手动强制激活**不迁移**到 spawn
    worker；worker 内 vllm 重新解析平台又落回 UnspecifiedPlatform（device_type="" →
    MemorySnapshot `assert device_fn is not None` 崩溃）。vllm general 插件在每个进程
    （process0/engine-core/worker）都执行 → 在此统一引导。此时 vllm_ascend 平台栈已
    由平台插件预热（日志 platform.py:62 先于本插件打印），import 安全。
    """
    try:
        import vllm.platforms as vp

        cur_type = getattr(vp.current_platform, "device_type", "")
        if cur_type == "npu":
            return
        if cur_type not in ("", None):
            # 非 Ascend 环境（cuda/rocm/cpu…）：不覆盖 —— 本插件只服务于 NPU 栈
            print(f"[oscar-ascend] 平台引导跳过（current device_type={cur_type!r} 非 NPU）")
            return
        from vllm_ascend.platform import NPUPlatform

        vp.current_platform = NPUPlatform()
        print(
            "[oscar-ascend] 平台引导: current_platform → NPUPlatform "
            f"(device_type={vp.current_platform.device_type!r}, pid={os.getpid()})"
        )
    except Exception as e:  # pragma: no cover
        print(
            f"[oscar-ascend] ⚠️ 平台引导失败（当前进程可能仍为 UnspecifiedPlatform）: "
            f"{type(e).__name__}: {e}"
        )


def load_plugin() -> None:
    """vllm.load_general_plugins() 调用（无参）。幂等 + fail-soft。

    ① 每个进程（含 spawn worker）先做**平台引导**（与注入开关无关）；
    ② 注入（OSCAR impl 类外科手术）仅在 OSCAR_ASCEND_ENABLE != "0" 时安装。
    """
    global _PATCHED
    if _PATCHED:
        return
    _bootstrap_platform()
    if os.environ.get("OSCAR_ASCEND_ENABLE", "auto") == "0" and \
            os.environ.get("OSCAR_ASCEND_CALIB", "0") != "1":
        print("[oscar-ascend] OSCAR_ASCEND_ENABLE=0 → 不注入（校准/诊断用原生路径）")
        _PATCHED = True
        return
    try:
        from vllm.model_executor.layers.attention.attention import Attention
    except Exception as e:  # pragma: no cover
        _HEARTBEAT["errors"].append(f"attention import: {e}")
        print(f"[oscar-ascend] 无法定位 Attention（跳过注入）: {e}")
        _PATCHED = True
        return

    try:
        from vllm.config import get_current_vllm_config  # noqa: F401  预热（无害）

        orig_init = Attention.__init__

        def _patched_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            if os.environ.get("OSCAR_ASCEND_CALIB", "0") == "1":
                # 校准模式：不替换 impl，只挂 K/V 捕获钩子（走引擎正常运行路径）
                try:
                    from .calib import register_attention_hook

                    register_attention_hook(self)
                except Exception as e:  # pragma: no cover
                    print(f"[oscar-ascend] 校准钩子注册失败: {e}")
                return None
            try:
                impl = getattr(self, "impl", None)
                if impl is None or not _should_oscar(self, impl):
                    return None
                from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

                if not isinstance(impl, AscendAttentionBackendImpl):
                    return None
                if impl.__class__.__name__.startswith("AscendAttentionCP"):
                    return None  # 上下文并行分支不替换（非本模型路径）
                from .backend import AscendOscarAttentionBackendImpl as OscarImpl

                impl.__class__ = OscarImpl
                impl._oscar_setup()
                cfg = impl._oscar
                _HEARTBEAT["loaded"] += 1
                layer_name = getattr(self, "layer_name", "?")
                _HEARTBEAT["layers"].append(
                    f"{layer_name}:heads={impl.num_heads}/kv={impl.num_kv_heads}/d={impl.head_size}"
                )
                # ★ 自证点 1：OSCAR impl 真实替换成功（FULL 层必有；GDN 层不出现）
                print(
                    f"[oscar-ascend] ★ 类外科手术生效: {layer_name} "
                    f"→ AscendOscarAttentionBackendImpl (Hq={impl.num_heads}, "
                    f"Hk={impl.num_kv_heads}, D={impl.head_size}, "
                    f"slot=160B, triton={cfg.use_triton}, "
                    f"sink={cfg.sink_tokens}/recent={cfg.recent_tokens})"
                )
            except Exception as e:  # pragma: no cover — fail-soft 回退原生
                _HEARTBEAT["errors"].append(str(e))
                print(f"[oscar-ascend] 注入失败(fail-soft 回退原生): {e}")
            return None

        Attention.__init__ = _patched_init
        _PATCHED = True
        print(
            "[oscar-ascend] plugin 注入 OK: Attention.__init__ after-hook "
            f"(enable={os.environ.get('OSCAR_ASCEND_ENABLE', 'auto')}, "
            f"triton={_triton_available()})"
        )
    except Exception as e:  # pragma: no cover
        _HEARTBEAT["errors"].append(str(e))
        print(f"[oscar-ascend] 插件初始化失败（原生路径继续）: {e}")
        _PATCHED = True


def _is_hybrid_config(model_config) -> bool:
    """hybrid 判定（可测试的纯函数）。

    ① 引擎 ModelConfig.is_hybrid（registry ModelInfo 标注）；
    ② 兜底：Qwen3.5 等架构在 registry 未标 is_hybrid，但 hf_config.layer_types
       含非 attention 层（linear_attention/mamba）→ 仍是混合模型
       （真机 2026-09-04：serve 时 is_hybrid=False 但层内含 GDN → 无手术的根因）。
    """
    if getattr(model_config, "is_hybrid", False):
        return True
    hf = getattr(model_config, "hf_text_config", None) or getattr(model_config, "hf_config", None)
    layer_types = getattr(hf, "layer_types", None) or []
    return any(lt != "attention" for lt in layer_types)


_SKIP_REASONS_LOGGED: set = set()


def _should_oscar(layer, impl) -> bool:
    """hybrid 模型的 decoder full-attention 层（plan §5.5）；失败打印拒绝原因（首个即可）。"""
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
        mc = cfg.model_config
        reasons = []
        if not _is_hybrid_config(mc):
            reasons.append("not-hybrid")
        if str(getattr(impl, "attn_type", "decoder")) != "decoder":
            reasons.append(f"attn_type={getattr(impl, 'attn_type', None)!r}")
        if getattr(impl, "sliding_window", None) is not None:
            reasons.append("sliding-window")
        if getattr(impl, "sinks", None) is not None:
            reasons.append("sinks")
        if reasons:
            key = ",".join(reasons)
            if key not in _SKIP_REASONS_LOGGED:
                _SKIP_REASONS_LOGGED.add(key)
                print(f"[oscar-ascend][SKIP] {getattr(layer, 'layer_name', '?')} → 不替换: {key}"
                      f"（is_hybrid={getattr(mc, 'is_hybrid', '?')}，"
                      f"layer_types={'含非attention层' if _is_hybrid_config(mc) else '无/全attention'}）")
            return False
        return True
    except Exception as e:  # pragma: no cover
        print(f"[oscar-ascend][SKIP] _should_oscar 异常: {e}")
        return False


def _triton_available() -> bool:
    try:
        from vllm.triton_utils import HAS_TRITON

        return bool(HAS_TRITON)
    except Exception:
        return False


def heartbeat() -> dict:
    """一键脚本读取：插件是否注入、升级了多少层。"""
    return dict(_HEARTBEAT)
