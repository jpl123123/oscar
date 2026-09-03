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


def load_plugin() -> None:
    """vllm.load_general_plugins() 调用（无参）。幂等 + fail-soft。"""
    global _PATCHED
    if _PATCHED:
        return
    if os.environ.get("OSCAR_ASCEND_ENABLE", "auto") == "0":
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
                _HEARTBEAT["loaded"] += 1
                _HEARTBEAT["layers"].append(
                    f"{getattr(self, 'layer_name', '?')}:heads={impl.num_heads}/kv={impl.num_kv_heads}/d={impl.head_size}"
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


def _should_oscar(layer, impl) -> bool:
    """hybrid 模型的 decoder full-attention 层（plan §5.5）。"""
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
        if not getattr(cfg.model_config, "is_hybrid", False):
            return False
        if str(getattr(impl, "attn_type", "decoder")) != "decoder":
            return False
        if getattr(impl, "sliding_window", None) is not None:
            return False
        if getattr(impl, "sinks", None) is not None:
            return False
        return True
    except Exception:
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
