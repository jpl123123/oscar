"""oscar_ascend.plugin — vllm.general_plugins 入口（零侵入运行期接入）。

职责（plan §5.5）：
  1. 环境检测：vllm-ascend 0.23.0 存在、AscendAttentionBackendImpl 可解析；
  2. 包裹 AscendAttentionBackendImpl.__init__：hybrid 模型 full-attention 层
     构造后把 impl.__class__ 升级为 AscendOscarAttentionBackendImpl（类外科手术，
     先例 kv_c8.py:130）并调用 _oscar_setup；
  3. fail-soft：任何异常 → 保持原生行为 + 打日志；
  4. 心跳计数：OSCAR_HEARTBEAT（供一键脚本三防线校验）。

入口点：pyproject [project.entry-points."vllm.general_plugins"]
    oscar_ascend = "oscar_ascend.plugin:load_plugin"
"""
from __future__ import annotations

import os

import torch

_HEARTBEAT: dict = {"loaded": 0, "layers": [], "errors": []}
_PATCHED = False


def load_plugin() -> None:
    """vllm.load_general_plugins() 调用（无参）。幂等。"""
    _install()


def _install() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("OSCAR_ASCEND_ENABLE", "auto") == "0":
        print("[oscar-ascend] OSCAR_ASCEND_ENABLE=0 → 不注入（原生路径）")
        _PATCHED = True
        return True
    try:
        import vllm_ascend  # noqa: F401
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    except Exception as e:
        _HEARTBEAT["errors"].append(f"platform import: {e}")
        print(f"[oscar-ascend] 平台不可用，跳过注入: {e}")
        _PATCHED = True
        return True

    from .backend import AscendOscarAttentionBackendImpl as OscarImpl

    orig_init = AscendAttentionBackendImpl.__init__

    def _patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            if _should_oscar(self):
                self.__class__ = OscarImpl
                self._oscar_setup()
                _HEARTBEAT["loaded"] += 1
                _HEARTBEAT["layers"].append(
                    f"heads={self.num_heads}/kv={self.num_kv_heads}/d={self.head_size}"
                )
            _HEARTBEAT["errors"].append("")  # 记录一次构造（诊断用）
        except Exception as e:  # pragma: no cover
            _HEARTBEAT["errors"].append(str(e))
            print(f"[oscar-ascend] 注入失败(fail-soft 回退原生): {e}")
        return None

    AscendAttentionBackendImpl.__init__ = _patched_init
    _PATCHED = True
    print(
        "[oscar-ascend] plugin 注入 OK: 包裹 AscendAttentionBackendImpl.__init__ "
        f"(hybrid 判定={os.environ.get('OSCAR_ASCEND_ENABLE', 'auto')}, "
        f"triton 可用={_triton_available()})"
    )
    return True


def _should_oscar(impl) -> bool:
    """判定该 impl 是否属于混合模型的 full-attention 层（plan §5.5）。"""
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
        if not getattr(cfg.model_config, "is_hybrid", False):
            return False
        attn_type = getattr(impl, "attn_type", "decoder")
        if str(attn_type) not in ("decoder", "AttentionType.DECODER"):
            return False
        if getattr(impl, "sliding_window", None) is not None:
            return False
        if getattr(impl, "sinks", None) is not None:
            return False
        # MTP/draft 层共享 full-attention 结构（qwen3_5_mtp.py:102 layer_type=full_attention）
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
