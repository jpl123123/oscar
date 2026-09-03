"""oscar_ascend — OSCAR INT2 KV 缓存量化 · vllm-ascend 0.23.0 零侵入插件。

包组成：
  plugin.py    vllm.general_plugins 入口（impl 类外科手术 + fail-soft + 心跳）
  config.py    OSCAR_ASCEND_* 配置
  format.py    160B 槽数值契约唯一权威（与 skill 沙盒 l3_numeric 同判据）
  rotation.py  per-layer 正交旋转加载
  kernels/     Triton + torch 双路径（store / decode / dequant）
  backend.py   AscendOscarAttentionBackendImpl（读/写/窗口三态）
"""
from . import format, config, rotation  # noqa: F401
from .kernels import HAS_TRITON  # noqa: F401

__version__ = "0.1.0"
__all__ = ["format", "config", "rotation", "HAS_TRITON", "__version__"]
