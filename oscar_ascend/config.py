"""oscar_ascend.config — 插件配置（OSCAR_ASCEND_* 环境变量，零侵入）。

默认对齐 OSCAR vLLM PR（config.py:25-176）：oscar_int2 预设；
sink/recent 窗口默认 64/256；k/v clip 默认 0（不裁剪）；旋转路径默认空（单位阵，退化为
clipped INT2）。所有键均为本插件私有（OSCAR_ASCEND_*），不触碰 vllm envs。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .format import D_DEFAULT, LEVELS


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


@dataclass
class OscarAscendConfig:
    head_dim: int = D_DEFAULT
    key_quant_bits: int = 2
    value_quant_bits: int = 2
    k_clip_ratio: float = 0.0
    v_clip_ratio: float = 0.0
    k_rotation_path: str = ""
    v_rotation_path: str = ""
    sink_tokens: int = 64
    recent_tokens: int = 256
    staging_tokens: int = 8192
    group_size: int = 0               # 0 => 每向量一组（>= head_dim 语义）
    use_triton: bool = True           # HAS_TRITON 且未强制 torch 时
    window_enabled: bool = True
    use_paged: bool = False  # enable only after NPU paged probe
    use_fused_prep: bool = False  # enable after native MTP preparation probe
    attention_mode: str = "streaming"
    stream_workspace_bytes: int = 32 * 1024 * 1024
    verbose: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def levels(self) -> int:
        return LEVELS

    @property
    def data_bytes(self) -> int:
        return (self.head_dim * self.key_quant_bits + 7) // 8  # 64 @ D=256

    @property
    def slot_bytes(self) -> int:     # 逻辑槽 = 96 + 64（拆分落位）
        return 160

    @property
    def num_groups(self) -> int:
        return 1 if self.group_size == 0 else -(-self.head_dim // self.group_size)

    @classmethod
    def from_env(cls, head_dim: int | None = None) -> OscarAscendConfig:
        D = head_dim if head_dim is not None else _env_int("OSCAR_ASCEND_HEAD_DIM", D_DEFAULT)
        cfg = cls(
            head_dim=D,
            k_clip_ratio=_env_float("OSCAR_ASCEND_K_CLIP_RATIO", 0.0),
            v_clip_ratio=_env_float("OSCAR_ASCEND_V_CLIP_RATIO", 0.0),
            k_rotation_path=os.environ.get("OSCAR_ASCEND_K_ROTATION_PATH", ""),
            v_rotation_path=os.environ.get("OSCAR_ASCEND_V_ROTATION_PATH", ""),
            sink_tokens=max(0, _env_int("OSCAR_ASCEND_SINK_TOKENS", 64)),
            recent_tokens=max(0, _env_int("OSCAR_ASCEND_RECENT_TOKENS", 256)),
            staging_tokens=max(0, _env_int("OSCAR_ASCEND_STAGING_TOKENS", 8192)),
            group_size=_env_int("OSCAR_ASCEND_GROUP_SIZE", 0),
            # 真机 01:18 实测 torch-npu 位运算偏 `>>` 向量广播 bug；Triton-ascend 未上机验证
            # → 默认 torch 参考路径（全 NPU 算子），显式 OSCAR_ASCEND_USE_TRITON=1 才启用 Triton
            use_triton=os.environ.get("OSCAR_ASCEND_USE_TRITON", "0") == "1",
            verbose=os.environ.get("OSCAR_ASCEND_VERBOSE", "1") != "0",
        )
        cfg.window_enabled = cfg.staging_tokens > 0 and (cfg.sink_tokens > 0 or cfg.recent_tokens > 0)
        cfg.use_paged = os.environ.get("OSCAR_ASCEND_USE_PAGED", "0") == "1"
        cfg.use_fused_prep = os.environ.get("OSCAR_ASCEND_FUSED_PREP", "0") == "1"
        cfg.attention_mode = os.environ.get("OSCAR_ASCEND_ATTENTION_MODE", "streaming")
        cfg.stream_workspace_bytes = max(0, _env_int("OSCAR_ASCEND_STREAM_WORKSPACE_MIB", 32)) * 1024 * 1024
        if cfg.attention_mode not in ("streaming", "native"):
            raise ValueError("OSCAR_ASCEND_ATTENTION_MODE must be streaming or native")
        if not all(0 <= r <= 1 for r in (cfg.k_clip_ratio, cfg.v_clip_ratio)):
            raise ValueError("OSCAR clip ratios must be finite and in [0, 1]")
        if cfg.group_size != 0:
            raise ValueError("Only per-vector OSCAR quantization (GROUP_SIZE=0) is supported")
        cfg.extra["enable"] = os.environ.get("OSCAR_ASCEND_ENABLE", "auto")
        return cfg
