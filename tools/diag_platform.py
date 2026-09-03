#!/usr/bin/env python3
"""tools/diag_platform.py — Docker 平台注册诊断（零侵入，任何环境可跑）。

用于真机现场定位 "Device string must not be empty"（platform device_type 为空）：
  1. 版本面：vllm / vllm-ascend / torch / torch-npu（importlib.metadata）
  2. vllm._C 是否存在（vendor fork interface.py 警告项）
  3. platform 插件发现/加载：ep.load('vllm_ascend:register') 是否成功
  4. resolve_current_platform_cls_qualname() → 实际 qualname
  5. 当前 current_platform 类 + device_type
  6. 强制激活 NPUPlatform 是否可行
  7. torch.npu 可用性/设备数
每项 try/except，不以本脚本自身失败掩盖现场事实。
"""
from __future__ import annotations

import importlib.metadata as md
import importlib.util
import sys

SECTION = "=" * 4


def _ver(name: str) -> str:
    try:
        return md.version(name)
    except Exception:
        return "NOT-FOUND"


def main() -> int:
    print(f"{SECTION} [1] 版本面")
    for n in ("vllm", "vllm-ascend", "torch", "torch-npu", "triton"):
        print(f"  {n}: {_ver(n)}")

    print(f"{SECTION} [2] vllm._C 是否存在（vendor interface.py:255 警告项）")
    print("  find_spec('vllm._C') =", importlib.util.find_spec("vllm._C"))

    print(f"{SECTION} [3] platform 插件发现/加载")
    try:
        from vllm.plugins import load_plugins_by_group
        from vllm.plugins import PLATFORM_PLUGINS_GROUP

        plugins = load_plugins_by_group(PLATFORM_PLUGINS_GROUP)
        print("  发现的 platform 插件:", list(plugins.keys()))
        if "ascend" in plugins:
            try:
                fn = plugins["ascend"]
                print("  register() 返回:", fn())
            except Exception as e:  # noqa: BLE001
                print(f"  register() 异常: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"  platform 插件枚举异常: {type(e).__name__}: {e}")

    print(f"{SECTION} [4] resolve_current_platform_cls_qualname()")
    try:
        from vllm.platforms import resolve_current_platform_cls_qualname

        print("  qualname =", resolve_current_platform_cls_qualname())
    except Exception as e:  # noqa: BLE001
        print(f"  解析异常: {type(e).__name__}: {e}")

    print(f"{SECTION} [5] 当前 platform / device_type（关键判据）")
    try:
        from vllm.platforms import current_platform

        print("  类 =", type(current_platform).__module__ + "." + type(current_platform).__name__)
        print("  device_type =", repr(getattr(current_platform, "device_type", "<none>")))
    except Exception as e:  # noqa: BLE001
        print(f"  current_platform 异常: {type(e).__name__}: {e}")

    print(f"{SECTION} [6] 强制激活 NPUPlatform 是否可行（gen_rotations 自愈路径）")
    try:
        import vllm.platforms as vp
        from vllm_ascend.platform import NPUPlatform

        vp.current_platform = NPUPlatform()
        print("  强制激活成功，device_type =", repr(vp.current_platform.device_type))
    except Exception as e:  # noqa: BLE001
        print(f"  强制激活异常: {type(e).__name__}: {e}")

    print(f"{SECTION} [7] torch.npu")
    try:
        import torch
        import torch_npu  # noqa: F401

        print("  is_available =", torch.npu.is_available())
        print("  device_count =", torch.npu.device_count())
    except Exception as e:  # noqa: BLE001
        print(f"  torch.npu 异常: {type(e).__name__}: {e}")

    print(f"{SECTION} diag done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
