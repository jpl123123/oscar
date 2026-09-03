# 合成"vllm_ascend"循环导入模拟（机制类别复现，不含任何真机字面量）
# 语义：register() 负责"前置准备"（设置标志），包体才导入 device 栈；
# 未经 register 直接导入 device → 循环导入（错误类别 = 真机日志）。
import os

DEVICE_TYPE = ""
_READY = os.environ.get("SIM_ASCEND_READY", "") == "1"

def register():
    os.environ["SIM_ASCEND_READY"] = "1"
    global _READY
    _READY = True
    global DEVICE_TYPE
    from .device import DeviceOperator
    DeviceOperator.bootstrap()
    DEVICE_TYPE = "npu"

def ensure_device():
    if not _READY:
        # 模拟插件直连导入路径：绕过 register 直接拉 device 栈
        from .device import DeviceOperator  # noqa: F401
    return DEVICE_TYPE

def get_device_type():
    return DEVICE_TYPE
