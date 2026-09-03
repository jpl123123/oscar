# 模拟 vllm_ascend/device/device_op.py：模块顶层先 import attention（循环边）
from . import attention  # noqa: F401

class DeviceOperator:
    @classmethod
    def bootstrap(cls):
        return True
