# 模拟 vllm_ascend/attention/attention_v1.py
from .device import DeviceOperator  # 无条件反引 device → 固有循环边

ATTN_READY = "ok"
