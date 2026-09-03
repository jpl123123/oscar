"""机制类别复现（无真机字面量）：
A（旧插件行为）：插件阶段绕过 register 直连 device 栈 → ImportError 类别同真机日志
B（修复后）：插件入口文件顶层（行首）零 vllm_ascend 导入 —— 只允许函数内延迟导入
"""
import importlib
import pathlib
import re
import sys

# 场景 A：旧行为 = 插件直接 import vllm_ascend 链路（device 栈）
try:
    importlib.import_module("pkg_ascend.device")
    print("A: UNEXPECTED-PASS")
except Exception as e:
    print("A: 错误类别 =", type(e).__name__, "|", e)

# 场景 B：插件入口文件顶层不得出现 vllm_ascend 导入（延迟导入仅限缩进内）
src = pathlib.Path("oscar_ascend/plugin.py").read_text(encoding="utf-8")
top = re.findall(r"^(?:import|from)\s+vllm_ascend[^\n]*", src, re.M)
assert not top, f"插件顶层仍含 vllm_ascend 导入: {top}"
print("B: plugin.py 顶层无 vllm_ascend 导入（静态断言 PASS）—— 旧路径已不存在")
