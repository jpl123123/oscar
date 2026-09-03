# R-20260903-docker-device-type — 真机反馈反思记录（二次失败：platform device_type 为空）

> 触发：真机 Docker 重跑一键脚本（插件修复 846bfca 后）→ gen_rotations 仍在
> `DeviceConfig(device=current_platform.device_type)` 报 `Device string must not be empty`；
> 关键新证据：`[oscar-ascend] OSCAR_ASCEND_ENABLE=0 → 不注入`（插件已排除），故该错误与插件无关。
> 真机指纹：HEAD=846bfca·Docker（vllm=/vllm-workspace/vllm 源码树·vendor interface.py:255 `vllm._C` 缺失警告）
> 状态：OPEN（卡在：现场 diag 输出未回传——需 tools/diag_platform.py 数据定位）

## §1 Q1 — 为什么真机会报这个错

**判定（机制级假设，等 diag 实测确认）：Docker 的 vllm-ascend 平台插件
`vllm_ascend:register` 虽被 vllm 发现，但未被成功激活（load/return 链上被吞或有 vendor
差异），`current_platform` 解析落到 `UnspecifiedPlatform`（device_type=""）→ pydantic
`torch.device("")` 抛 `Device string must not be empty`。**

| 机制环节 | 证据 | 置信 |
|---|---|---|
| 插件被禁用仍复现 | 日志 `OSCAR_ASCEND_ENABLE=0 → 不注入` + 相同报错 | CONFIRMED（排除插件） |
| vllm 平台解析路径 | `vllm/platforms/__init__.py:203-250`（register() 返回 qualname；`except Exception: pass` 吞插件异常）；arg_utils.py:1712-1714（`current_platform.device_type` → DeviceConfig） | CONFIRMED（参考树） |
| vendor 差异嫌疑 | Docker vllm 含 `vllm/interface.py:255`（参考树 0.23.0 无此文件）+ `vllm._C` 缺失警告 ×4；vllm 位于 /vllm-workspace 源码树 | CONFIRMED（日志路径） |
| 空字符串来源 | `UnspecifiedPlatform.device_type` 类默认 ""（interface.py Platform 基类） | CONFIRMED（参考树推断/待 diag 实证） |

- 尚缺的证据（→ diag 五点）：① `ep.load('vllm_ascend:register')` 成败；②
  `resolve_current_platform_cls_qualname()` 返回值；③ `current_platform` 实际类名/device_type；
  ④ 强制 `NPUPlatform()` 是否可行；⑤ vendor vllm-ascend 与参考树（skill-ref-023）的差异点。
- 排除的候选：插件循环导入（本次 enable=0 仍复现，排除）；HAS_TRITON（无关）；
  模型目录/依赖缺失（引擎已走到 create_engine_config）。

## §2 Q2 — 为什么递交前沙盒没有检测到

- 缺口分类：**GAP-SIGNATURE**（"vendor docker 的 platform 激活成功"契约无检查；
  沙盒无 vendor 树，不可复现）。
- 沙盒现状：只建模算子数值/引擎静态路径；对 vendor fork（interface.py / vllm._C 缺失/
  平台插件激活）无任何探针。
- 缺口证据：二次失败均为"平台未激活"类，属环境契约，而非本项目算子/插件逻辑。

## §3 沙盒复现（机理、可追溯、禁止拷贝真机数值）

### §3.1 机制推导链
参考树：`vllm/platforms/__init__.py` `resolve_current_platform_cls_qualname()`：
load_plugins_by_group → register() 返回 qualname → 唯一 OOT 插件激活 →
`resolve_obj_by_qualname(qualname)()`。任一环节异常被 `except Exception: pass` 吞 →
`UnspecifiedPlatform`（device_type=""）。Docker 现象与该"静默回退"完全吻合。

### §3.2 新增检查与签名
- 检查项名：`platform-activation-npu`；判定词：FAIL（device_type 非 "npu"）。
- 签名结构：`current_platform.device_type == ""`（类别：平台未激活）。
- 新增探针：`tools/diag_platform.py`（七段输出现场取证，已接入一键自检）。

### §3.3 复现代码位置
- `tools/diag_platform.py`（新）；`tools/gen_rotations.py::_force_ascend_platform`（自愈尝试）。
- 沙盒不可复现声明：无 vendor vllm-ascend 树（rpc 限制）；以参考树契约推导 + 现场
  diag 五点确认。

### §3.4 无拷值声明
判据仅用 device_type=="npu"/"" 的类别语义；不含真机日志字面量。

## §4 修复（与 Q1 逐行对应；本次为"取证+自愈"，根因待 diag）

| 文件:行 | 改动 | 对应 Q1 机制 |
|---|---|---|
| `tools/diag_platform.py`（新建） | 七段诊断（版本/`vllm._C`/ep.load/qualname/current_platform/强制 NPU/npu 设备） | 定位①-⑤ |
| `tools/gen_rotations.py` | `_force_ascend_platform()`：device_type=="npu" 则继续；否则强制 `vp.current_platform=NPUPlatform()`；失败退出码 3 + 指引跑 diag | 自愈（若激活只是惰性/可强置） |
| `delivery/install_and_launch.sh` | 自检阶段接入 `tools/diag_platform.py`（tee 落盘，不阻断） | 现场取证即得 |

## §5 门禁重放结果

```text
py_compile gen_rotations.py / diag_platform.py OK；bash -n install_and_launch.sh OK
tests/test_numeric.py → 8 PASS / 0 FAIL（未受影响）
reflect gate（本记录）→ 待跑
```

## §6 退出条件（逐条 verdict）

| # | 条件 | verdict / 证据 |
|---|---|---|
| 1 | Q1 全证据根因 | OPEN：机制假设成立，①-⑤ 待 diag |
| 2 | Q2 分类+位置 | PASS：GAP-SIGNATURE |
| 3 | 沙盒旧 FAIL 新 PASS + 无拷值声明 | OPEN：vendor 树缺失（声明见 §3.3） |
| 4 | 修复逐行对应 | PASS（§4，本轮为取证/自愈层修复） |
| 5 | 门禁重放全绿 | PASS（§5 本地项） |
| 6 | 记录提交 | 随工作区提交 |

**最终 verdict：OPEN（卡在：真机 diag_platform.py 输出未回传；回传后按 Q1 五证据收敛）**


## §7 后续进展（2026-09-03 12:53 真机日志二）

- 平台问题**已自愈收敛**：`[gen-rotations] 强制激活 NPUPlatform: OK (device_type='npu')`，
  EngineCore 以 `device_config=npu` 启动到模型装载。Q1 假设成立（vendor 栈平台为惰性注册，
  强制激活即生效）——该环节由 8411ad8 的 `_force_ascend_platform()` 闭环。
- **新失败（OOM）与根因**：模型装载在 `w8a8_dynamic.get_weight → torch.empty(int8)` 处
  `NPU out of memory`（29.49 GiB 总容量 / 28.97 GiB 已用）。日志自证校准
  `tensor_parallel_size=1`——27B W8A8 权重 ≈27GB > 单卡容量，TP=1 必然 OOM（目标
  serve 用 TP4 正是为此）。**修复：gen_rotations 默认 `{"tensor_parallel_size":4,
  "gpu_memory_utilization":0.9}`（env `OSCAR_ASCEND_GEN_LLM_ARGS` 可覆盖）+ 一键脚本
  预检 `npu-smi` 快照 + 仅清理本模型残留进程（`pkill -f $MODEL_PATH`）。**
- 环境确认：vendor `patch_mamba_config.py:104` 把 attention block size 设为 **1536**
  （hybrid 双网格物理页=1536 / kernel=128），与本方案 N-06/双网格契约一致，无需改代码。
- 本记录保持 OPEN；退出条件改为：TP4 校准成功生成 oscar_rotations.pt → probe → serve。
