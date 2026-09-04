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


## §8 后续进展（2026-09-03 12:58 真机日志三）—— fork 线程池崩溃

- TP4 已生效（EngineCore world_size=4；rank1/2/3 workers 开始初始化），但 EngineCore
  在 autograd 线程初始化时 C++ 硬崩溃：`pool INTERNAL ASSERT FAILED at
  "/pytorch/aten/src/ATen/ParallelOpenMP.cpp":64 Invalid thread pool!`
  （`set_num_threads` → torch::autograd::Engine::thread_init）。
- 根因：vllm 多进程 worker 默认 **fork** 启动；父进程（gen 校准，pid=765）已多线程
  （torch_npu/FunctionLoader），fork 后子进程 OpenMP 线程池失效（父日志自己警告
  `use of fork() may lead to deadlocks in the child`）。
- 修复：任何 vllm 导入前 `VLLM_WORKER_MULTIPROC_METHOD=spawn`（vllm.envs.py:68/892 支持
  literal ["fork","spawn"]）——gen_rotations 顶部 setdefault + install/serve 脚本默认导出。
- 该崩溃与插件/算子无关；同因很可能会在 TP4 的 vllm serve 上复现，故一并写入 serve 默认。
- 本记录保持 OPEN；退出条件：spawn 后校准成功 → probe → serve。


## §9 后续进展（2026-09-03 13:02 真机日志四）—— spawn worker 平台未迁移

- spawn 修复生效（EngineCore/worker 均不再有 ParallelOpenMP 崩溃），但 (Worker pid=1057)
  `Current platform  does not have 'current_device' attribute.` +
  `MemorySnapshot __post_init__ assert device_fn is not None`（mem_utils.py:88）。
- 根因：父进程 `_force_ascend_platform()` 的强制激活**不迁移**到 spawn 子进程；子进程
  重新解析平台又落回 UnspecifiedPlatform（device_type="" → 无 current_device）。
- 修复：vllm general 插件在 process0/engine-core/worker 都会执行 → `load_plugin()` 首步
  新增 `_bootstrap_platform()`：device_type 非 npu 且非空则跳过；为空则强制
  `current_platform = NPUPlatform()`（带 pid 日志）；该引导与 OSCAR_ASCEND_ENABLE 无关
  （enable=0 也引导，仅跳过算子注入）。此时 vllm_ascend 平台栈已由平台插件预热
  （日志 platform.py:62 先于插件打印），import 安全；非 NPU 环境不覆盖。
- 本记录保持 OPEN；退出条件：worker 平台引导日志 `平台引导: ... pid=<worker>` +
  校准成功 → probe → serve。


## §10 真根因确认（2026-09-04 00:00 diag 数据）—— VLLM_PLUGINS 白名单过滤

- diag [3]：`发现的 platform 插件: []`（尽管 Available 列表打印了 ascend）
  + [4] `qualname = vllm.platforms.interface.UnspecifiedPlatform` + [5] `device_type=''`
  + [6] 强制激活成功——完全吻合"发现但被过滤"。
- 根因：vllm `VLLM_PLUGINS` = 逗号分隔**跨组白名单**（envs.py:1041-1044，精确成员匹配）；
  我方脚本默认导出 `VLLM_PLUGINS=oscar_ascend` → platform 插件 `ascend` 被过滤 →
  `load_plugins_by_group(PLATFORM_PLUGINS_GROUP)` 返回空 → 平台未激活。
- 修复：白名单默认 `ascend,oscar_ascend`（install/serve 脚本 + README）。届时
  vllm_ascend:register 自动激活，spawn worker 亦自动正确；_bootstrap_platform 保留为
  兜底（device_type 已 npu 时 no-op）。
- 保留项：VLLM_WORKER_MULTIPROC_METHOD=spawn（fork 线程池崩溃独立问题，仍有效）。
- 本记录保持 OPEN；退出条件：平台自动激活（不再依赖强制引导日志）→ 校准成功 →
  probe → serve。


## §11 后续进展（2026-09-04 00:06 真机日志五）—— 平台全通，采样 API 修复

- 平台链路**完全打通**：diag [3] `发现的 platform 插件: ['ascend']`、[4]
  `qualname=vllm_ascend.platform.NPUPlatform`、[5] `device_type='npu'`；EngineCore + 4
  workers 全部启动，TP4 装载 8.5GB/rank 权重，KV cache 16.36GiB/rank，engine init 59.7s。
- 新失败（我们工具链自身）：`gen_rotations.py:138 AttributeError: 'LLM' object has no
  attribute 'model'` —— vLLM 0.23 的 LLM 不暴露 model（模型在 worker 进程）。
- 修复：改用官方通道 `llm.llm_engine.apply_model(fn)`（llm_engine.py:419 →
  worker_base.py:128 `fn(self.get_model())`）：新增 `oscar_ascend/calib.py::capture_cov`
  （模块级可 pickle），worker 内对已加载模型做一次纯文本前向，Attention 前向钩子捕获
  未量化 K/V → 逐层协方差（TP 分片累计）→ 父进程跨 rank 加权合并 → eigh → 保存。
- 本记录保持 OPEN；退出条件：saved oscar_rotations.pt → probe → serve。


## §12 后续进展（2026-09-04 00:18 真机日志六）—— RPC 序列化出口

- 平台/装载已稳定复现通过；新失败：`TypeError: Object of type <class 'function'> is not
  serializable. Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 to allow fallback to pickle-based
  serialization.`（vendor vllm `serial_utils.enc_hook`，collective_rpc 传函数默认拒绝）。
- 修复：按 vendor 官方提示出口，**任何 vllm import 前** `VLLM_ALLOW_INSECURE_SERIALIZATION=1`
  （gen_rotations 顶部 setdefault + install 脚本默认导出，serve 继承）；
  `calib.capture_cov` 为模块级函数（pickle 安全）。
- 一次性复查（本提交前完成）：跨 rank 协方差合并数学（两分片==全量，atol=1e-5）、
  R 正交性（RᵀR=RRᵀ=I，atol=1e-4）、特征值降序，全部本地数值验证 PASS；
  tests 8/8；后续链路（probe ref/triton → serve 注入）静态核对无已知缺口。
- 本记录保持 OPEN；退出条件：saved oscar_rotations.pt → probe → serve。
