# R-20260903-docker-circimport — 真机反馈反思记录（Docker：gen_rotations 启动失败）

> 触发：真机 Docker 运行 `delivery/install_and_launch.sh` → 阶段4 生成旋转检查点 FAIL
> （原始日志：用户粘贴，日期 2026-09-03 12:42-12:43；`/tmp/oscar_ascend_logs/genrot_*.log`）
> 真机指纹：HEAD=9a9ff68（修复前）· 设备=Docker（vllm/vllm-ascend 预装）· 触发命令=`bash delivery/install_and_launch.sh`
> 状态：RESOLVED

## §1 Q1 — 为什么真机会报这个错

**判定：插件在 `vllm.general_plugins` 阶段直接 `import vllm_ascend`，触发该包在"未经
`vllm_ascend:register`（platform 插件）前置准备"导入时的 `device_op → attention →
device_op` 循环导入（`DeviceOperator` 部分初始化），并连累注册进程里 platform
`device_type` 未就绪（`Device string must not be empty`）。**

| 机制环节 | 证据（file:line / commit / 契约行） | 置信 |
|---|---|---|
| 旧插件导入点 | `oscar_ascend/plugin.py`（9a9ff68）`_install()` 首行 `import vllm_ascend` | CONFIRMED（真机日志错误文本） |
| 循环导入类别 | 真机日志：`cannot import name 'DeviceOperator' from partially initialized module 'vllm_ascend.device.device_op' (most likely due to a circular import)` | CONFIRMED（类别；机制复现见 §3） |
| 平台注册顺序 | vllm/plugins：platform 组在 `current_platform` 首次访问时经 `vllm_ascend:register` 导入（`vllm/plugins/__init__.py:60-90`）；general 组在 engine args 组装时先执行（`engine/arg_utils.py:745-747`） | CONFIRMED（参考树） |
| 连累 device_type | 真机日志：`RuntimeError: Device string must not be empty`（`vllm/config/device.py:78`） | CONFIRMED（真机日志，属注册未完成的下游表现） |

- 尚缺的证据：`vllm_ascend/__init__` 内部精确的循环边（真机容器内可 `python -c
  "import vllm_ascend"` 复现——留给用户下一条现场输出确认；不影响修复正确性）。
- 排除的候选：
  - "HAS_TRITON=False 导致"：不成立——错误与 triton 无关，且日志显示 triton 已预装。
  - "gen_rotations 直接构造 LLM 不合法"：不成立——同容器的 `vllm serve` 用同一
    engine 路径；根因是插件抢先导入破坏了平台注册时序。

## §2 Q2 — 为什么递交前沙盒没有检测到

- 缺口分类：**GAP-SIGNATURE**（"插件导入时序不得抢先 vllm_ascend 平台注册"这一
  契约行不存在；沙盒只模拟算子数值/合同，不模拟 Python 包导入时序）。
- 沙盒现状（检查在哪里、检查了什么）：`sandbox/` 只建模 CANN 算子数值/契约与
  engine 静态路径（无包导入时序仿真）；`plugins` 加载语义仅在 plan §6 E10 引用了
  `vllm/plugins/__init__.py:58-80`，未转成强制规则。
- 缺口证据：静态门禁（design gate/knowledge gate）只检查 references 路径存在性，
  不检查插件源码导入面——本次错误由**真机首个插件导入**暴露。

## §3 沙盒复现（机理、可追溯、禁止拷贝真机数值）

### §3.1 机制推导链
1. vLLM 启动顺序：general 插件（含本插件）在 engine args 组装时执行；
   platform 插件（`vllm_ascend:register`）在 `current_platform` 首次被访问时执行。
2. vllm-ascend 正常路径：`register()` 先做前置准备再导入设备栈；
   旧插件绕过该前置直接 `import vllm_ascend` → 包内 `device → attention → device`
   循环边在 device 部分初始化时命中 → ImportError（类别=真机日志）。
3. 修复路径：插件**零顶层 vllm_ascend 导入**，仅在 `Attention.__init__` after-hook
   （vllm_ascend 已由平台插件正常加载完成）里做 impl 类外科手术。

### §3.2 新增检查与签名
- 检查项名：`plugin-top-level-no-vllm-ascend-import`；判定词：FAIL；
- 签名结构：`oscar_ascend/plugin.py` 顶层（行首）出现 `import/from vllm_ascend ...`
- 与真机故障签名的对应：类别一致（"插件抢先导入 vllm_ascend"）；字节/文本不比对。

### §3.3 复现代码位置
- `plan/reflections/repro_circimport/`（合成 `pkg_ascend` 循环边，模拟包）
- 运行：`PYTHONPATH=plan/reflections/repro_circimport python3 plan/reflections/repro_circimport/sim.py`
  → 场景 A（旧行为）：`ImportError: cannot import name 'DeviceOperator' from partially
  initialized module 'pkg_ascend.device' (most likely due to a circular import)`
  （错误类别与真机一致）；场景 B（固定后）：`plugin.py 顶层无 vllm_ascend 导入（静态断言 PASS）`
- 沙盒不可完整复现声明：无 vllm-ascend 包体（合法）；以合成循环边复现**错误类别**。

### §3.4 无拷值声明
签名/期望/阈值全部来源于：合成包结构 + 插件源码静态断言；不含任何真机日志字面量；
真机数值仅用于"类别一致性"人工比对。

## §4 修复（与 Q1 逐行对应）

| 文件:行 | 改动 | 对应 Q1 机制 |
|---|---|---|
| `oscar_ascend/plugin.py`（`load_plugin`/`_patched_init` 整体重写） | 删除插件阶段 `import vllm_ascend`；改为包装 `Attention.__init__`，其 after-hook 内（vllm_ascend 已加载）再 `from vllm_ascend.attention.attention_v1 import ...` 并做 `impl.__class__` 外科手术 + `_oscar_setup()` | 消除抢先导入；注入时机后移 |
| `oscar_ascend/plugin.py`（`_should_oscar`） | 判定改用 `layer/impl`（hybrid + decoder + 非 sliding_window + 非 sinks + 非 CP 类） | 同 Q1 |
| `delivery/install_and_launch.sh`（阶段4） | 校准进程前加 `OSCAR_ASCEND_ENABLE=0`（BF16 原路径采集 K/V，插件不参与） | 隔离校准与注入路径 |
| `tests/test_numeric.py` | 新增 `t_plugin_purity` 静态守卫（顶层 vllm_ascend 导入即 FAIL） | 防重漏（§3.2） |

## §5 门禁重放结果

```text
python3 tests/test_numeric.py
  → 8 PASS / 0 FAIL（含 plugin 顶层静态守卫）
bash -n delivery/install_and_launch.sh && bash -n delivery/serve_oscar.sh → OK
skill: gates.sh --knowledge → errors=0；--level L3 --op store → ALL GREEN / PASS
design gate → PASS（plan 修订后重跑）
```

## §6 退出条件（逐条 verdict）

| # | 条件 | verdict / 证据 |
|---|---|---|
| 1 | Q1 全证据根因 | PASS：循环导入类别 + 插件导入点（§1） |
| 2 | Q2 分类+位置 | PASS：GAP-SIGNATURE（§2） |
| 3 | 沙盒旧 FAIL 新 PASS + 无拷值声明 | PASS：sim.py 场景 A FAIL（类别一致）/ B PASS（§3，§3.4） |
| 4 | 修复逐行对应 | PASS（§4） |
| 5 | 门禁重放全绿 | PASS（§5） |
| 6 | 记录提交 | 本记录随工作区提交 |

**最终 verdict：RESOLVED**
