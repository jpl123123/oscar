# R-20260904-calib-device-dtype — 真机反馈反思记录：finalize_cov 设备/精度混用（Expected all tensors to be on the same device + ERR01002）

> 触发：真机一键输出（2026-09-04 05:02:24，node93，`R-20260904-calib-capture-2d` 修复后第二次重校准）：
> `calib.py:135 cov_q += qtq` → `RuntimeError: Expected all tensors to be on the same device. Expected NPU tensor, ...`，
> 随后 `[ERROR] ERR01002 OPS invalid type`；Worker_TP0..TP3 全部命中 → `apply_model` 失败 → `❌ 旋转检查点生成失败`。
> 原始日志指针：用户会话消息（genrot 日志尾部；`/tmp/oscar_ascend_logs/*20260904_0501*`，node93）。
> 真机指纹：HEAD=`edf6e6b`（2D 捕获修复后）· 设备=node93 docker TP4 · 触发命令=`bash delivery/install_and_launch.sh`（自动重校准分支）
> 状态：RESOLVED

## §1 Q1 — 为什么真机会报这个错

**判定：`finalize_cov` 的协方差累加器用 `torch.zeros(dtype=torch.float64)` 在默认设备（CPU）创建，而被累加的 `qtq` 在 NPU（捕获张量 `.detach().float()` 后仍留 NPU）→ 设备混用运行时错误；且 torch-npu 不支持 float64（ERR01002 OPS invalid type）。本地 CPU 单测全部通过，因为本地一切都在 CPU + fp64 可用。**

| 机制环节 | 证据（file:line / commit / 契约行） | 置信 |
|---|---|---|
| 捕获张量留在 NPU（钩子只做 `.view/.detach/.float()`，无 `.cpu()`） | 工作区 `oscar_ascend/calib.py::register_attention_hook`（edf6e6b） | CONFIRMED |
| 累加器 `torch.zeros(q.shape[-1], q.shape[-1], dtype=torch.float64)` 无 device → 默认 CPU | 工作区 `calib.py:128-129`（edf6e6b 引入） | CONFIRMED |
| `cov_q += qtq`：CPU fp64 宿主 += NPU fp32 → RuntimeError（"Expected NPU tensor"），全 4 worker 同栈 | 真机日志（上） | CONFIRMED |
| torch-npu 不支持 float64 → 即便设备一致也会 ERR01002 OPS invalid type | 真机日志 ERR01002；torch-npu dtype 支持面（fp16/fp32/bf16） | CONFIRMED（次因） |
| 旧管线没踩此坑：旧 `_stats` 在 NPU 上按同一 dtype 计算（`x.float()` 全程），仅最终 `.cpu()` | 旧版 `calib.py::finalize_cov`（_stats 实现）；真机旧校准成功 | CONFIRMED |

- 尚缺的证据：无。
- 排除的候选与排除证据：
  - *qtq 本身在 CPU*：排除——matmul 输入 `qg = q[…]` 源自 NPU 捕获张量，运算结果必在 NPU；错误文案"Expected NPU tensor"与"累加器为 CPU"一致。
  - *EARR01002 为独立 dtype 失败*：不可完全分离（同栈各报一次），两类均已被本次修复同源消除（设备绑定 + fp32）。

## §2 Q2 — 为什么递交前沙盒没有检测到

- 缺口分类：**GAP-FIXTURE**（本地 fixture 运行在纯 CPU 环境——CPU 上"默认设备"恰好等于数据设备、fp64 恰好可用 → 设备绑定/dtype 契约完全不可见；真实执行环境为 NPU。类别归属按协议五类取 FIXTURE，另附注：本质是"执行环境语义漂移"，可在协议中提议新类 GAP-DEVICE）。
- 沙盒现状：`tests/test_numeric.py::t_calib_cov_q_sst`（edf6e6b）只断言 CPU 语义（形状/对称性/Σ_S≠Σ_V），从不检查"累加器必须绑定输入设备 + 禁用 fp64"。
- 缺口证据：`torch.zeros(…, dtype=torch.float64)` 在 CPU 测试中完全合法；没有一个检查项把"NPU 不支持 fp64 / 默认设备≠数据设备"写成契约。

## §3 沙盒复现（机理、可追溯、禁止拷贝真机数值）

### §3.1 机制推导链
1. 捕获张量 device = 运行设备（NPU）（钩子无 .cpu()）；
2. 累加器 `torch.zeros(dtype=float64)` 默认 device=CPU；
3. `+=`（NPU 张量）→ 运行时"同一设备"断言失败——**这是纯 Python/设备语义推导，与真机错误类别完全一致**。

### §3.2 新增检查与签名
- 检查项名：`calib_rpc_payload_cpu_fp32` + `calib_no_fp64_no_default_device`；判定词：FAIL；
- 签名结构（类别级）："累加器设备/dtype 与数据不一致 → 全 worker 同一运行时错误、校准管线终止" ↔ 真机观测类别一致；
- 双向证据：旧代码（CPU fp64 累加器）在"设备契约守卫"下 FAIL；新代码（device=dev + fp32）PASS。CPU 上无法直接复现运行时错误 → 采用**静态契约守卫 + RPC 载荷断言**双检查（机制等价、可本地执行）。

### §3.3 复现代码位置
- `tests/test_numeric.py::t_calib_cov_q_sst`：新增 `cov_q.device.type=="cpu"`、`dtype==float32` 断言 + 源码守卫（`calib.py` 禁 `torch.float64`、必须出现 `device=dev`）。
- 复现运行：`sandbox/.venv/bin/python tests/test_numeric.py` → 16 PASS / 0 FAIL；旧代码在源码守卫上 FAIL。

### §3.4 无拷值声明
签名/期望全部来源于：**torch 设备/dtype 语义 + torch-npu 支持面（fp16/fp32/bf16）+ 校准管线接口契约（apply_model RPC 回传 CPU）**；不含任何真机日志字面量（错误消息仅类别比对）。

## §4 修复（与 Q1 逐行对应）

| 文件 | 改动 | 对应 Q1 机制 |
|---|---|---|
| `oscar_ascend/calib.py::finalize_cov` | 累加器改 `torch.zeros(D, D, device=dev, dtype=torch.float32)`（dev = q.device），全程 fp32；末尾 `.detach().cpu()` | 设备混用 + fp64 |
| `oscar_ascend/calib.py::finalize_cov` | 删除未使用的 `_eigh_dims_stats`（死代码） | 整洁性 |
| `tools/gen_rotations.py::_rotation_from_stats` | eigh/组合改为**父进程 CPU fp64 固定执行**（移除 NPU 尝试路径；一次性 256×256 小矩阵，消除 NPU fp64 不确定性） | 同族风险消除 |
| `tests/test_numeric.py::t_calib_cov_q_sst` | RPC 载荷 CPU/fp32 断言 + 源码守卫（禁 fp64、累加器必须设备绑定） | Q3 |

## §5 门禁重放结果

```text
python3 -m py_compile oscar_ascend/calib.py tools/gen_rotations.py tests/test_numeric.py  → OK
bash -n delivery/install_and_launch.sh delivery/serve_oscar.sh                            → OK
sandbox/.venv/bin/python tests/test_numeric.py                                           → 16 PASS / 0 FAIL
  ✅ 校准 Σ_Q/Σ_S（qqt/sst）：含 RPC 载荷断言 + 静态设备/dtype 守卫（旧代码 FAIL 双向）
（L2/L5/CANN：纯 Python 修复，不适用；真机由 delivery 阻塞 probe 复核）
```

## §6 退出条件（逐条 verdict）

| # | 条件 | verdict / 证据 |
|---|---|---|
| 1 | Q1 全证据根因 | ✅ 设备混用 + fp64（torch-npu 不支持），4 worker 同栈 |
| 2 | Q2 分类+位置 | ✅ GAP-FIXTURE（附注拟提 GAP-DEVICE 类）；检查点 file:line |
| 3 | 沙盒旧 FAIL 新 PASS + 无拷值声明 | ✅ §3.3/§3.4（静态守卫双向 + 载荷断言） |
| 4 | 修复逐行对应 | ✅ §4 |
| 5 | 门禁重放全绿 | ✅ §5 |
| 6 | 记录提交 + 摘要 | ✅ 本记录 + TASK_STATUS 更新 |

**最终 verdict：RESOLVED**（修复=累加器绑定 NPU 设备 fp32 + 末尾 .cpu()；父进程固定 CPU fp64 eigh；真机复跑 `git pull && bash delivery/install_and_launch.sh` 预期校准通过并产出 v2 pt）。
