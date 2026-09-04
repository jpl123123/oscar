# R-20260904-calib-capture-2d — 真机反馈反思记录：校准 finalize_cov 4 worker 全挂（IndexError: too many indices for tensor of dimension 2）

> 触发：真机一键输出（2026-09-04 03:43:51，node93，`R-20260904-oscar-int2-mtp-precision` 修复后首次重校准）：
> `calib.py:104 qg = q[:, h*g:(h+1)*g, :]` → `IndexError: too many indices for tensor of dimension 2`
> （Worker_TP0..TP3 全部命中；随后 `collective_rpc` 失败 → `gen_rotations.py` 退出 1 → `❌ 旋转检查点生成失败`）。
> 原始日志指针：用户会话消息（serve.log/genrot 日志尾部；`/tmp/oscar_ascend_logs/*20260904_033923*`，node93）。
> 真机指纹：HEAD=`ed6340a`（上一轮修复即触发点）· 设备=node93 docker TP4 · 触发命令=`bash delivery/install_and_launch.sh`（自动重校准分支）
> 状态：RESOLVED

## §1 Q1 — 为什么真机会报这个错

**判定：校准钩子捕获的是 `Attention.forward` 的**入口实参**——2D `[N, H*D]`（头维未展开），而上一轮修的 `finalize_cov` 按 3D `[N, Hq, D]` 切片 → 4 worker 同时 IndexError。**

| 机制环节 | 证据（file:line / commit / 契约行） | 置信 |
|---|---|---|
| vllm `Attention.forward` 入口 query/key/value 为 2D `[N, H*D]`；**3D view 发生在 forward 内部**（`query.view(-1, num_heads, head_size)` 等） | `references/vllm/vllm/model_executor/layers/attention/attention.py:483-488`（"Handle both 2D and 3D query"）；`AscendQwen3NextAttention` 调 `self.attn(q, k, v)` 直接传 2D（`references/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py:51-81`） | CONFIRMED |
| 旧钩子（上一轮之前）只取 `args[1]/args[2]`（k/v），`_stats` 用 `reshape(-1, D)` —— 对 2D `[N, Hk*D]` 恰好按"token×head 行"展开 → 旧管线从不触碰 Q 的头维度 → 从未暴露该形状差异 | 旧版 `oscar_ascend/calib.py::finalize_cov`（`_stats` reshape(-1, D)）；真机历史：旧校准成功生成 16 层 pt | CONFIRMED |
| 上一轮修复引入 3D 假设：`finalize_cov` 新代码 `q[:, h*g:(h+1)*g, :]`（qqt 分组需要头维）→ 与钩子实际捕获的 2D 契约冲突 | 工作区 `oscar_ascend/calib.py`（ed6340a 引入，:104 处崩溃） | CONFIRMED |
| MTP 层捕获未排除（旧钩子同样捕获 `mtp.layers.0...`，引入无意义的层 0 旋转） | `register_attention_hook`（修复前只排 `.linear_attn`） | CONFIRMED（次要） |

- 尚缺的证据：无（根因=形状契约，本地已逐行对到 attention.py:483-488 与崩溃行）。
- 排除的候选与排除证据：
  - *捕获内容错误（Q/K/V 张量不匹配）*：排除——旧管线用同一钩子的 k/v 生成过成功 pt；崩溃前 `Processed prompts 1.16s/it` 说明常规前向与捕获均正常，问题只在 finalize 的切片形状。
  - *NPU 侧 view/维度问题*：排除——错误是纯 Python 张量索引（所有 rank 同一 traceback），发生在 CPU/worker 内统计阶段。

## §2 Q2 — 为什么递交前沙盒没有检测到

- 缺口分类：**GAP-FIXTURE**（本地 fixture 与真实接口形状漂移：测试直接用手工 3D 捕获喂 `finalize_cov`，从未经过真实 `register_attention_hook` 的 2D 入口）。
- 沙盒现状：`tests/test_numeric.py::t_calib_cov_q_sst`（ed6340a 新增）**绕过了钩子**，手工构造 `calib._captures = {… "q": [q(64,8,32) 3D] …}` → 测试"3D 正常"但真机钩子给 2D → 测试通过≠真实契约。
- 缺口证据：参考树 `attention.py:483-488` 已明示"2D 与 3D 双形态"，但知识库/契约表无"Attention.forward 入口 2D"这一条；上一轮编写的测试 fixture 按 impl 层（3D）而非模块入口（2D）建模（[UNKNOWN→现已升级为 CONFIRMED 契约]）。

## §3 沙盒复现（机理、可追溯、禁止拷贝真机数值）

### §3.1 机制推导链
1. vllm `Attention.forward(query, key, value…)` 的 docstring/实现："Handle both 2D [num_tokens, hidden] and 3D [num_tokens, heads, head_dim] query"，且内部 `query.view(-1, num_heads, head_size)`（attention.py:483-488）→ 模型调用侧 2D 是合法契约；
2. 模块级 `forward_pre_hook` 看到的是 **入口实参**（`args`），未做 view → 捕获为 2D；
3. `finalize_cov` 若按 3D 索引 → `IndexError`（与真机栈一致）。

### §3.2 新增检查与签名
- 检查项名：`calib_capture_2d_entry_view`；判定词：FAIL；
- 签名结构（类别级）："捕获张量维度 ≠ 处理代码假定维度 → 全 worker IndexError/静默错统计" ↔ 真机观测类别 = "所有 worker 同一 Python 索引异常、校准管线终止"；
- 双向证据：旧代码（3D 切片 + 2D 捕获）FAIL；新代码（钩子 2D→3D 还原 + finalize 防御）PASS。

### §3.3 复现代码位置
- `tests/test_numeric.py::t_calib_cov_q_sst`：真实 FakeAttention（num_heads=8/num_kv_heads=2/head_size=32）→ `register_attention_hook` → **2D** (64, 256)/(64, 64)/(64, 64) 调用 → 断言 3D 视图 `(64,8,32)/(64,2,32)` + `finalize_cov` Σ_Q/Σ_S + MTP 层不捕获。
- 复现运行：`sandbox/.venv/bin/python tests/test_numeric.py --` → 上轮代码：`❌ 校准 Σ_Q/Σ_S: IndexError too many indices`（同真机）；本轮：✅ 16/16。

### §3.4 无拷值声明
签名/期望全部来源于：**vllm 参考树 attention.py:483-488 的形状契约 + 模块头参数（num_heads/num_kv_heads/head_size）**；不含任何真机日志字面量（错误消息仅类别比对）。

## §4 修复（与 Q1 逐行对应）

| 文件 | 改动 | 对应 Q1 机制 |
|---|---|---|
| `oscar_ascend/calib.py::register_attention_hook` | 钩子内用模块 `num_heads/num_kv_heads/head_size` 把 2D 捕获**还原为 3D**（与 attention.py:483-488 同语义；保留 3D 输入兼容） | 2D 入口契约 |
| `oscar_ascend/calib.py::register_attention_hook` | 跳过 `mtp.` 层（与 plugin._should_oscar 策略一致，草稿层不校准不量化） | MTP 层次要问题 |
| `oscar_ascend/calib.py::finalize_cov` | 3D 校验 + 头数整除校验，异常形态**跳过并告警**（不再 IndexError 炸整条校准） | 防御：同类形状回归不再击穿校准 |
| `tests/test_numeric.py::t_calib_cov_q_sst` | 改为**真实钩子路径 + 2D 入口**（GAP-FIXTURE 修复） | Q3 |

## §5 门禁重放结果

```text
python3 -m py_compile oscar_ascend/calib.py tests/test_numeric.py        → OK
sandbox/.venv/bin/python tests/test_numeric.py                          → 16 PASS / 0 FAIL
  （含新 t_calib_cov_q_sst：真实钩子 2D→3D 还原；旧代码在该测试上 IndexError = 双向证据）
delivery/install_and_launch.sh 阶段4 逻辑不变（v2 配方自检→自动重校准）
（L2/L5/CANN：纯 Python 修复，不适用）
```

## §6 退出条件（逐条 verdict）

| # | 条件 | verdict / 证据 |
|---|---|---|
| 1 | Q1 全证据根因 | ✅ attention.py:483-488 + 崩溃行 + 旧钩子契约对比 |
| 2 | Q2 分类+位置 | ✅ GAP-FIXTURE（t_calib_cov_q_sst 绕过钩子喂 3D fixture） |
| 3 | 沙盒旧 FAIL 新 PASS + 无拷值声明 | ✅ §3.3/§3.4 |
| 4 | 修复逐行对应 | ✅ §4 |
| 5 | 门禁重放全绿 | ✅ §5 |
| 6 | 记录提交 + 摘要 | ✅ 本记录 + TASK_STATUS 更新 |

**最终 verdict：RESOLVED**（修复=钩子按模块头参数还原 3D + finalize 防御跳过；真机复跑 `git pull && bash delivery/install_and_launch.sh` 预期校准通过并产出 v2 pt）。
