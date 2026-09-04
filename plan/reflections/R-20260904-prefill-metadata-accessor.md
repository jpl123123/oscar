# R-20260904-prefill-metadata-accessor — 真机反馈反思记录：serve 崩溃（Boolean value of Tensor with more than one value is ambiguous）

> 触发：真机一键输出（2026-09-04 05:21:38，node93，`9950764` 后首次 serve 请求阶段）：
> `backend.py:179 forward → _prefill_attention → backend.py:237 (getattr(attn_metadata,"seq_lens_cpu",None) or attn_metadata.seq_lens).tolist()`
> → `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`（Worker_TP3 等）。
> 原始日志指针：用户会话消息（serve.log 05:20-05:21 段；`/tmp/oscar_ascend_logs/serve.log`，node93）。
> 真机指纹：HEAD=`9950764` · 设备=node93 docker TP4 · 触发命令=`bash delivery/install_and_launch.sh` + 用户 ais_bench
> 状态：RESOLVED

## §1 Q1 — 为什么真机会报这个错

**判定：`_prefill_attention` 用 `(seq_lens_cpu or seq_lens)` 的 `or` 短路取备选，而运行时 `attn_metadata.seq_lens_cpu` 是**多元素 Tensor**——把 Tensor 当布尔值必然抛 Boolean ambiguity。该行在 MTP 目标层 decode/续写（attn_state=SpecDecoding/ChunkedPrefill，走 _prefill_attention 分支）时触达。**

| 机制环节 | 证据（file:line / commit / 契约行） | 置信 |
|---|---|---|
| `seq_lens_cpu` 在 vllm-ascend `AscendMetadata` 中恒为 torch.Tensor（builder.build 里 `seq_lens_cpu=seq_lens`） | `references/vllm-ascend/vllm_ascend/attention/attention_v1.py:353-366`（`seq_lens_cpu=seq_lens`，:293-298 选定 CPU 张量） | CONFIRMED |
| 旧写法 `(X or Y)` 对多元素 Tensor 抛 `Boolean value of Tensor with more than one value is ambiguous` | 真机栈 + PyTorch 语义（`bool(tensor)` 仅 0/1 元素合法）；本地测试 `(torch.tensor([1,2]) or None)` 必然 RuntimeError（双向证据） | CONFIRMED |
| MTP 下目标层在 decode 各步的 attn_state=SpecDecoding → 插件 forward 落入 `_prefill_attention`（仅 DecodeOnly 分支例外） | `oscar_ascend/backend.py::forward`（state==DecodeOnly→decode，else→prefill）；`references/vllm-ascend/worker/model_runner_v1.py:1481-1492`（MTP 且 num_scheduled_tokens==1 → SpecDecoding） | CONFIRMED |
| 同文件 query_start_loc 访问已用 `is not None`（安全；唯独 seq_lens 用 `or`） | `oscar_ascend/backend.py:230-238`（修复前） | CONFIRMED |
| 为何早期运行（03:17 接受率 9.5-25% 那次）未在此行崩溃 | 属未确证差异（[UNKNOWN]：可能元数据路径不同/请求时序不同；该行自 a07e5a2 即存在，说明触发条件=seq_lens_cpu 为多元素 Tensor 的路径此前未被走到）——本轮修复样式上完全消除此类风险，不再依赖区分 | [UNKNOWN] |

- 尚缺的证据：早期运行未崩的精确元数据差异（不影响修复）。
- 排除的候选与排除证据：
  - *seq_lens 语义错误（非形状问题）*：排除——异常是纯 Python 布尔判定，发生在取值阶段，且 qsl 同模式用 `is not None` 一直安全。
  - *clip/pack/量化路径*：排除——崩溃点远在量化算子之后；写路径 ★ 16 层已全部成功打印。

## §2 Q2 — 为什么递交前沙盒没有检测到

- 缺口分类：**GAP-VERBATIM**（检查/测试只看"取到列表"纯 CPU 语义，未把"元数据字段可能是 GPU/多元素 Tensor、禁止作布尔值"写成契约；本地测试从未构造 `seq_lens_cpu` 为 Tensors 的元数据）。
- 沙盒现状：`tests/test_numeric.py::t_prefill_ref` 只验证 prefill 数值路径，传入的是**手工 list**；无任何元数据访问器测试。
- 缺口证据：`metadata_batch_lists` 概念不存在；`or` 模式与 `is not None` 模式混用而无守卫；本地 fixture 从不包含 Tensor 型元数据字段。

## §3 沙盒复现（机理、可追溯、禁止拷贝真机数值）

### §3.1 机制推导链
1. PyTorch：`bool(t)` 仅允许 0/1 元素；多元素 Tensor 在布尔上下文抛 RuntimeError；
2. `a or b` 等价于 `bool(a) and a or b` 的求值序 → 多元素 Tensor 的 `a` 直接触发异常；
3. vllm-ascend 构建的 `AscendMetadata.seq_lens_cpu` 为长度=B 的张量 → 任何走到该行且 `seq_lens_cpu` 非 None 的执行即崩溃（与真机栈一致）。

### §3.2 新增检查与签名
- 检查项名：`prefill_metadata_no_tensor_bool`；判定词：FAIL；
- 签名结构（类别级）："元数据 Tensor 字段被当作布尔值 → 运行时 Boolean ambiguity，worker 崩溃" ↔ 真机类别一致；
- 双向证据：旧表达式 `(torch.tensor([16384,5]) or None)` 必然 RuntimeError（测试内显式断言）；新 helper `metadata_batch_lists` 在 A（Tensor）、B（None 回退）、C（裸 list）三形态均返回正确 list。

### §3.3 复现代码位置
- `oscar_ascend/backend.py::metadata_batch_lists`（新纯函数，无 vllm 依赖，可单测）；
- `tests/test_numeric.py::t_prefill_metadata_accessor`（三形态 + 旧模式必错断言）；
- 复现运行：`sandbox/.venv/bin/python tests/test_numeric.py` → 17 PASS / 0 FAIL。

### §3.4 无拷值声明
期望值来自 PyTorch 布尔语义 + vllm-ascend 元数据契约（attention_v1.py:353-366）；测试中的 16384/5 为形状构架值，非真机日志字面量（真机只用于类别比对）。

## §4 修复（与 Q1 逐行对应）

| 文件 | 改动 | 对应 Q1 机制 |
|---|---|---|
| `oscar_ascend/backend.py` | 新增 `metadata_batch_lists(attn_metadata)`：`is not None` 判定 + `tolist()`（Tensor 任意设备）/裸 list 兼容 + `_seq_lens_cpu` 回退；`_prefill_attention` 改用该函数（替换 `or` 写法） | Tensor 作布尔值 |
| `tests/test_numeric.py` | 新增 `t_prefill_metadata_accessor`（旧模式必错 = 双向证据） | Q3 |

## §5 门禁重放结果

```text
python3 -m py_compile oscar_ascend/backend.py tests/test_numeric.py  → OK
sandbox/.venv/bin/python tests/test_numeric.py                       → 17 PASS / 0 FAIL
（含 t_prefill_metadata_accessor：旧 `or` 模式失败断言 + 新 helper 三形态 PASS）
（L2/L5/CANN：纯 Python 修复，不适用；真机由 delivery 阻塞 probe + serve 复核）
```

## §6 退出条件（逐条 verdict）

| # | 条件 | verdict / 证据 |
|---|---|---|
| 1 | Q1 全证据根因 | ✅ Tensor 布尔化 + `or` 短路；qsl 对照行证明模式混用 |
| 2 | Q2 分类+位置 | ✅ GAP-VERBATIM（t_prefill_ref 手工 list fixture） |
| 3 | 沙盒旧 FAIL 新 PASS + 无拷值声明 | ✅ §3.3/§3.4 |
| 4 | 修复逐行对应 | ✅ §4 |
| 5 | 门禁重放全绿 | ✅ §5 |
| 6 | 记录提交 + 摘要 | ✅ 本记录 + TASK_STATUS 更新 |

**最终 verdict：RESOLVED**

**附：同轮真机观测（重要）**——修复前同一 run 的 SpecDecoding metrics 已显示精度大幅恢复：
单位置接受率 0.765~1.000 / 0.278~0.800 / 0.118~0.700，Avg Draft acceptance rate **42.6%~83.3%**（修复前 9.5%~25%），Mean acceptance length **2.28~3.50**（修复前 1.29~1.75）——**U·H·P + clip 0.96/0.92 + MTP 层排除的精度修复已生效**；本崩溃为独立元数据访问器缺陷，修复后即可完整跑通。
