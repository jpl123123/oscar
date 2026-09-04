# R-20260904-oscar-int2-mtp-precision — 真机反馈反思记录：OSCAR 激活后 MTP 接受率骤降（9.5%~25% vs 正常 ~80%）

> 触发：用户转述真机 serve 日志（2026-09-04 03:17-03:18）：OSCAR 激活后 `SpecDecoding metrics: Avg Draft acceptance rate: 9.5%→25%`（正常应为 ~80%），
> 且首行判定 `🎉 VERDICT: OSCAR ACTIVE`（注入/外科手术/配置生效均真机成立：类外科手术 68 次 = 17 层 × 4 rank；写路径 16 层 + mtp 层首执行）。
> 原始日志指针：用户会话消息（serve.log 尾部摘录，未落盘；真机日志 = `/tmp/oscar_ascend_logs/serve.log`，node93）。
> 真机指纹：HEAD 修复前 ≈ `f5bc2b4`（用户运行） · 设备=node93 docker TP4 · 触发命令=`bash delivery/install_and_launch.sh` + 用户 ais_bench
> 状态：RESOLVED（≤3 条未知：见 §6）

## §1 Q1 — 为什么真机会报这个错

**判定：serve 实例的 INT2 KV 量化精度不在"论文/PR 已验证质量"区间，而是处于"近似噪声"区间——根因是旋转检查点为纯特征向量（无 U·H·P_br 组合、hessian 目标错误）+ 裁剪 clip=0 + per-vector（非 group16）+ MTP 草稿层 KV 也被 INT2 量化；量化噪声经 68 层放大后把草稿/目标一致性击穿，接受率从 ~80% 崩到 9.5%~25%。**

| 机制环节 | 证据（file:line / commit / 契约行） | 置信 |
|---|---|---|
| 旋转组合应为 **R = U·H·P_br**（默认已验证配方 `r_h_pbr`），而旧 gen_rotations 只存纯特征向量 U（降序），无 Hadamard、无位反置换 | `references/oscar-paper/rotation/compute_kv_rotation.py:234-265`（`r_h_pbr` = "default, validated"）；`:23-46`（H/P_br 构造）；README.md:318（P_br "interleaves high-variance directions evenly across quant groups"）；工作区修复前 `tools/gen_rotations.py::_rotation_from_stats`（只 `evecs[:, idx]`） | CONFIRMED |
| 校准 hessian 目标错误：K 旋转应取 **Q** 协方差（qqt），V 应取 **score-weighted**（sst）；旧实现用 K 的 K^TK 与 V 的 V^TV | `compute_kv_rotation.py:93-136`；README.md:312（`Σ_K = (1/H_kv)·Σ_h Q_h^T Q_h/n`）；工作区修复前 `oscar_ascend/calib.py::finalize_cov`（只统计 K/V） | CONFIRMED |
| 裁剪 = 0（per-vector min/max 被尾部离群拉大 → bulk 分量塌缩到 1-2 电平）；PR 评估配方 = K 0.96 / V 0.92 | `references/oscar-vllm-pr46774/tests/quantization/oscar_gpqa_eval.py:71-72`；论文内核测试 clip 0.95/0.90（`sglang-research/test/registered/kernels/test_oscar_rotation_clip_int2.py:86-87`）；工作区修复前 `delivery/serve_oscar.sh:30-31`（0.0） | CONFIRMED |
| 量化粒度：per-vector（1 组/向量，D=256，fp16 meta）vs 论文 **group16 + bf16** | 工作区 `oscar_ascend/config.py:44-67`（group_size=0 → num_groups=1，且内核忽略 group_size）；论文 `kv_cache_quant_group_size=16`（test_oscar_rotation_clip_int2.py:326；数值权威镜像 `sandbox/l3_numeric/group_int2.py:17-20`）；vLLM PR 端口明确单组/向量（`oscar/config.py:154-163` 强校验 group_size≥head_dim + 依赖 clip） | CONFIRMED |
| BF16 sink 窗口在 block_size=128 下失效：sink=64 < 128 → `sink_eff=(64//128)*128=0`，开头 token 无 BF16 保护 | 工作区 `oscar_ascend/backend.py::_ensure_staging`（`sink_eff = (cfg.sink_tokens // bs) * bs`）；vllm-ascend 强制 block_size=128（`references/vllm-ascend/vllm_ascend/utils.py:1381`、`ascend_config.py:696`）；参考实现测试用 bs=16 + sink=16（PR test_oscar.py:372-389） | CONFIRMED |
| MTP 草稿层 KV 也被 INT2 量化（日志 `★ INT2 写路径首次执行: mtp.layers.0.self_attn.attn`）——参考实现从未验证投机解码下的 INT2 草稿 KV；论文 README:292 明确 hybrid（Qwen3.5 GDN）不在支持范围 | 工作区修复前 `plugin.py::_should_oscar`（mtp 层通过）；`references/oscar-paper/README.md:292`；vllm-ascend 草稿层 attention 走 `attn_state=SpecDecoding`（`references/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:1483-1486`） | CONFIRMED |
| 插件对 `attn_state=SpecDecoding` 无专用分支：原生走 `forward_fused_infer_attention`（消费 metadata attn_mask），插件落入 prefill 分支重建因果掩码 | 工作区 `oscar_ascend/backend.py::forward`（仅 DecodeOnly/else 二分）；原生 `attention_v1.py:1456-1477`（DecodeOnly→paged，其余→FIA）、`step3p5.py:173-174`（mask 语义按 metadata） | [UNKNOWN]（机制存在、语义未逐字核验；已通过"MTP 层排除"彻底规避） |

**数值证据（合成数据实验，非真机拷贝）**：`/tmp/oscar_int2_quality/exp1.py`、`exp2.py`（sandbox venv torch 2.13，合成"离群通道 + 幅值漂移"K/V，D=256，N=4096）：

```
旋转(校准式)         量化                 K relL2   attn out relL2   attn KL
纯U(工作区旧配方)     pervec fp16 clip=0   1.5762       2.2180       18.6589   ← ≈噪声
纯U(工作区旧配方)     group16 clip=0       0.3826       1.0261        2.5820
U@H@P(论文/PR配方)   pervec fp16 clip=0   0.3925       0.9433        1.5116
U@H@P(论文/PR配方)   pervec clip=.96/.92  0.3329       0.8405        1.7217
U@H@P(论文/PR配方)   group16 bf16 clip=0  0.0995       0.2702        0.0897   ← 近无损
U@H@P(论文/PR配方)   group16 clip=.95/.90 0.1122       0.3278        0.1738
```

- 结论：**旋转组合是最大杠杆（KL 18.66→1.51，12×；再 group16 → 0.09，再 16×）**；per-vector+clip 是 PR 端口级的最小修复（KL 1.72），group16 是论文级最优（kernel 槽位改造，本轮未实施——见 §6 未知）。
- 尚缺的证据（真机侧复核项）：① 用户运行日志中 `★ OSCAR 配置生效` 行的 `K旋转=已加载/单位阵`、`窗口(sink=.., recent=..)` 与 [SKIP] 行（判断旧 pt 是否真被加载、窗口是否真启用）；② SpecDecoding 元数据逐字段（判插件 prefill 分支是否产错）；③ MTP 草稿层排除后的接受率对照。
- 排除的候选与排除证据：
  - *字节级存储/解包错误*：排除。workspace `format.py` 与 sandbox 权威 `l3_numeric/format.py` 数值语义一致（N-01..N-04），CPU 数值镜像 12/12 PASS（store 字节差=0、dequant≤1e-5、decode≤1e-4）。
  - *旋转正交性/方向错误*：排除。U·H·P 正交误差实测 ≤7e-7（fp32）；Q/K 同 R_k、V 用 R_v 的旋转不变式有 `t_rotation_invariance ≤1e-3` 覆盖。
  - *decode 未逆旋转/窗口拼接错位*：排除。`test_decode_window_matches_reference`（PR）与工作区 `_stage_splice` 逻辑一致（逐行对照）；窗口数学由 PR 测试 rel<2e-2 覆盖。
  - *clip 路径的 torch.quantile NPU 不支持*：部分排除。旧实现失败会静默跳过（质量与 clip=0 相同）→ 新实现改 sort-based（论文内核同款），不再依赖 quantile。

## §2 Q2 — 为什么递交前沙盒没有检测到

- 缺口分类：**GAP-SIGNATURE**（知识库/沙盒缺失"量化质量地板"与"旋转配方""MTP/SpecDecoding 契约"三条契约行；现有检查全为**格式正确性**，无一条是**质量**）。
- 沙盒现状（检查在哪里、检查了什么）：
  - `sandbox/l3_numeric/format.py::check_case`：量化-存储-解包-解码的**往返一致性**（k_error==0.0 / rotation_error≤1e-4）——只证明"自己与自己一致"，不证明"与 fp32 精度足够近"；
  - `sandbox/l3_numeric/format.py::rand_ortho` + `check_case(clip=0.0)`：测试用**随机正交**旋转（而非校准配方），Gaussian 数据无离群通道 → 恰好避开"纯 U 在离群数据上坍缩"的窗口；
  - `sandbox/l3_numeric/group_int2.py`：作者已按论文 group16 建模为**独立镜像**，但 `knowledge/numerical-contract.md N-01..N-09` 只钉了 per-row 160B 契约 → 工作区 per-vector 实现与论文 group16 的差异无契约行（OQ）；
  - `knowledge/` 无 MTP/投机解码/SpecDecoding 条目（grep acceptance/MTP/mtp → 0 命中）。
- 缺口证据：per-vector INT2 + 纯 U 的组合在检查语义上"完全合法"（字节差=0 成立），但质量地板（合成离群数据下 relL2 ≤ 0.6 / KL ≤ 某阈值）不存在 → 任何"字节正确但噪声级"的实现都能全绿；参考树的"配方"（qqt/sst + r_h_pbr + clip + group16）从未进入沙盒契约表。

## §3 沙盒复现（机理、可追溯、禁止拷贝真机数值）

### §3.1 机制推导链
1. 论文语义（`compute_kv_rotation.py:234-265`）：旋转 = U·H·P_br；U 来自 attention-aware hessian 特征向量，**H（Hadamard）把高方差方向能量摊开**，P_br 交错高方差方向 → 旋转后各分量幅值均衡 → per-vector min/max 紧致。
2. 旧实现只取 U → 旋转后分量幅值高度不均（前几个特征方向主导）→ per-vector min/max 被少数分量主导 → **其余分量只占 1-2 个电平**（量化语义推导，非真机数值）。
3. 沙盒建模：合成"离群通道"K/V（4 通道 ×24、8 ×8、12 ×2.5 + lognormal 幅值漂移——LLM KV 文献典型 massive/outlier 通道结构；与真机数值无关），按工作区 `format.quantize/dequant` 的同一公式量化。

### §3.2 新增检查与签名
- 检查项名：`int2_quality_floor_uhp_vs_plain_u`；判定词：FAIL；
- 签名结构（类别级）：per-vector INT2 反量化 rel-L2 阈值（旧配方 ≥1.0 判 FAIL 期望；新配方 ≤0.60 判 PASS 期望）——类别 = "重建信噪比低于可用区间" ↔ 真机观测类别 = "MTP 接受率从 80% 量级跌至 9.5%~25%（输出精度崩坏）"；
- 双向证据（同一夹具、同一判据函数）：旧 `r_old=1.568`（FAIL 地板）、新 `r_new=0.377`（PASS 地板）。

### §3.3 复现代码位置
- `tests/test_numeric.py::t_rotation_composition_quality_floor`（旧 FAIL / 新 PASS 双向）；`t_clip_sort_threshold`；`t_should_oscar_rejects_mtp`；`t_calib_cov_q_sst`；
- 量化矩阵实验：`/tmp/oscar_int2_quality/exp1.py`、`exp2.py`（输出见 §1），沙盒 venv 下运行；
- 复现运行：`sandbox/.venv/bin/python tests/test_numeric.py` → 16 PASS / 0 FAIL（修复后）；修复前版本（纯 U + 无质量地板测试）数字见 §1（1.5762 列）。

### §3.4 无拷值声明
签名/期望/阈值全部来源于：**合成分布参数（chan=[24,8,2.5]、amp=exp(0.4·N)）+ 量化公式（scale=(max-min)/3、fp16 舍入、floor(+0.5)）+ 判据公式（rel-L2 与 KL）**，不含任何真机日志中的字面量（9.5%/25%/80% 仅用于"类别一致性"人工比对，未进入任何代码/期望值）。

## §4 修复（与 Q1 逐行对应）

| 文件 | 改动 | 对应 Q1 机制 |
|---|---|---|
| `oscar_ascend/calib.py` | 捕获 Q/K/V（钩子占 `args[0]`）；`finalize_cov` 计算 Σ_Q（qqt，GQA 分组）与 Σ_S（sst，score-weighted，w_i=k_i^T Σ_Qh k_i） | 校准 hessian 目标错误（K←Q^TQ、V←sst） |
| `tools/gen_rotations.py` | 新增 `build_hadamard/bit_reversal_perm/make_br_perm_matrix/compose_rotation`（论文移植）；`_rotation_from_stats` 输出 **R = U·H·P_br**（fp64 计算，正交自检）；`format_version=2` + objective 标记 | 旋转组合缺失（纯 U） |
| `oscar_ascend/plugin.py` | `_should_oscar` 最先拒绝 `.mtp.` 层（纯函数、无 vllm 依赖），草稿 KV 回退 BF16 原生 SpecDecoding 路径 | MTP 草稿层被 INT2 量化 + SpecDecoding 无专用分支（规避） |
| `oscar_ascend/backend.py::_rotate_clip` | `torch.quantile` → **sort-based 阈值**（论文内核同款 `sorted_abs[..., int(ratio*D)]`；不再依赖 NPU 未验证的 quantile） | clip=0 实际生效条件（NPU 上可执行） |
| `delivery/serve_oscar.sh` | 默认 `K_CLIP_RATIO=0.96`、`V_CLIP_RATIO=0.92`（PR 评估配方）；`SINK_TOKENS=128`（≥ block 128，sink 页不再静默失效） | clip=0（默认）+ sink=64 < bs=128 |
| `delivery/install_and_launch.sh` | 阶段 4 增加 v2 配方检查（`format_version>=2 且 objective 含 r_h_pbr`）：旧 pt 自动触发重校准（否则用户现状的旧 pt 会继续复用 → 修复无效） | 旧旋转检查点复用 |
| `oscar_ascend/rotation.py` | 加载非 v2 检查点打印醒目告警（防御：手动 serve 复用旧 pt 场景） | 同上 |
| `tests/test_numeric.py` | +4 项：质量地板（旧 FAIL/新 PASS）、裁剪语义、MTP 拒绝、Σ_Q/Σ_S | Q3 防重漏 |

未实施（本轮）——**group16 量化**：槽位布局（group16 → K/V meta 各 2×16=64B，slot 256B vs 160B）+ store/decode/dequant 三条路径改造，属 kernel 级手术；本轮回滚风险最小化优先。给出明确结论：论文级精度（KL 0.09）依赖 group16，per-vector+U·H·P+clip 为 PR 端口级基线（KL 1.7）。

## §5 门禁重放结果

```text
python3 -m py_compile tools/gen_rotations.py oscar_ascend/{calib,plugin,backend,rotation}.py tests/test_numeric.py  → OK
bash -n delivery/{install_and_launch,serve_oscar,check_oscar_active}.sh                                  → OK
sandbox/.venv/bin/python tests/test_numeric.py                                                           → 16 PASS / 0 FAIL
   [quality] 纯U relL2=1.568  U@H@P relL2=0.377  （旧→FAIL 期望 / 新→PASS 期望）
E2E pipeline（fake per-rank stats → merge → R=U·H·P -> pt v2 -> loader 正交 err≈5e-7 → 缺层单位阵回退 → v2 检查器） → OK
skill sandbox L3 判据：workspace format.py 与 sandbox/l3_numeric/format.py 数值语义一致（diff 仅命名/结构，公式一致）
（L2/L5、CANN 编译：本轮纯 Python/脚本改动，无内核文件 → 不适用；真机 probe 由 delivery/install_and_launch.sh 阻塞执行）
```

## §6 退出条件（逐条 verdict）

| # | 条件 | verdict / 证据 |
|---|---|---|
| 1 | Q1 全证据根因 | ✅ 7 条 CONFIRMED + 1 条 [UNKNOWN]（SpecDecoding 逐字段语义；已被 MTP 排除规避，待真机复核） |
| 2 | Q2 分类+位置 | ✅ GAP-SIGNATURE（含 4 个具体检查点 file:line + 为何抓不到） |
| 3 | 沙盒旧 FAIL 新 PASS + 无拷值声明 | ✅ §3.3/§3.4（1.568→0.377 双向；合成数据） |
| 4 | 修复逐行对应 | ✅ §4 表 |
| 5 | 门禁重放全绿 | ✅ §5（纯 Python 改动；L5/CANN 不适用） |
| 6 | 记录提交 + docs/04 摘要 | ✅ 本记录；工作区无 docs/04 → 摘要同步 TASK_STATUS.md（唯一状态权威） |

**最终 verdict：RESOLVED** —— 根因=旋转检查点非已验证配方（U·H·P_br 缺失 + qqt/sst 目标错误）+ clip=0 + per-vector + MTP 草稿层 INT2（+sink<bs 静默失效）；修复=校准 v2 配方、sort 裁剪默认 0.96/0.92、sink=128、MTP 层排除；本地 16/16 含质量地板双向证据；交由真机 `git pull && bash delivery/install_and_launch.sh` 复核（预期自动强制重校准 + 写/读路径 16 层 ×4 rank 且**不含 mtp 层**）。
