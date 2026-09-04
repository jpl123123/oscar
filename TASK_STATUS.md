# TASK_STATUS — OSCAR INT2 → vllm-ascend 0.23.0 当前状态（窗口压缩交接）

> 2026-09-04 16:00 快照。**工作区状态唯一权威文件**；详细机制证据在
> `plan/reflections/R-20260904-oscar-int2-mtp-precision.md`（本轮：OSCAR 激活后 MTP 接受率
> 9.5%~25% vs 正常 ~80% 的**系统性精度排查**，RESOLVED）、
> `plan/reflections/R-20260903-docker-device-type.md`（§1-§24 逐轮）、
> `plan/PLAN-1-oscar-int2-vllm-ascend.md`（方案）、`README.md`（用法）。
> 恢复点：git tag **`v0.1.0-stable-20260904`** = `b37a8ae`（校准/数值/性能已验的稳定基线）。
> 最新 main：本轮回调（v2 旋转配方 + clip/sink 默认 + MTP 草稿层排除 + 精度地板测试）。

## 目标（一句话）
让 OSCAR INT2 KV 量化在真机 docker 的 vllm-ascend serve（Qwen3.5-27B-w8a8-mtp，TP4，
8989）上**激活且精度达标**（MTP 接受率回到 ~80% 量级），用户只跑一条
`git pull && bash delivery/install_and_launch.sh`，随后自己用 ais_bench 跑精度/性能。

## 当前真实状态
- ✅ 平台/装载/校准全链路真机打通；一键 ACTIVE 判定闭环（上一轮已真机 VERDICT: ACTIVE）。
- 🚧 **本轮（2026-09-04）**：真机 ACTIVE 后 MTP 接受率 9.5%~25%（正常 ~80%），判定为
  **量化精度坍塌**（非字节/存储 bug）。根因 6 项已定位并修复（见下）；本地 16/16 PASS
  含**精度地板双向证据**（纯 U 旋转 relL2=1.568 ≈ 噪声 → U@H@P 0.377）。
- ⏳ **待用户复跑一键**：install_and_launch 检测到旧配方 pt（format_version<2）会自动
  **强制重校准**（U·H·P_br + qqt/sst），随后 serve 写/读路径应 **16 层 ×4 rank 且不含
  mtp 层**（MTP 草稿层保持 BF16）；用户 ais_bench 复核接受率。

## 本轮已定位并修复的精度链（6 项，详见 R-20260904 记录）
1. **旋转检查点非已验证配方**（最大杠杆）：旧 gen_rotations 只存特征向量 U（降序）；
   论文/PR 默认验证 = **R = U @ H @ P_br**（Hadamard + 位反置换，`compute_kv_rotation.py:234-265`）
   → 修复：`tools/gen_rotations.py` 移植组合，`format_version=2`。
2. **校准 hessian 目标错误**：K 旋转应取 **Q 协方差（qqt）**、V 应取 **score-weighted（sst）**；
   旧实现用 K^TK / V^TV（`compute_kv_rotation.py:93-136`、README:312）。
   → 修复：`calib.py` 捕获 Q/K/V + worker 内计算 Σ_Q/Σ_S。
3. **clip=0**（per-vector min/max 被离群拉大 → bulk 塌缩）：PR 评估配方 0.96/0.92
   （oscar_gpqa_eval.py:71-72）。→ 修复：serve 默认 0.96/0.92 + 裁剪实现改 **sort-based**
   （论文内核同款，替代 NPU 未验证的 torch.quantile）。
4. **per-vector vs 论文 group16**：论文级精度（KL 0.09）依赖 group16+bf16（slot 160B→256B
   布局改造，kernel 级手术）——**本轮未实施**，作为下一步精度杠杆（记录 §4/§6）。
5. **sink=64 < block_size=128 → sink_eff=0**（BF16 sink 窗口静默失效）→ 默认 128。
6. **MTP 草稿层 KV 被 INT2 量化**（日志 `★ ... mtp.layers.0.self_attn.attn`）：参考从未验证
   投机解码草稿 INT2；vllm-ascend 草稿层 attn_state=SpecDecoding，插件无专用分支
   （原生走 FIA + metadata attn_mask）→ `_should_oscar` **拒绝 mtp 层**（BF16 原生路径）。

## 已修复问题链（按提交序，全部在 gitcode main）
（前序 7 项见上一版快照：ddf8cb7→74b529f→041bc30→770cc9c→…→33c6469→…→f5bc2b4；
本轮回调=上述 6 项，交付前 commit+push）

## 关键约定（用户钉死，后续 Agent 必须遵守）
- **不自动发任何请求**（/health、curl、样例请求一律禁止；仅 grep 日志判定就绪）——
  请求全部由用户 ais_bench 发起。
- **serve 前台实时输出**（不要 nohup/setsid），同时 `tee /tmp/oscar_ascend_logs/serve.log`
  （固定名、每次启动原地覆盖）。
- **一键自动化、无手工步骤**：预检→pt（v2 配方自检）→probe→serve→自动激活判定。
- 用户会自己跑 `git pull && bash delivery/install_and_launch.sh`；回传日志即反馈回路。

## 下一步（按序）
1. 提交本轮回调并 push（gitcode）；用户 `git pull && bash delivery/install_and_launch.sh`
   ——预期：自动"旧配方 pt → 强制重校准"（genrot 日志出现 `R_k/R_v ok (U@H@Pbr)`）。
2. 用户 ais_bench → 复核：接受率应回 ~80% 量级；`grep "★" serve.log | cut -d: -f2 | sort -u`
   应为 16 层 ×4 rank（**无 mtp 层**）；`grep "[SKIP]" serve.log` 应见 `mtp-draft` 4 条。
3. 若仍低于预期：按记录 §6 未知项取证（配置生效行的 K旋转=已加载/窗口值）→ 下一步
   实施 **group16**（slot 256B 布局 + store/decode/dequant 三路径）。
4. 例行：`sandbox/.venv/bin/python tests/test_numeric.py`（16 项；本机无 torch 时用
   技能沙盒 venv）。

## 关键文件
- 一键主流程 `delivery/install_and_launch.sh`（阶段4 = v2 配方自检+强制重校准）；判定
  `delivery/check_oscar_active.sh`；启动器 `delivery/serve_oscar.sh`（clip 0.96/0.92、
  sink 128、字面 --enforce-eager）。
- 插件：`oscar_ascend/{plugin,backend,format,rotation,config,calib}.py`；算子 `kernels/`；
  校准 `tools/gen_rotations.py`（U·H·P_br）；诊断 `tools/diag_platform.py`。
- 测试 `tests/test_numeric.py`（16/16，含精度地板/裁剪/MTP拒绝/Σ_QΣ_S）；方案
  `plan/PLAN-1-...`；真机反思 `plan/reflections/R-20260904-oscar-int2-mtp-precision.md`。

## Suggested skills（新 Agent 开始前必载）
- `quantc-vllm-ascend-integration` — 本任务总纲（铁律/参考树/沙盒/门禁/交付契约）
- `vllm-ascend-debugging` — 引擎/接缝/数值类运行时问题定位方法论
- `ascendc-loop-debug` — 若涉及 Ascend-C / CANN 本地编译回路（当前 OSCAR 走 torch/Triton，
  未用到 Ascend-C；仅在转"候选 B"或 Triton 内核深入时按需引用）
