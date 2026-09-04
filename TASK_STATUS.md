# TASK_STATUS — OSCAR INT2 → vllm-ascend 0.23.0 当前状态（窗口压缩交接）

> 2026-09-04 11:30 快照。**工作区状态唯一权威文件**；详细机制证据在
> `plan/reflections/R-20260903-docker-device-type.md`（§1-§24 逐轮）、
> `plan/PLAN-1-oscar-int2-vllm-ascend.md`（方案）、`README.md`（用法）。
> 恢复点：git tag **`v0.1.0-stable-20260904`** = `b37a8ae`（校准/数值/性能已验的稳定基线）。
> 最新 main：`f5bc2b4`（零自动请求版一键流程）。

## 目标（一句话）
让 OSCAR INT2 KV 量化在真机 docker 的 vllm-ascend serve（Qwen3.5-27B-w8a8-mtp，TP4，
8989）**真正激活且可验证**，用户只跑一条 `git pull && bash delivery/install_and_launch.sh`，
随后自己用 ais_bench 跑精度/性能。

## 当前真实状态
- ✅ 平台/装载/校准全链路真机打通：probe ref+triton **0/0/0**；
  `oscar_rotations.pt`（16 FULL 层，D=256）已生成并被复用。
- ✅ 类外科手术**已真机生效**（栈进入过 impl.forward→do_kv_cache_update→rotation）。
- 🚧 **最近一轮实弹**：serve 运行时报窗口 staging `ERR01002 设备混用`（已硬化+单条告警，
  自动降级纯 INT2）；**是否已达到"一键 ACTIVE + ais_bench"闭环仍待下一次运行确认**
  （期间共修 6 个问题，见下）。
- 📌 数值/性能对比（LongBenchv2 55.9 = 无 OSCAR 基线）尚未做——等用户 ais_bench。

## 已修复问题链（按提交序，全部在 gitcode main）
1. 平台插件未激活真根因：`VLLM_PLUGINS` 白名单漏 `ascend`（ddf8cb7）
2. fork OpenMP 线程池崩溃 → `VLLM_WORKER_MULTIPROC_METHOD=spawn`（74b529f）
3. spawn worker 平台未迁移 → `_bootstrap_platform` 进程级引导（041bc30）
4. 校准管线 6 阶梯：apply_model / 序列化回退 / generate 路径 / 钩子签名 / GDN bf16 dtype /
   层号解析（770cc9c→e652232）
5. 数值：`aclnnRightShift` 广播缺陷→算术链；`floor(x+0.5)` 契约（N-02）；triton
   scale 预计算+精确转换；`vector_scales` .float() 守卫（57faf35→b37a8ae）
6. serve 激活 3 连坑：`is_hybrid` 兜底（ccf123b）→ `str(Enum)` 值比较（1527998）→
   pt 键 `int(k)`（33c6469）
7. 用户交互迭代：`${OSCAR_EXTRA_ARGS:-}`；serve.log 固定名覆盖；**前台+tee（去 nohup）**；
   **零自动 HTTP/curl（日志就绪判定）**；staging 设备硬化+单条告警（72a2bb1→f5bc2b4）

## 关键约定（用户钉死，后续 Agent 必须遵守）
- **不自动发任何请求**（/health、curl、样例请求一律禁止；仅 grep 日志判定就绪）——
  请求全部由用户 ais_bench 发起。
- **serve 前台实时输出**（不要 nohup/setsid），同时 `tee /tmp/oscar_ascend_logs/serve.log`
  （固定名、每次启动原地覆盖）。
- **一键自动化、无手工步骤**：预检→pt→probe→serve→自动激活判定（后台观察者打印）。
- 用户会自己跑 `git pull && bash delivery/install_and_launch.sh`；回传日志即反馈回路。

## 下一步（按序）
1. 用户复跑一键 → `🎉 VERDICT: OSCAR ACTIVE`（观测：★外科手术×16、★配置×16、
   `enforce_eager=False=0`）。若仍 [SKIP] 或新异常 → 取 `serve.log` 栈顶 10 行。
2. 用户 ais_bench → 复核 `grep "★" /tmp/oscar_ascend_logs/serve.log | wc -l`（写/读各 16）。
3. 精度/性能对比（对基线 55.9）；预期 KV **读写带宽** -84%（引擎池占用率%不变属候选 A 取舍；
   引擎可见显存压缩 = 候选 B，另开分支基于 tag 实验）。
4. 例行：`cd /Users/sunao2000/new_oscar_vllmascend && python3 tests/test_numeric.py`（12 项）。

## 关键文件
- 一键主流程 `delivery/install_and_launch.sh`；判定 `delivery/check_oscar_active.sh`
  （`--preflight` / 默认 serve.log）；启动器 `delivery/serve_oscar.sh`（字面 --enforce-eager）。
- 插件：`oscar_ascend/{plugin,backend,format,rotation,config,calib}.py`；算子 `kernels/`；
  校准 `tools/gen_rotations.py`；诊断 `tools/diag_platform.py`。
- 测试 `tests/test_numeric.py`（12/12）；方案 `plan/PLAN-1-...`；真机反思 `plan/reflections/R-...`。

## Suggested skills（新 Agent 开始前必载）
- `quantc-vllm-ascend-integration` — 本任务总纲（铁律/参考树/沙盒/门禁/交付契约）
- `vllm-ascend-debugging` — 引擎/接缝/数值类运行时问题定位方法论
- `ascendc-loop-debug` — 若涉及 Ascend-C / CANN 本地编译回路（当前 OSCAR 走 torch/Triton，
  未用到 Ascend-C；仅在转"候选 B"或 Triton 内核深入时按需引用）
