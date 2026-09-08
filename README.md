# oscar-ascend — OSCAR INT2 KV cache for Ascend

目标运行环境：**vllm 0.23.0 + vllm-ascend 0.23.1.dev0+g5cb98caaa.d20260822**（用户指定镜像版本）。

面向 Qwen3.5-27B W8A8 + MTP、TP4、eager 执行的插件。通过
`vllm.general_plugins` 替换 FULL attention impl，并包装 worker 的显存预算与缓存初始化接口；不修改 reference 或安装目录内的 vllm/vllm-ascend 源码。GDN 继续使用原生实现。

**当前版本 0.2.0：用户已通过基础 NPU 数值门禁并成功启动，但服务日志显示严重性能退化。本轮新增原生融合 prefill、staging 排序优化及 paged 分块调整；CPU 回归通过，改动仍待 NPU 数值与性能复验。不能将 CPU PASS 或 ACTIVE 判定视为吞吐验收。**

## 本轮变化

- 修复窗口每步清空、负 slot 写坏尾部缓存、CPU seq metadata 误用、decode 异常回退、类替换失败未恢复以及多 KV head 的几何检查。
- sink/recent 保留未量化数据，写入时旋转到 FP32 空间；避免历史 KV 反复逆旋转。窗口现在跨步保留，哈希碰撞选择同一个 owner/value 写入者，非窗口重写会使旧 tag 失效。
- 新增 MTP 多 query 分页 Triton attention：直接从 INT2 历史缓存读取，当前 chunk 使用未量化 K/V，窗口按 owner tag 覆盖。每请求 q_len≤16 时可走此路径；混合批次中的长 prefill 单独走 dense 路径。
- dense continuation 在旋转域计算，只对新 Q/K/V 和最终输出旋转；NPU 使用原生融合注意力，仍会物化历史 KV。CPU 保留 SDPA oracle。
- staging 在槽编号可由 FP32 精确表示时使用浮点稳定排序，避免整数 ArgSort 的 AiCPU 回退；启动门禁增加16K前缀与真实1536-token页。paged分块32在目标910B4编译时出现UB溢出，默认已恢复为通过过真机门禁的4。
- MTP BF16 影子池、FP32 窗口、旋转矩阵纳入常驻内存预算，并在缓存初始化时分配、检查。临时算子的峰值内存仍需 NPU 压测。
- 启动 READY manifest 按 TP rank 检查 FULL 层覆盖和源码指纹。安装器默认重装当前 checkout 的 editable 包，避免复用旧 wheel。
- probe 显式拒绝 NaN/Inf。新内核有独立跨页、多请求和空段 NPU 门禁。

FP32 staging 的容量约为旧 BF16 staging 的两倍（D=256/Hk=1/8192 tokens 时，每层 K/V 约16 MiB，加 owner）；已计入预算。packed×2 是 FULL 缓存的槽密度变化，不能解释为整机容量或吞吐翻倍。

## 本地验证

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch pytest numpy
.venv/bin/python tests/test_numeric.py
.venv/bin/python -m pytest tests/test_backend.py -q
.venv/bin/python delivery/probe_paged.py --device cpu
```

本轮执行：backend/prefill/版本/类型/分块配置回归51项通过，新增prefill因果与16K分页CPU镜像通过。CPU 使用 PyTorch 2.14.0，不代表部署环境 torch-npu 的结果。

历史问题审查见 `plan/audits/REVIEW-20260908.md`；历史复现脚本固定读取审查提交 `e451ca6`，不用于验证当前代码。实现与验收记录见 `plan/IMPLEMENTATION-20260908.md`。

## NPU 验证与启动

容器需预装 vllm 0.23.0、vllm-ascend **0.23.1.dev0+g5cb98caaa.d20260822**、torch-npu、triton-ascend，并挂载模型和本仓库。该 Ascend 构建的 commit 标识对应本地 reference 的 v0.23.0 tag（5cb98caaa），不能仅凭包版本中的0.23.1拒绝它。提供的 reference HEAD 还附加了PR#12607 GDN补丁，仍需区分部署构建与参考树。安装器使用 --no-deps，保留容器预装框架。

```bash
# 在仓库根目录安装本次代码；不更新容器的框架依赖
python3 -m pip install --no-deps --no-build-isolation -e .

# 新内核独立门禁（不会向外部服务发请求）
python3 delivery/probe_paged.py --device npu --triton
python3 delivery/probe_prefill.py --device npu

# 一键：环境/版本检查、安装、校准、数值门禁，再前台启动服务
bash delivery/install_and_launch.sh
```

一键脚本在旧内核门禁及 `probe_paged.py` 通过后设置 `OSCAR_ASCEND_USE_PAGED=1`。
默认硬门禁下，paged probe失败会阻断启动；可显式 `OSCAR_ASCEND_USE_PAGED=0` 验证dense路径。
直接运行 `serve_oscar.sh` 默认不启用新分页内核，且要求 K/V 旋转文件存在。

服务默认：模型 `/softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp`、TP4、NPU 0–3、端口8989、MTP草稿3 tokens、eager。可用 `MODEL_PATH`、`ASCEND_RT_VISIBLE_DEVICES` 和 `OSCAR_EXTRA_ARGS` 调整。

```bash
# 核验每rank的缓存初始化、层集合和实际源码
bash delivery/check_oscar_active.sh /tmp/oscar_ascend_logs/serve.log

# attention单层微基准；先通过paged probe。不是端到端吞吐测试。
python3 tools/benchmark_attention.py --device npu --triton --lengths 1024 8192 32768

# 15360-token长prefill：旧SDPA与原生融合注意力（先停服务，避免争抢NPU）
python3 tools/benchmark_attention.py --device npu --mode prefill --prefill-tokens 15360 --iterations 3
```

真实验收还需原生 BF16 / legacy OSCAR / packed OSCAR 在相同模型、上下文和并发下的任务精度、MTP接受率、TTFT、TPOT、吞吐和峰值内存对比。

服务仍慢时，先停旧服务，再采集一次有界诊断：

```bash
OSCAR_ASCEND_PROFILE_STEPS=6 bash delivery/install_and_launch.sh
# 使用原有客户端发送同一批请求后，从另一终端提取诊断行
grep '\[oscar-ascend\] PERF' /tmp/oscar_ascend_logs/serve.log
```

只计时rank 0前6个非空调度步骤及其采样，输出真实forward的加载路径/代码指纹、形状/开关、各OSCAR阶段耗时、整数sort调用栈。`inclusive_ms`包含嵌套阶段，不能相加；`wait_before_ms`记录进入该阶段前等待已提交设备任务的时间。同步和Python算子追踪会改变这几个步骤的吞吐，首轮也可能包含JIT，不能把诊断吞吐当作正式benchmark。默认关闭；达到步数后自动停止采集，不修改计算结果。

## 关键配置

| 环境变量 | 默认/含义 |
|---|---|
| `VLLM_PLUGINS` | 脚本默认 `ascend,oscar_ascend`，两个插件都必须存在 |
| `OSCAR_ASCEND_ENABLE` | `auto`；`0` 禁用接入 |
| `OSCAR_ASCEND_PACKED` | serve默认1；0使用BF16物理槽几何 |
| `OSCAR_ASCEND_USE_TRITON` | serve默认1；0使用torch参考内核 |
| `OSCAR_ASCEND_USE_PAGED` | 插件默认0；一键paged门禁通过后设为1；每请求q_len≤16使用新内核 |
| `OSCAR_ASCEND_PAGED_BLOCK_KV` | 默认4；16/32/64/128仅供显式实验，32已在目标910B4出现UB溢出；调整后必须重新运行paged门禁 |
| `OSCAR_ASCEND_PROFILE_STEPS` | 默认0关闭；正数表示rank 0需要采集的真实调度步数，包含execute_model和sample_tokens |
| `OSCAR_ASCEND_REQUIRE_TRITON` | 一键默认1，门禁失败阻断；0为诊断降级模式 |
| `OSCAR_ASCEND_K/V_ROTATION_PATH` | serve默认仓库内 `oscar_rotations.pt`；启动缓存初始化时检查目标层覆盖和正交性 |
| `OSCAR_ASCEND_K/V_CLIP_RATIO` | serve默认0.96/0.92；插件直接加载时默认0，范围[0,1] |
| `OSCAR_ASCEND_SINK_TOKENS` | serve默认128；按实际kernel block size向下对齐 |
| `OSCAR_ASCEND_RECENT_TOKENS` | 默认256 |
| `OSCAR_ASCEND_STAGING_TOKENS` | 默认8192；0禁用窗口 |
| `OSCAR_ASCEND_BATCHED_TOKENS` | packed默认15360，legacy默认16384 |

内存接缝当前要求 PP=1，并拒绝绕过预算的 `num_gpu_blocks_override`。当前路径依赖 eager，尚未支持图捕获、KV传输/换出对插件额外arena的同步；这些不是已验证能力。
