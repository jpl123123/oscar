# oscar-ascend — OSCAR INT2 KV cache for Ascend

目标运行环境：**vllm 0.23.0 + vllm-ascend 0.23.1.dev0+g5cb98caaa.d20260822**（用户指定镜像版本）。

面向 Qwen3.5-27B W8A8 + MTP、TP4、eager 执行的插件。通过
`vllm.general_plugins` 替换 FULL attention impl，并包装 worker 的显存预算与缓存初始化接口；不修改 reference 或安装目录内的 vllm/vllm-ascend 源码。GDN 继续使用原生实现。

**当前代码默认使用 streaming 分块矩阵 attention：有历史的MTP、decode和continuation直接按小块读取INT2页，不再生成完整历史K/V临时张量。无历史prefill保留原生快路径。新内核已通过CPU数学/索引/生命周期回归，尚待目标NPU编译、数值和性能门禁验收。**

历史实测：原生融合读取曾使重复MTP步骤由约4.72秒降至0.556秒，但仍反复物化完整历史；后续KV准备融合未证明提速。该路径现在作为显式`native`对照保留。streaming为新的矩阵实现，不能沿用之前8.5倍的数字宣称自身收益。实现细节见 [streaming改造与验收](plan/IMPLEMENTATION-STREAMING-20260908.md)。

## 本轮变化

- 修复窗口每步清空、负 slot 写坏尾部缓存、CPU seq metadata 误用、decode 异常回退、类替换失败未恢复以及多 KV head 的几何检查。
- sink/recent 保留未量化数据，写入时旋转到 FP32 空间；避免历史 KV 反复逆旋转。窗口现在跨步保留，哈希碰撞选择同一个 owner/value 写入者，非窗口重写会使旧 tag 失效。
- 默认按32个KV token、64个通道的小块解码，在一组query/head之间共享并用`tl.dot`计算，在线softmax归并；不生成随历史长度增长的全局dense KV。
- cache-only decode、混合请求和有历史的长continuation都走新路径；混批中的无历史prefill单独使用原生attention，保留当前chunk的未量化语义。
- standalone KV update会立即使被重写位置的staging标签失效，防止cache-only读取旧窗口值。MTP拒绝后的尾部重写、block回收和padding有回归覆盖。
- staging 在槽编号可由 FP32 精确表示时使用浮点稳定排序，避免整数 ArgSort 的 AiCPU 回退；启动门禁增加16K前缀与真实1536-token页。paged分块32在目标910B4编译时出现UB溢出，默认已恢复为通过过真机门禁的4。
- MTP BF16影子池、FP32窗口、旋转矩阵继续计入常驻预算；新split暂存和本步query变换的保守临时预留按worker计入预算，不按层复制大工作区。
- 启动 READY manifest 按 TP rank 检查 FULL 层覆盖和源码指纹。安装器默认重装当前 checkout 的 editable 包，避免复用旧 wheel。
- probe 显式拒绝 NaN/Inf。新内核有独立跨页、多请求和空段 NPU 门禁。

FP32 staging 的容量约为旧 BF16 staging 的两倍（D=256/Hk=1/8192 tokens 时，每层 K/V 约16 MiB，加 owner）；已计入预算。packed×2 是 FULL 缓存的槽密度变化，不能解释为整机容量或吞吐翻倍。

## 本地验证

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch pytest numpy
.venv/bin/python tests/test_numeric.py
.venv/bin/python -m pytest -q
.venv/bin/python delivery/probe_streaming.py --device cpu
```

本轮验证包含实际JIT源码的CPU数学与越界检查模拟、在线softmax独立对照，以及禁止调用全历史反量化/拼接的backend测试。CPU使用PyTorch 2.14.0，不验证Ascend编译器的UB规划或设备性能；pytest默认只收集本仓库tests，不运行reference的手动测试。

历史问题审查见 `plan/audits/REVIEW-20260908.md`；历史复现脚本固定读取审查提交 `e451ca6`，不用于验证当前代码。实现与验收记录见 `plan/IMPLEMENTATION-20260908.md`。

完整评测约88分钟对比base约10分钟的ac94f31代码快照分析、改进顺序与有条件的加速预期，见 [端到端慢速分析与改进计划](plan/audits/END_TO_END_SLOWNESS_AND_PLAN-20260908.md)。该历史报告不代表新streaming实现已经达到其中的目标。

## NPU 验证与启动

容器需预装 vllm 0.23.0、vllm-ascend **0.23.1.dev0+g5cb98caaa.d20260822**、torch-npu、triton-ascend，并挂载模型和本仓库。该 Ascend 构建的 commit 标识对应本地 reference 的 v0.23.0 tag（5cb98caaa），不能仅凭包版本中的0.23.1拒绝它。提供的 reference HEAD 还附加了PR#12607 GDN补丁，仍需区分部署构建与参考树。安装器使用 --no-deps，保留容器预装框架。

```bash
# 在仓库根目录安装本次代码；不更新容器的框架依赖
python3 -m pip install --no-deps --no-build-isolation -e .

# 新内核独立门禁（不会向外部服务发请求）
python3 delivery/probe_streaming.py --device npu
./bench

# 一键：环境/版本检查、安装、校准、数值门禁，再前台启动服务
bash delivery/install_and_launch.sh
```

一键默认`OSCAR_ASCEND_ATTENTION_MODE=streaming`。先跑数值和生命周期门禁，再在同一输入上比较单请求24K MTP、8请求24K MTP及15360+9256 continuation。任何形状的预热中位耗时若比native对照高超过10%，会停止启动并留下`streaming_bench_*.log`；不会静默回退完整历史物化。此门槛只是防止明显回退，不能替代完整LongBench/吞吐验收。
显式`OSCAR_ASCEND_ATTENTION_MODE=native`保留上一版对照。直接运行`serve_oscar.sh`也默认streaming，但不执行前置门禁，正式复测应使用一键入口。

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

停服务后运行`./bench`，现在比较native完整历史对照和新的streaming路径，包括1/8请求MTP及continuation。输出`STREAM BENCH`中的首次调用、6次交错采样、中位数和`baseline_over_streaming`；>1表示新路径更快。暂存字节是计算量，不是设备峰值。旧KV准备融合的A/B工具仍在`tools/benchmark_prep.py`。工具不启动服务、不切换默认值。

服务仍慢时，先停旧服务，再采集一次有界诊断：

```bash
./diag
# 使用原有客户端发送同一批请求后，从另一终端提取诊断行
grep '\[oscar-ascend\] PERF' /tmp/oscar_ascend_logs/serve.log
```

只计时rank 0前6个非空调度步骤及其采样，输出真实forward的加载路径/代码指纹、形状/开关、各OSCAR阶段耗时、整数sort调用栈。`inclusive_ms`包含嵌套阶段，不能相加；`wait_before_ms`记录进入该阶段前等待已提交设备任务的时间。同步和Python算子追踪会改变这几个步骤的吞吐，首轮也可能包含JIT，不能把诊断吞吐当作正式benchmark。默认关闭；达到步数后自动停止采集，不修改计算结果。

并发诊断用 `./load`：同样安装、检查并启动服务，等待rank 0出现至少8个正调度请求、每请求最多4个token的步骤，只采集4个匹配步骤及其采样。用原有客户端发起32并发负载；它不会自动发送请求。不匹配的prefill/单请求步骤不消耗采样次数，日志会显示 `PERF waiting`。PERF的batch字段给出实际调度请求数及token数；不要用APIServer的Running数代替单步实际调度数。

## 关键配置

| 环境变量 | 默认/含义 |
|---|---|
| `VLLM_PLUGINS` | 脚本默认 `ascend,oscar_ascend`，两个插件都必须存在 |
| `OSCAR_ASCEND_ENABLE` | `auto`；`0` 禁用接入 |
| `OSCAR_ASCEND_PACKED` | serve默认1；0使用BF16物理槽几何 |
| `OSCAR_ASCEND_USE_TRITON` | serve默认1；0使用torch参考内核 |
| `OSCAR_ASCEND_ATTENTION_MODE` | 默认streaming；native显式启用原完整历史对照。streaming在NPU要求Triton，不提供dense fallback |
| `OSCAR_ASCEND_STREAM_WORKSPACE_MIB` | 默认32，仅限制split暂存；0使用单split。查询输入/输出和变换临时量另作worker预留，不代表整模型峰值上限 |
| `OSCAR_ASCEND_USE_PAGED` | 仅native对照模式下有效；默认0，1启用旧向量paged实验 |
| `OSCAR_ASCEND_PAGED_BLOCK_KV` | 默认4；16/32/64/128仅供显式实验，32已在目标910B4出现UB溢出；调整后必须重新运行paged门禁 |
| `OSCAR_ASCEND_PROFILE_STEPS` | 默认0关闭；正数表示rank 0需要采集的真实调度步数，包含execute_model和sample_tokens |
| `OSCAR_ASCEND_PROFILE_MIN_REQUESTS` | 默认0不筛选；只采集至少此数量的正调度请求，`./load`设为8 |
| `OSCAR_ASCEND_PROFILE_MAX_TOKENS_PER_REQUEST` | 默认0不限制；过滤单请求调度token数超过上限的步骤，`./load`设为4 |
| `OSCAR_ASCEND_FUSED_PREP` | 默认0使用独立反量化/窗口拼接；1显式测试准备融合，仍需通过数值门禁且不代表性能验收 |
| `OSCAR_ASCEND_REQUIRE_TRITON` | 一键默认1，门禁失败阻断；0为诊断降级模式 |
| `OSCAR_ASCEND_K/V_ROTATION_PATH` | serve默认仓库内 `oscar_rotations.pt`；启动缓存初始化时检查目标层覆盖和正交性 |
| `OSCAR_ASCEND_K/V_CLIP_RATIO` | serve默认0.96/0.92；插件直接加载时默认0，范围[0,1] |
| `OSCAR_ASCEND_SINK_TOKENS` | serve默认128；按实际kernel block size向下对齐 |
| `OSCAR_ASCEND_RECENT_TOKENS` | 默认256 |
| `OSCAR_ASCEND_STAGING_TOKENS` | 默认8192；0禁用窗口 |
| `OSCAR_ASCEND_BATCHED_TOKENS` | packed默认15360，legacy默认16384 |

内存接缝当前要求 PP=1，并拒绝绕过预算的 `num_gpu_blocks_override`。当前路径依赖 eager，尚未支持图捕获、KV传输/换出对插件额外arena的同步；这些不是已验证能力。
