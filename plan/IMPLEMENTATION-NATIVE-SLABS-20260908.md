# 固定容量KV块与原生attention：替换退化的直接streaming内核

当前`OSCAR_ASCEND_ATTENTION_MODE=streaming`的backend使用`native_slabs`。这次修改了实际计算路径；不是只改门禁或输出提示。本地没有NPU，CPU验证不代表Ascend编译或真机性能已通过。

## 触发问题

用户在`4978d1b`/`e9f4a7e`回传的日志确认，直接Triton streaming的MTP稳定耗时约37毫秒，对照约17毫秒；15360历史+9256 query的continuation稳定约30.47秒，对照30.46毫秒。这些baseline是仍带OSCAR的旧native路径，不是关闭OSCAR的vLLM服务。

旧实现的32行query/head tile使长continuation形成1736组独立历史扫描，累计约108万次KV tile循环、868万次小`tl.dot`。本次从默认backend移除这条路径。旧函数保留用于数学测试和独立兼容实验，不用于默认服务或性能门禁候选。

## 计算路径

1. 保留原来的旋转、裁剪、INT2写入、staging窗口及owner失效规则。
2. 当前未量化K/V与当前query单独调用原生TND因果attention，取得输出和FP32 LSE。当前段不再与完整历史拼接；大于8192 token的当前段仍直接使用已有当前K/V。
3. 每请求每次读取不超过8192个历史token，多请求受共享工作区约束后按更小块批量读取。两个Triton解码调用分别把K/V填入复用缓冲，融合owner检查、窗口覆盖和原有FP16中间舍入，再存为query dtype。
4. 一次原生TND attention让该批请求的所有query使用对应KV块，返回部分输出及LSE。矩阵乘法交给原生融合算子，不再在query小块里重复解码历史。
5. 用FP32归并结果，随后覆盖同一对K/V缓冲处理下一块。最终仅对输出逆旋转。

对于有新K/V的请求，每个历史token在一次forward中只解码到块缓冲一次。精确attention仍需要读取全部可见历史；本次消除的是全历史浮点临时张量及按query小块重复解码，并未减少可见上下文，也未建立跨步完整BF16历史副本。

当前query的原生因果部分与历史块之间按以下关系合并，而不是直接平均输出：

```text
L = log(exp(L_old) + exp(L_part))
O = exp(L_old - L) * O_old + exp(L_part - L) * O_part
```

实际实现先减去最大LSE，避免上溢。空部分的LSE为-inf、权重为零；空输出即使含未定义数值也不会进入累加。真实NaN不会被当作空段吞掉。

## 原生接口依据

使用提供的vllm-ascend reference已有接口，不修改reference或安装目录：

- `attention/context_parallel/attention_cp.py`中的nomask/mask及prefix-chunk路径使用TND、`actual_seq_lengths`、`actual_seq_lengths_kv`、`softmax_lse_flag=True`。
- 历史部分使用`sparse_mode=0`，当前因果部分使用`sparse_mode=3`及原有2048×2048压缩因果mask。
- `attention/context_parallel/common_cp.py::_update_out_and_lse`给出相同的自然对数LSE加权合并关系。

新wrapper要求TND LSE形状为`[query_tokens, query_heads, 1]`，形状不符直接报错；不凭元素数猜测或静默转置。模型/query仍限BF16、实际kernel cache block仍限128，以匹配当前目标部署。新解码launcher和归并kernel仍需此环境的编译/数值门禁。

## cache-only、混批及生命周期

- 无新K/V时，所有可见K/V按固定块从缓存读取。对跨越当前query位置的块，分别处理需要因果mask的query片段与可见整个块的后续query片段。只有这类边界块可能解码两次；没有按每个query重复扫描的循环。
- `seq_len < q_len`时，落在序列起点之前的query保持0输出/-inf LSE，不把全mask空行交给原生算子。
- 当前有K/V时，历史读取止于`seq_len - q_len`；当前段使用原始旋转K/V，避免MTP新token被量化后再参与当前attention。
- 混批中的无历史prefill继续由backend原生快路径处理，其他请求进入有界块路径。padding不提交给原生算子。
- 多请求的query/KV累计长度分别打包。普通连续请求直接使用query视图；需要跳过请求时只打包当前query，整数页表另行规范成连续布局。
- staging依然使用物理block owner检查。standalone写入先失效旧tag，MTP拒绝重写、未接受尾部不可见及block复用规则保持。
- 计算/编译失败直接抛出，不回退完整历史展开，也不回退旧直接streaming内核。

## 工作区与调用数量

默认浮点K/V工作区上限32 MiB，由`OSCAR_ASCEND_STREAM_WORKSPACE_MIB`控制；大小取决于该上限、请求数、Hk、D和dtype，不依赖历史长度。每组最多64请求；请求更多时分组，复用同一缓冲。0不能容纳K/V块，现明确拒绝此配置。

目标BF16、Hq6/Hk1/D256下的计划如下。原生attention调用次数不等于底层矩阵运算次数，更不能直接换算加速倍数。

| 形状 | 每请求KV块上限 | K+V缓冲 | 历史原生attention调用 | 当前段原生attention调用 |
|---|---:|---:|---:|---:|
| 1请求，24579历史+4 query | 8192 tokens | 8 MiB | 4 | 1 |
| 8请求，每请求24579历史+4 query | 4096 tokens | 32 MiB | 7（批量） | 1（批量） |
| 1请求，15360历史+9256 query | 8192 tokens | 8 MiB | 2 | 1 |

其中长continuation的15360个历史token各解码一次，不再由1736个query/head tile重复解码。小于块上限的历史可能一次全部放入块缓冲，但分配容量有固定上限，不会随上下文增长为整段历史大小。

查询输出、部分attention输出、LSE、旋转及当前写入暂存按本步query数量计入worker保守预算；共享4 MiB因果mask也计入。目标15360调度token/Hq6/Hk1/D256约预留457.05 MiB/worker，较旧split实现多约95 MiB。这个值按worker峰值取最大，不乘16个FULL层，也不代表额外分配同等大小的常驻张量。CANN内部workspace和整模型峰值仍需真机验证。

## 验证与交付状态

本地专项测试覆盖：

- 独立在线softmax参考对照；强制5/7-token小块以跨越页、请求和因果边界。
- 跟踪实际调用，验证33+53历史仅解码86个token，所有query复用、同一对K/V缓冲反复覆盖，不按query tile分配历史。
- cache-only多query、序列前空query、窗口开关、legacy/packed、Hk2、非连续物理页、MTP重写及padding。
- 真实Triton launcher参数与JIT源码在带指针越界检查的CPU模拟中执行，检查非零读取偏移、多请求打包、staging以及LSE归并的极端值/空段/NaN。
- 原生TND接口参数及LSE布局检查；这些mock测试不代替真实CANN调用。
- 默认CPU probe包含20个数值用例，并执行真实backend和standalone更新检查。

当前本地验证通过；NPU速度尚未测量，不给出已实现的加速倍数。精确任务验收还需同样本LongBench、MTP接受率、TTFT/TPOT和峰值内存。

```bash
git pull && ./diag
```

入口依次执行数值/生命周期门禁、三个同输入性能用例，再启动服务。任何性能用例超过旧native对照110%立即停止并跳过后续用例；必须全部通过才允许serve。`STREAM BENCH plan`应出现`impl=native_slabs`，JSON中有`streaming_kv_scratch_bytes`，不再报告旧query split暂存。

服务启动后的六步诊断增加`slab_dequant`、`slab_native_attention`、`slab_merge`，帮助区分解码、原生算子和归并耗时。`inclusive_ms`包含嵌套阶段，不可相加；同步诊断不作为正式吞吐数据。
