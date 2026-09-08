# 2026-09-08 NPU 服务性能回归

> 后续真机纠正：原生融合prefill门禁通过；分块32在paged stage1出现UB溢出及编译器重试失败。默认值现恢复4，微基准也不再默认运行32。下文的4→32描述保留为本次实验记录，不能视为有效优化或当前默认值。

用户提供两段日志，并说明除 OSCAR 外其余配置相同。OSCAR 片段为09-08 03:03–03:08，原生片段为09-07；不是对齐的完整benchmark报告，不能用逐行均值作严格速度比或推导TTFT/TPOT。

| 指标 | OSCAR 日志 | 原生日志 |
|---|---:|---:|
| generation throughput 范围 | 0–2.6 tokens/s | 4.5–262.6 tokens/s |
| prompt throughput 峰值 | 3104.5 tokens/s | 8979.9 tokens/s |
| Running=5时观察值 | 1.2、0 tokens/s | 57.6 tokens/s |
| Running=10时观察值 | 2.6、0 tokens/s | 93.8 tokens/s |

同Running数的片段仍有prefill、队列和上下文差异，但数量级退化明确。OSCAR在低KV占用下仍积压大量请求，日志没有给出能够把慢速归因于KV池耗尽的证据；早期MTP接受率100%时吞吐仍极低，不能只用接受率解释。基础数值门禁与READY成功没有验证性能。

## 代码与日志证据

1. 长prefill路径仍调用 `oscar_prefill_ref`：15360个query按512分块，单请求单FULL层至少30次SDPA，且每块生成带prefix偏移的显式mask。TP4各rank有16个FULL层，每rank仅一次这样的prefill就至少480次SDPA调用。是否落到math后端需设备trace确认，但该调用结构已偏离原生融合注意力。
2. `_staging_write` 每层调用int64稳定排序，用户日志明确报告四个rank的ArgSort在AiCPU运行。改为对有界arena槽号进行FP32排序；容量大于2^24时保留整数路径以防编号舍入，碰撞仍选择最后一个writer。
3. paged stage1固定 `BLOCK_KV=4`。以32768上下文、16个split计，每program约512轮，且每个query/head重复读取和反量化历史。改为默认32后约64轮；这是循环次数变化，不是已测的8倍加速。该内核仍以向量运算实现，尚未获得原生Cube融合内核同等效率的证据。

## 修改与边界

- NPU dense prefill调用参考源码已有的 `torch_npu.npu_fused_infer_attention_score`，TND、GQA、sparse_mode=3、2048压缩因果mask；prefix=0直接使用原始K/V，prefix>0拼接旋转域反量化历史。只接受bf16/fp16，不静默降精度。历史反量化、窗口拼接和旋转仍有成本。
- CPU使用原有SDPA参考算法。新增原生API契约测试以及独立NPU probe：空前缀、单query带前缀、前缀与query超过2048、bf16/fp16。
- paged门禁新增16387-token前缀、1536-token物理页、16:1 GQA、窗口命中/缺失、bf16/int8两种物理槽，覆盖长循环和尾块。
- 微基准能直接比较KV分块4/32，以及15360-token prefill旧SDPA/新融合路径。先warmup，再计时，并核对数值；不把JIT编译耗时算作稳态。

## 本地验证与真机复验

48项pytest通过；8个prefill CPU数值case及小/长分页CPU oracle通过。无本地NPU，未测CANN编译、融合算子执行及端到端吞吐，不能宣称性能回归已消除。

停服务后可运行：

```bash
python3 tools/benchmark_attention.py --device npu --triton --lengths 1024 8192 32768 --iterations 3
python3 tools/benchmark_attention.py --device npu --mode prefill --prefill-tokens 15360 --iterations 3
```

这两项分别测单层读取和dense prefill；不包含完整forward的store、staging更新、调度、通信、模型其它层。最终需相同请求集、相同eager/图设置、固定并发与上下文的端到端benchmark，记录TTFT/TPOT、generation吞吐与MTP接受率。若仍慢，应采集NPU trace区分写入/旋转/裁剪、staging、attention、CPU同步与编译停顿，不能继续把小probe PASS当作性能证据。

## 分块32的编译失败

用户03:24复测日志在 `_oscar_paged_stage1` 出现 `IR Dump After PlanMemory Failed (hivm-plan-memory)`，随后报告 `Ub overflow detected`；编译器关闭code-motion重试又报 `Failed to obtain op buffer shape size which should be static`。IR包含大量32×256临时张量以及reduce临时缓冲，直接把KV分块放大8倍没有解决临时数据占用问题。

生产与微基准默认均恢复分块4；保留原生融合prefill、浮点稳定排序、长上下文门禁和计时工具。没有自动尝试16或吞掉编译失败，也没有关闭数值门禁。今后扩大分块需要先降低K/V临时张量的同时存活量并在目标编译器测定UB占用；不能根据循环数下降直接认定可用或提速。

## 03:36–03:37复测：仍无明显服务收益，增加运行诊断

新片段在Running=3/4、Waiting=27–29时生成吞吐仍只有0.1–0.9 tokens/s，MTP接受率100%。四个rank仍报告整数ArgSort回退AiCPU。日志没有实际attention路径或分段耗时，无法确认两项保留优化的耗时占比，也不能据此指认旧代码仍被加载。

核对指定commit 5cb98caaa：`vllm_ascend/ops/gdn_attn_builder.py`中的 `_stable_argsort_for_npu` 把bool转成int32再稳定排序；混合spec/non-spec token元数据会调用它。这是当前警告的另一个明确候选来源。未改动GDN，也未把这条warning继续当作OSCAR staging的独占证据。

新增默认关闭的 `OSCAR_ASCEND_PROFILE_STEPS`：对rank 0的前N个非空execute_model和后续sample_tokens采集同步墙钟时间、实际forward加载位置和已加载函数代码hash、OSCAR子阶段计数与耗时、PyTorch整数sort调用栈。正数显式开启；达到N后停止同步和算子追踪。嵌套阶段耗时重叠，冷调用包含编译，采集时吞吐会受到干扰。这次改动是取证工具，不宣称解决性能回归。

55项本地pytest通过，包含关闭时不安装wrapper、真实CPU sort调用栈、保留结果/异常、清理诊断上下文、跳过空步骤和达到次数后停止采集。NPU同步及vendor调度接缝仍待用户环境验证。

## 04:18 PERF：确认向量paged内核为主要瓶颈

实际加载 `/workspace/oscar/oscar_ascend/backend.py`，Q为bf16，形状[4,6,256]，use_paged/use_triton均为true。staging容量8192。连续步骤数据（rank 0，16个FULL层累加）：

| 步骤 | execute_model | paged_attention | staging_write | staging_sort |
|---|---:|---:|---:|---:|
| 3 | 4865 ms | 4368 ms | 60 ms | 2.21 ms |
| 4 | 4721 ms | 4351 ms | 54 ms | 1.88 ms |
| 5 | 4725 ms | 4353 ms | 54 ms | 1.92 ms |

paged约占模型执行92%，每FULL层约272 ms；后续步骤没有随首轮编译结束而下降。此前先优化prefill/排序不能解决这项主要开销。长prefill的原生融合调用已经生效：步骤1、2、6的16层native_prefill时间分别为96、144、94 ms。整数ArgSort warning出现在6步诊断结束之后；sorts为空不能据此声称实际运行没有排序（staging_sort计时已经说明有），vendor执行还可能绕过Python dispatch追踪。

本轮默认禁用自写paged内核，改用每请求一次历史反量化 + 窗口拼接 + 原生融合attention，涵盖MTP和有新K/V的DecodeOnly；显式USE_PAGED=1保留实验入口。INT2常驻缓存、旋转、当前chunk未量化K/V与窗口owner语义保留；计算使用bf16/fp16融合算子，舍入顺序与FP32向量内核不同。该路线需要临时dense KV，临时显存和端到端质量仍需复验。

扩展门禁：实际6:1 GQA，24579前缀，4-token query，非单位旋转，128/1536物理页，窗口开/关，比较真实backend.forward与独立CPU分页oracle，并在预热后报告单层完整forward耗时。新增CPU回归验证DecodeOnly/SpecDecoding、非连续物理页、只反量化一次、padding不参与注意力。59项pytest及8个原有prefill case、4个24K backend case通过。本机仍无NPU，不把CPU耗时当成NPU提速结果。

## 04:38复测：模型执行约8.5倍改善，优化KV准备

真实cache shape=[11772,128,1,256]，use_paged=false。第4/5步execute_model约556 ms（此前4721/4725 ms），16层OSCAR forward约279 ms（此前4446/4449 ms）。这是相同同步诊断口径下的模型执行改善，不是端到端吞吐倍数。诊断后的服务generation达到11.3 tokens/s，但尚未达到此前原生水平。

剩余分段：prefill_attention约197 ms，native_prefill自身约8.4 ms，进入该调用前wait_before约165.7 ms；staging_write约53 ms、store约25 ms。进入原生调用前提交了历史反量化、窗口拼接、类型转换及Q/K/V旋转，因此165.7 ms只能归到这段准备工作，不能全部算成原生算子时间，也不能在没有细分测量时全部算给dequant。

新增 `prepare_native_kv`：4-token tile、K/V分别launch以缩小同时存活数据；把INT2解码、owner匹配窗口值、FP16中间舍入和最终输入类型转换合到输出写入。输出直接按prefix+current长度分配，只有current尾部copy，避免独立全prefix gather/where/cast和concat。新的 `npu_prefill_prepared`直接消费该缓冲。保留旧路径通过FUSED_PREP=0对照；一键默认先经过新门禁才启用，Triton关闭或跳过门禁时不自动开启。

门禁增加准备缓冲的逐元素比较：257个历史token（尾tile）、非连续页、Hk=1/2、bf16/int8物理cache、fp16/bf16输出、窗口命中/缺失和owner碰撞。随后继续跑完整24K MTP及计时。74项本地pytest和CPU门禁通过；尚未在本机执行新Triton内核，不能宣称新增融合已有NPU收益。

PERF新增prepare_native_kv计时；原生算子改标为native_attention，边界为已准备好完整输入的调用，避免与旧native_prefill（包含concat）的计时混淆。比较收益仍以execute_model和forward的相同边界为主。

## 05:05复测：准备融合未证明提速，恢复默认并加入同进程A/B

use_fused_prep=true。第4–6步execute_model约599 ms，forward约295 ms，prepare_native_kv约181 ms；前一轮分别约556/279 ms，准备区间wait_before约166 ms（计时边界不同）。本轮未证明新增融合有收益。两轮其他阶段也有变化，因此不能把全部约43 ms差额严格归因于融合代码；后续需要同进程交错比较。

第2步prepare_native_kv合计10018 ms，远高于后续181 ms；可能包含未被此前odd-prefix门禁预热的aligned-prefix编译特化，但没有编译事件trace，不能断言10秒全部为编译。新的PERF增加各阶段first_ms/max_ms，以区分首个调用与其余调用的贡献。

恢复一键和直接probe的FUSED_PREP默认0，保留已验证的原生融合读取。新增 `./bench`：同一impl/cache/输入，交错运行基线与候选，分别记录first_call_ms和6次预热采样的中位数；先比较结果，数值不符不报告有效性能结果，复制输出避免复用输出buffer掩盖错误。覆盖MTP与实际continuation形状、6:1 GQA、128页、窗口开启、0.96/0.92裁剪。不会修改服务默认值。76项CPU回归通过，A/B工具已在CPU MTP形状执行；CPU结果不用于推断NPU性能。

## 05:30同进程A/B：继续保留基线，转向并发诊断

用户NPU实测，数值比较均通过：

| 形状 | 基线中位数 | KV融合中位数 | 候选相对基线 |
|---|---:|---:|---:|
| 24579历史 + 4-token MTP | 28.233 ms | 28.637 ms | 高约1.4% |
| 15360历史 + 9256-token continuation | 83.903 ms | 93.751 ms | 高约11.7% |

各6个样本波动较大：例如continuation基线47.30–121.35 ms、候选49.26–115.61 ms。不能声称有统计显著差异，但没有证据支持把候选恢复为默认。首次调用基线446.74/116.42 ms、候选45.32/55.73 ms也不能代替稳态结论；它们受到已有编译缓存及初始化顺序影响。默认保持USE_PAGED=0、FUSED_PREP=0。

下一步需要并发短query步骤的实际分段数据：当前实现每请求独立准备历史并调用原生attention，调用数会随请求数增长；但此前PERF主要覆盖单请求，不能直接断定并发时耗时占比。参考原生_get_fia_params确认dense TND可以使用累计query/KV长度表达多个请求；全量拼接32×24K、Hk1/D256/bf16的K+V就约768 MiB（还不含临时workspace），应先设计内存受限分组，不能直接全批物化。

新增 `./load` 条件诊断：至少8个正调度请求，单请求调度token数最多4，仅采集4个匹配步骤。保留默认计算路径，不自动发请求；PERF包含batch信息。77项CPU回归通过，验证跳过空请求、单请求、长prefill时不消耗采集预算，匹配步骤仍保留返回值且达到上限停止。NPU并发数据仍待收集。
