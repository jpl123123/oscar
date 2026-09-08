# 移除每步完整历史KV物化：streaming实现记录

本次是计算路径改造，不再以调整KV准备小内核作为主要修复。本地没有NPU；用户11:19开始的复测已确认实际BF16/128配置的16个内核case及backend通过，但三个性能用例全部失败，长continuation甚至比旧OSCAR native对照慢约1000倍。结构上消除了全历史浮点临时张量，性能目标仍未完成。

## 改了什么

默认`OSCAR_ASCEND_ATTENTION_MODE=streaming`。有历史的MTP、DecodeOnly和continuation不再调用`oscar_full_dequant`、`_stage_splice`或`prepare_native_kv`来产生全历史张量。`native`显式保留为A/B对照；旧向量paged没有重新成为默认。

`oscar_ascend/kernels/streaming_attention.py`包含：

- CPU在线softmax参考实现：仅解码有界KV小块，用于数学、生命周期和内存复杂度测试，不作为NPU性能回退。
- 新Triton矩阵实现：一个program处理一组query/head行，共享32-token KV块；通道维分成64，使用`tl.dot`完成QK和PV。它与此前每query/head逐元素循环的向量内核不同。
- FP32在线max/sum/output累积，必要时写出每query/head的split结果，再归并。空split和空cache-only序列输出0/-inf；NaN不能被归并静默丢弃。
- 只在片上展开小块K/V，KV相关全局浮点暂存只有本步查询输出、LSE和受限split结果，另有本步旋转输入及整数页表/调度元数据。历史长度影响循环次数与页表信息，不进入全局dense KV分配尺寸。

无历史prefill保留原生快路径，因为它只使用当前chunk。混批时把无历史请求单独交给原生快路径，已有历史的请求打包后走streaming；不会因一个新长prefill把已缓存MTP重新送回全历史展开。

修复的是“把完整历史写成全局临时张量”，不是让精确attention不再读取历史。每组query仍需遍历可见的压缩页；不同query tile也可能重读同一KV tile，其性能必须实测。

## 内存边界

设本步查询token数为Nq、query heads为Hq、D为head_dim、split数为S：

```text
S > 1时，split暂存 = Nq × Hq × S × (D+1) × 4 bytes
S = 1时，不分配split暂存，直接写查询输出
```

S由本步query工作量和预算决定，不由历史长度决定。预算不足时减少S直到1。默认split预算32 MiB；查询输出及当前Q/K/V旋转不是历史缓存，另计入预留。

| 场景，Hq6/Hk1/D256 | S | split暂存 | 24K完整FP16 K+V历史对照 |
|---|---:|---:|---:|
| 1请求，每请求4个query | 32 | 789504 B，约0.75 MiB | 约24 MiB/层 |
| 8请求，每请求4个query | 16 | 3158016 B，约3.01 MiB | 约192 MiB/层 |
| 32请求，每请求4个query | 4 | 3158016 B，约3.01 MiB | 约768 MiB/层 |
| 9256个query的continuation | 1 | 0 | 原对照还需完整历史K/V |

这是张量尺寸计算，不是已测设备峰值。当前代码还保留固定大小的FP32 staging及原有MTP BF16影子池；没有增加FULL层的完整BF16历史副本。

worker预算新增一次共享峰值预留：split预算加上按`max_num_batched_tokens`、Hq/Hk/D计算的当前query变换/输出/写入暂存保守量。按目标15360、6/1、D256估算约362 MiB/worker，作为不分配给KV池的余量，不是每层各分配362 MiB。真实编译器workspace和整模型峰值仍需真机压力测试。

## 正确性与生命周期

1. K/V仍按原INT2协议写入，读取时按物理页和token偏移解码。
2. 历史窗口值仅在owner匹配时覆盖；保留与上一版native路径一致的历史FP16中间舍入，再转到query数据类型参与矩阵计算。
3. 当前chunk从未量化的旋转K/V读取，按每个query的绝对位置应用因果边界。即使chunk超过staging容量，也不读取量化后的当前值来代替它。
4. standalone `do_kv_cache_update`现在先使写入位置的staging标签失效。普通forward随后刷新保留窗口；cache-only decode也不会错误读到更新前的窗口值。
5. MTP拒绝后的重写、未接受尾部不可见、物理block回收、负slot和padding继续受测试约束。每层KV和窗口状态保持独立。
6. NPU streaming缺少Triton或内核运行失败时抛出异常，不吞掉错误转到完整历史物化。CPU运行的是有界参考实现。
7. 长continuation也走分块路径；prefix-free快路径明确不读取历史。这不是只修短decode、让其他有历史路径悄悄保留大展开。

FP32 softmax统计配合bf16/fp16矩阵乘法的累加/舍入顺序与原生融合算子存在差异，因此仍需任务精度及MTP接受率验收，不能只看elementwise probe。

## 本地验证范围

- 实际JIT函数源码被提取后，在带指针越界检查的NumPy张量模拟中执行，测试分块、GQA、D64/D256、Hk1/2、不同split、空段、NaN传播与归并。这验证源码数学与索引，不验证Triton-Ascend lowering、UB分配或NPU性能。
- CPU独立在线softmax参考及原有dense oracle对照。
- backend测试把全历史反量化、整段准备、窗口splice入口替换成失败函数，验证新路径不调用它们。
- 多请求/非连续页/padding、混批中的无历史prefill、cache-only更新、MTP拒绝重写、block回收和无dense fallback测试。
- 预算测试确认history从24K增至262K时相同查询的split计划不变；worker预留按峰值而不是按16层求和。
- pytest入口限定到本仓库tests，避免收集reference的手动/平台测试。

## NPU门禁与运行

`delivery/probe_streaming.py --device npu`验证实际BF16 query/128-token kernel页：混合query、16/32/64请求、单split长query、24K历史、空cache-only、legacy/packed槽、Hk2、窗口开关、真实backend及standalone更新。BF16指query dtype，不要求把INT2存储改成BF16。

`./bench`现在比较native对照和streaming，使用同输入交错测量三个实际形状：

- 1请求、24579历史、4-token MTP；
- 8请求、每请求24579历史、4-token MTP；
- 1请求、15360历史、9256-token continuation。

一键启动的streaming模式先跑数值/生命周期门禁，再按顺序跑这三项性能门禁。任何一项预热中位耗时超过native基线110%就立即停止，跳过尚未执行的性能用例，并保留`streaming_bench_*.log`。10%的阈值仅用于阻止明显性能回退；六个样本仍可能波动，它不是完整LongBench验收或统计显著性证明。

```bash
git pull
./diag
```

`./diag`会执行上述门禁再启动服务并采集前6步。若需要单独检查性能，停服务后运行`./bench`。显式`OSCAR_ASCEND_ATTENTION_MODE=native`可运行历史对照，但它仍有原来的全历史物化，不属于修复路径。

在NPU编译、数值、真实形状性能和完整同样本LongBench评测完成前，只能确认结构性实现与本地回归，不报告新增加速倍数。

## 10:45复测：部署内核通过，额外兼容组合编译失败

用户日志确认前8组×窗口开关共16个case在NPU通过，包括24K、16/32/64请求、cache-only及Hk2 legacy槽。随后`large_page_fp16`的第一个window=False用例编译报`Failed to obtain op buffer shape size which should be static`。该用例同时改变query dtype和kernel页大小，不能从当前错误判断究竟是哪一项触发，也没有UB溢出的明确日志。

用户实际服务PERF此前为query bf16、cache shape=[11772,128,1,256]。此次修正没有改动三个JIT计算函数，而是把默认门禁明确限定到这份部署配置；其他组合拆成独立研究入口：

```bash
python3 delivery/probe_streaming.py --device npu --compat fp16
python3 delivery/probe_streaming.py --device npu --compat large-page
python3 delivery/probe_streaming.py --device npu --compat fp16-large-page
```

这些命令分别固定另一轴以便定位，只输出COMPAT结果，不会启用服务支持。对应编译器问题仍未解决，不计为PASS。默认门禁保留全部16个已验证内核case并继续执行真实backend及性能门禁。

worker缓存初始化、backend入口和公共Triton调用均检查NPU query/model dtype及实际kernel block size；未验证配置会明确失败，不转换精度、不修改缓存几何、不回退完整历史物化。只有独立兼容probe可显式放行实验profile。106项本地pytest与默认CPU门禁通过，JIT函数AST与上一个提交一致。

## 11:19复测：长continuation运行约30秒/次，性能目标失败

证据是用户回传的`streaming_bench_20260908_111650.log`，运行提交`4978d1b`。16个NPU内核case、`STREAM BACKEND PASS no_dense=True cache_only_rewrite=True`及prefix-free检查都通过，随后性能门禁三个case全部失败。

| 单层完整forward，同输入交错采样 | 旧OSCAR native中位数 | 新streaming中位数 | 新/旧耗时 |
|---|---:|---:|---:|
| 1请求，24579历史+4 query | 17.834 ms | 37.681 ms | 2.11倍 |
| 8请求，每请求24579历史+4 query | 97.224 ms | 215.255 ms | 2.21倍 |
| 1请求，15360历史+9256 query | 30.458 ms | 30472.514 ms | 1000.47倍 |

这里的baseline仍是带OSCAR的旧native路径，不是关闭OSCAR的vLLM服务；单层倍数也不等于端到端LongBench倍数。

长continuation首次调用30495.117 ms，六次预热样本30464.812–30497.722 ms。热调用同样慢，排除了“只是首次编译、再等一次就快”的解释。首次加六次采样共约213秒，旧工具在这段时间没有采样进度；最后返回性能失败并退出，没有启动模型服务。8请求MTP的首次调用另有37.364秒开销，但不能拿它解释长continuation的稳定30秒耗时。

### 从当前代码能够确认的计算问题

`BM=32, BN=32, BK=64, Hq/Hk=6, D=256`。每个query tile只有32个query/head行，折算约5.33个query token。9256-token continuation形成`ceil(9256*6/32)=1736`个query tile，每个tile都从历史起点扫描；split=1只表示不需要归并暂存，不代表计算量少。

按`plan_stream`与`_stream_stage1`的实际循环边界计算：

| 用例 | query tiles | splits | 跨所有program的KV tile循环数 | QK/PV小矩阵乘法次数 |
|---|---:|---:|---:|---:|
| 单请求MTP | 1 | 32 | 769 | 6152 |
| 8请求MTP | 8 | 16 | 6152 | 49216 |
| 长continuation | 1736 | 1 | 1085290 | 8682320 |

这里的次数是源码循环和`tl.dot`计算数量，不是CPU发射了868万个内核。每个KV小块又按D拆成四次QK和四次PV，并重复解码及查询窗口owner；Q加载也写在KV循环内部，编译器能否提升或缓存需要检查生成代码。各query tile之间重读历史、重复解码，只有单个32行tile内共享。去掉全历史物化并不会自动带来加速，这次布局把复用范围缩得过小。

这些结构问题与实测退化一致，但尚无设备profiler来量化反量化、矩阵计算、Vector/Cube交互、搬运及循环开销各占多少；不能把推断写成已证实的单一硬件根因，也不能保证只改tile大小就能消除1000倍差距。

### 本次修正及仍需完成的工作

- 性能门禁改为首个失败case立即退出。此前MTP已确定失败，却仍耗时测并发和长continuation；现在该日志对应的运行会在MTP结果后停止，明确列出跳过的case。所有case都通过才打印总门禁PASS。
- 独立benchmark保留完整采样，每次首次调用完成、预热采样开始/完成都打印，输出放在计时区间外。长调用依然会等待设备同步，这不是超时中止机制，也不会把设备尚未完成的提交耗时当作执行时间。
- 这次没有改动attention数值内核；修复的是诊断可见性和无意义的后续等待，不能宣称30秒的内核已修好。README同步标明真实性能失败，不要求用户为了这次日志修正再跑完整NPU/LongBench。

本次本地验证：14项benchmark/门禁测试通过，覆盖首项或后续项失败后不再执行剩余用例、非有限计时不能放行、CPU/部分形状不能放行、全部通过才放行，以及日志输出不进入计时区间。另用17-token历史+4-token query实际执行CPU backend的首次及六轮A/B采样，数值对照通过；这些结果不代表NPU性能复测。

下一步的内核改造必须将MTP与长continuation分开设计：短query重点减少每个KV小块的解码/元数据和小矩阵开销；长query重点扩大同一解码KV块的query复用范围。优先评估固定容量KV工作区配合原生矩阵attention及在线LSE归并，工作区容量必须独立于历史长度，不能恢复每请求每层每步完整历史展开。若继续融合Triton路径，需要先用设备profile和生成代码确认布局/搬运瓶颈，再验证更大tile及流水线，而不是仅凭CPU模拟上调常量。

新候选先做数值、causal mask、窗口owner、MTP重写和空序列检查，再在同一组真实形状测预热性能。当前没有依据承诺新的加速倍数；端到端目标仍需同样本、同输出长度及MTP接受率的评测。
