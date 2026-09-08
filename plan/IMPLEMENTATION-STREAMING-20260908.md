# 移除每步完整历史KV物化：streaming实现记录

本次是计算路径改造，不再以调整KV准备小内核作为主要修复。当前本地没有NPU，新代码尚不能宣称已通过Ascend编译或已获得端到端加速。

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

`delivery/probe_streaming.py --device npu`验证：混合query、16/32/64请求、单split长query、24K历史、空cache-only、legacy/packed槽、Hk2、fp16/bf16、窗口开关、真实backend及standalone更新。没有本地NPU，这些case仍需在目标机器运行。

`./bench`现在比较native对照和streaming，使用同输入交错测量三个实际形状：

- 1请求、24579历史、4-token MTP；
- 8请求、每请求24579历史、4-token MTP；
- 1请求、15360历史、9256-token continuation。

一键启动的streaming模式先跑数值/生命周期门禁，再跑这三项性能门禁。任何一项预热中位耗时超过native基线110%就停止启动，并保留`streaming_bench_*.log`。10%的阈值仅用于阻止明显性能回退；六个样本仍可能波动，它不是完整LongBench验收或统计显著性证明。

```bash
git pull
./diag
```

`./diag`会执行上述门禁再启动服务并采集前6步。若需要单独检查性能，停服务后运行`./bench`。显式`OSCAR_ASCEND_ATTENTION_MODE=native`可运行历史对照，但它仍有原来的全历史物化，不属于修复路径。

在NPU编译、数值、真实形状性能和完整同样本LongBench评测完成前，只能确认结构性实现与本地回归，不报告新增加速倍数。
