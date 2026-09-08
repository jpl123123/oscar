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
