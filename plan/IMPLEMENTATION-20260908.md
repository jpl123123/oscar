# 2026-09-08 实现与验收记录

> 版本纠正：用户指定的部署环境是 vllm 0.23.0 + vllm-ascend **0.23.1.dev0+g5cb98caaa.d20260822**。启动自检与依赖声明已按此修正；下文0.23接缝指API/源码基线，不要求Ascend包元数据为0.23.0。新增6项版本回归测试。

本轮按系统审查先处理正确性，再优化注意力路径。未修改 references；没有 NPU 或线上模型可供本轮执行。

## 已实现

| 审查项 | 实现与回归 |
|---|---|
| 1 staging重复清空 | layer ready标记一致；跨步arena复用；重复hash seat选择确定的同一owner/value；非窗口重写使tag失效 |
| 2 无效slot | ref过滤负槽；bf16/int8 × Hk=1/2/4测试，纯负槽为no-op |
| 3 CPU seq metadata | decode显式迁移到query设备；优先用Ascend现成host累计query ends，避免每层query_start_loc.tolist() |
| 4 decode fallback | 统一(out,lse)解包；故障注入后与直接ref对照 |
| 5 MTP性能 | 新多query分页内核；当前chunk未量化、历史INT2和旋转域staging；混合长prefill/短MTP批次分流。dense路径也去掉历史KV逆旋转 |
| 6 影子池预算 | 包装NPUWorker.determine_available_memory，从原生cache groups/config算每block池成本，再扣除影子池和固定arena；初始化时分配并对账 |
| 7 类替换异常 | 保存原class和实例状态，setup失败恢复；注入失败中止构造，不打印虚假的原生回退 |
| 8 probe漏洞 | 旧probe拒绝非finite误差；新增多请求/跨页/窗口/空序列/CPU seq Triton门禁 |
| 9 ACTIVE | worker输出完整层验证manifest；检查所有TP rank、层集合、源码sha，而不是统计一条成功日志 |
| 10 安装旧包 | 默认重新安装当前editable checkout；worker指纹必须匹配当前源码 |
| 11 head stride | 使用stride(2)检查每head槽宽；成功后才置geometry ready |

附加：fp16 scale floor采用最小正常值2^-14，常量向量不再除0；生产旋转加载要求目标层存在且矩阵有限/正交；serve缺文件在启动前失败；依赖限定vllm0.23.0及用户指定Ascend构建0.23.1.dev0+g5cb98caaa.d20260822，并在一键自检核对版本。

## 设计取舍

- 未量化窗口在写入时转换为FP32旋转域，容量由8MiB/层升至约16MiB/层（D256/Hk1/8192 tokens，不含owner）。这避免历史旋转，并保留BF16输入的未量化信息；需要精度评测确认浮点运算顺序变化的影响。
- fused多query内核返回旋转域输出，每个query/head/split一个program，按请求的prefix和query位置限制因果可见范围；只逆旋转最终输出。长prefill使用dense SDPA，仍有历史KV临时内存和请求循环，尚未实现大prefill融合。
- 与原始实现相比，修正后的窗口会真正保留历史未量化值，因此输出不会与有bug的旧版本逐位一致。测试基准是应有的窗口/因果语义。
- 新paged内核没有自动吞掉异常并回退。只有独立NPU门禁成功后，一键脚本才启用；否则硬门禁终止，可显式关闭paged测试其它路径。
- worker内存接缝依赖0.23私有API，PP=1；固定预留包含staging、owner和四份旋转矩阵/转置。native+shadow+fixed的实际常驻容量在初始化再次检查；临时算子workspace/长prefill峰值仍待真机验收。

## 已执行验证

环境：macOS ARM64、独立.venv、Python3.12、PyTorch2.14.0 CPU。

- `.venv/bin/python tests/test_numeric.py`：17 PASS。
- `.venv/bin/python -m pytest tests/test_backend.py -q`：23 PASS（包括真实forward分流和模拟0.23 worker预算/初始化接缝）。
- `.venv/bin/python delivery/probe_paged.py --device cpu`：bf16/int8两种槽的CPU oracle执行通过；这不代表Triton内核通过。
- 新增文件Ruff检查、Python编译、delivery shell语法检查、git diff whitespace检查通过。

单层CPU微基准（D256/Hq8/Hk1/q_len4，无staging，5次平均；只用于局部对比）：

| 历史长度 | 原历史逆旋转路径 | 旋转域dense路径 |
|---|---:|---:|
| 1024 | 2.16 ms | 1.79 ms |
| 4096 | 7.61 ms | 5.95 ms |

这些数字不是NPU性能或服务吞吐承诺。更大的结构性收益来自新fused路径不物化历史KV，需实测。

## NPU验收仍未执行

1. 安装当前代码后执行旧probe的ref/triton两种模式和两种物理槽宽。
2. `python3 delivery/probe_paged.py --device npu --triton`：新增内核编译、跨页、每请求q_len=1/4/2、prefix=0/129/257、窗口有/无、空decode、负slot。
3. 一键启动时检查每rank MEMORY/READY manifest、实际KV几何、MTP影子池、旋转覆盖和预算。真实vendor版本差异可能在这里需要适配；CPU替身测试不验证CANN接口。
4. 相同数据与参数对照原生BF16、legacy OSCAR和packed OSCAR；检查MTP接受率和任务精度，再测1K/8K/32K以及并发1/8/32的TTFT、TPOT、吞吐、显存峰值。
5. 覆盖请求结束后的block复用、MTP连续拒绝/重写、长短请求混批、接近容量上限的持续负载。KV传输/换出、图捕获和PP>1当前未实现额外arena生命周期同步。

校准数据代表性、模型指纹及全链路质量基准仍需进一步工作；本轮的严格旋转加载解决了缺层/坏矩阵静默回退，但没有把单段校准文本改造成代表性数据集。

## Ascend decode stage2 编译故障修复

用户在 Ascend910B4 上反馈：ref probe、Triton store 字节一致性和反量化检查通过，但 decode stage2 编译报 `unsupported datatype for arith::ExtFOp to hfusion`。此前新增的分母下限 `1e-38` 低于 FP32 最小正规数，可能被 Triton 推断为 FP64，与报错吻合（上游类型推断讨论：https://github.com/triton-lang/triton/issues/6251）。尚未取得目标环境 IR，因此这是依据代码与日志作出的定位，需真机复验。

归并内核改用 `tl.where(e_sum > 0.0, e_sum, 1.0)`，除法和 log 共用安全分母；非空结果保持原公式，空序列输出零、LSE 为负无穷。普通 decode 与 paged attention 共用此修复。

新增 JIT 浮点字面量守卫，以及执行实际归并尾部表达式的 CPU 类型/数值回归。33 项 pytest、17 项 CPU 数值镜像、测试文件 Ruff 和 diff whitespace 检查通过。未在本机执行 Ascend 编译；需重新运行一键启动门禁确认。

## Paged probe 原生后端导入故障

用户复测确认旧 Triton probe 的 512B/256B 几何均通过，decode 误差为 1.907e-06。随后 paged probe 在真实 backend.forward 中因 `AscendAttentionState = object` 失败；此前捕获所有导入异常的逻辑隐藏了原始错误，不能从该日志确定缺失的依赖或符号。

独立 NPU probe 现在先解析 current_platform，并调用原生 CLI 同样使用的 `pre_register_and_update()` 应用全局补丁，再加载真实后端和 SpecDecoding 状态。后端只有在 vllm 与 vllm_ascend 均未安装时才提供 CPU 参考类及完整状态枚举；已安装环境的导入失败保留异常链并立即终止，不再假装加载成功。后端导入也先解析平台，避免在 attention 导入途中才启动懒加载平台。

新增回归覆盖缺省 attn_state 的实际 forward、平台导入顺序和导入故障保留；移除 MTP 测试替换状态枚举的遮蔽。38 项 pytest、bf16/int8 两种 CPU paged oracle、测试/probe Ruff、Python 编译及 diff 检查通过。该初始化调整仍待 NPU 复验，尚不能确认底层导入问题已消除。日志中的整数 ArgSort 回退 AiCPU 属于另一个待测性能问题，本次未改动其碰撞处理语义。
