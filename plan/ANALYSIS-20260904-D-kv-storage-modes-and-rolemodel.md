# ANALYSIS-20260904-D — KV 存放与读取三模式对比 + Role Model 判决

> **聚焦范围**：只讲一件事——KV cache 的**存放布局**（池怎么建、字节怎么摆）与**读取模式**
> （每步算子怎么把 KV 用回去）。三种模式：① 当前插件版 OSCAR（本仓）、② OSCAR 的
> vLLM PR（`references/oscar-vllm-pr46774` @57286d5d）、③ TurboQuant 的 vLLM 官方实现
> （upstream vLLM v0.23.0 @0fc695fc，本地 `references/vllm` 全量在库）。
> 所有行号均实测。前置结论（布局未动、显存零收益、110s 归因）见
> `ANALYSIS-20260904-C-kv-chain-and-oscar-q1q2.md`，本文不重复推导。

---

## 0. 一句话判决

**当前版本约等于无效（用户判断正确）：池按原生 BF16 几何分配，INT2 只是把每 token·head
1024B 里的前 160B 重新解释，KV usage、池大小、nb、块表与不开 OSCAR 完全一样。**
两个成熟方案的共同点是：**在"分配期"就把池按压缩槽建小**（OSCAR PR：`get_kv_cache_shape`
第 4 维换成压缩槽；TQ：`TQFullAttentionSpec` 重定义页字节公式），读写路径只是兑现手段。
**Role model 判决：TurboQuant（工程蓝图）+ OSCAR PR（后端本体模板 + 数值配方），
二者互补缺一不可**——TQ 赢在三件我们最缺的事：分配期缩池的完整官方先例、
**hybrid（16 full + 48 GDN）与 Mamba 共存的官方修法（vllm PR#39931）**、小续算直接
复用 decode 内核（恰好命中我们 MTP verify=4 token 的热点）；OSCAR PR 给出"OSCAR
后端本体该长什么样"（classvar/独立写缝/内核/窗口——我们已移植 90%）。对称落地
映射见 §6.3，三池字节级方块图见 §2。

---

## 1. 三种模式分别是什么、代码在哪（先钉死事实）

| 模式 | 代码位置 | 状态 |
|---|---|---|
| ① 插件版 OSCAR | 本仓 `oscar_ascend/`（HEAD `63132db`） | 真机 ACTIVE；显存 0 收益 |
| ② OSCAR vLLM PR#46774 | `/Users/sunao2000/oscar_zai/references/oscar-vllm-pr46774`（快照仅含 PR 新增的 11 个文件） | PR 分支快照；**未处理 hybrid/Mamba**（`oscar/config.py:178-192` hybrid 模型直接禁用边界跳层；PR 无 mamba 分支） |
| ③ TurboQuant | **upstream vLLM v0.23.0**（本地 `references/vllm`：backend + 双 triton 内核 + 配置全套） | 已合入上游（KV 路径 = vllm PR#38479，2026-04-15；hybrid 修复 = PR#39931，2026-05-05）；**CUDA/ROCm/XPU 生态** |

**关于 TurboQuant 归属的两个取证（2026-09-04 实测，防乱说）**：

1. **vllm-ascend 里没有 TurboQuant**。本地参考树 v0.23.0 grep 零命中；官方仓库 main
   （`4fe7ddb`，稀疏克隆全量 grep）源码零命中；release notes（271,793 字节）零提及。
   用户指定的 `github.com/varjoranta/turboquant-vllm`（HEAD `c8a7e0a`）README:18 自己
   声明：**GQA/MHA 的 KV 压缩已上上游 vLLM（PR#38479，`--kv-cache-dtype turboquant_3bit_nc`
   等），本仓库今后只做权重压缩 + MLA legacy KV**；README:160 进一步说明"真正的
   TurboQuant（ICLR 2026 论文）应用就是上游 KV 路径"。
2. **README 模型表（README:126 表头 `Model family | Attention | Weight quant | Legacy KV
   monkey-patch | Notes`）里 Qwen3.6-27B 那行**：第 3 列 Works(v0.13.5) 是**权重**量化；
   第 4 列（KV monkey-patch）是 **Untested**；"15.4 GB (3.6×)" 是权重检查点压缩
   （54GB BF16 → 15.37GB）。**它证明的是"16 full + 48 GDN 混合架构在此插件生态可跑"，
   不是 KV 压缩数字。** 混合模型 KV 压缩的可用证据是上游 #39931（修 Qwen3.5 Mamba 层
   NotImplementedError），该修复就在我们本地 v0.23.0 树里（§5 hybrid/Mamba 段）。

---

## 2. 三池方块图：从 torch.zeros 到每一个槽（每格带参数依据）

> 统一口径：我们的模型每 rank（TP4）= **Hk=1, Hq=8, D=256**（真机日志 `heads=1`）；
> 每层 GDN 状态 = conv **15,360B** + ssm **393,216B** = **408,576B**
> （vllm-ascend `patch_mamba_config.py:64-70` 取 max/min，真机 P=801,792 反推吻合；
> 上游同构公式 `mamba_utils.py:218-233`）。下列 ②③ 的"推算"均由所引公式代入
> D=256/Hk=1 直接计算，正文逐个标注。

### 2.1 模式①（当前插件版）：16 池 × 三大条页，OSCAR 只用每槽前 160B

> **⚠️ 勘误（2026-09-05，用户追问触发）**：下图把"页"画成了页内 [条A|条B|条C]
> 连续交织——**字节级不精确**。真相是：整根池张量按**三个稠密区**排布（区A 全部
> conv 格、区B 全部 K/ssm 格、区C 全部 V 格），801,792B 的"页"只是记账单位
> （一个块号在三区各占一格）。精确图与逐字节推导见 **§2.5**（含 32K/TP4/单卡
> 完整排布图）。下图仅保留"区域宽度"语义正确。

```text
建池（每 rank 16 个，逐字真实代码）:
  torch.zeros(kv_cache_tensor.size, dtype=torch.int8)      ← [VA] model_runner_v1.py:4124
  size = P × nb，P = 801,792B                              ← patch_mamba_config.py:113-117
  16 个池：桶{48 GDN, 16 FULL} → group_size=16 → 池 i 被
  [GDN组0第i层, GDN组1第i层, GDN组2第i层, FULL第i层] 共享    ← [V] kv_cache_utils.py:1142-1195,1318-1326

一个池 = nb 个 768-token 物理页（每页 801,792B，三大条）:
┌──────────────────────────── 一个物理页（768 token）────────────────────────────┐
│ 条A conv   15,360B   │ 条B K/ssm 393,216B                │ 条C V 393,216B      │
│ GDN 专属              │ FULL 页装 K ←→ GDN 页装 ssm（同条不同页）│ FULL 页装 V          │
└──────────────────────────────────────────────────────────────────────────────┘
  ↑ GDN 层每请求领 1 页: conv 装条A + ssm 装条B（GDN 不碰条C）
  ↑ FULL 层每请求领 cdiv(seq/768) 页: K 装条B + V 装条C

FULL 页内一个 token·head 的原生槽（kernel 块 128 = 768/6，逻辑块=phys×6+i
  ← [VA] attention_v1.py:142-144 + block_table.py:288-303）:
  K 槽 512B = 256×bf16          V 槽 512B = 256×bf16        ← 原生每 token·head 1024B
  ┌────────────K 槽 512B────────────┐┌────────────V 槽 512B────────────┐
  │ meta 8B │pad 24B│ K idx 64B │ 死 ││ V idx 64B │        死 448B       │
  │ ←── OSCAR 只写这 96B ──→ │区 ││←─ 只写这 64B ─→│                  │
  └──────────────────────────┼─────┘└────────────────┼─────────────────┘
                             └──── 其余 864B 永远没人写（84%）────┘
  槽内字节图 = [W] format.py:25-33（K: scale/zero@0-7 + idx@32-95；V: idx@0-63）
```

| 参数 | 值 | 依据 |
|---|---|---|
| 分配 block_size | **768 token** | patch_mamba_config.py:93-103（`block_size := attn_block_size`） |
| kernel 块 | **128**（每页 6 逻辑块） | attention_v1.py:142-144；block_table.py:288-303 |
| FULL 每 token·head | 原生 **1024B**，OSCAR 写 **160B**（96+64） | 原生=D×2B×2；160B=[W] format.py:26 |
| 页字节 | **801,792B**（=15,360+393,216×2） | patch_mamba_config.py:113-117（真机日志吻合） |
| GDN 每层每请求 | 1 页内的 408,576B（条A+条B格） | patch_mamba_config.py:64-70 |
| 字节利用率 | **160/1024 = 15.6%**（其余 84% 死区） | 本表直接相除 |

### 2.2 模式②（OSCAR PR#46774）：uint8 单池，槽宽进 shape —— 但只有 dense 布局

```text
建池（引擎分配期，形状由后端给出）:
  get_kv_cache_shape → (num_blocks, block_size, num_kv_heads, slot_size_aligned)
                                       ← [PR] oscar_attn.py:87-103（uint8、无前导 2 维）
  slot = key_packed + value_packed:
    key  = ceil(D×2/8)=64B idx + 4B meta = 68B   ← [PR] config.py:99-118
    value= ceil(D×2/8)=64B idx + 4B meta = 68B   ← [PR] config.py:118-121
    slot = 136B（D=256 推算；D=128 实测断言 72B ← tests/quantization/test_oscar.py:71）
  引擎记账魔法: effective_head_size = slot//2 = 68 → 标准页公式
    2×bs×Hk×68×1B(uint8) = bs×Hk×136 ✓            ← [PR] config.py:124-128 注释原文

dense 模型一页（CUDA 默认 bs=16 ← [V] config/cache.py:46 DEFAULT_BLOCK_SIZE=16）:
┌────────── 一个物理页 = 16 token × Hk=1 × 136B = 2,176B ──────────┐
│ slot[0] 136B │ slot[1] 136B │ … │ slot[15] │        （uint8 池）      │
│ ┌key 68B┬value 68B┐ 每槽写满，无死区                          │
│ │4B meta+64B idx ×2 │                                          │
└ └──────────────┘───────────────────────────────────────────────┘
  kernel 块可选 [16,32,64,128]  ← [PR] oscar_attn.py:67-69

hybrid（3 ssm + 1 full）: ✗ 不存在
  - hybrid 模型 boundary skip 直接禁用（[PR] config.py:178-192 `if is_hybrid: return []`）
  - PR 无 Mamba 分支/无页协调代码（快照 11 文件 grep mamba=0）
  - 若强套上游统一页机制会复现 TQ 修前的 page-merger 断言崩溃（#39931 修的正是这个）
```

| 参数 | 值 | 依据 |
|---|---|---|
| 分配 block_size | dense: **16**（CUDA 默认）；hybrid: **无** | [V] cache.py:46；[PR] 无 hybrid 路径 |
| kernel 块 | [16, 32, 64, 128] | [PR] oscar_attn.py:67-69 |
| FULL 每 token·head | **136B**（vs 原生 1024B ≈ **7.5×**，推算） | [PR] config.py:99-128 公式代入 D=256 |
| 页字节（dense） | 16×1×136 = **2,176B**（无 mamba） | 同上 |
| GDN 层 | **不在池内**（PR 未处理 hybrid） | config.py:178-192 |
| 字节利用率 | **100%**（槽即页，页即槽） | 布局定义 |

### 2.3 模式③（TurboQuant @ upstream v0.23.0，post-#39931）：packed 页自己定义，mamba 页 pad 上来

```text
建池（引擎分配期）:
  get_kv_cache_shape → (nb, bs, Hk, slot_size_aligned) uint8
                                       ← [V] turboquant_attn.py:139-161
  slot = key_packed + value_packed（k3v4_nc, D=256 推算）:
    key  = ceil(256×3/8)=96B MSE idx + 2B vec_norm(fp16) = 98B   ← [V] turboquant/config.py:128-142
    value= ceil(256×4/8)=128B idx + 4B(scale/zero fp16)  = 132B  ← [V] turboquant/config.py:150-156
    slot = 230B（偶对齐规则 :166-174；3bit_nc 档=198B ≈ 5.2×）
  页字节公式重定义: real_page_size_bytes = block_size × Hk × slot
                                       ← [V] kv_cache_interface.py:327-349（TQFullAttentionSpec）

hybrid 页协调（TQ packed 页自己定大小，mamba 页向上 pad——与 vllm-ascend
patch_mamba_config 的"attention 对齐到 mamba + assert 相等"同机制、反方向）:
  attn_page_1_token = Hk×slot = 230B                 ← [V] platforms/interface.py:573-608
  kernel 对齐 = max(min([16,32,64,128]), bs=16) = 16  ← turboquant_attn.py:112-113 + cache.py:46
  attn_block_size = 16 × cdiv(408,576, 16×230) = 16×112 = 1,792 token   ← interface.py:649-655
                                                  （16×230=3,680B/token；3,680×111=408,480<408,576）
  attn_page = 1,792×230 = 412,160B ≥ 408,576 ✓
  mamba_page_size_padded = 412,160B（pad 3,584B，+0.88%）              ← interface.py:685-699

一个共享池的一页（1,792 token；3 层 GDN + 1 层 FULL 共享同一池/同一页号空间）:
┌────────────── 一个物理页 = 412,160B ──────────────┐
│ FULL 页: 1,792 个 230B 槽排满（K/V 拼在一个槽）      │
│ ┌─slot 230B────────────┐ ×1,792                     │
│ │ key 98B │ value 132B │   每槽写满，无死区           │
│ │2B norm+96B idx│4B meta+128B idx│                  │
│ └───────────────────────                            │
│ GDN 页: conv 15,360B + ssm 393,216B + pad 3,584B    │
└────────────────────────────────────────────────────┘
```

| 参数 | 值 | 依据 |
|---|---|---|
| 分配 block_size | **1,792 token**（k3v4_nc 推算；公式具象） | interface.py:649-655 公式代入 |
| kernel 块 | 16（[16,32,64,128] 取 min 对齐） | turboquant_attn.py:112-113 |
| FULL 每 token·head | **230B**（vs 1024B ≈ **4.45×**；3bit_nc 198B≈5.2×） | turboquant/config.py:128-174 代入 |
| 页字节 | **412,160B** = 1,792×230（mamba pad +3,584B/0.88%） | interface.py:685-699 |
| GDN 每层每请求 | 1 页（408,576B 状态 + 3,584B pad） | 同上 |
| 字节利用率 | FULL 区 **100%**；mamba 区 99.1%（pad 损耗） | 本表相除 |

### 2.4 三池一眼对比

| | ① 插件版 | ② OSCAR PR（dense） | ③ TQ（hybrid） |
|---|---|---|---|
| 池 dtype/形状 | int8 一维 `P×nb`（BF16 几何切三大条） | uint8 `(nb, 16, 1, 136)` | uint8 `(nb, 1792, 1, 230)` |
| 分配 block_size | 768 | 16（无 hybrid） | 1,792 |
| 每 token·head 槽 | 1024B 只用 160B | 136B 写满 | 230B 写满 |
| GDN 与 FULL 共池 | 是（三大条，assert 等宽） | 否（未支持） | 是（mamba pad 到 attention 页） |
| 显存效率 | 15.6% | ~100% | ~99% |
| 页字节 | 801,792B | 2,176B | 412,160B |

> 同样装下 GDN 408,576B 状态的约束下：① 用"K 条宽=ssm 条宽"的硬等式
> （512×768=393,216）；③ 用"mamba 页 pad 到 attention 页"的软等式
> （412,160 ≥ 408,576）。这就是 §6 判决里"TQ 是破对齐锁参考答案"的方块图级证据。

### 2.5 勘误与正图：nb、三稠密区、以及一条 32K 请求在单卡上的完整排布

> 本节由用户追问驱动（"条B K/ssm 393,216B——有些页的 B 是 K、有些是 ssm？nb 怎么
> 定义？A 和 C 怎么排布？32K/TP4 单卡的 torch.zeros 到底长什么样？"）。
> 所有断言基于下列**逐行亲读**的原文，零推导想象：
> `[VA] model_runner_v1.py:4569-4577`（FULL 切分，:4571 `attn_tensor_page_size =
> prod(shape[1:])×dtype_size` 是**整根张量级**字节数 = nb×393,216；:4574
> `conv_block_padding_size = raw_k_tensor.numel() − attn_page×2` = **15,360×nb**）、
> `:4696-4714`（GDN 从 raw 头部**顺序稠密切** conv→ssm，官方注释原文
> "tensor1: [(kv_padding), conv, ...] / tensor2: [k, ssm, ...] / tensor3: [v,
> (mamba_padding), ...]"）、`:4685`（GDN num_blocks = raw.numel()//page_size）、
> `[V] kv_cache_utils.py:1318-1326`（KVCacheTensor.size = page×nb）、`:967`
> （nb = available ÷ page ÷ 16）、`:911`（memory_per_block = page×16）、
> `[VA] block_table.py:288-303/:210-229`（逻辑块/槽映射）。

#### 2.5.1 先纠正一个容易画错的点：三区是"稠密排布"，页只是"记账单位"

一根池张量 `torch.zeros(801,792 × nb, int8)`（[VA] :4124）的真实字节布局：

```text
字节偏移      0 ────────── 15,360·nb ────────── 408,576·nb ────────── 801,792·nb
              ┌──────────────┬──────────────────────┬──────────────────┐
              │ 区A: conv 格   │ 区B: K/ssm 双视图共区   │ 区C: V 格（FULL 专用）│
              │ nb × 15,360B  │ nb × 393,216B         │ nb × 393,216B    │
              │ （GDN 专用）    │                      │                  │
              └──────────────┴──────────────────────┴──────────────────┘
   区A 视图（GDN 层）: raw[0 : 15,360nb].view((nb, 2560, 3))            ← :4705-4713 顺序切
   区B 视图一（FULL）: raw[15,360nb : 408,576nb].view((nb×6, 128, 1, 256))  ← :4575-4576+4637
   区B 视图二（GDN）: raw[15,360nb : 408,576nb].view((nb, 12, 128, 128))   ← :4705-4713 第二刀
   区C 视图（FULL）: raw[408,576nb : 801,792nb].view((nb×6, 128, 1, 256))  ← :4577+4642
```

**"页"（块号 b，记账 801,792B）的真实含义**：同一个块号 b 在三个区各占一格——
区A 的第 b 格（15,360B）⊕ 区B 的第 b 格（393,216B）⊕ 区C 的第 b 格（393,216B），
三笔字节分散在三区。引擎按 `memory_per_block = page_size × 16`（16 池各一页，
[V] kv_cache_utils.py:911）记账。

#### 2.5.2 回答"有些页的 B 是 K、有些页的 B 是 ssm？"——对，且这是对齐等式的真正目的

- **块号归属决定区B 格子的身份**：块号 b 被 FULL 组领走 → 区B 第 b 格（=逻辑块
  6b..6b+5，共 6×65,536B）装该请求的 K、区C 第 b 格装 V、**区A 第 b 格死置**；
  块号 b 被 GDN 组领走 → 区B 第 b 格装 ssm 状态、区A 第 b 格装 conv 状态、
  **区C 第 b 格死置**（该块 801,792B 记账中约 49% 为死字节——这正是上游
  "Padding mamba page size" 语义的代价，vllm-ascend 同款）。
- 块号由**全局唯一 BlockPool** 分配（nb 个块号、引用计数），FULL 组与 GDN 组从同一
  自由链表领号，天然互斥 → **同一字节永远不会被两种视图同时写**。
- **对齐等式 `512×768 == 393,216`（[VA] patch_mamba_config.py:95-97）的真正作用**：
  让"GDN 的一个 ssm 格（393,216B）"与"FULL 的一个物理页的 K 跨度（768×512B
  = 6 个逻辑块）"**字节级同一**——这样两种视图才能共用一个块号空间、一个页宽
  记账，互斥而不撕裂。（packed×2 下等式变 `256×1,536 == 393,216`，作用相同：
  ssm 格 == 12 个 32,768B 逻辑块。）

#### 2.5.3 回答"nb 怎么定义"——nb 是池容量，不是请求属性（附 910B4 完整算例）

```python
# [V] v1/core/kv_cache_utils.py
:967   num_blocks = int(available_memory // page_size // num_layers)
       #          = KV 显存预算 ÷ 801,792 ÷ 16  → nb（启动时一次算死）
:1318-1326  for i in range(16): KVCacheTensor(size = 801,792 × nb, shared_by=[3 GDN + 1 FULL])
:911        memory_per_block = page_size × num_layer_per_group = 801,792 × 16
```

**公式直觉**：把块号想成"车位"——**一个块号的全卡单价 = 801,792B × 16 池 =
12,828,672B ≈ 12.23MiB**（因为一个块号必须在 16 个池里同步各占一页，64 层混合
结构里"翻一页"所有相关层都要有地方放）。所以 **nb = 预算 ÷ 单价**："还剩多少钱
÷ 一个车位多少钱"。除数里的 16 不是层数巧合，而是"每池都住着 1 个 FULL 层 +
3 个 GDN 层"这一分组结构的账本折射（§2.2 分组代码）。

**一张 910B4 的预算瀑布（TP4 单卡视角，32GiB HBM，gpu-memory-utilization=0.9）**：

| 步 | 项 | 值 | 性质 |
|---|---|---|---|
| 1 | 卡显存 | 32 GiB（npu-smi 32768MB） | 事实 |
| 2 | × 0.9 目标水位 | ≈ 28.8 GiB | 事实（serve 参数） |
| 3 | − 权重（27B w8a8 ÷ TP4） | ≈ 7 GiB | **估算** |
| 4 | − 激活峰值/运行时/NPU 上下文 | ≈ 5~6 GiB | **估算** |
| 5 | = KV 可用预算 | ≈ 16 GiB | — |
| 6 | **实证锚点**：上轮真机启动日志 `GPU KV cache size: 16.36 GiB` | 16.36 GiB | 真机日志原文 |

代入公式（步骤 6 为锚）：

```text
nb = int( 16.36 × 2^30 ÷ 801,792 ÷ 16 ) = int(21,909.4) = 1,369
实际吃满 = 1,369 × 12,828,672B = 17,562,451,968B = 16.356 GiB ✓（与日志对账）
单池张量 = 801,792 × 1,369 = 1,097,653,248B ≈ 1.022 GiB；全卡 16 池 ≈ 16.36 GiB
三区边界（nb=1,369，字节偏移）:
   区A [0,          21,027,840)      宽 20.05 MiB   ← 15,360×1,369
   区B [21,027,840, 559,340,544)     宽 513.38 MiB  ← 393,216×1,369
   区C [559,340,544, 1,097,653,248)  宽 513.38 MiB
```

**一个块号 b = 16 个池各自的第 b 页**（:911）。请求消耗的才是变量：

| 组 | 每请求块数公式 | 32K 请求实例 | 依据 |
|---|---|---|---|
| FULL（16 层共用一份块表） | `cdiv(seq_len + lookahead 3, block_size)` | native **43** / packed **22** | [V] single_type_kv_cache_manager.py:276-277；lookahead [V] scheduler.py:462 |
| GDN（3 组 × 16 层，align 模式） | 每组记账 `2 + num_spec(3) = 5` → 记账 15 / **驻留 12** | **15**（不变） | 记账：[V] kv_cache_interface.py:629-631；驻留：块表 = cdiv(num_tokens,768)+3 项，其中仅最后 `1+3=4` 个为真实块、其余为共享 null_block 占位（[V] single_type_kv_cache_manager.py:1142-1165）→ 3 组 × 4 = 12 |

32K 请求的"账单"（nb=1,369）：native 占 **58 块号 = 4.24%**（有效字节 38.1MiB/池 ×
16 池 ≈ 609MiB；记账占用 58×12.23MiB ≈ 709MiB，差值为死格）；packed 占 **37 块号
= 2.70%**（22.3MiB/池 ×16 ≈ 357MiB）。→ 剩余 1,311 / 1,332 个块号可服务其他请求，
这就是"密度×2 → 并发×2"的账本形态。

#### 2.5.4 图一：原生 vllm-ascend（bf16, block=768）——32K 请求，TP4，单卡 16 池之一

请求参数：seq=32,768 token，MTP lookahead 3 → FULL 需 `cdiv(32,771, 768) = 43` 块；
GDN 需 15 块；合计 58 个块号（nb=1,369 的 4.24%；块号具体取值由分配器决定，下图按
新池顺序分配示意）。

```text
单卡 = 16 个 torch.zeros(1,097,653,248B)（=801,792×1,369），每个池被
[GDN组0第i层, GDN组1第i层, GDN组2第i层, FULL第i层] 共享（画一个池；其余 15 个同构）

═══ 统一主轴 = 物理块号（0…1,368；一个块号 = 每区各一格；分配器按块号发牌）═══
本请求领走块号 0…57（新池顺序分配示意）：0–42 给 FULL 组（43 个）、43–57 给 GDN 组（15 个）

区A  conv 尺（GDN 专用）字节 [0, 21,027,840)
  每格 15,360B 的算法：**conv 状态 (2,560, 3) bf16 = 2,560 × 3 × 2B**
    · 2,560 = conv_dim ÷ TP4 = (2×128×16 + 128×48) ÷ 4 —— key 方向 16 头×2
      （卷积同时吃 q,k 两路）+ value 方向 48 头，头维 128（公式 [V] mamba_utils.py:
      218-226 `conv_dim = head_k×num_k_heads×2 + head_v×num_v_heads`；头数分解
      由真机 15,360B 数值对账，标注[推断]）
    · 3 = conv_kernel − 1（因果短卷积需要保留的历史抽头数）
    · 区A 总宽 = 15,360 × 1,369 = 21,027,840B ✓
  块号 →   0 ………… 42      │ 43…47  │ 48…52  │ 53…57  │   58 ………… 1368
        ┌───────────────┬─────────┬─────────┬─────────┬────────────────┐
        │  死 ×43 格      │组0·第i层 │组1·第i层 │组2·第i层 │    空 ×1,311 格  │
        │ （FULL 块）     │ conv×5格 │ conv×5格 │ conv×5格 │                │
        └───────────────┴─────────┴─────────┴─────────┴────────────────┘
                                        ↑ 3 个 GDN 层各领各的 5 个块号、各存各的卷积抽头

区B  K/ssm 双视图共区 字节 [21,027,840, 559,340,544)
  每格 393,216B 的算法——**两种身份必须同宽，这就是对齐等式的全部意义**：
    · GDN 侧算：**ssm 状态 (12, 128, 128) bf16 = 12×128×128×2B = 393,216B**
      （12 = num_value_heads ÷ TP4 = 48÷4；128×128 = head_v × head_k，
      [V] mamba_utils.py:228-232 `temporal_state_shape=(num_v/tp, head_v, head_k)`）
    · FULL 侧算：**一个物理页的 K 跨度 = 768 token × Hk 1 × D 256 × 2B = 393,216B**
      （768 = 对齐公式选出的 attn_block_size，patch_mamba_config.py:93-97）
    · 两算同值 ⇔ `assert 512×768 == 393,216`（:95-97）——格子双重身份（K 页/ssm 态）
      字节级同宽，一个块号空间才能两种解释互斥共存
    · K 视图每格再切 6 份：6 = 768÷128（kernel 块），每份 65,536B = 128×1×256×2B
    · 区B 总宽 = 393,216 × 1,369 = 538,312,704B ✓
  块号 →   0 …………………… 42（FULL 领）      43…47   48…52   53…57（GDN 领）  58 … 1368
        ┌───────────────────────────────┬────────┬────────┬────────┬──────────────┐
        │  K 视图把每格再切 6 份：          │组0·第i层│组1·第i层│组2·第i层│     空 ×1,311   │
        │  块0→K[0..5]   块1→K[6..11]      │ ssm×5格 │ ssm×5格 │ ssm×5格 │      格        │
        │  ……  块42→K[252..257]           │(实4/记账5)×3 层同构               │               │
        └───────────────────────────────┴────────┴────────┴────────┴──────────────┘
  GDN 段每层的 5 格 = 【1 提交态 + 3 投机态 + 1 过渡缓冲】：
     提交态 = 已验证 token 前推后的 ssm 状态；投机态 ×3 = MTP 3 个 draft token
     各自前推出的状态（拒绝时回滚用）；缓冲 = 记账公式 2+spec 的余量
     （稳态真实块 = 1+3 = 4，[V] single_type_kv_cache_manager.py:1162-1165
      "save the running state at the last (1+num_spec) block"）
  两种读法（数字 ↔ 上图格子一一对应）：
    K   视图 (8,214 = 1,369×6, 128, 1, 256) bf16 → 本请求 = 43 格 × 6 = 258 逻辑块
                                                    × 65,536B/块 = 16.9MB
    ssm 视图 (1,369, 12, 128, 128) bf16            → 本请求 = 3 层 × 5 格 × 393,216B
                                                    = 5.9MB（每层稳态实块 4）

  ★ 跨池方向（"×3 层"与"×16 池"是两个正交方向）：
    组0 领的块号 43–47 **同时存在于 16 个池**：池 0 装组0第0层、池 1 装组0第1层、…
    池 i 装组0第 i 层的 ssm 状态——一个 GDN 组的 16 个层靠"同一批块号 × 16 个池"
    全覆盖；池内看到的"×3"是三个**组**的第 i 层挤在同一个池里。

区C  V 尺（FULL 专用）字节 [559,340,544, 1,097,653,248)
  每格 393,216B 的算法：**一个物理页的 V 跨度 = 768 token × Hk 1 × D 256 × 2B**
    · 与区B 同宽的原因：K 和 V 每 token·head 字节相同（都是 D×2B=512B）→ 页跨度
      必然同构；head_size_v == head_size（非 MLA）→ 形状也同构
    · 区C 只有 FULL 一种视图（GDN 不用），无双重身份、无对齐约束——宽度纯粹跟随 K 侧
    · 每格再切 6 逻辑块 × 65,536B（128×1×256×2B，与 K 完全同构）
    · 区C 总宽 = 393,216 × 1,369 = 538,312,704B ✓；三区合计 21,027,840+538,312,704×2
      = 1,097,653,248B = 801,792×1,369 ✓（与 P×nb 闭合）
  块号 →   0 ………… 42                43 ………… 57       58 ………… 1368
        ┌───────────────────────┬───────────────┬────────────────┐
        │ V[0..5] … V[252..257]  │   死 ×15 格    │    空 ×1,311 格  │
        │ = 258 逻辑块 = 16.9MB   │  ↑GDN 块的 V 格没人写            │
        └───────────────────────┴───────────────┴────────────────┘

本池占用合计 = FULL 43×(65,536×6×2) + GDN 3层×5格×(15,360+393,216) = 33.8MB + 6.1MB
              ≈ 39.9MB（记账 58/1,369 = 4.24%）
全卡 16 池 ≈ 609MiB 有效字节（同批块号在 16 池各占一页 → 64 层全覆盖）
```

**token → 字节的完整寻址链**（t=20,000；nb=1,369；设请求块表[26]=物理 7）：

```text
t=20,000
 → 请求内第 t//768 = 26 个 FULL 物理页 → 查请求块表 = 物理块号 7   （[V] 块表=请求私有）
 → 页内偏移 t%768 = 32；kernel 块内 t%128 = 32
 → 逻辑块 L = 7×6 + 32//128 = 42                                      （[VA] block_table.py:288-303）
 → K 字节 = 21,027,840 + 42×65,536 + 32×512 = 23,796,736
   V 字节 = 559,340,544 + 42×65,536 + 32×512 = 561,609,536
 → slot_mapping 值 = 42×128 + 32 = 5,408                               （[VA] block_table.py:210-229）
（Hk=1：h 维度偏移 0；若 Hk>1 再加 h×512）
```

**图一逐数字解说**（用户逐问的答案，全部有出处）：

1. **"43×6" 的 6 怎么来的**：6 = `spec.block_size(768) ÷ kernel_block(128)`
   ——[VA] model_runner_v1.py:4562 `block_size_chunk = current_kv_cache_spec.block_size
   // block_size`。**分配器按 768 token/块记账，注意力内核按 128 token 翻页**，所以
   一个物理块在块表里被展开成 6 个逻辑块（`logical = phys×6 + i`，block_table.py:
   288-303）。K 视图第 0 维因此是 `1,369×6 = 8,214` 个逻辑块，43 个物理块 × 6 =
   258 个逻辑块即本请求的 K 跨度。
2. **128 是不是 NPU 片上 UB 的大小？——不能这么说**。128 是 Ascend 后端**声明**的
   kernel block size（attention_v1.py:142-144 `get_supported_kernel_block_sizes() →
   [128]`；硬编码处 patch_mamba_config.py:58），语义是"块表/slot_mapping 的分页
   粒度"。aclnn 算子内部按多大 tile 进 UB 是 CANN 算子实现细节，不由该常数决定，
   本文不对此下结论。768 则来自对齐等式（`attn_block_size = 128 × cdiv(393,216,
   128×512) = 768`，patch_mamba_config.py:93-97）。
3. **K 视图为什么是 (1,369×6, 128, 1, 256)**：区B 的 538,312,704B 稠密字节流按
   "（逻辑块数, 每逻辑块 token 数, 每 rank KV 头数, 头维）"重新解释——128 是每逻辑
   块 token 数、**1 是 Hk/rank**（模型共 4 个 KV 头，TP4 → 每卡 1 个，真机日志
   `heads=1` 自证）、256 是 D、bf16 每元素 2B → 每逻辑块 128×1×256×2 = 65,536B。
   代码：get_kv_cache_shape 返回 `(2, nb×6, 128, Hk, 256)`（model_runner:4563-4568 +
   attention_v1.py:104-112），k_cache 取 shape[1:]（:4637）。
4. **ssm 与 GDN 的关系；ssm 存什么**：GDN（Gated DeltaNet 线性注意力层）不存
   per-token K/V，而是存一个**定尺寸**的循环状态——每层两份：
   - **conv state (2560, 3) bf16 = 15,360B**（区A）：因果短卷积的最后 3 个抽头，
     形状 = (conv_dim/TP4, conv_kernel−1)（[V] mamba_utils.py:218-226）；
   - **ssm state (12, 128, 128) bf16 = 393,216B**（区B）：DeltaNet 的递推状态矩阵
     S ∈ R^{Hv × d_v × d_k}，形状 = (num_v_heads/TP4 = 48/4 = 12, head_v=128,
     head_k=128)（[V] mamba_utils.py:228-232 `temporal_state_shape =
     (divide(num_v_heads, tp_world_size), head_v_dim, head_k_dim)`）。
   线性注意力的全部历史被压缩进这一个固定矩阵 → **GDN 存储不随序列长度增长**
   （与 FULL 的线性增长相对；这也是 chunked prefill 必须按块对齐的原因：状态只能在
   整块边界 checkpoint）。
5. **ssm 视图为什么是 (1,369, 12, 128, 128)**：同一区B 字节流换一把尺子——
   `(块数 nb, 单块状态形状)`，每格 12×128×128×2B = 393,216B = **恰好 6 个连续
   K 逻辑块**（6×65,536）——对齐等式 512×768==393,216 的存在目的。代码：
   model_runner:4705-4713 `target_shape=(num_blocks, *shape)` 顺序稠密切片。
   两视图同一字节；格子的身份（K 还是 ssm）由块号归属决定（§2.5.2）。
6. **"我以为只存 GDN 三层的 state？"——对**：一个池住着 3 个 GDN 层（三个组各一
   层）+ 1 个 FULL 层；每个 GDN 层存**自己的** conv+ssm 两份，占**自己领的块**。
   15 的来源 = **记账公式**：每组 align 模式 `2 + num_spec(3) = 5` 块
   （kv_cache_interface.py:629-631）× 3 组；**实际驻留**真实块 = 每组块表
   cdiv(32,768,768)+3 = 46 项中仅最后 `1+3=4` 个真实块（其余 42 个为共享
   null_block 占位，不占块号），3 组 × 4 = **12 个真实块/池**（single_type:
   1142-1165）。图一按记账口径画 15（保守上界）。
7. **区A 的 FULL 格装什么？——什么都不装，纯死字节**。conv 视图只给 GDN 层切
   （model_runner_v1.py:4682-4714），FULL 层只拿 (k_cache, v_cache)（:4679-4680），
   没有任何指向区A 的张量；但统一页记账下块号必须三区同价（801,792B），故 FULL
   块的区A 格（15,360B，~1.9%/块）成为无人可寻址的死字节——官方注释自证：区A 在
   K/V 视角的名字就是 "(kv_padding)"、区C 在 mamba 视角叫 "(mamba_padding)"
   （:4701-4704）。对称地，GDN 块的区C 格（393,216B，~49%/块）同样死置。32K 请求
   量化：FULL 死格 ≈ 43×16×15,360 ≈ 10.1MiB/卡；GDN 死格 ≈ 15×16×393,216 ≈
   89.8MiB/卡。死格随块号归属变化（FREE 后改派 GDN 组即被 conv 复用）。
8. **为什么 ssm 要和 K 混存，而不是"K+V 同区、ssm 独占一区"？——三个层次**：
   （先定义**页宽** = 一个块号在三区各占一格的字节总和 = 它的记账单价 =
   代码里的 page_size_bytes；nb = 预算 ÷ 页宽 ÷ 16，故页宽直接决定同预算能买
   多少块号。现状页宽 801,792B = conv 格 15,360 + B 格 393,216（K/ssm 二选一）
   + C 格 393,216——ssm 无专属格，免费搭乘 K 的字节。）
   a. **先算账（用户的方案更贵）**：不重叠 ⇒ 每块号需四格（conv/K/V/ssm 各一）⇒
      页宽 = 15,360 + 393,216×3 = **1,195,008B**（现 801,792B = 三格，+49%；
      差值恰为一格 393,216B）。同预算（16.36GiB）下
      nb = 17,562,451,968÷1,195,008÷16 = **918**（现 1,369，**-33% 容量**）；且
      每块的浪费暴涨——FULL 块浪费 408,576B（34.2%），GDN 块浪费 786,432B
      （65.8%）（现状分别 1.9% / 49%）。**混存=让"FULL 的 K 格"与"GDN 的 ssm 格"
      是同一批字节，一个块号才能只花 801,792 就当两种身份用**。
   b. **框架约束（为什么必须等宽）**：上游整套 hybrid 机制建立在"单一 BlockPool、
      单一 num_blocks、每个 KVCacheTensor 被 4 组各一层共享、统一页记账
      （memory_per_block = page×16）"之上（分组代码 kv_cache_utils.py:1142-1195
      的存在意义就是让 48+16 层挤进 16 个等宽池、避免逐组 padding——:1146-1157
      的 FIXME 注释原文讨论的正是这个）。一个块号要能发给**任何**组，页宽就必须
      全局唯一 ⇒ GDN 页（conv+ssm 跨两区）与 FULL 页（K+V 跨两区）必须**等宽**；
      而等宽 + 总量不膨胀的唯一办法就是让 ssm 字节与 K 字节重叠（块号互斥保证
      两种写者永不同时碰同一格，等宽 assert 保证视图不撕裂）。上游 vllm 走同一条
      路（mamba_page_size_padded = attention 页宽），非 vllm-ascend 独创。
   c. **几何必然（为什么恰好叠在 K 不是 V）**：GDN 的两个状态视图必须从 offset 0
      起**连续**切（conv→ssm 顺序循环，model_runner:4705-4713）；FULL 的 K|V 必须
      在**尾部**连续切（raw[15,360nb:] 前 K 后 V，:4574-4577）。一头一尾锚定后，
      两段 span 的交集被几何强制为 [15,360nb, 408,576nb) = **恰好 K 那一半**。
      若把 FULL 内部顺序换成 [V|K]，ssm 就会叠在 V 上——功能完全等价；叠在 K 纯属
      切片顺序的偶然，不是语义选择（官方注释 "tensor2: [k, ssm, ...]" :4703）。
      （前提：ssm 宽 393,216 == K 半宽，即对齐等式；不满足则交集错位、视图撕裂。）

#### 2.5.5 图二：当前 OSCAR packed×2（int8, block=1,536）——同一请求，同一张卡

池的三区**宽度一字不变**（nb 也不变：page 仍 801,792B、÷16 仍一样——1,369 块号；
packed 的收益在"每块装 1,536 token 而非 768"，不在于更多块）；变的只有 FULL 视图
的"每逻辑块字节"（65,536→32,768）与"每页 token"（768→1,536）：

```text
区A（不变，[0, 21,027,840)）：15 个 GDN conv 格 + FULL 块死置（同图一）

区B（K/ssm 共区 [21,027,840, 559,340,544)；一格 = 12 个 32,768B 逻辑块）
  ┌ K[0..11] ┬ K[12..23] ┬─…─┬ K[252..263] ┬ ssm[22] ┬─…─┬─────────────┐
  │←物理页0   │←物理页1    │ … │←物理页21     │←GDN     │ … │ 空(至块1368) │
  └──────────┴───────────┴─…─┴──────────────┴─────────┴─…─┴─────────────┘
   K 视图 (1,369×12, 128, 1, 256) int8：FULL 只需 cdiv(32,771, 1,536) = 22 块
     = 264 逻辑块 × 32,768B = 8.65MB（图一的一半）
   ssm 视图 (1,369, 12,128,128) bf16：GDN 15 格不变（GDN 与 cache_dtype 无关）
   ★ OSCAR 只写每 token·head 512B（K 槽 256B 的前 96B + V 槽 256B 的前 64B）

区C（V 格 [559,340,544, 1,097,653,248)）：V 视图 (1,369×12,128,1,256) int8，
   22 物理页；GDN 块死置同图一

MTP 草稿层（第 17 个 attention 层）：不在池内写任何东西
 → BF16 影子池（页外，形状 (1,369×12,128,1,256)×2 ≈ 2.15GiB/rank，块号空间与主池一一对应）
   ← [W] oscar_ascend/mtp_shadow.py（DESIGN-E 方案 A）
   ⚠️ 影子池在引擎算完 nb 之后才懒分配 → 花的是 utilization=0.9 之外的 10% 余量
   （32GiB×10% = 3.2GiB ≥ 2.15GiB ✓，但吃掉安全垫，nb 不因此变小）

该请求在本池占用 = 22×786,432 + 15×408,576 = 23,430,144B ≈ 22.3MiB（37/1,369 = 2.70%）
→ 与图一同池对比：同一 32K 请求，FULL 侧块号 43→22（-49%），即"密度 ×2"的来源；
  剩余块号 1,311→1,332 → 同池可容纳的并发 32K 请求翻倍
```

同一 token t=20,000 的寻址链（packed 版，设请求块表[13]=物理 7）：

```text
t=20,000 → t//1,536 = 13 → 块表[13] = 物理 7；t%1,536 = 32；t%128 = 32
 → 逻辑块 L = 7×12 + 32//128 = 84
 → K 字节 = 21,027,840 + 84×32,768 + 32×256 = 23,788,544
   ★ OSCAR K 槽内写：[+0..+8) meta ∥ [+32..+96) K idx（共 96B，槽其余 160B 闲置）
 → V 字节 = 559,340,544 + 84×32,768 + 32×256 = 562,101,248
   ★ OSCAR V 槽内写：[+0..+64) V idx（共 64B，槽其余 192B 闲置）
 → slot_mapping 值 = 84×128 + 32 = 10,784
```

#### 2.5.6 三点诚实备注

1. 图中块号（0..42=FULL、43..57=GDN）是**顺序分配示意**；真实块号由 BlockPool
   自由链表决定（复用/前缀命中会打散），布局语义与块号取值无关。
2. GDN 块在区C 的死置（~49%/块）与 FULL 块在区A 的死置（~1.9%/块）是 vllm-ascend
   统一页记账的固有代价（上游同款 "Padding mamba page size"），与 OSCAR 无关。
3. 真机 16K 请求显示 `GPU KV cache usage: 4.1%`，本节公式给出 22+15=37 块
   ≈2.7%（nb≈1,369）——差的 ~1.4% 属引擎 usage 统计口径（含预留/水印类），
   未逐行核验，不硬凑数字。

#### 2.5.7 演讲主线：从一张 910B4 到一条 32K 请求（五步完整叙事）

> 应用户演讲需求整理：每一步回答上一步留下的问题，所有数字均有前文出处。

**第 1 步 · 需求——这个模型有"两种记忆"**
Qwen3.5-27B = 48 个 GDN 线性注意力层 + 16 个 FULL 全注意力层（+1 MTP 草稿层）。
FULL 层的记忆 = **每个 token 的 K 和 V**：D=256、bf16 → 每 token·head 1,024B，
随序列长度**线性增长**；GDN 层的记忆 = **一个固定大小状态**：conv 抽头 15,360B +
ssm 状态矩阵 (12,128,128)bf16 = 393,216B，**永不增长**。两种记忆、两个尺寸，
要住进同一片显存。

**第 2 步 · 结构——16 个池，每池三个区，页宽 801,792B**
把 48+16 层按"同型分桶、交错分组"分成 4 组（每组 16 层：3 个 GDN 组 + 1 个 FULL
组），于是只需要 **16 个池**，第 i 个池住 [GDN组0第i层, GDN组1第i层, GDN组2第i层,
FULL第i层]——4 层共享，避免逐组 padding。每个池是一根 `torch.zeros`，按**三个稠密
区**排布：区A（conv 格）、区B（K/ssm 共用格）、区C（V 格）。
**一个块号在每区各占一格**，三格相加 = 它的"页宽"（记账单价）：

  页宽 = 区A 格 15,360 + 区B 格 393,216 + 区C 格 393,216 = **801,792B**

为什么三格就够、ssm 不用第四格？因为对齐等式 512×768==393,216 保证"一份 ssm
（393,216B）"与"一页 K（768×512B）"**字节级同宽**——区B 的每格可以有两种身份：
块号归 FULL 就是 K，归 GDN 就是 ssm。同一批字节，两种解释，互斥使用。

**第 3 步 · 容量——nb = 预算 ÷ 单价**
一张 910B4（32GiB）按 gpu-memory-utilization=0.9 留 28.8GiB 水位，扣权重
（27B w8a8 ÷ TP4 ≈ 7GiB）与运行/激活（≈6GiB），KV 预算 ≈ 16.36GiB（真机日志）。
一个块号的全卡单价 = 页宽 × 16 池 = 801,792×16 ≈ **12.23MiB**（它必须在 16 个池
里同步各占一页）。所以：

  **nb = 16.36GiB ÷ 12.23MiB ≈ 1,369 个块号**（全局唯一 BlockPool，引用计数发牌）

**第 4 步 · 使用——一条 32K 请求进场**
请求找分配器领块号：FULL 组要 cdiv(32,768+3, 768) = **43 个**（装 token 的 K/V）；
每个 GDN 组要 2+3 = **5 个**（1 提交态 + 3 投机态 + 1 缓冲；MTP 拒绝回滚用），
3 组共 **15 个**。合计 **58 个块号**，身份决定用法：

  · 块 0–42（FULL 领）：区B 格当 K——每格再切 6 个 128-token 逻辑块
    （块 p → K[6p..6p+5]）；区C 格当 V（同构再切 6）；区A 格死置（1.9% 浪费）
  · 块 43–57（GDN 领）：组0/组1/组2 各 5 格——区A 装 conv、区B 装 ssm；
    区C 格死置（49% 浪费，但每层只驻留 4-5 格，绝对量小）
  · 寻址示例：token 20,000 → 请求块表[26]=物理 7 → 逻辑块 7×6+0=42
    → K 字节 = 21,027,840 + 42×65,536 + 32×512 = 23,796,736

**第 5 步 · 总账与取舍**
本池占用 = FULL 43×786,432 + GDN 15×408,576 ≈ 39.9MB（58/1,369 = 4.24% 块号）；
全卡 16 池 ≈ 609MiB。设计的三个聪明处：① ssm 搭乘 K（页宽省 1/3，否则 1,195,008B、
容量 -33%）；② 4 组共享 16 池（免逐组 padding）；③ 单一块号空间（调度/前缀缓存/
MTP lookahead 统一）。代价：死格（统一记账的学费）。延伸：OSCAR packed×2 把区B/C
每格从装 768 token 变 1,536 token（每 token·head 1024B→512B），同一 32K 请求
FULL 侧 43→22 块号——同池并发翻倍（§2.5.5 图二）。

---

## 3. 模式①（当前插件版）：布局不动，槽内重解释

```text
┌─ 存放 ──────────────────────────────────────────────────────────────────┐
│ 引擎建池：16 个 torch.zeros(P×nb, int8)，P=801,792B/页（BF16 页几何）      │
│   [VA] model_runner_v1.py:4124；页几何由 patch_mamba_config.py:95-97 锁死  │
│   （assert 512×768==393,216：K 条宽 = GDN ssm 条宽，插件无法参与）          │
│        ↓                                                                 │
│ 手术：Attention.__init__ 后 hook 换 impl.__class__（不碰池/页表/块大小）    │
│   [W] plugin.py:111                                                       │
│        ↓                                                                 │
│ 写：每步新 token → 旋转 matmul → sort 分位裁剪 → 量化打包                  │
│   → 写进原生 K 槽(512B) 的前 96B + V 槽(512B) 的前 64B = 逻辑 160B 槽      │
│   [W] backend.py:108-146；槽内字节图 format.py:25-33；散写 store_kernel.py:54-95 │
│   → 窗口 token(sink/recent) 再往页外 staging arena 双写一份 BF16            │
│   [W] backend.py:316-346（~8.4MB/层，16 层 ≈ +134MB/rank）                 │
│   ★ 池的其余 84% 字节永远没人写 → KV usage 与原生一模一样（真机日志 4.1%）  │
├─ 读取 ──────────────────────────────────────────────────────────────────┤
│ 本部署(MTP) 每步 attn_state=SpecDecoding → 全部落入 prefill 分支：          │
│   全前缀(16k+) INT2 反量化物化成 fp32 [C,1,256]×2 → 逆旋转 matmul×2        │
│   → staging splice → SDPA                                                 │
│   [W] backend.py:202-205（路由）、:253-293（执行）、dequant_kernel.py:105-116 │
│ 设计中的 fused INT2 decode + 窗口 LSE 合并 = 死代码（DecodeOnly 永不出现）   │
│   证据链：[VA] model_runner_v1.py:1477-1505 × backend.py:202（ANALYSIS-C §5.3）│
└─ 账本 ────────────────────────────────────────────────────────────────────┘
   显存：0（净 +134MB staging）；稳态写：160B+1024B(双写)=1184B > 原生 1024B；
   读：每步全前缀物化+逆旋转 ≈ 原生 fused 读的 10~15× 带宽 → warmup 110s 主因。
```

**一句话**：把"压缩"做在了**写的内容**上，没做在**存的结构**上；读路径又因 MTP 路由
绕开了唯一的压缩域算子。两头都没兑现。

---

## 4. 模式②（OSCAR vLLM PR#46774）：分配期缩池 + 压缩域读

```text
┌─ 存放 ──────────────────────────────────────────────────────────────────┐
│ 用户开 --kv-cache-dtype oscar_int2                                        │
│   [PR] oscar_attn.py:60（supported_kv_cache_dtypes）、config.py:25（preset）│
│        ↓                                                                 │
│ 池形状在分配期就按压缩槽建：get_kv_cache_shape 返回                       │
│   (num_blocks, block_size, num_kv_heads, slot_size_aligned)  ← uint8 池    │
│   [PR] oscar_attn.py:87-103（模块 docstring :11-14 同款布局说明）          │
│   slot = key_packed + value_packed：D=128 → 72B/槽（vs BF16 512B ≈ 7.1×）  │
│   [PR] config.py:120-128；实测断言 tests/quantization/test_oscar.py:71     │
│   （按同公式推到我们 D=256 → 136B/槽 vs 1024B ≈ 7.5×）                     │
│   → 页字节变小 → 引擎同一显存预算推出更多 num_blocks = 显存收益在"建池"这步 │
│        ↓                                                                 │
│ 写：与 attention forward 分离的独立 custom op（每层每步一次）               │
│   forward_includes_kv_cache_update=False  [PR] oscar_attn.py:58           │
│   unified_kv_cache_update → do_kv_cache_update → 旋转+裁剪+oscar_store     │
│   [V-host] model_executor/.../attention.py:690；[PR] oscar_attn.py:338-345 │
│   store 内核 grid=(N×H,)，每 program 写满一个压缩槽                        │
│   [PR] triton_oscar_store.py:99-112,160-166                               │
├─ 读取 ──────────────────────────────────────────────────────────────────┤
│ decode：压缩域三段窗口——sink/recent 用页外 BF16 staging（owner-tag 防错配）│
│   + 中段 INT2 走 split-KV triton 内核（块表平移跳过 sink 页，内核零改动）   │
│   + 两段 LSE 加权合并                                                     │
│   [PR] oscar_attn.py:253-336（staging）、:618-685（窗口+平移）、:745-750（合并）│
│ prefill 首 chunk：原生 flash_attn_varlen（不读缓存）                        │
│ prefill 续读：前缀 INT2 全量反量化物化 → 逆旋转 → staging splice → varlen   │
│   [PR] oscar_attn.py:515-518                                              │
└─ 边界 ────────────────────────────────────────────────────────────────────┘
   仅 eager（_cudagraph_support=NEVER，[PR] oscar_attn.py:134-141）；
   spec 解码显式不放宽（supports_spec_as_decode=False，:141）；
   hybrid/Mamba：未处理——hybrid 模型禁用边界跳层后无兜底（config.py:178-192）。
```

---

## 5. 模式③（TurboQuant @ upstream vLLM v0.23.0）：与②同骨架，工程更完整

```text
┌─ 存放 ──────────────────────────────────────────────────────────────────┐
│ 用户开 --kv-cache-dtype turboquant_{k8v4|4bit_nc|k3v4_nc|3bit_nc}          │
│   [V] config/cache.py:19-35（CacheDType 字面量）                           │
│   [V] turboquant/config.py:20-41（4 preset + PPL 标注 :71-75）             │
│        ↓                                                                 │
│ 平台"淘汰制"选 backend：FA/FlashInfer 只认 auto/fp16/bf16 被 dtype 淘汰，  │
│   TQ 剩者为王（无需改 selector）                                           │
│   [V] platforms/cuda.py:131-147 + flash_attn.py:183-193 +                 │
│       turboquant_attn.py:163-167（supports_kv_cache_dtype）                │
│        ↓                                                                 │
│ 池形状在分配期按压缩槽建（与②同型，uint8 无前导 2 维）：                    │
│   get_kv_cache_shape → (nb, bs, Hk, slot_size_aligned)                     │
│   [V] turboquant_attn.py:139-161；页字节公式重定义在 Spec：                 │
│   real_page_size_bytes = block_size × num_kv_heads × tq_slot_size          │
│   [V] kv_cache_interface.py:327-349（TQFullAttentionSpec）                 │
│   槽内 = [key_packed | value_packed]：K=Hadamard 旋转+Lloyd-Max 查表码      │
│   （离线解 N(0,1/d) 的 2^bits centroids，centroids.py:82-86）+vec_norm fp16；│
│   V=均匀量化+scale/zero fp16（config.py:128-174）                          │
│   D=128 k3v4_nc → 118B/槽（vs 512B ≈ 4.3×）；推到 D=256 → 230B（≈4.5×）    │
│        ↓                                                                 │
│ 写：与 forward 分离的独立 custom op（同②）                                  │
│   forward_includes_kv_cache_update=False  [V] turboquant_attn.py:93-94     │
│   do_kv_cache_update → 旋转走外部 cuBLAS GEMM → 单 triton kernel 写满一槽   │
│   [V] turboquant_attn.py:363-386, :537-557；store.py:412-447（grid=(NH,)）  │
├─ 读取 ──────────────────────────────────────────────────────────────────┤
│ decode：压缩域 split-KV——q 先旋转到 Hadamard 域，内核内位解包+centroids     │
│   查表 gather + 在线 softmax（K 全程不解旋转、不物化 BF16）                 │
│   [V] triton_turboquant_decode.py:552-611（grid=(B,Hq,32), stage2 复用     │
│   社区内核；centroids gather :192-197）                                    │
│ prefill 首 chunk：原生 flash_attn_varlen（同②）  [V] turboquant_attn.py:576-588 │
│ ★ 小续算（新 token ≤128）：不物化——直接复用 decode 内核，合成 seq_lens     │
│   做 causal 掩码                                                          │
│   [V] turboquant_attn.py:69（阈值 128）、:666-692                          │
│ 大续算（>128）：才全量 dequant 物化→逆旋转→flash_attn_varlen                │
│   [V] turboquant_attn.py:712-834                                          │
├─ hybrid/Mamba（#39931 修法，我们同构模型的官方先例）───────────────────────┤
│ TQFullAttentionSpec 注册进 FullAttentionManager 分组（与普通 FULL 同池记账）│
│   [V] single_type_kv_cache_manager.py:1383-1387                           │
│ hybrid 页协调：TQ packed 页用 TQ 专属公式（标准公式会算错页宽触发断言），    │
│   skip 层页取 lcm，Mamba 页 pad 到 attention 页                            │
│   [V] platforms/interface.py:573-608（lcm/packed 公式）、:618-699（pad）    │
│ 被排除层（mamba/sliding）自动回退 "auto" 原生 dtype → 各自原生池            │
│   [V] attention.py:252-268；hybrid 层识别 config.py:193-202                │
└─ 边界 ────────────────────────────────────────────────────────────────────┘
   无 sink/recent 窗口/staging（纯全量压缩缓存；grep 零命中，§证据索引）；
   cudagraph=UNIFORM_BATCH（比②的 NEVER 强一档）[V] turboquant_attn.py:199；
   spec-as-decode 同样 False（:203）。
```

---

## 6. 三模式对照 + Role Model 判决

### 6.1 对照表（一行一维度，人话）

| 维度 | ① 插件版（现状） | ② OSCAR PR | ③ TurboQuant（upstream） |
|---|---|---|---|
| 池在分配期变小？ | **否**（原生 BF16 池原样，84% 字节浪费） | **是**：槽宽进 `get_kv_cache_shape`（oscar_attn.py:87-103） | **是**：槽宽进 `TQFullAttentionSpec` 页字节公式（kv_cache_interface.py:327-349） |
| 每槽字节（我们 D=256 口径） | 160B 写在 1024B 里（有效比 15.6%） | ~136B/1024B ≈ **7.5×**（公式推算） | k3v4_nc ~230B/1024B ≈ **4.5×**；k8v4 ~2.6×（公式推算；PPL 标注 config.py:71-75） |
| 引擎记账（usage/nb/块表） | 全按原生（假象：没省） | 引擎原生按小页记账（真省） | 同② |
| 写路径 | forward 内联；旋转+sort+散写+staging 双写 | 独立 custom op + 单内核写满槽 | 同②（旋转在 cuBLAS，内核只做量化打包） |
| decode 读 | **死代码**（MTP 路由进 prefill 分支） | 压缩域 split-KV triton + 三段窗口 LSE 合并 | 压缩域 split-KV triton（centroids 查表），无窗口 |
| 续读（前缀命中/chunk 后续） | 每步**全前缀物化**+逆旋转（10-15× 放大） | 全量物化（同左，仅 prefill 时发生） | **≤128 新 token 直接复用 decode 内核不物化**；>128 才物化 |
| hybrid（16 full + 48 GDN） | 能跑（因为什么都没动） | **未处理**（hybrid 禁用边界跳层，无 mamba 分支） | **已解决**（#39931：Spec 并组 + lcm 页协调 + mamba 页 pad） |
| Mamba 页对齐锁怎么破 | 没破（所以池缩不了） | 未涉及 | attention 页自己定义 + **mamba 页 pad 到 attention 页**（interface.py:618-699） |
| 投机解码/MTP | verify 步落错分支（110s 主因） | 显式不支持（supports_spec_as_decode=False） | 同样 False；但小续算复用 decode 内核天然覆盖 verify 形态 |
| CUDA Graph | NEVER（--enforce-eager） | NEVER | UNIFORM_BATCH |
| 在 vllm-ascend 可用？ | 是（本插件） | 否（CUDA 生态，且 PR 未合入） | **否**（官方 vllm-ascend main 零 TQ，2026-09-04 实测） |

### 6.2 判决：Role Model = TurboQuant 的工程蓝图 + OSCAR PR 的后端本体与数值配方

**为什么 TQ 是更合适的 role model（按分量排序）**：

1. **它解决了我们最缺的"分配期缩池"且给出 hybrid 修法**。我们要的模型（16 full +
   48 GDN + Mamba 页对齐锁 `512×768==393,216`）正是 #39931 处理的形态：TQ 让 attention
   页按自己的 packed 槽宽定义、Mamba 页 pad 到 attention 页——**方向与 vllm-ascend
   `patch_mamba_config` 现在的做法（attention 页对齐到 mamba 页）相反但机制相同**，
   这是"破对齐锁"的官方参考答案。OSCAR PR 完全没碰 hybrid。
2. **读路径分级直击我们 110s 的根因**。`_CONTINUATION_DECODE_THRESHOLD=128`：新 token
   ≤128 时直接复用压缩域 decode 内核、不物化全前缀（turboquant_attn.py:69,:666-692）——
   我们 MTP verify 每步恰是 4 token，正落在这个桶里；TQ 从设计上就没有"每步全前缀
   反量化"这条路径。
3. **工程完备度高且本地就有全部参考代码**：dtype 淘汰制接入（不动 selector）、独立写缝
   custom op、cudagraph UNIFORM_BATCH、preset 带 PPL 代价标注、统一 workspace 管理。
   OSCAR PR 是未合入的分支快照，集成侧文件（CacheDType/selector 改动）甚至不在快照里。

**OSCAR PR 仍然要当"数值配方库"**：U·H·P_br 旋转、sort 分位裁剪、per-vector INT2
槽公式、sink/recent BF16 窗口 + owner-tag staging——这些我们已经移植且真机验证过字节
级正确（store 内核），TQ 没有窗口机制。

### 6.3 落地映射：两个 role model 的件套 → vllm-ascend 接缝（对称，零侵入边界标注）

**表 A：OSCAR PR 六件套**（"OSCAR 后端本体该长什么样"——classvar/写缝/内核/窗口，
我们已移植约 90%）：

| OSCAR PR 件套 | vllm 位置（[PR]/宿主） | vllm-ascend 对应接缝 | 零侵入插件内可做？ |
|---|---|---|---|
| backend 选择：`--attention-backend OSCAR` + `supported_kv_cache_dtypes=["oscar_int2"]` | oscar_attn.py:60,64-65 | platform `get_attn_backend_cls`；vllm-ascend 无 dtype 淘汰制 | ✗（改源码；插件用 `impl.__class__` 手术等价绕过，已在做） |
| 槽宽进分配：`get_kv_cache_shape→(nb,bs,Hk,slot)` + `effective_head_size=slot//2` 记账 | oscar_attn.py:87-103；config.py:124-128 | model_runner_v1.py:4559-4577 切分 + patch_mamba_config 页公式（对齐锁） | ✗（改源码） |
| hybrid 页协调 | **缺失**（config.py:178-192 hybrid 禁用；无 mamba 分支） | 需自创 → 只能借 TQ #39931 模板（interface.py:573-699） | ✗（且 PR 无作业可抄） |
| 独立写缝：`forward_includes_kv_cache_update=False` + `do_kv_cache_update` custom op | oscar_attn.py:58,338-345；宿主 attention.py:690 | attention_v1.py:1529-1534 forward 内联写 | ✓（impl 子类等价，已在做） |
| 压缩域 decode：split-KV stage1/2 + 三段窗口 LSE 合并 | triton_oscar_decode.py:267-335；oscar_attn.py:618-750 | impl 读路径（含 SpecDecoding 分支接线） | ✓（内核已 port+stage2 已修；差分支接线） |
| 续读物化 + staging splice | oscar_attn.py:515-518,317-336 | `_prefill_attention` | ✓（已在做，同 PR 语义） |
| eager-only（`_cudagraph_support=NEVER`） | oscar_attn.py:134-141 | vllm-ascend 已 `--enforce-eager` | ✓ 兼容 |

**表 B：TQ 五件套**（"怎么接进引擎 + hybrid 共存"）：

| TQ 件套 | vllm 位置 | vllm-ascend 对应接缝 | 零侵入插件内可做？ |
|---|---|---|---|
| CacheDType 字面量 + dtype 淘汰制 | cache.py:19-35 / cuda.py:131-147 | backend 选择/platform 层 | ✗（改源码） |
| TQFullAttentionSpec 页字节公式 | kv_cache_interface.py:327-349 | `get_kv_cache_spec`/patch_mamba_config（对齐锁所在地） | ✗（改源码） |
| lcm 页协调 + mamba 页 pad | interface.py:573-699 | patch_mamba_config.py:58-117（同功能反方向） | ✗（改源码） |
| 独立写缝 do_kv_cache_update | attention.py:690 | impl 子类可等价实现 | ✓（我们已在做） |
| 压缩域 decode + 小续算(≤128)复用 decode 内核 | triton_turboquant_decode.py / turboquant_attn.py:666-692 | impl 读路径（SpecDecoding 分支） | ✓（**当前最值得先做的一件**：MTP verify=4 token 正落在 ≤128 桶） |

**组合判决**：给 vllm-ascend 写"真 OSCAR 后端"上游 PR 时，两份参考缺一不可——
**PR#46774 告诉你 OSCAR 后端本体长什么样**（表 A 第 1/2/4/5/6/7 行，即我们已移植的
那部分 + 分配期槽宽），**TQ#38479/#39931 告诉你怎么把它接进引擎并和 Mamba 共存**
（表 B 第 1/2/3 行：dtype 淘汰、Spec 页公式、lcm+pad——恰是 PR 缺失、而我们的混合
模型必需的三件）。零侵入插件期内两张表的可兑现行完全一致：写缝（已做）+
读路径接线（待做，且 TQ 的 ≤128 复用 decode 是现成设计）。

---

## 7. 证据索引

**模式①（本仓 [W]，HEAD 63132db）**：plugin.py:88-132(111)；backend.py:108-146,187-199,
202-205,253-293,316-346；format.py:25-33；store_kernel.py:54-95；dequant_kernel.py:105-116；
serve_oscar.sh:32-37,48-70。参考树 [VA]：model_runner_v1.py:4124,1477-1505；
patch_mamba_config.py:95-97；attention_v1.py:1529-1534。

**模式②（[PR] oscar-vllm-pr46774@57286d5d）**：oscar_attn.py:11-14,54-60,87-103,134-141,
253-336,338-345,515-518,618-685,745-750；config.py:25,120-128,154-163,178-192；
triton_oscar_store.py:99-112,160-166；triton_oscar_decode.py:21,58,267-292,320,335；
tests/quantization/test_oscar.py:71。宿主接缝（vllm@0fc695fc）：
model_executor/layers/attention/attention.py:690；v1/attention/backend.py:66。

**模式③（[V] vllm@0fc695fc）**：config/cache.py:19-35；
model_executor/layers/quantization/turboquant/config.py:20-41,46-65,71-75,128-174,193-202,235-260；
turboquant/centroids.py:31-49,82-86；v1/kv_cache_interface.py:327-349；
v1/attention/backends/turboquant_attn.py:69,93-105,112-121,139-167,199,203,363-386,
537-557,576-588,638-692,712-834；v1/attention/ops/triton_turboquant_store.py:144-211,278-325,412-447；
v1/attention/ops/triton_turboquant_decode.py:178-206,223-306,321-455,519-611；
v1/core/single_type_kv_cache_manager.py:1383-1387；platforms/interface.py:573-699；
platforms/cuda.py:131-147,270-290；platforms/rocm.py:411-421；platforms/xpu.py:64-68；
v1/attention/backends/registry.py:98；utils/torch_utils.py:46-49；
engine/arg_utils.py:1780-1789。

**归属/版本取证（2026-09-04）**：
- vllm-ascend 无 TQ：本地 v0.23.0 树 grep 0 命中；官方 main `4fe7ddb` 稀疏克隆源码
  grep 0 命中；release notes（raw 271,793B）grep 0 命中。
- varjoranta/turboquant-vllm（`c8a7e0a`）README:18（KV 已上上游 #38479）、:126-137
  （模型表：Weight quant 列 vs Legacy KV 列）、:81/:135（Qwen3.6-27B：权重 Works
  v0.13.5 / KV Untested / 15.4GB(3.6×)=权重）、:160（论文归属说明）。
- vllm PR#39931（2026-05-05 合入，`4f2af1a`）：hybrid 修复内容（仅 FULL 层量化、
  TQ packed 页公式、排除层回退、ROCm 包装）来自 PR 页面摘要。

**§2 方块图专项（本轮新增核验）**：
- [VA] attention_v1.py:142-144（kernel 块=[128]）；block_table.py:288-303（6 逻辑块/页）。
- [V] platforms/interface.py:565-699 全文亲读（TQ packed 页公式 :573-608、lcm :603-607、
  kernel 对齐 :644-648、attn_block_size 公式 :649-655、mamba pad :685-699）——②③ 的
  1,792/412,160/3,584 等数字全部由这些公式代入 Hk=1/D=256/slot=230 计算得出。
- [V] config/cache.py:46（DEFAULT_BLOCK_SIZE=16）；turboquant_attn.py:110-113（kernel 块表）。
- [V] turboquant/config.py:128-174 亲读（key/value/slot 公式原文，:150-156 value +4B、
  :166-174 偶对齐）；mamba_utils.py:218-233（conv_dim/temporal 形状公式）。
- [PR] oscar_attn.py:54-69 亲读（classvar、get_name、kernel 块表 [16,32,64,128]）；
  config.py:99-132 亲读（key/value/slot/effective_head_size=slot//2 原文）。

---

*分析模式产物（零代码改动）。本文所有"公式推算"（②的 136B、③的 230B/390B）均由
所引 config 公式对 D=256 直接计算得出，已在正文标注"推算"；其余数字全部来自所引
file:line 的原文或真机日志。*
