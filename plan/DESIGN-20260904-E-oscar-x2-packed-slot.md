# DESIGN-20260904-E — OSCAR ×2 packed 槽方案：对齐 PR 消灭槽浪费，零 vllm-ascend 源码改动

> 目标（用户原话归纳）：跟 OSCAR vLLM PR 对齐（不要槽位浪费），**必须兼容 vllm-ascend
> 的总体 KV 管理框架、不能大改动、必须兼容 Qwen3.5 固有的 3+1（3 GDN + 1 FULL 共池）
> 模式**。承接 `ANALYSIS-20260904-D` §2 的三池方块图（现状利用率 15.6%）。
> 本文所有机制断言均已逐行核验（file:line 见 §6），数字全部可复算。

---

## 0. 一句话方案

**给 serve 加一个参数 `--kv-cache-dtype int8_per_token_head`**（upstream CacheDType
合法字面量，映射 torch.int8，dtype 尺寸 1 字节）。vllm-ascend 的页几何函数
`patch_mamba_config` 会**自动**把 FULL 层几何从"768 token/页 × 1024B/token"改成
"**1,536 token/页 × 512B/token**"，而**页字节（801,792B）、三大条结构、16 池、GDN
状态布局、断言等式全部原样成立**——OSCAR 的 160B 槽从"塞在 1024B 里用 15.6%"变成
"塞在 512B 里用 31.3%"，**FULL token 密度 ×2 = 真·显存收益（KV usage 减半、并发容量
翻倍）**，且 FULL 层内核的槽内偏移（96B/64B）完全不用改。唯一的硬约束是 MTP 草稿层
（§3，推荐"影子池"解）。

---

## 1. 为什么是这一个参数：页几何的完整推导（每步带证据）

`patch_mamba_config` 的每-token 字节数**直接由 `cache_config.cache_dtype` 驱动**：

```python
# [VA] patch/platform/patch_mamba_config.py
:53-56  kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]   # int8_per_token_head → torch.int8
:91     attn_single_token_k_page_size = attn_head_size(256) × Hk(1) × get_dtype_size(int8=1) = 256B
:92     attn_token_page_size = 2 × 256 = 512B        （原 bf16：1024B）
:93     attn_block_size = 128 × cdiv(393,216, 128×256=32,768) = 128 × 12 = 1,536   （原：768）
:95-97  assert 256 × 1,536 == 393,216 ✓            （393,216 = 3×2^17；256=2^8 整除 ✓ 断言不炸）
:101-103 cache_config.block_size := 1,536           （引擎全局记账自动跟上）
:110-117 attn_page = 1,536 × 512 = 786,432B；
        mamba_page_size_padded = 786,432 + conv 15,360 = 801,792B   ← 与原生一字不差
```

**关键结论**：P=801,792B 不变 → `nb = available ÷ 801,792 ÷ 16` 不变 → 16 个池、池大小、
GDN 侧全部不动。变的只有一件事：**每个 FULL 页从装 768 个 token 变成装 1,536 个**。

### 1.1 分配视图为什么自动正确（不需要碰 model_runner 源码）

FULL 层 spec 的 dtype 来自 Attention 模块实例属性 `self.kv_cache_torch_dtype`
（构造时 `attention.py:225 kv_cache_dtype = cache_config.cache_dtype` → `:276`
`kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(...)` → int8），于是原生视图构造
代码自己算对：

```python
# [VA] worker/model_runner_v1.py
:4562   block_size_chunk = spec.block_size(1,536) // kernel_block(128) = 12     （原：6）
:4563-4568  kv_cache_shape = (2, nb×12, 128, Hk=1, head_size=256)
:4571   attn_tensor_page_size = nb×12×128×256×1B = nb×393,216B                  （K 条宽，与 ssm 同宽 ✓）
:4574   conv_block_padding_size = nb×801,792 − 2×nb×393,216 = nb×15,360          （与原生同值 ✓）
:4575-4577   raw_kv = raw[15,360nb:]; k = 前半; v = 后半
:4637-4642   k_cache = raw_k.view(int8).view((nb×12, 128, 1, 256))
```

→ **三大条 [conv 15,360 | K/ssm 393,216 | V 393,216] 字节级与原生同构**；K 条从
"768 × bf16(512B)"变成"1,536 × int8(256B)"，V 条同理。GDN 的切分（:4696-4714，
从条头顺序切 conv 15,360 + ssm 393,216）与 K/V 无关，一字不动。

### 1.2 引擎记账为什么自动跟上（这就是"兼容框架"）

| 环节 | 机制 | 证据 |
|---|---|---|
| 块表展开 | `blocks_per_kv_block = block_size(1,536)//kernel(128) = 12` 逻辑块/物理页（原 6），`logical = phys×12 + i` | [VA] block_table.py:47-68, :288-303 |
| slot_mapping | `slot = 逻辑块号×128 + 块内偏移`——公式不变，号空间自动变大 | [VA] block_table.py:210-229 |
| 每请求块数 | FULL 组 `cdiv(seq, 1,536)`（原 768）→ **减半** → 同池可装请求翻倍 | [V] single_type_kv_cache_manager.py:276-277 |
| GDN 组 | 1 block = 1 state 不变（mamba spec 的 dtype 由 `--mamba-cache-dtype bfloat16` 独立控制，serve 已显式传） | serve_oscar.sh:66-67；patch_mamba_config.py:64-70 |
| KV usage 显示 | 池占用率口径自动反映减半的块需求 | — |

### 1.3 OSCAR 读写路径为什么零改动

- `k8 = k_cache.view(uint8)` 形状 `(nb×12, 128, 1, 256)`；**bs = k8.shape[1] = 128
  不变**；`stride(1) = 256B`（原 512B）——所有偏移都从**活视图 stride** 计算
  （[W] store_kernel.py:35-48），自动适配。
- 槽内偏移不变：K 槽 256B 用前 96B、V 槽 256B 用前 64B（96/64 < 256 ✓）。
  **store/decode/dequant 内核、format.py 槽数值契约、probe 全部不需要改**。
- （可选优化，非必需）：把 160B 逻辑槽整体挪进 K 槽连续区
  `[0..7 meta | 32..95 Kidx | 96..159 Vidx]`，V 槽完全闲置——省一次跨槽写；本轮不做。

### 1.4 收益账本（对比 D 文 §2.4 三池表）

| | 每 token·head 池字节 | 其中 OSCAR 有效 | 槽利用率 | FULL token 密度 |
|---|---|---|---|---|
| 原生 BF16 | 1,024B | — | — | 1×（基准） |
| 现插件（D 文①） | 1,024B | 160B | 15.6% | **1×（假压缩）** |
| **方案 E（本设计）** | **512B** | 160B | **31.3%** | **2×（真收益）** |
| OSCAR PR（D 文②，需源码） | 136B | 136B | 100% | 7.5× |

---

## 2. 方块图：一页在方案 E 下的样子（与 D 文 §2.1 同口径）

> ⚠️ 勘误指针（2026-09-05）：下图与 D 文 §2.1 同样把"页"画成页内 [条A|条B|条C]
> 交织——**区域宽度正确，字节排布不精确**。真相：池张量按三**稠密区**排布
> （区A 全部 conv 格 / 区B 全部 K·ssm 共区格 / 区C 全部 V 格），"页"= 一个块号在
> 三区各占一格的记账单位。逐字节正图与 32K/TP4/单卡完整排布见
> `ANALYSIS-20260904-D` **§2.5**（packed 版为 §2.5.5 图二）。

```text
建池：torch.zeros(P×nb, int8) 不变（P=801,792B，16 池，3 GDN + 1 FULL 共享） ← [VA] model_runner_v1.py:4124

一个物理页（1,536 token；原生 768）:
┌────────────────────────── 一个物理页 801,792B ──────────────────────────┐
│ 条A conv 15,360B │ 条B K/ssm 393,216B              │ 条C V 393,216B    │
│ GDN conv state   │ FULL 页: 1,536 × 256B K 槽      │ FULL 页: 1,536 × 256B V 槽 │
│                  │ GDN 页: ssm state 393,216B      │ （GDN 不用）        │
└───────────────────────────────────────────────────────────────────────┘
  kernel 块 128 不变；逻辑块 = phys×12 + i（原 ×6）

FULL 页内一个 token·head（新）:
  K 槽 256B（int8 视图 D=256）          V 槽 256B
  ┌meta 8B│pad24B│K idx 64B│空闲 160B┐  ┌V idx 64B│空闲 192B┐
  │ ← OSCAR 用 96B →│              │  │← 用 64B →│        │
  └────────────────┼──────────────┘  └──────────┼────────┘
                   └── 每 token·head 总占用 512B（原 1,024B）──┘
```

---

## 3. 唯一硬约束：MTP 草稿层（三选一，需拍板）

**事实**：MTP 草稿层 = 普通 `Qwen3_5DecoderLayer`（[V] qwen3_5_mtp.py:25,:100），
其 KV **在池内**（翻转前真机日志 `★ INT2 写路径首次执行: mtp.layers.0...` 印证）。
全局 dtype 翻转后，它也会拿到 int8 cache，但它的原生路径（`reshape_and_cache`
bf16→cache + FIA 读）**没有 per_token_head 量化算子**（vllm-ascend grep 零命中）→
不可用。三个解：

| 方案 | 机制 | 代价 | 风险 |
|---|---|---|---|
| **A（推荐）影子池** | 插件包 MTP 层 `impl.forward`，把 `kv_cache` 入参替换为插件自建 **BF16 影子池**（形状与原生 `(nb×12,128,1,256)` bf16 同构、逻辑块号与主池一一对应），写/读其余 100% 走原生算子 | ≈ **2.16GB/rank** 固定显存（nb=1,369 × 12×128×512×2B） | 低（纯 impl 层替换，复用既有类手术机制；块号空间天然对齐） |
| B 混组 spec | 插件 after-hook `Attention.get_kv_cache_spec`，仅对 MTP 层返回 `block_size=768, dtype=bf16` 的 spec（页 786,432B pad 到 801,792 ✓ 可同池） | 0 额外显存 | 中：同池两种 block_size 组（FULL 1,536 / MTP 768）——upstream 有 lcm/gcd 协调机制，vllm-ascend `use_hybrid_blocks` 单一 block_size 假设**未验证** |
| C MTP 也量化 | int8 per-head（idx 256B + fp16 scale 2B = 258B > 256B 槽）**装不下** | — | ✗ 本轮排除（需 384B 槽 = 源码级） |

净账（方案 A）：+2.16GB/rank 影子池，换 FULL 池 token 容量 ×2（16 池 K/V 区 ≈
16.05GB → 有效容量翻倍）——**净赚 ≈ 14GB/rank 的 token 容量**，且 MTP 精度保持
BF16（不回退到"量化草稿击穿接受率"的老坑，R-20260904 §1）。

---

## 4. 改动清单（全部在插件层 + 1 个 serve 参数；零 vllm/vllm-ascend 源码改动）

| # | 改动 | 文件 | 量级 |
|---|---|---|---|
| 1 | serve 加 `--kv-cache-dtype int8_per_token_head` | delivery/serve_oscar.sh | 1 行 |
| 2 | MTP 影子池 shim（方案 A）：impl 包装 + 影子池懒分配 + fail-soft | oscar_ascend/plugin.py + 新 mtp_shadow.py | ~80 行 |
| 3 | 防御断言：启动时校验 k8.stride(1)==256、k8.shape[1]==128、block 12 逻辑块（几何对账，防 fork 漂移） | backend.py `_oscar_setup` | ~15 行 |
| 4 | probe 增加几何档（256B 槽）用例 | delivery/probe_oscar.py | ~10 行 |
| 5 | 文档/环境变量（开关 `OSCAR_ASCEND_PACKED=1` 才启用影子池+断言；关=现状兼容） | README/config | 小 |

不改动：format.py 槽契约、三个 triton 内核、backend 读写路径、旋转/裁剪、窗口
staging（staging 是页外 buffer，与池几何无关）。

---

## 5. 代价与风险（诚实清单）

1. **prefix cache 粒度变粗**：hash/命中粒度从 768 token 变 1,536（[V]
   kv_cache_utils.py:593-627 的 scheduler/hash block size 由组 block_size 推导）→
   短前缀复用率下降。可接受性需真机 ais_bench 复核（`Prefix cache hit rate` 对比）。
2. **chunked prefill 对齐粒度 1,536**：`max_num_batched_tokens=16,384` 不是 1,536 的
   倍数（16,384/1,536=10.67）→ mamba 对齐切分后每 step 实际调度 15,360 token
   （-6.25% 预算利用率，[V] scheduler.py:293-338）。**实施已定为 15,360（=1,536×10；
   初稿"16,128"为笔误——16,128/1,536=10.5 亦非整倍数；其他可选 16,896/18,432）**。
3. **MTP 影子池固定 2.16GB/rank**：即使FULL 密度收益兑现，也要在 max-model-len
   262,144 的高水位下复核总显存（gpu-memory-utilization 0.9 已含 KV 预算重算）。
4. `int8_per_token_head` 的 `kv_quant_mode` 位只被 FULL 层 spec 携带、无消费者
   （OSCAR impl 接管读写）；若未来 vllm-ascend 对该 dtype 加原生逻辑，需回归本设计。
5. 31.3% 槽利用率仍非 PR 的 100%——**再往上（192B 槽=83%、384B 槽=group16-ready）
   必须改 `get_kv_cache_shape`/视图构造（源码 PR）**，那是 D 文 §6.3 表 A/B 的
   "上游 PR"阶段，不在零侵入范围。

## 6. 证据索引（本轮新核验）

- [VA] patch/platform/patch_mamba_config.py:40-120 全文亲读（:53-56 dtype 入口、
  :91-97 页公式与断言、:101-117 block_size/mamba pad）；
  worker/model_runner_v1.py:4559-4577,4637-4642（视图构造，§1.1 逐行）；
  block_table.py:47-68,210-229,288-303（记账自动性）；
  vllm-ascend 全树 grep `int8_per_token_head` 零命中（无校验亦无算子——§3 依据）。
- [V] model_executor/layers/attention/attention.py:225（`kv_cache_dtype =
  cache_config.cache_dtype` 全层默认入口）、:276-278（`kv_cache_torch_dtype` 为
  实例属性——插件钩子可 per-layer 覆盖）、:563-606（`get_kv_cache_spec` 读实例
  属性构 spec）；utils/torch_utils.py:32-44（STR_DTYPE 表：int8_per_token_head→
  torch.int8）；config/cache.py:19-35,258-275（CacheDType 合法性+校验器仅打日志）；
  v1/core/sched/scheduler.py:293-338（mamba 对齐切分）；qwen3_5_mtp.py:25,100
  （MTP 复用 FULL DecoderLayer）。
- 数字复算：256×1,536=393,216 ✓（断言）；1,536×512=786,432；786,432+15,360=801,792 ✓；
  影子池 = 1,369×12×128×(256×2B)×2 = 2.154GB/rank。

---

*状态：DESIGN PROPOSAL（未实施）。决策点一个：§3 MTP 方案 A/B 拍板（推荐 A）。
实施顺序建议：①serve 参数+防御断言+probe 档（半天可验"页几何自动正确"）→
②影子池 shim → ③真机 ais_bench 复核 usage 减半/接受率不变/prefix hit 代价。*
