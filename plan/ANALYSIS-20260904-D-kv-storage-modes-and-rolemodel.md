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
**Role model 判决：TurboQuant（工程蓝图）+ OSCAR PR（数值配方）**——TQ 赢在三件我们
最缺的事：分配期缩池的完整官方先例、**hybrid（16 full + 48 GDN）与 Mamba 共存的官方
修法（vllm PR#39931）**、小续算直接复用 decode 内核（恰好命中我们 MTP verify=4 token
的热点）。详见 §5。

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
   NotImplementedError），该修复就在我们本地 v0.23.0 树里（§4.6）。

---

## 2. 模式①（当前插件版）：布局不动，槽内重解释

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

## 3. 模式②（OSCAR vLLM PR#46774）：分配期缩池 + 压缩域读

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

## 4. 模式③（TurboQuant @ upstream vLLM v0.23.0）：与②同骨架，工程更完整

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

## 5. 三模式对照 + Role Model 判决

### 5.1 对照表（一行一维度，人话）

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

### 5.2 判决：Role Model = TurboQuant 的工程蓝图 + OSCAR PR 的数值配方

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

### 5.3 落地映射（TQ 蓝图 → vllm-ascend 接缝；诚实标注零侵入边界）

| TQ 五件套 | vllm 位置 | vllm-ascend 对应接缝 | 零侵入插件内可做？ |
|---|---|---|---|
| CacheDType 字面量 + dtype 淘汰 | cache.py:19-35 / cuda.py:131-147 | `AscendAttentionBackendImpl` 无 dtype 淘汰机制 → 需在 backend 选择/platform 层加 | ✗（改源码） |
| TQFullAttentionSpec 页字节公式 | kv_cache_interface.py:327-349 | `get_kv_cache_spec`/`patch_mamba_config` 页公式（对齐锁所在地） | ✗（改源码） |
| lcm 页协调 + mamba 页 pad | interface.py:573-699 | `patch_mamba_config.py:58-117`（同功能不同方向） | ✗（改源码） |
| 独立写缝 do_kv_cache_update | attention.py:690 | vllm-ascend `forward` 内联写（attention_v1.py:1529-1534）——impl 子类可等价实现 | ✓（我们已在做） |
| 压缩域 decode + 小续算复用 | triton_turboquant_decode.py / turboquant_attn.py:666-692 | impl 子类的读路径（含 SpecDecoding 分支） | ✓（当前死代码的 decode 内核正是为此准备） |

**结论**：显存收益（真·KV 优化）在零侵入约束内拿不到——这正是现状"约等于无效"的
结构性原因（与 ANALYSIS-C §4.2 一致）。TQ 蓝图中**唯一能在插件内先兑现的一件**是
第五行：给 SpecDecoding/小续算分支接压缩域 decode（把现在"每步全前缀物化"替换成
复用已验证的 triton decode 内核）——这不省显存，但直接砍 110s 的主因。其余四件
（真缩池）需要给 vllm-ascend 提 PR 或 fork，TQ 的 #39931 就是那份 PR 的模板。

---

## 6. 证据索引

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

---

*分析模式产物（零代码改动）。本文所有"公式推算"（②的 136B、③的 230B/390B）均由
所引 config 公式对 D=256 直接计算得出，已在正文标注"推算"；其余数字全部来自所引
file:line 的原文或真机日志。*
