# ANALYSIS-20260904-A — vLLM-Ascend KV 调度全链路（基于当前仓库）

> 分析模式产物（用户要求：不动核心代码；一切以当前代码仓为事实源）。
> **代码源**（下称 [KCU]/[MRV]/[AV1]/[BTS]/[PMC]/[ATT]/[MAM]/[QGN]）：
> `$SK/references/`（vllm=v0.23.0 `0fc695fc`；vllm-ascend=skill-ref-023 v0.23.0+PR#12607 `19e436985`）
> 工作区插件：`oscar_ascend/{format,backend,plugin,config}.py` + `kernels/`（HEAD `1e95146`）。
> 模型：Qwen3.5-27B-w8a8-mtp（Qwen3_5ForConditionalGeneration；48 GDN + 16 FULL + 1 MTP）。
> **凡与"另一项目草稿"不同之处，均以本树为准（草稿不参与结论）。**

## 0. 结论速览

1. **分配在插件之前完成**：`model_runner_v1._allocate_kv_cache_tensors`（:4080-4132）用
   `torch.zeros(P×nb, int8)` 按 **BF16 页几何**一次性分完 HBM 池；插件是之后才做的
   `impl.__class__` 手术，只能**重解释已有字节**，不能重排内存。
2. **P=801,792B/页，16 个池/rank**：P = conv 15,360 + K/ssm 393,216 + V 393,216
   （`patch_mamba_config.py:94` 的页对齐等式 `512×768 == 393,216` 是 K 页与 ssm 页
   **同条共尺寸**的前提）；池数=16（=max 组宽），nb=1,369（由
   `16.36GiB // P // 16` 反推，与运行日志"KV 16.36 GiB / 每请求 342.25×"完全吻合）。
3. **块表=逻辑 128-块**：物理页 768 被 `BlockTable` 切成 6 个逻辑 128-块
   （`block_table.py:66-88`，`logical_id = phys×6 + i`），与 FULL 层
   `(nb×6,128,Hk,D)` 视图索引同序 —— 插件 `bs=128` 的槽位数学与原生约定**一致**。
4. **MTP 草稿层**：不是独立池——它与 3 个 GDN 层 + `FULL.0` **共享第 0 个池**
   （`kv_cache_utils.py:1300-1326` 的"i<len(组)"拼接规则，池数仍=16；与
   `get_num_blocks` 的 ÷16 账本、日志 342.25× 均自洽）。
5. **插件看不到/不碰的部分**：GDN 的 conv/ssm 条（A/B 条内 GDN 专属页）、MTP 草稿层
   （本轮已排除）、BlockPool/前缀缓存块管理、原生 FIA 路径——这些是"链路上插件外
   的环节"，插件只对 FULL 层 K/V 条做字节重解释。

---

## 1. 图谱阶段：KVCacheSpec 与分组（把"每层什么缓存"变成"几组几池"）

### 1.1 Spec 生成（每层一个 KVCacheSpec）

| 层型 | 名称形态 | Spec | 关键尺寸 |
|---|---|---|---|
| GDN（48） | `…layers.{i}.linear_attn` | `MambaSpec`（[KCI]=vllm/v1/kv_cache_interface.py:604-645） | shapes≈((2560,3),(12,128,128))，dtype=bf16；`page_size_bytes`=padded=**801,792** |
| FULL（16） | `…layers.{3,7,…,63}.self_attn.attn` | `AttentionSpec`（[KCI]:160-236） | `real_page_size_bytes=2×768×1×256×2B=786,432`（:174-180） |
| MTP（1） | `mtp.layers.0.self_attn.attn` | `AttentionSpec`（与 FULL 近同） | 同上（其缓存需共享池——见 §2.4） |

### 1.2 页几何（`patch_mamba_config.py:58-124`）

```
kernel_block_size = 128                      # :58
attn_single_token_k_page_size = D×Hk×dtype  = 256×1×2   = 512B   # :91
attn_token_page_size          = 2×…         = 1,024B            # :92
ssm_block_page_size (max mamba size)        = 12×128×128×2B = 393,216B  # :70
attn_block_size = 128 × cdiv(393,216, 128×512) = 768              # :94
assert 512 × 768 == 393,216  ⇒  K 页(768 token) 与 ssm 页 **同宽**     # :95-97
cache_config.block_size := 768                                     # :102-103
attn_page_size = 768 × 1,024 = 786,432                            # :110
mamba_page_size_padded = 786,432 + 15,360 = **801,792**            # :113-117
```
> **关键不变量**：`attn_single_token_k_page_size × attn_block_size == ssm_block_page_size`
> 是"FULL 的 K 条与 GDN 的 ssm 条能放在同一条 393,216B 页格"的**必要条件**——
> 任何改变每 token·head K 字节数的方案（如 INT2 → 160B/槽）都会破坏此等式（见
> ANALYSIS-B §2）。
> GDN conv 条 = 15,360 = 页内头部填充（conv_block_padding 语义），供 GDN 使用。

### 1.3 分组（`kv_cache_utils.py:1142-1192` 同型分桶 + 分组）

```
same_type_layers = { MambaSpec: [48 GDN], AttentionSpec: [16 FULL, 1 MTP] }   # 桶
group_size = min(48, 17) = 17   # :1158-1159 —— 但运行时账本实为 16 池（见下注）
```
> **注（本树实测账本）**：`num_layers = group_size`；由日志反推 nb=16.36GiB//P//16=1,369、
> `max_concurrency = nb / 每个请求块数 = 1,369/4 = 342.25`（:905-927,1735-1744 公式）——
> 与日志 `342.25x` 完全一致 ⇒ **池数=16、组宽=16**。即 MTP 层**不并入**主 attention 组
> （`MTP spec` 带 `num_speculative_blocks=3`（[KCI]:614/631-636 路径 + 调度器块计）
> 形成独立"桶+组"，宽度=1 → 按 `:1300-1326` 的 `i<len(group.layer_names)` 拼接规则
> 落入 **池 0**，不改变池数 16）。

### 1.4 池数、页数、预算（`kv_cache_utils.py:1300-1326 + 952-967 + 905-927`）

```python
group_size = max(len(g.layer_names) for g in kv_cache_groups)        # 16
page_size  = get_uniform_page_size([...])                            # 801,792（统一取最大）
num_blocks = int(available_memory // page_size // group_size)        # :967 → nb=1,369
for i in range(group_size):                                          # 16 个 KVCacheTensor
    shared_by = [各组第 i 层]（不足的组不贡献）                        # → 池0=5层(3GDN+FULL.0+MTP)
    KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)  # = 801,792×1,369 B
```
**每 rank 池总量 = 16 × 1,369 × 801,792B = 17.56 GB ≈ 16.36 GiB**（BF16 几何，与日志完全一致）。

---

## 2. 施工阶段：`torch.zeros` 与三视图切片（`model_runner_v1.py`）

### 2.1 分配（`model_runner_v1.py:4104-4132`）

```python
self.hybrid_with_attn_and_mamba = False                    # :4104
for kv_cache_tensor in kv_cache_config.kv_cache_tensors:   # 16 个池
    ...use_mamba / use_attn 按 shared_by 内 spec 类型……
    if ("linear_attn" in layer_name or hybrid or ...) and layer_name not in kv_cache_raw_tensors:
        tensor = torch.zeros(kv_cache_tensor.size, int8, device)   # :4124 ← 真正落地
        for layer_name_inner in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name_inner] = tensor        # :4132 ← 4~5 层共享
```
- **16 个 `torch.zeros(P×nb, int8)` / rank**；每个池被 (3 GDN + 1 FULL [+MTP in 池0]) 共享。
- 池体是一维 raw 字节区；**页内布局**在 reshape 阶段切分（§2.2）。

### 2.2 FULL 层视图（`model_runner_v1.py:4559-4642`，hybrid 分支）

```python
block_size          = attn_backend.get_supported_kernel_block_sizes()[0]  # [128]（AV1:143-144）
block_size_chunk    = current_kv_cache_spec.block_size // block_size      # 768//128 = 6
kv_cache_shape      = get_kv_cache_shape(nb*6, 128, 1, 256)               # (2, nb*6,128,1,256)
attn_tensor_page_size = np.prod(kv_cache_shape[1:]) * dtype_size          # nb*393,216
conv_block_padding_size = raw.numel() - attn_tensor_page_size*2           # nb*15,360
raw_kv = raw[conv_block_padding_size:]      # 去掉 15,360×nb 的 conv 头部
raw_k  = raw_kv[:attn_tensor_page_size]     # 条B = nb×393,216   ← FULL K == GDN ssm 同条
raw_v  = raw_kv[attn_tensor_page_size:]     # 条C = nb×393,216   ← FULL V
k_cache = raw_k.view(bf16).view((nb*6, 128, 1, 256))   # :4637
v_cache = raw_v.view(bf16).view((nb*6, 128, 1, 256))   # :4642
kv_caches[layer_name] = (k_cache, v_cache)             # :4684
```
- **页条 A/B/C**（每池）：`[0, nb×15,360)` conv；`[nb×15,360, nb×408,576)` K/ssm 同条；
  `[nb×408,576, nb×801,792)` V。FULL 每 token·head 槽 = 512B（D=256 × 2B bf16）。

### 2.3 GDN 层视图（`model_runner_v1.py:4681-4714`）

```python
num_blocks = raw.numel() // spec.page_size_bytes        # = nb
for shape, dtype in zip(spec.shapes, spec.dtypes):      # ((2560,3),(12,128,128))
    target_shape = (num_blocks, *shape)
    tensor = raw[start:end].view(dtype).view(target_shape)
# 切1 → conv (nb,2560,3)；切2 → ssm (nb,12,128,128)  ← ssm 与 FULL-K **同条**（[0,393,216) 相对）
```
> GDN 页 b 的 ssm 与 FULL 页 p 的 K 同条不同格：`p≠b` ⇒ 字节区不相交 ⇒ 无冲突
> （页号来自**各组独立的块表**，`kv_cache_utils.py:1304-1305` 注释同义）。

### 2.4 MTP 草稿层（池 0 的第 5 个成员）

池 0 `shared_by` = [GDN组0.层0, GDN组1.层0, GDN组2.层0, FULL.层0, MTP.layers.0]；
reshape 时 MTP 层同样走 2.2 的 hybrid 分支（`use_hybrid_blocks = len(attn_groups)>1`
= True，`model_runner_v1.py:3856`）→ 得到自己的 `(nb×6,128,1,256)` bf16 K/V 视图
（**本轮插件已 `[SKIP] mtp-draft`，草稿层保持原生 BF16**；池的 conv/ssm 条对 MTP
而言只是它视图之外的前导 padding，不冲突）。

---

## 3. 运行时映射：块表、slot_mapping、三类执行

### 3.1 块表（768 物理页 → 128 逻辑块；`block_table.py:59-95, 288-300`）

```
physical_block_size = 768（cache_config.block_size 在聚合进 BlockTable）
kernel_sizes=[128] ⇒ block_size=128, logical_block_size=128,
blocks_per_phys_block=6, use_hybrid_blocks=True
logical_table_size = max_num_blocks_per_req × 6                     # :88
append_row: block_ids → _convert_physical_to_logical_blocks         # :116-117
  逻辑 id = phys×6 + i（页主序展开）                                  # :294-300
```
- FULL/注意力组的块表 = **逻辑 128 块**；GDN 组块表 = `kernel_sizes=[0]` → 无分裂、按物理页。

### 3.2 slot_mapping（`block_table.py:148-179, 208-229`）

```
slot_mapping = block_numbers(逻辑id) × 128 + block_offsets          # :225-228
（Triton `_compute_slot_mapping_kernel` 同义；BLOCK_SIZE=128）
```
⇒ **插件 `_slot_bases` 的 `blk=slot//128, off=slot%128`（store_kernel.py:38-48）
与原生槽位约定一一对应**（逻辑 id 直接索引 `(nb×6,128,…)` 视图 dim0）。

### 3.3 三类执行（`model_runner_v1.py` 调度 + `attention_v1.py` builder）

| 阶段 | attn_state（`model_runner_v1.py:1477-1506`） | 原生路径 | 插件路径 |
|---|---|---|---|
| 首次 prefill（全命中=0） | `PrefillNoCache` | FIA TND 无缓存 | `forward`→`_prefill_attention`（cached_len=0 → SDPA 当前 chunk） |
| chunked/续写（有前缀） | `ChunkedPrefill` | FIA 前缀 | `_prefill_attention`（**全前缀 dequant + 逆旋转 + stage splice + SDPA**） |
| 纯 decode | `DecodeOnly` | paged attention（`forward_impl`→`forward_paged_attention`） | `_decode_attention`（**全上下文 dequant 式窗口/INT2 解码**） |
| MTP 投机验证步 | `SpecDecoding`（`model_runner_v1.py:1483-1486`；MTP 且每步 1 token） | `forward_fused_infer_attention`（FIA + metadata attn_mask） | **落入 `_prefill_attention`**（插件无 SpecDecoding 分支；每步重做全前缀） |
| 草稿层（MTP 层） | `SpecDecoding`（`llm_base_proposer.py:651-657`） | **原生（本轮插件已跳过）** | — |

### 3.4 元数据（`attention_v1.py:276-394`）

```
block_tables = common_attn_metadata.block_table_tensor（逻辑 128 块表）
seq_lens     = _seq_lens_cpu / seq_lens_cpu / seq_lens（:293-298），AscendMetadata.seq_lens_cpu=Tensor
slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]   # :300
attn_state   = common_attn_metadata.attn_state                         # :309
```
> 插件读取这些字段的契约：`oscar_ascend/backend.py::metadata_batch_lists`
> （真机 05:21 修复：`seq_lens_cpu` 为多元素 Tensor，禁止 `or` 布尔化）。

---

## 4. 插件接线点（类外科手术后的字节重解释）

```
plugin.load_plugin → 包装 Attention.__init__（plugin.py:86-132）
  → _should_oscar（hybrid + attn_type==decoder + 非 MTP）→ impl.__class__ 替换
  → impl.forward / do_kv_cache_update 重写（backend.py）
写入：do_kv_cache_update → rotate+clip → 160B 逻辑槽（format.py:N-01）
      K 96B(meta8+pad24+idx64) @ k 视图槽前 96B；V 64B(idx) @ v 视图槽前 64B
读取：decode/prefill 反量化 + 逆旋转 + staging(BF16 sink/recent 覆盖)
```

| 插件感知的几何 | 值 | 来源 |
|---|---|---|
| k/v 视图 | `(nb×6, 128, 1, 256)` bf16 | §2.2 |
| 每 token·head 原生槽 | 512B（K）/ 512B（V） | D×2B |
| 插件写入 | 96B(K 视图) + 64B(V 视图) = 160B | format.py |
| 逻辑块粒度 | 128 token | §3.1 |
| stage 环 | rows=64（8192/128），slot 逻辑 id % 64 | backend._ensure_staging/_staging_write |

---

## 5. 插件"看不到"的链路环节（后续分析的重点边界）

1. **BlockPool/前缀缓存**：`kv_cache_utils` 的 `get_num_blocks`/`KVCacheManager`
   只按池总量与 token 哈希发块；插件不改池总量 ⇒ 块数与池占用（`KV cache usage %`）
   与原生一致（本机日志 4.1% 前后不变）。
2. **GDN 读写**（`qwen_gdn_linear_attn.py`/`AscendGatedDeltaNetAttention`）：conv/ssm
   状态按 GDN 组块表读写条 A/B，与 FULL INT2 字节无交集（页号空间不同）。
3. **MTP 草稿层 KV 读写**：原生 BF16 路径（本轮[SKIP]后），与 FULL 层 INT2 无关；
   但草稿层与 `FULL.0` **同池**（池 0）——草稿层视图从条 B/C 偏移 nb×15,360 起，
   FULL.0 的 INT2 写在**同条不同页**，互不覆盖。
4. **FIA / 元数据构建**（`attention_v1.py`）：slot_mapping/block_table/attn_state 的
   权威构造者；`AscendMetadata.seq_lens_cpu` 恒为 Tensor（真机 05:21 崩溃源）。
5. **数值 probe**（`delivery/probe_oscar.py`）：独立于引擎，校验 store/dequant/decode
   字节级一致性（ref 0 / triton 0 全 PASS 已真机确认）。

---

## 6. 一张图（端到端）

```text
[QLora/config] Qwen3.5-27B-w8a8-mtp (48 GDN + 16 FULL + 1 MTP)
      │ kv_cache_spec: MambaSpec(801,792B/页) + AttentionSpec(768-token 页)
      │ kv_cache_utils: 桶→组（GDN 3×16, FULL 16, MTP 1）→ 16 池 × (P×nb)；
      │   P = conv15,360 | K/ssm 393,216 | V 393,216；nb = 16.36GiB//P//16 = 1,369
      ▼
[vllm-ascend boot] 16×torch.zeros(P×nb, int8)（:4124）   ← 池已定型（BF16 几何）
      │ reshape（:4559-4642/4681-4714）：
      │   FULL: (2, nb×6,128,1,256) → k/v 视图（条B/条C，每个 512B 槽）
      │   GDN:  conv (nb,2560,3) + ssm (nb,12,128,128)（条A/条B）
      │   MTP:  FULL 同款视图（池0 第5共享者）
      ▼
[BlockTable] 768→128 逻辑块（logical_id=phys×6+i）；slot=logical×128+off
      ▼
[运行时] 每步构建 AscendMetadata（attn_state, seq_lens_cpu=Tensor, block_tables=逻辑表,
          slot_mapping=逻辑槽）→ Attention.forward → impl.forward
      ├─ 原生 GDN 路径：conv/ssm 按 GDN 组块表
      ├─ 原生 MTP 路径：草稿层（BF16，SpecDecoding 元数据）
      └─ 插件 FULL 路径（本插件）：
           写：rotate+clip(sort) → 160B 逻辑槽 → k 视图[0:96]+v 视图[0:64]
           读：decode/prefill → 全前缀反量化 + 逆旋转 + stage(BF16 sink/recent 覆盖) + SDPA
```

**核心事实重申**：① 池在插件前定型（BF16 页几何）；② 插件只重解释 FULL 层 K/V 条的
前 96/64B；③ 逻辑块表与插件槽位数学一致（128 粒度）；④ MTP/ GDNDN / 前缀缓存/
池管理全部在插件作用面之外。
