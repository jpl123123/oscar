# ANALYSIS-20260904-C — vLLM-Ascend KV 调度全链路 + OSCAR 布局/读写双问答（纯分析模式）

> **模式声明**：本轮为纯分析，未改任何核心代码（工作区 HEAD 保持 `1e95146`）。
> 所有断言锚定两类事实源，均给出 file:line：
> ① **本仓插件代码**（`oscar_ascend/`、`delivery/`，即"我自己写的代码"）；
> ② **本机参考树** `/Users/sunao2000/oscar_zai/references/`（PROVENANCE：vllm=`0fc695fc`、
> vllm-ascend=`19e436985`(v0.23.0+PR#12607)、oscar-vllm-pr46774=`57286d5d`）。
> ③ 真机日志：用户回传 2026-09-04 05:35 起的 serve.log 摘录（warmup 110.12s/case，
> 对照无插件 base 4.10s/case）。
> 与已删除的 A/B 两篇的区别：本轮所有关键行号**逐一人工复核**（非转引）；新增
> **MTP attn_state 实际路由链**与 **110s 定量归因**两个此前缺失的承重环节。
>
> 路径缩写：`[VA]`=references/vllm-ascend、`[V]`=references/vllm、`[PR]`=references/oscar-vllm-pr46774、
> `[W]`=本仓工作区。模型：Qwen3.5-27B-w8a8-mtp（48 GDN + 16 FULL + 1 MTP 草稿层，TP4，
> D=256，Hk=1/rank，Hq=8/rank（日志 `heads=1` 自证 Hk）。

---

## 0. 结论速览（每条后附证据节号）

| # | 结论 | 节 |
|---|---|---|
| C1 | **手术时序其实先于分配**（`Attention.__init__` 在 `load_model` 内，早于 `initialize_kv_cache` 的 `torch.zeros`）。真正的约束不是"先后"，而是**插件根本不参与页几何协商**（候选 A 的自觉取舍）——几何在 config 期已由 `patch_mamba_config` 定死 | §4.1 |
| C2 | **混合精度的显存收益 = 0（结构性）**：16 个池仍按 BF16 几何 `torch.zeros(P×nb)`，本仓代码从不改 shape；对照原版 PR，其显存收益来自**分配期** `get_kv_cache_shape` 第 4 维换成压缩槽（72B/槽 vs 512B → ~7×），插件版没有这一环 | §4.2 |
| C3 | **带宽收益只在"中段 prefill 写入"成立**（160B vs 1024B，-84%）；**decode 稳态写入反而 +15.6%**（INT2 160B + staging BF16 双写 1024B = 1184B）；**读路径带宽被放大约 10~15×**（全前缀 fp32 物化 + 逆旋转，见 C4） | §4.3 |
| C4 | **★ 本部署下 `_decode_attention`（含窗口 LSE 合并）是死代码**：`method=qwen3_5_mtp` 时 `_build_attn_state` 永不产出 `DecodeOnly`（连 1-token 步也标 `SpecDecoding`），插件 forward 的二分把 SpecDecoding 落入 `_prefill_attention` → **每个 verify 步、每个 FULL 层都全前缀反量化 + 逆旋转 + SDPA**。这是 warmup 110s（vs base 4.1s）的主因 | §5.3, §6 |
| C5 | **sink BF16 窗口从第一个 verify 步起失效**（16384-token 上下文）：staging 环行号 = 逻辑块号 % 64，首个 verify 步的新 token 落在逻辑块 128 → 行 0，恰好覆盖 sink 块 0 的 owner tag。通式：**活跃序列长度每跨过 8192 的倍数，sink 窗口即被杀死**（此后静默回退 INT2，无任何告警） | §5.4 |
| C6 | **地址/偏移/字节契约本身是对的**（stride 全部取自运行时活视图；块表逻辑粒度 phys×6+i 与内核假设一致；probe 字节差=0）。问题全部是**性能型/路由型**：triton 默认关（`config.py:84`）、eager ~1500 次算子发射/步、32+ 次 host 同步/步、`torch.sort`/`index_put_`/高级索引等 AICPU/慢核倾向算子 | §5.1, §5.7 |

---

## 1. 事实源与部署形态锚定

### 1.1 本轮 serve 的关键配置（`[W] delivery/serve_oscar.sh`）

```bash
# :32-37  裁剪与窗口（本轮默认）
export OSCAR_ASCEND_K_CLIP_RATIO=0.96      # :32
export OSCAR_ASCEND_V_CLIP_RATIO=0.92      # :33
export OSCAR_ASCEND_SINK_TOKENS=128        # :35  （≥block_size 128 才有 sink 页）
export OSCAR_ASCEND_RECENT_TOKENS=256      # :36
export OSCAR_ASCEND_STAGING_TOKENS=8192    # :37
# :48-70  vllm serve 关键参数
    --tensor-parallel-size 4 \                                   # :53
    --max-model-len 262144 \                                     # :54
    --max-num-batched-tokens 16384 \                             # :55  ← warmup 16k 单 chunk 的来源
    --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \  # :59
    --enforce-eager \                                            # :68  ← 全程 eager，无图捕获
```

**未设置 `OSCAR_ASCEND_USE_TRITON`** → `[W] oscar_ascend/config.py:84` 默认 `"0"`：

```python
        # 真机 01:18 实测 torch-npu 位运算偏 `>>` 向量广播 bug；Triton-ascend 未上机验证
        # → 默认 torch 参考路径（全 NPU 算子），显式 OSCAR_ASCEND_USE_TRITON=1 才启用 Triton
        use_triton=os.environ.get("OSCAR_ASCEND_USE_TRITON", "0") == "1",
```

⇒ **当前真机 serve 的 store/dequant/decode 全部走纯 torch 参考实现**；且 launch 脚本对
triton probe 只做观察不做门禁（`[W] delivery/install_and_launch.sh:181-185`：
`OSCAR_ASCEND_REQUIRE_TRITON=1` 才是硬门禁，默认"服务走 torch 参考路径"）。

### 1.2 层面账目（与用户日志互证）

- 类外科手术 64 次 = 16 FULL 层 × 4 rank（GDN 层无 Attention impl，天然不出现）。
- `★ INT2 写路径首次执行` 16 条（层 3,7,…,63），每条 `tokens=16384 heads=1`：首 chunk
  全长 = `max_num_batched_tokens`；Hk=1/rank。
- 每层一条 `裁剪实现 = 排序分位数` = `_rotate_clip` 首次调用打印
  （`[W] oscar_ascend/backend.py:90-95`）。

---

## 2. vLLM-Ascend 调度全链路（六阶段，每格真实代码）

### 2.0 总图

```text
━━━ 阶段① config 期：页几何定死（插件不可能参与）━━━━━━━━━━━━━━━━━━━━━━
 vllm serve 启动 → [VA] patch/platform/patch_mamba_config.py
   kernel_block_size=128 (:58)
   attn_single_token_k_page_size = D×Hk×2B = 512B        (:89-91)
   ssm_block_page_size = max(mamba sizes) = 393,216B     (:64-70)
   attn_block_size = 128 × cdiv(393216, 128×512) = 768   (:93)
   assert 512×768 == 393216   ← K条与ssm条同宽的对齐等式  (:95-97)
   cache_config.block_size := 768                         (:101-103)
   mamba_page_size_padded = 768×1024+15360 = 801,792B     (:113-117)
        │
━━━ 阶段② 图谱期：每层 spec → 分桶分组 → 16 张图纸 ━━━━━━━━━━━━━━━━━━━
 [V] v1/core/kv_cache_utils.py:1142-1195
   same_type_layers 分桶 = { MambaSpec: 48层 , FullAttentionSpec: 16层 }
   min_num_layers=16 → group_size=16（48 ≥ 16×1.5，不触发 max 启发式 :1169）
   mamba → 3 组（layers[i::3] 交错 :1189）、FULL → 1 组 ⇒ 4 个 kv_cache_group
 [V] kv_cache_utils.py:1318-1326（一般分支）
   for i in range(group_size):            # 16
       shared_by = [组0第i层, 组1第i层, 组2第i层(GDN), FULL组第i层]
       KVCacheTensor(size = page_size×nb, shared_by)      # 16 张图纸
 [V] kv_cache_utils.py:967
   nb = available_memory // page_size(801,792) // 16
        │
━━━ 阶段③ 施工期：每 rank 16 个 torch.zeros + 三大条切分 ━━━━━━━━━━━━━━
 [VA] worker/model_runner_v1.py:4116-4132
   for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
       tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, ...)  # :4124
       for layer_name_inner in kv_cache_tensor.shared_by:
           kv_cache_raw_tensors[layer_name_inner] = tensor                # :4130-4132
 FULL 层视图（同文件 :4559-4577, :4597-4642）：
   block_size_chunk = 768//128 = 6                                        # :4562
   kv_cache_shape = get_kv_cache_shape(nb*6, 128, Hk, 256) = (2, nb*6, 128, 1, 256)
   attn_tensor_page_size = nb*6*128*256*2B = nb*393,216                   # :4571
   conv_pad = raw.numel() - 2×attn_page = nb*15,360                       # :4574
   raw_kv = raw[conv_pad:]; raw_k = raw_kv[:attn_page]; raw_v = raw_kv[attn_page:]
   k_cache = raw_k.view(bf16).view((nb*6,128,1,256))                      # :4637
   kv_caches[layer] = (k_cache, v_cache)                                  # :4679-4680
 GDN 层（同文件 :4696-4714）：同池顺序切 conv (nb,2560,3) + ssm (nb,12,128,128)
        │
━━━ 阶段④ 调度期：token 追赶 + MTP spec + 前缀 hash（每 step）━━━━━━━━━
 [V] v1/core/sched/scheduler.py:342ff
   num_new_tokens = num_tokens_with_spec - num_computed_tokens  (:403-407)
   MTP verify 步：spec_token_ids 下发 → 每请求调度 1+3=4 token  (:519-531)
   allocate_slots(..., num_lookahead_tokens=3)                 (:462)
   chunked prefill 按 mamba 块对齐切（block_size=768）          (:293-338)
   前缀缓存：块链 hash = hash(父hash, token_ids, group_id)     [V kv_cache_utils.py:584]
        │
━━━ 阶段⑤ worker 执行期：逻辑块表 + slot_mapping + attn_state ━━━━━━━━━
 [VA] worker/block_table.py
   物理块(768) → 6 逻辑块(128)：logical = phys*6 + i      (:288-305)
   slot = 逻辑块号×128 + 块内偏移                          (:210-229)
 [VA] model_runner_v1.py:1477-1505  _build_attn_state（★ 见 §5.3 判定表）
 [VA] attention/attention_v1.py:276-365  AscendMetadata（qsl/seq_lens/block_tables/slot_mapping）
        │
━━━ 阶段⑥ 层执行期：写 KV + 读注意力 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 原生 [VA] attention_v1.py：
   写：reshape_and_cache（aclnn 算子）                     (:1442-1453)
   读：DecodeOnly → torch_npu._npu_paged_attention         (:1363-1382, :1465-1470)
       其余（含 SpecDecoding/ChunkedPrefill）→ FIA 整池视图 (:1471-1473, :1174-1232)
 插件 [W]（load_model 期手术替换，见 §4.1）：
   写：do_kv_cache_update（旋转+sort裁剪+INT2 打包散写）  backend.py:108-146
   读：DecodeOnly → _decode_attention（★本部署永不触发）  backend.py:202-205
       其余     → _prefill_attention（全前缀反量化+逆旋转+SDPA）backend.py:253-293
```

### 2.1 "各种类型的 KV store"归属总表

| 存储体 | 分配处 | 生命周期 | 消费者 | 本轮状态 |
|---|---|---|---|---|
| 混合池 16×`(P×nb,) int8` | `[VA] model_runner_v1.py:4124` | 进程级，serve 全程 | FULL+GDN 4 组共享 | 不变（usage 4.1% 同量级） |
| FULL K/V 视图 `(nb*6,128,1,256)`×2 | 同上 `:4637-4642` 切出 | 同池 | 原生 paged/FIA；**插件重解释为 160B 槽** | 每槽仅前 96B(K)+64B(V) 被写 |
| GDN conv/ssm `(nb,2560,3)`/`(nb,12,128,128)` | 同上 `:4705-4713` | 同池 | GDN 原生路径 | 完全原生，未动 |
| 插件 staging arena（BF16，页外） | `[W] backend.py:296-314` 首请求懒分配 | per-layer 常驻（64 行×128×1×256×2B×2 ≈ 8MB/层） | sink/recent BF16 窗口 | 写路径活，decode 读路径死（C4） |
| INT2 逻辑槽（in-pool 重解释） | 无实体——写路径 `store_kernel.py:35-95` 按偏移落位 | 与池同生命周期 | 插件读路径 | 字节契约 probe 已验 |

---

## 3. warmup 110.12s 的端到端时间线拆解（真机日志互证）

| 时段 | 证据 | 推算 |
|---|---|---|
| 05:35:18→05:35:25 | ais_bench 建连 + 首请求进入 | ~7s（建连/排队） |
| prefill 16384 tok | `Avg prompt throughput: 2461.5 tok/s`（05:35:32 行） | 16384/2461.5 ≈ **6.7s**（base 2542.4 → 6.4s，几乎持平：写路径+prefill SDPA 只带来 ~5% 差异） |
| decode/verify ~100s | 生成吞吐 4.9~6.5 tok/s 持续到 05:37:0x；acceptance 2.88~3.65 | 设生成 512 tok：步数 ≈ 512/3.35 ≈ 153 步 → **~653ms/verify 步**；base 同模型正常 MTP 步为几十 ms 量级 |

**每个 verify 步的插件侧成本构成**（16 FULL 层/rank × 下列项，机制推导，取证法见 §7）：

```text
每层每步（SpecDecoding → _prefill_attention 分支，backend.py:253-293）：
  写路径   : 旋转 matmul×2 + sort×2 + clamp×2 + quantize + index_put_×10 + staging 写 ~15 op
  读路径   : tolist 同步×2（backend.py:256, metadata_batch_lists）
             oscar_full_dequant(ref) ≈ 15+ op：4 次高级索引 gather（[C,1,8]/[C,1,64]×2）
               + 算术解包链 q0..q3（format.py:99-110，NPU `>>` 不可用故用 //、%）+ 2 次 dequant
               → 物化 fp32 [16388,1,256]×2 ≈ 33.6MB
             逆旋转 matmul×2：[C,256]×[256,256] fp32 ≈ 2×2.1 GFLOP（backend.py:282-283）
             _stage_splice：gather [128,128,1,256]×2 + where（backend.py:348-363）
             SDPA（attn_mask 版）+ cat/transpose ≈ 10 op（decode_kernel.py:88-93）
  合计     : ≈ 90-100 次算子发射 + 2 次 host 同步 + ~50MB 物化 + ~4.2 GFLOP
×16 层 ⇒  ≈ 1500+ 次发射/步 + 32 次同步/步 + ~0.8GB 物化/步 + ~67 GFLOP/步
eager 下 NPU 每算子发射 ~0.2-0.4ms ⇒ 发射项即 ~300-600ms/步 ← 与 653ms 观测吻合
```

对照 base 同一步：每 FULL 层 = `reshape_and_cache`(1 op) + `npu_fused_infer_attention_score`(1 op)。

---

## 4. Q1 — 布局先于插件：sink/int2/recent 混合精度到底释放了没有？

### 4.0 先修正用户示意草图的两处

1. **"plugin 在 HBM 分完之后才进来"——不准确**。手术点在 `Attention.__init__`
   after-hook（`[W] plugin.py:88-132`），发生在 `load_model` 内（`[VA] worker.py:681-692`），
   **早于** `initialize_kv_cache`（`[VA] worker.py:918` → `model_runner_v1.py:4124` 的
   `torch.zeros`）。但先后顺序不是本质——**插件在时间上本来来得及参与几何，却在设计上
   主动放弃**（候选 A：不改 shape/页表/块大小，见 `[W] plan/PLAN-1-§3`）。
2. **"OSCAR store 只往原来的 K/V slot 写 96B+64B"——对**，且不止：窗口 token 还要
   再往 staging arena 双写一份 BF16（`[W] backend.py:316-346`；PR 同款语义
   `[PR] oscar_attn.py:282` docstring："INT2 copy is still written"）。

### 4.1 时序事实链

```text
config 期（几何定死） → load_model（★手术在这里） → profile_run → torch.zeros×16
→ 首请求（写/读路径首次执行）
```

插件能触及的只有最后一环：把已按 BF16 几何切好的 `(nb*6,128,1,256)` 视图按 uint8
重解释，在每 token·head 的 512B K 槽前 96B + 512B V 槽前 64B 写入 INT2 逻辑槽
（`[W] format.py:25-33` 槽常量；`[W] store_kernel.py:54-95` 落位）。

### 4.2 显存维度：结构性为零

- 池字节 = `16 × P × nb` 与原生完全相同（分配代码 `[VA] model_runner_v1.py:4124` 不感知
  插件）。日志侧互证：`GPU KV cache usage: 4.1%`（池占用率口径）与无插件部署同量级。
- **为什么不能改**（本模型的对齐锁）：`[VA] patch_mamba_config.py:95-97`
  `assert 512 × 768 == 393,216`——K 条每 token·head 512B 是"FULL 的 K 条与 GDN 的
  ssm 条放进同一 393,216B 页格"的前提。若 K 槽改 160B：160×768=122,880≠393,216，
  反解 attn_block_size=393,216/160=**2457.6 非整数** ⇒ 不重排 GDN 页宽（12×128×128×2B）
  就无法满足同宽约束。这是候选 B 被否决的数学根因（`[W] plan/PLAN-1-§3`）。
- **原版 PR 的收益环**（对照）：`[PR] oscar_attn.py:87-103` `get_kv_cache_shape` 直接把
  第 4 维换成 `slot_size_aligned`（`[PR] oscar/config.py:120-128`，D=128 时 72B/槽 vs
  BF16 512B → ~7×），让引擎在**分配期**按小页推导 num_blocks——插件版没有这个接缝
  （`[VA] attention_v1.py:104-112` 原生 shape 被原样使用）。

### 4.3 带宽维度：只有一处为正

| 路径 | 每 token·head 字节 | vs 原生 1024B | 证据 |
|---|---|---|---|
| 写：中段 prefill token（非窗口） | 160B（K96+V64） | **-84.4%** | `[W] backend.py:131-135`（日志 -68.8% 是对 512B 单边口径） |
| 写：窗口 token（sink/recent 内，decode 步全部新 token 都属此类） | 160B + staging BF16 1024B = **1184B** | **+15.6%** | keep 判据 `backend.py:330-332`；双写 `:343-346` |
| 读：native decode（对照） | 1024B×seq，单算子 fused | 1× | `[VA] attention_v1.py:1363-1382` |
| 读：插件 verify 步（实际路径） | 反量化物化 fp32 33.6MB + 逆旋转读写 ~134MB + splice ~17MB ≈ **10~15×** 放大 | ≈+1000% | `backend.py:277-291` 链 |

⇒ "IO -68.8%" 的日志口径**只在写路径、且只对非窗口 token 成立**；本部署稳态
（MTP verify）下写 +15.6%、读约一个数量级放大。

### 4.4 语义维度：三段混合"存在但未按设计运行"

| 机制 | 设计 | 本部署实际 |
|---|---|---|
| INT2 中段 | decode/verify 读时内核内反量化 | verify 步全量物化反量化（fp32 TND）——语义对、代价错（§5.3） |
| recent BF16 | decode 走窗口 LSE 合并 + prefill 走 splice | **窗口 decode 死代码**；仅 splice 生效（§5.3/5.4） |
| sink BF16 | 同上 | **从首个 verify 步起失效**（§5.4 环冲突） |

### 4.5 Q1 判决

**"sink/int2/recent 混合精度"作为内存特性没有释放、也不可能在该零侵入布局下释放
（显存收益 0、稳态净带宽为负）；作为精度分层语义部分存在（INT2 中段 + recent splice），
但其中 sink 窗口在长上下文本部署中名存实亡、窗口 decode 合并路径整体未被执行。**
根因不是时序，而是候选 A 把干预点放在读写期而非分配期 + 部署形态（MTP）使设计中的
decode 读路径从未被路由到。

---

## 5. Q2 — 当前布局能否系统性正确读写？逐路径审计

### 5.1 地址/偏移正确性：✓（by construction）

- 所有槽偏移用**运行时活视图的 stride**，无硬编码：`[W] store_kernel.py:35-48`
  （`k_off = blk*k8.stride(0) + off*k8.stride(1) + h*k8.stride(2)`），decode/dequant 同
  （`decode_kernel.py:272-284`、`dequant_kernel.py:86-97`）。
- 块表语义一致：插件假设 `block_tables/slot_mapping` 为 128-token **逻辑块**粒度
  （`bs = k8.shape[1] = 128`），与 `[VA] block_table.py:288-305`（logical=phys×6+i）、
  `:210-229`（slot=逻辑块号×128+偏移）完全一致；k_cache 第 0 维正是 nb×6
  （`[VA] model_runner_v1.py:4559-4568`），stride(0)=65,536B=一个逻辑块。
- 字节契约：`[W] format.py`（N-01..N-04）为唯一权威；probe 判据 store 字节差=0
  （`[W] delivery/probe_oscar.py`），本轮真机 ACTIVE。
- 逻辑块号取模环（staging）与池大小无越界风险：rows=64 << nb×6。

### 5.2 写路径：✓ 语义等价原生 `reshape_and_cache`

- 每步只写 `slot_mapping[:num_actual_tokens]`（`[W] backend.py:187-191`），与原生
  截断口径一致（`[VA] attention_v1.py:1444-1448`）。
- MTP 语义：verify 步 4 个 token（含 draft）全部写入；被拒 draft 的槽位下步被新 token
  覆盖——与原生行为一致（复用同 slot_mapping）。
- 数值链：rotate（fp32 matmul）→ sort 分位裁剪 → per-vector INT2（fp16 meta 先舍入
  N-02）→ 打包（N-03）→ 散写；CPU 镜像 16/16（含精度地板双向）。

### 5.3 读路径实际路由：★核心发现——decode 读路径是死代码

`[VA] model_runner_v1.py:1477-1505` `_build_attn_state`（人工复核原文）：

```python
        if np.all(...num_computed_tokens_cpu... == 0):
            attn_state = AscendAttentionState.PrefillNoCache          # :1478-1479
        elif np.all(num_scheduled_tokens == 1):
            attn_state = AscendAttentionState.DecodeOnly               # :1481-1482
            if self.speculative_config and self.speculative_config.method == "mtp":
                attn_state = AscendAttentionState.SpecDecoding         # :1483-1486 ← 即使 1-token 也是 SpecDecoding
        elif np.all(num_valid_tokens == 1):                            # verify 步: 4 调度-3 spec=1 有效
            if self.speculative_config:
                attn_state = AscendAttentionState.SpecDecoding         # :1487-1490
        ...
        if attn_state == SpecDecoding and method != "mtp":
            self.attn_state = ChunkedPrefill                           # :1500-1503 ← eagle3 降级，mtp 保持
```

本 serve `method=qwen3_5_mtp` ⇒ **DecodeOnly 永不出现**。而插件分派：

```python
        # [W] backend.py:202-205
        if state == getattr(AscendAttentionState, "DecodeOnly", None):
            attn_out = self._decode_attention(...)          # ← 死代码（含 _decode_attention_windowed / _oscar_int2_decode）
        else:
            attn_out = self._prefill_attention(...)         # ← PrefillNoCache/ChunkedPrefill/SpecDecoding 全走这里
```

实际读路径路由表：

| attn_state | 原生读（`[VA] attention_v1.py:1456-1475`） | 插件读 | 正确性 | 代价 |
|---|---|---|---|---|
| PrefillNoCache（首 chunk） | FIA（TND current K/V） | `oscar_prefill_ref`（is_causal SDPA） | ✓ | ~持平（§3 prefill 差 5%） |
| ChunkedPrefill（续 chunk，>16384 的 prompt） | FIA（整池视图+block_tables） | 全前缀 dequant+逆旋转+splice+SDPA | ✓ | O(prefix)/层/chunk（§5.8） |
| PrefillCacheHit（前缀命中） | 同上 | 同上 | ✓（INT2 槽可读回，几何未变） | 同上 |
| **SpecDecoding（MTP verify，稳态）** | FIA（sparse_mode=3 + current K/V） | **同上（cached_len=seq-4）** | ✓ | **每步每层全前缀反量化 → §3 的 653ms** |
| DecodeOnly | npu_paged_attention | （写了但不会被路由到） | — | — |

正确性结论：**数学上是对的**（旋转正交、dequant 精确重建、因果掩码 mask=k_pos≤q_pos
`[W] decode_kernel.py:88-93`、`query_start_loc/seq_lens` 访问器已修），真机接受率
42~88% 也证明非崩溃级错误；**系统性问题是路由+代价**：为 SpecDecoding 设计的 fused
INT2 decode（`[W] kernels/decode_kernel.py`，port 自 PR）从未被执行，稳态跑的是
"每步全量反量化"这条被 PLAN-1 明确否决过的候选 D 形态（`[W] plan/PLAN-1-§3` 候选 D：
"每步 O(seq_len×D) 带宽+scratch 分配…仅作 fallback"）。

### 5.4 窗口机制审计

- **staging 写**：✓ 每步执行（`backend.py:192-199`）；keep=sink ∪ recent（`:330-332`）；
  环行号 `rows = (slot//bs) % 64`（`:336`，rows_total=max(⌈8192/128⌉,1+3+2)=64）。
- **splice（prefill/verify 读侧）**：✓ 执行（`:284-287`）——owner tag 匹配处用 BF16
  精确值替换 INT2 重建（`:348-363`），不匹配静默回退 INT2（PR 同语义）。
- **窗口 decode（LSE 合并 + 块表平移）**：✗ 死代码（§5.3）。`si`/`bt_eff`/`cut`/
  `logaddexp` 整段（`backend.py:365-453`）在本部署 0 次执行。
- **★ sink 环冲突时序**（修正已删文档的"prefill 覆盖"说法，真凶是 decode 期新块）：
  - prefill 16k 单 chunk 只写 sink 块 0（row 0）+ 尾块 125/126/127（row 61/62/63）——
    **prefill 自身不冲突**（keep 掩码挡住了中段块）。
  - 首个 verify 步：4 个新 token 落逻辑块 128 → `128 % 64 = 0` → **row 0**，且
    forward 内写路径先于读路径（`backend.py:187-205`）→ 同一步内 splice 就看到
    owner[row0]=128≠0 ⇒ **sink 从第一个 verify 步起回退 INT2，且永不恢复**（块 0
    不会再被写）。
  - 通式：**活跃 seq 每跨过 8192 的倍数（=64 逻辑块），最先牺牲的就是 sink 行**。
    warmup 16384 恰在第 128 块起 decode——首发即失效。
  - recent 窗口存活期更长（尾块行 61-63 直到块 ~190/191 才被撞，seq>24320）。
  - 无任何告警（设计内"静默降级"，`[PR] oscar_attn.py:282` 语义）。

### 5.5 MTP 兼容性

- 草稿层排除：✓ `[W] plugin.py:176-184`（`.mtp.` 先拒，草稿 KV 保持 BF16 原生
  SpecDecoding→FIA 路径）。
- 目标层：SpecDecoding 落入 `_prefill_attention` 的语义等价性——q_len=4、
  cached_len=seq-4、因果掩码逐 token 正确（`decode_kernel.py:88-93`）⇒ 数学等价
  （真机 mean acceptance 3.0~3.65 vs base 4.0，差距来自 INT2 量化+sink 失效，符合
  R-20260904 修复后的量级预期）。
- `rejection_sampler fallback` 警告：来自 vllm 上游 `enable_reduce_sample=False`
  的常规告警（vendor 版本路径），与本插件无数据依赖；判据：base 侧同配置同样出现
  （base 摘录未覆盖该时段，列为待用户 grep 复核项，§7-⑥）。

### 5.6 前缀缓存 / 块复用 / zeroing / GDN 隔离

- 前缀缓存：✓ hash/块复用/命中链全部原生（`[V] kv_cache_utils.py:584`、
  `single_type_kv_cache_manager.py:552-568`、hybrid 收敛 `kv_cache_coordinator.py:651-686`），
  几何未变 ⇒ 命中块读回 INT2 槽与写入自洽（PLAN-1 R8 论证 + probe）。
- zeroing：✓ `needs_kv_cache_zeroing`（`[V] kv_cache_interface.py:877-879`）对 FULL 新块
  清零由原生执行；插件只读 `cached_len` 以内已写段。
- GDN 共池隔离：✓ 4 组各自从 block_pool 领块（互斥），FULL 只写自己块上的 K/V 条格；
  conv 条(A)/ssm 条(B) 的 GDN 专属页与 FULL 页是不同块号，无字节交叠。

### 5.7 副作用与慢算子清单（"AICPU 一直在多余操作"的具体化）

| # | 算子/行为 | 位置 | 为什么慢/多余 | 量级/步 |
|---|---|---|---|---|
| S1 | `torch.sort(dim=-1)`（clip 阈值） | `backend.py:100` | NPU sort 常落 AICPU/向量核；verify 步 N=4 仍每层×2 次 | 32 次/步（小张量但发射+调度重）；prefill 时 [16384,256]×2×16 层 |
| S2 | `index_put_`×10（store ref 散写） | `store_kernel.py:82-95` | 高级索引散写，逐次发射 | 160 次/步 |
| S3 | 高级索引 gather×4+（dequant） | `store_kernel.py:115-122` | 随机行聚集 + 中间 [C,1,64/8] 索引张量 | 64 次/步 + ~34MB 物化 |
| S4 | 算术解包链 //、%（NPU `>>` 缺陷绕行） | `format.py:99-110` | 每 64B 槽 8 个额外逐元素核 | 与 S3 同链 |
| S5 | `.tolist()`×2 + `.max().item()` | `backend.py:256,457` | host-device 同步 | 32+ 次/步 |
| S6 | eager 无图 | serve_oscar.sh:68 | ~1500 算子发射无法合并 | ~300-600ms/步 |
| S7 | triton 路径默认关 | `config.py:84` | 唯一能 fused 化的读路径从未启用；probe 只观察不门禁（install_and_launch.sh:181-185） | — |
| S8 | （若启用 triton）`BLOCK_KV=4` | `decode_kernel.py:291` | 每内层迭代仅 4 个 KV 位置、num_warps=1——与 PR 原版同为小 tile（PR `triton_oscar_decode.py:320` 同值），16k 上下文仍是慢核 | — |
| S9 | `repeat_interleave` GQA 展开 | `decode_kernel.py:57-58` | 死路径（DecodeOnly 不出现），但若出现则 [L,Hq,D] fp32 双物化 | 0（死代码） |

### 5.8 正确性风险残留（非本轮现象根因，但真实存在）

1. `_oscar_int2_decode` 静默吞异常（`backend.py:470-471` `except Exception: pass`）——
   若未来启用 DecodeOnly 路径，triton 失败无告警直接掉 ref。
2. `_prefill_attention` 按请求 Python 串行（`backend.py:259`）——并发 32 的 ais_bench
   实测段会把"每层每请求全前缀反量化"再乘以批内请求数（长上下文并发才是放大器；
   warmup 单请求只是下限）。
3. chunked prefill（prompt>16384）时每 chunk 全前缀反量化 O(prefix)——262k 满长下
   第 16 chunk 需反量化 ~245k token/层。
4. SDPA `enable_gqa` 在 torch-npu 的后端落点未钉死（math/fused 取舍随版本）——
   prefill 16k 实测持平，但属未钉死契约。

---

## 6. 110s vs 4.1s 定量归因总表

| 环节 | base | OSCAR | 差异来源（节） |
|---|---|---|---|
| 建连+首请求 | ~1s | ~7s | 日志窗口口径差异（非插件） |
| prefill 16k | 6.4s（2542 tok/s） | 6.7s（2461 tok/s） | S1 sort + ref store + SDPA ≈ +5% |
| decode/verify 每步 | 数十 ms（fused×2 op/层） | **~653ms** | §3 清单：路由错位（C4）+ eager 发射（S6）+ 反量化物化（S3/S4）+ 同步（S5）+ 双写（C3） |
| 精度副产物 | acceptance 4.0 | 3.0~3.65 | INT2 中段 + **sink 失效（C5）** + per-vector 粒度（R-20260904 已知项） |

**链路上没搞好的那一环，按权重排序**：
① 读路径路由（SpecDecoding→prefill 分支，设计中的 fused INT2 decode 未被使用）；
② triton 默认关 + eager（所有实现都是逐算子参考路径）；
③ staging/sink 的环形冲突（精度项）。
写路径、地址偏移、字节契约、前缀缓存兼容性均验证无恙。

---

## 7. 真机取证清单（全部非侵入：grep / 环境变量 ablation，不改代码）

1. `grep -c "★ INT2 读路径(decode)" /tmp/oscar_ascend_logs/serve.log` → **预期 0**
   （死代码直接证据；若 >0 说明本分析路由判断有误，立即回报）。
2. `grep "★ OSCAR 配置生效" serve.log | head -1` → 确认 `triton=torch参考路径`
   （C4/S7 前提）。
3. 计时 ablation A：`OSCAR_ASCEND_SINK_TOKENS=0 OSCAR_ASCEND_RECENT_TOKENS=0` 重跑
   warmup → 预期 decode 略提速（去掉 staging 写/拼接），接受率略降（纯 INT2）。
4. 计时 ablation B：`OSCAR_ASCEND_USE_TRITON=1` → 预期 prefill 写提速（store 单核）；
   decode 不变（读路径不走 triton decode——C4）。
5. 结构 ablation C：临时把 `--speculative_config` 去掉跑 1 条（用户自行决定）→
   attn_state 变 DecodeOnly → `_decode_attention` 才会首次执行——可直接验证 §5.3。
6. `grep rejection_sampler` base 侧日志同时段 → 确认 S5.5 的"与插件无关"判断。
7. （可选）npu profilermind 采一个 verify 步 → 对 §3 的 ~1500 发射/32 同步计数。

---

## 8. 与用户草稿的对照（草稿仅供风格参考，数字以本树为准）

| 草稿项 | 本树核验 |
|---|---|
| 每 rank 16 个 torch.zeros | ✓（§2.2-2.3；桶 {48,16}→group_size=16） |
| P=801,792 = 15,360+393,216+393,216 | ✓（§2.1 对齐等式链） |
| 条A conv / 条B K≡ssm / 条C V | ✓（`[VA] model_runner_v1.py:4569-4577` 注释原文 "tensor1:[(kv_padding),conv] tensor2:[k,ssm] tensor3:[v,(mamba_padding)]"） |
| 页号→三小格偏移 | ✓ 且补充：kernel 粒度是**逻辑块**（phys×6+i），插件全部按逻辑块对齐（§5.1） |
| nb=available÷P÷16 | ✓ 公式（`[V] kv_cache_utils.py:967`）；具体 nb 数值属环境量（上轮真机日志曾反推 1369，本轮无池日志，不锚死） |
| MTP 层与 FULL.0 共享池 0 | **不作断言**（草稿该点未证实；empirical 16 池账本与"17 层并入会得 17 池"矛盾，草稿解释牵强。本轮标注 [UNKNOWN-非承重]：MTP 草稿层 Attention 模块存在（手术 68→64 的差值）但其池归属不影响本文任何结论） |

---

## 9. 证据总索引（file:line）

**本仓 [W]**（HEAD `1e95146`）：
backend.py:19-42,87-105(100 sort),108-146(131-135 写日志),148-153,166-211(187-205 路由),
214-250,253-293(256 tolist),296-314(302 sink_eff),316-346(330-336 keep/环),348-363,
365-453(413-415 bt_eff,450-453 LSE),455-475(457 item,470-471 吞异常)；
config.py:44-50,84；format.py:19-33,51-64,67-88,99-110,113-125；
store_kernel.py:35-48,54-95,98-126,129-147,240-272；
decode_kernel.py:37-63(54-58 死路径),66-94(88-93 掩码),97-111,259-303(275 item,291 BLOCK_KV=4)；
dequant_kernel.py:20-116；plugin.py:88-132(111 手术),145-157,163-214(176-184 MTP 拒绝)；
delivery/serve_oscar.sh:16,32-37,44-46,48-70；delivery/install_and_launch.sh:173-191。

**[VA] vllm-ascend@19e436985**：
model_runner_v1.py:1477-1505(_build_attn_state),3119-3121,3161-3191,4116-4132(4124 zeros),
4493-4502,4559-4577,4597-4602,4637-4642,4679-4680,4696-4714,5025-5034；
attention/attention_v1.py:104-112,142-144,1174-1232,1302-1357,1363-1382,1408-1453,
1456-1475,1529-1534；worker/block_table.py:88,116-117,165-179,210-229,288-305；
worker/worker.py:681-692,918；utils.py:1369-1419(1380-1406)；ascend_config.py:696-701；
patch/platform/patch_mamba_config.py:58-117；patch/worker/patch_qwen3_5.py:153-192；
quantization/methods/kv_c8.py:119-139。

**[V] vllm@0fc695fc**：
v1/core/sched/scheduler.py:105-109,217-221,293-344,403-407,462-505,519-531,684-699,926-930,1127；
v1/core/kv_cache_manager.py:58-84,221,385-434,446,549-551；
v1/core/kv_cache_coordinator.py:451-463,544-546,582-686,689-741；
v1/core/single_type_kv_cache_manager.py:276-289,363,552-568,1007-1013,1121-1195；
v1/core/kv_cache_utils.py:584,1142-1195,1318-1326,952-969,899-917；
v1/kv_cache_interface.py:101-112,167-200,614-636,829-871,877-879；
v1/worker/gpu_model_runner.py:1399,1891,1899,1985,2101,2146-2174,2232-2248,2289-2305,2445-2446；
v1/request.py:247。

**[PR] oscar-vllm-pr46774@57286d5d**：
vllm/v1/attention/backends/oscar_attn.py:54-58,87-103,126-157,194-199,253-336(282 双写,303-317 tag),
338-345,515-518,618-750(671-685 cut/bt_eff,745-750 LSE)；
model_executor/layers/quantization/oscar/config.py:25,81,99-130,120-128,154-163,178-192；
vllm/v1/attention/ops/triton_oscar_store.py:64-73,99-112,152-190；
vllm/v1/attention/ops/triton_oscar_decode.py:21,58,61-63,267-292,320,335。

---

*写作：2026-09-04（分析模式，工作区零改动）。本文所有"机制推导"类结论（§3 计数、§5.4 时序）
均已给出可证伪的取证命令（§7）；凡未能亲自复现的行号均标注来源（agent 探查 + 人工抽核）。*
