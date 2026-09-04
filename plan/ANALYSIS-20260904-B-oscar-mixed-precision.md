# ANALYSIS-20260904-B — OSCAR 混合精度在"先分配后插件"布局下的可行性与副作用

> 配套：ANALYSIS-20260904-A（链路与证据）。本文件回答两个问题：
> **Q1 混合精度（sink/recent BF16 + 中段 INT2）能否兑现"显存收益"？**
> **Q2 当前布局下 INT2 写/读是否系统性正确？sink/INT2/recent 混合是否真落地？副作用（AiCpu、地址、MTP、前缀缓存）？**
> 结论证据均来自当前代码仓（工作区 `oscar_ascend/` + `$SK/references/`，file:line 见 A 文）。

## Q1 — 混合精度能否释放 KV 显存？→ 不能（当前实现只做带宽，不做显存）

### 1.1 池子定型顺序（铁的事实）

1. `model_runner_v1.py:4080-4132`：**16×`torch.zeros(P×nb, int8)`**，P=801,792（BF16 页几何：
   `patch_mamba_config.py:91-117`：512B/token·head×768 token 页 + ssm 393,216 同条 + conv 15,360）。
2. `torch.zeros` 完成时，**HBM 池已按"每 token·head K/V 各 512B"的 bf16 合同落定**。
3. 插件在此**之后**经 `impl.__class__` 手术接入（`plugin.py:111`），唯一能做的是
   **在已分配的 512B 槽里重解释前 96B(K)/64B(V)** —— 池形状、nb、块表长度、`KV usage%`
   全部与原生一致（本机日志：前后均为 `KV cache usage: 4.1%`；池总量 16.36GiB 不变）。

### 1.2 因此"混合精度"只存在于**读路径语义**，不在**内存布局**：

```text
内存：每个 FULL token·head 槽 = 512B×2（K/V）——被插件写成 160B 逻辑槽 + 其余字节归零
读取：中段(INT2 主体)   ← 反量化 160B 逻辑槽
      recent 256 token ← `_oscar_stage_k/v`（**额外**的 BF16 舞台，rows×bs×Hk×D×2B×2）
      sink 128 token   ← 同上舞台（但见 1.3 的环碰撞）
```
→ **"sink/INT2/recent 混合精度"是读侧三层覆盖，不是三档内存分区。** 显存总量要么是
  分配时定死的 16.36GiB（+插件额外 ~128MB/rank 舞台，实为**净增**），要么等另一个
  dtype/几何方案（见 1.4）。

### 1.3 舞台（sink/recent BF16）现状：sink 在长序列上**静默失效**

- `_ensure_staging`：`rows = max(⌈8192/128⌉, …) = 64` 行；`_staging_write` 的环键：
  `rows = (slot//bs) % 64`（`backend.py:_staging_write`）—— **128-逻辑块 id 取模 64**。
- 16384-token 序列：块 0（sink）与块 64..128（中段/尾段）**同环行 0..63**；prefill 一次性写入
  时，块 64..127 依次**覆盖**块 0..63 的舞台条目 ⇒ sink 槽 owner 键被尾段块顶替。
- `_decode_attention_windowed` 的 `sink_active = (seq>S_eff) & s_staged.all()` 判假 ⇒
  **sink 段自动回退 INT2**（降级无告警）。即：**任何 seq ≥ ~8192（且尾页行号与 0 碰撞，本例
  16384 恰中）时，BF16-sink 保护实际为 0；recent 256 仍生效**（尾段行未被后续覆盖）。
- 参考实现（oscar-vllm PR oscar_attn.py:245-336/618-750）同款代码同款行为——其测试
  仅覆盖 S=100/50 的短序列，故该特性从未在长上下文上验证 → **这是"混合精度"在
  本部署中名存实亡的第一处**。

### 1.4 真正压缩为何"不可行/需大改"（hybrid 对齐约束）

- 参考 PR 的显存收益来自 `kv_cache_dtype="oscar_int2"` + `get_kv_cache_shape` 返回
  `(num_blocks, bs, Hk, slot_size_aligned)` —— **分配期**就按压缩槽大小。
- 但本混合模型的对齐等式（`patch_mamba_config.py:95-97`）：
  `attn_single_token_k_page_size × attn_block_size == ssm_block_page_size`
  `512 × 768 == 393,216` —— K 页每 token 512B 是**前提**；若 K 槽改 160B：
  `160 × 768 = 122,880 ≠ 393,216`；反解 `attn_block_size = 393,216/160 = 2,457.6` **非整数**
  ⇒ 无法在不动 GDN 页宽（12×128×128×2B）的前提下满足"K 条与 ssm 条同宽"。
- 结论：**要在本模型兑现真正的显存压缩，必须同时改三处**（比"改槽位布局"大得多）：
  ① `patch_mamba_config` 的页对齐公式/ssm 页宽；② vllm `cache_config.block_size`
  与 KVCacheSpec 的 page_size_bytes；③ 分配器/reshape 分支。当前插件坚持"零侵入不动
  几何"（L-20260903-01 决策），因此**它交付的是带宽收益（写 160B vs 512B，-68.8% 为日志
  自证），显存收益为 0（甚至 -128MB/rank 舞台）**。
- 文档佐证：`README.md` 的"IO 160B (原生 512B)"本就只宣称写入开销；`KV usage 4.1%`
  不变即池不变。

---

## Q2 — 当前布局能否系统性正确读写？混合机制与副作用

### 2.1 地址正确性：✅（槽位/块表/视图三层一致，有旁证）

| 层 | 插件假设 | 原生契约 | 证据 |
|---|---|---|---|
| k/v 视图 | `(nb×6,128,Hk,D)` 连续 bf16 | `model_runner_v1.py:4637-4642` | 一致（视图=raw 切片 .view，连续） |
| 逻辑块粒度 | `bs=128`（取视图 dim1） | `block_table.py:66-88`（logical=phys×6+i） | 一致（视图 dim0 恰为逻辑 id 序） |
| 槽号 | `slot//128, slot%128` | `block_table.py:225-228`（logical×128+off） | 一致（双向同式） |
| 槽内字节 | K meta0-7+pad8-31+idx32-95；V idx0-63 | N-01 契约 + probe | **真机 ref/triton 字节差=0**（05:14 日志） |
| 页格不相交 | 只写 FULL 所属页格 | GDN 页号空间不同 | 无交集（ANALYSIS-A §2.3） |
| 与 MTP 同池 | FULL.0 与 MTP 同池 0（条 B/C 不同页） | 池 0 shared_by=5 | 草稿层本轮已原生化，无插件写入 |

**结论**：写/读不"读错地址"——字节级 probe 全绿 + 接受率恢复（下节）是最强旁证。

### 2.2 混合机制真正落地状态

| 机制 | 是否生效 | 说明 |
|---|---|---|
| INT2 主体 | ✅ | 160B 逻辑槽写/读；decode 全上下文反量化 |
| recent 256 BF16 | ✅（本负载） | 舞台尾段行未被覆盖；窗口化合并 LSE |
| sink 128 BF16 | ❌（16384 序列） | §1.3 环碰撞 → 静默降级 INT2 |
| MTP 草稿层 | ✅（本轮修复后为原生 BF16） | 不再被 INT2 污染；接受率 9.5→66-88% 主因之一 |
| 前缀命中复用 | ⚠️ [UNKNOWN] | OSCAR 跑批 Prefix hit 0.0% vs base 48.4%；vLLM 前缀缓存按 token 哈希建块，
   与缓存内容无关，疑似**请求序列/调度差异**或缓存标记路径差异——需同请求 A/B 复核 |
| clip 0.96/0.92 | ✅（sort-based 已真机打印） | 每 token·head 一次 sort —— 见副作用 S2 |

### 2.3 副作用清单（按观测/机制排序）

**S1 · 性能坍缩（主诉：warmup 110.12s vs base 4.10s，约 27×）** — 机制：
- MTP 下目标层每投机验证步 `attn_state=SpecDecoding`（`model_runner_v1.py:1483-1486`），
  插件 `forward` 只有 DecodeOnly/else 二分（`backend.py:172-175`）→ 每步走
  `_prefill_attention`：**全前缀** dequant（16 层 × ~16K 上下文 × K/V）+ 2 次逆旋转 matmul
  + stage splice + 每请求 SDPA；外加每层 2 次 `.tolist()` CPU 同步（`metadata_batch_lists`）。
- warmup 单样例 = 16384-token prefill：16 层 × (16384×256×2×2B 读 + dequant + 16K×16K SDPA)；
  base 等同工作由原生 FIA paged 路径完成（无逐 token 反量化）。
- 观测：generation 5-6 tok/s 且 `Avg prompt throughput 2461` 后无 prefill 吞吐（长上下文
  每步都全量反量化）——与"每次 decode 都在做全上下文反量化"一致。

**S2 · AiCpu 多余操作** — 日志 05:21:38 四 rank：
`ArgSortKernelNpuOpApi: kernel [ArgSort] can not support dtype int32 or int64 on AiCore,
Now this kernel is running on AiCpu` —— 来自 `backend.py::_rotate_clip` 的
`x_rot.abs().sort(dim=-1)`（sort 返回 values+indices，indices=int64；仅需要 values）。
每个 token·head×层都触发：16384×16×4rank 次/预填 ⇒ AiCpu 是铁证（并非"猜想"）。修复只需
取 `.values`（或 `torch.topk`），但属于"核心代码改动"，本轮不改，仅记录。

**S3 · 显存净增（而非压缩）** — 舞台 `_oscar_stage_k/v`：
rows 64×bs 128×Hk 1×D 256×2B×2 ≈ 8MB/层 → ×16 层 = 128MB/rank（相对 16.36GiB 池占比极小，
但方向相反：**+0.75% 而非 -84%**）。"KV 带宽 -84%"的承诺只对齐 IO 字节，不对齐池占用。

**S4 · 草稿层已修复（正向）**：MTP 层 `[SKIP] mtp-draft`（`plugin.py:_should_oscar`），
草稿 KV 全 BF16 原生 SpecDecoding 路径 —— 这是接受率从 9.5-25% → 66-88% 的两大主因之一
（另一主因=U·H·P 旋转 + clip）。

**S5 · 残留接受率差距（vs base 100.0% 单位置/均值）**：
- base 05:33 warmup：`1.000/1.000/1.000, Avg 100.0%`；OSCAR 本轮：单位置 0.765~1.000、
  均值 42.6~88.2%（多数窗口 64-88%）。
- 机制候选（按实验证据强度排序）：
  ① **per-vector INT2 噪声**（合成实验：KL ≈ 1.7，group16 → 0.09）——靶向层 K/V 噪声
     使验证步 logits 与草稿预测偏差 → 2/3 位接受率下降（观察到的恰是 pos2/3 降）；
  ② **SpecDecoding→prefill 分支**：插件重建 `k_pos <= q_pos` 因果掩码，原生 FIA 用
     metadata attn_mask（`step3p5.py:173-174` 语义）——两者对 MTP 验证步**未逐字核验**
     [UNKNOWN]；且每步全前缀反量化将其放大（S1 同源）。
  ③ sink 降级（§1.3）使 16384 上下文"开头 token"也为 INT2 —— 位置 2/3 的验证更依赖
     早期上下文时受损略高于理论。

**S6 · [SKIP] 计数与[2]/[3] 计数**：`[SKIP] mtp-draft×4`（4 rank）符合预期；[2]/[3]=64
=16 层×4 rank（不再 68）；`check_oscar_active.sh` 的"应为 16"措辞应理解为"每 rank 16 层"。

**S7 · 前缀命中 0% 待查**：见 §2.2（[UNKNOWN]，需同负载 A/B 排除调度差异）。

---

## 3. 一页结论

| # | 判断 | 证据强度 |
|---|---|---|
| Q1 | **当前实现不释放 KV 显存**：池在插件前按 BF16 几何定型；"160B"只是 512B 槽内的
        重解释；显存收益=0（+128MB/rank 舞台）。"sink/recent INT2 混合"只是**读侧三层
        覆盖**，且 sink 在 ≥8K 序列因环碰撞静默失效 | 运行日志（16.36GiB/KV 4.1% 不变）
        + 代码几何 + 参考 PR 的对比 |
| Q1' | **真正压缩在 hybrid 上不可行（不动 GDN 页前提下）**：512×768=393,216 对齐等式
        与 160B/槽互斥（2,457.6 非整数）⇒ 需同时改页对齐/spec/分配三层 | 等式算术（必演） |
| Q2 | **写读地址系统性正确**（槽位/块表/视图三层一致；probe 字节差=0；接受率回升） | 代码双式对拍 + 真机 probe |
| Q2 | **副作用实锤**：S1 每步全前缀反量化（warmup 27×）；S2 ArgSort AiCpu；S3 显存净增；
        S5 pos2/3 残差（per-vector INT2 为主因，group16 是下一杠杆） | 真机日志 110s/4.1s、
        ArgSort 警告 ×4rank、接受率窗口 |
| Q2 | 待验：S7 前缀缓存 0%（[UNKNOWN]）——建议同负载 A/B；S2 修复=取 `.values`（属核心代码改动，本模式不动） | [UNKNOWN] |
