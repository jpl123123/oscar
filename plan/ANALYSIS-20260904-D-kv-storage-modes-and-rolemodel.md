# ANALYSIS-20260904-D — KV Cache 布局/读写全景 与 Role Model 判决（v2 全量重构）

> **v2 说明**：本文由多轮真机问答验证的事实重构而成，删除了 v1 的过程性冗余（勘误
> 记录、重复图、重复解释）；所有数字经"源码亲读 + python 独立复算 + 真机日志对账"
> 三重验证。**版本指纹**：工作区 `a874b5d`（packed×2 已上线）；
> 参考树 `/Users/sunao2000/oscar_zai/references/`：vllm=`0fc695fc`（v0.23.0）、
> vllm-ascend=`19e436985`（v0.23.0+PR#12607，下称 [VA]）、oscar-vllm-pr46774=`57286d5d`
> （下称 [PR]）、vllm 下称 [V]、本仓 [W]。模型：Qwen3.5-27B-w8a8-mtp
> （48 GDN + 16 FULL + 1 MTP，TP4；单卡 Hk=1、Hq=8、D=256——真机日志 heads=1 自证）。

---

## 0. 结论速览

| # | 结论 | 节 |
|---|---|---|
| C1 | 池 = 16 根一维 int8 张量；字节序 = 三个**稠密区**（conv / K≡ssm 共区 / V），"页"（801,792B）只是记账单位 | §2.2 |
| C2 | 对齐等式 `512×768==393,216` 的意义：让"GDN 一份 ssm 态"与"FULL 一页 K"字节级同宽，双视图共用一个块号空间互斥共存 | §2.2 |
| C3 | nb = 预算 ÷ (页宽×16 池)；910B4 实算 nb=1,369（锚：真机日志 16.36GiB） | §2.3 |
| C4 | 最初版 OSCAR 是"假压缩"（512B 格装 64B，引擎不知道，0 节省）；packed×2 把槽宽真改成 256B → **×2 真节省**；PR 紧排是 ×7.5（需源码） | §4 |
| C5 | MTP 下 decode 读路径是死代码（attn_state 恒 SpecDecoding）→ 窗口三段合并 0 次执行；sink 受环冲突在长上下文失效 | §5 |
| C6 | 地址/字节契约本身正确（stride 取活视图 + 逻辑块表一致 + probe 字节差=0）；问题全部是性能型/路由型 | §3.3 |
| C7 | Role model = **TQ 工程蓝图（分配期缩池 + hybrid 修法 #39931 + 小续算复用 decode）+ OSCAR PR（后端本体 + 数值配方）** | §7 |

---

## 1. 三模式定位与归属取证

| 模式 | 代码位置 | 状态 |
|---|---|---|
| ① 插件版 OSCAR（packed×2） | 本仓 `oscar_ascend/` | 真机 ACTIVE；×2 密度已兑现 |
| ② OSCAR vLLM PR#46774 | [PR]（快照仅 PR 新增 11 文件） | PR 分支；**未处理 hybrid/Mamba**（config.py:178-192） |
| ③ TurboQuant | **upstream vllm v0.23.0**（本地 [V] 全套） | 已合入（KV=#38479 2026-04-15；hybrid 修复=#39931 2026-05-05）；CUDA/ROCm/XPU |

**归属取证（2026-09-04/05 实测）**：
- vllm-ascend 官方**无 TQ**：本地 v0.23.0 树、main 分支（`4fe7ddb` 稀疏克隆）、release
  notes（271,793B）三处 grep 零命中。
- `varjoranta/turboquant-vllm`（`c8a7e0a`）README:18 自述：GQA/MHA 的 KV 压缩已上上游
  vLLM（#38479，`--kv-cache-dtype turboquant_3bit_nc` 等），本仓库今后只做权重压缩。
  其 README:126-137 模型表中 Qwen3.6-27B（16 full + 48 GDN 同构）的 "15.4GB (3.6×)"
  是**权重检查点**压缩（KV 列 Untested）。
- **在 vllm-ascend 上传 `turboquant_*` 字符串**：经 [V] torch_utils.py:46-49 映射 uint8
  → patch_mamba_config 算出 256B/token、block 1,536——**与 `int8_per_token_head` 完全
  同款**；spec 会生成 TQFullAttentionSpec，但 vllm-ascend 分配器不读 `tq_slot_size` →
  int4 语义从未落地。"借入口"= 换名字的同一档。

---

## 2. 池的解剖（五步主线）

### 2.1 需求：两种记忆

FULL 层的记忆 = 每 token 的 K/V（D=256 bf16 → 每 token·head 1,024B，随序列**线性增长**）；
GDN 层的记忆 = **定尺寸**循环状态（conv 抽头 15,360B + ssm 状态矩阵 393,216B，**永不增长**）。
两种记忆、两个尺寸，要住进同一片显存。

### 2.2 结构：16 池 × 三稠密区 × 页宽 801,792B

**分组**：48 GDN + 16 FULL 按同型分桶、交错分组（[V] kv_cache_utils.py:1142-1195，
FIXME 注释 :1146-1157 讲的就是避免逐组 padding）→ 4 组各 16 层 → **16 个池**，第 i 池住
[GDN组0第i层, 组1第i层, 组2第i层, FULL第i层]。每池一根 `torch.zeros(P×nb, dtype=int8)`
（[VA] model_runner_v1.py:4124）——**池 dtype 永远 int8，与 `--kv-cache-dtype` 无关**。

**三稠密区（一根张量的真实字节序，非页内交织）**——GDN 从头顺序切（:4696-4714，官方
注释 :4701-4704），FULL 在尾部切（:4574-4577）：

```text
字节偏移      0 ────────── 15,360·nb ────────── 408,576·nb ────────── 801,792·nb
              ┌──────────────┬──────────────────────┬──────────────────┐
              │ 区A: conv 格   │ 区B: K/ssm 双视图共区   │ 区C: V 格（FULL 专用）│
              └──────────────┴──────────────────────┴──────────────────┘
  区A 视图: raw[0:15,360nb].view((nb,2560,3))                     ← GDN 专用
  区B 视图一(FULL): raw[15,360nb:408,576nb].view((nb×6,128,1,256))bf16
  区B 视图二(GDN): 同一段字节 .view((nb,12,128,128))bf16            ← K≡ssm 同字节双视图
  区C 视图(FULL): raw[408,576nb:801,792nb].view((nb×6,128,1,256))
```

**每格字节的算法**：
- 区A 每格 15,360B = conv 状态 (2,560,3)bf16；2,560 = conv_dim÷TP4 =
  (2×128×16 + 128×48)÷4（[V] mamba_utils.py:218-226 公式；头数分解由真机值对账[推断]）；
  3 = conv_kernel−1（卷积历史抽头）。
- 区B 每格 393,216B，**两种身份必须同宽**：GDN 侧 = ssm 状态 (12,128,128)bf16 =
  12×128×128×2B（12 = num_v_heads÷TP4=48÷4；[V] mamba_utils.py:228-232）；FULL 侧 =
  一物理页 K 跨度 768×1×256×2B。两算同值 ⇔ `assert 512×768==393,216`
  （patch_mamba_config.py:95-97）。K 视图每格再切 6 份（6=768÷128 kernel 块，每份
  65,536B=128×1×256×2B）。
- 区C 每格 393,216B = 一页 V 跨度（K/V 每 token·head 同为 D×2B → 必然同宽；单一视图，
  无对齐约束，宽度跟随 K）。

**页 = 记账单位**：块号 b 在三区各占一格，合计 801,792B = 页宽（[V]
kv_cache_interface.py:830 "size in bytes"；memory_per_block = 页宽×16，
kv_cache_utils.py:911）。

**块号身份决定格子用法**（块号由全局唯一 BlockPool 发牌、引用计数互斥）：
- FULL 领走 → 区B 格当 K（=6 逻辑块）+ 区C 格当 V；**区A 格死置**（~1.9%/块，官方注释
  名 "(kv_padding)"）；
- GDN 领走 → 区A 装 conv + 区B 装 ssm；**区C 格死置**（~49%/块，"(mamba_padding)"）。
  死格随块号归属流动（改派即复活）。

**为什么 ssm 必须与 K 混存（而不是 K+V 一区、ssm 独占一区）**——三层：
① 算账：不重叠则每块号四格（conv/K/V/ssm）→ 页宽 1,195,008B（+49%），同预算
nb 1,369→918（-33%）；混存让 K 格与 ssm 格**是同一批字节**，一个块号只花 801,792。
② 框架：单一 BlockPool/统一页记账要求 GDN 页与 FULL 页等宽（分组机制的存在意义）；
等宽且总量不膨胀的唯一解 = 字节重叠。③ 几何：GDN span 锚定头部、FULL span 锚定尾部
→ 交集被强制为 [15,360nb, 408,576nb) = 恰好 K 半区（叠 K 还是 V 纯属切片顺序偶然）。

**三层 dtype**：① 池 = int8（:4124，原生/packed 同一行代码）；② K/V 视图 dtype 才是
被 `--kv-cache-dtype` 改的东西（attention.py:225→276→spec.dtype→model_runner:4628→
:4637 `.view(dtype)`）：auto→bf16（512B 槽）/ int8_per_token_head→int8（256B 槽）；
③ OSCAR 无 dtype——`k_cache.view(torch.uint8)`（store_kernel.py:61）仅按字节寻址，
int2 是字节约定。该参数喂两个独立消费者：patch_mamba_config:53-56（只吃字节数→页
几何）与视图 dtype；**不喂池**。命名陷阱：`int8_per_token_head` 的量化语义在
vllm-ascend 无消费者（grep 零命中）——只借"1 字节"当门把手。

### 2.3 容量：nb = 预算 ÷ 单价（910B4 算例）

公式（[V] kv_cache_utils.py:967）：`nb = available_memory ÷ 801,792 ÷ 16`。
**直觉**：块号=车位，全卡单价 = 页宽×16 池 = 12.23MiB/块号（一个块号必须在 16 个池
同步各占一页）。

```text
910B4 预算瀑布（TP4 单卡）：32GiB × 0.9 = 28.8 − 权重~7(估) − 运行/激活~6(估) ≈ 16
实证锚：上轮真机日志 "GPU KV cache size: 16.36 GiB"
→ nb = int(16.36×2³⁰ ÷ 801,792 ÷ 16) = 1,369（吃满 16.356GiB 对账 ✓）
   单池 = 801,792×1,369 = 1,097,653,248B ≈ 1.022GiB；全卡 16 池 ≈ 16.36GiB
   三区边界：区A [0, 21,027,840) 宽 20.05MiB；区B [21,027,840, 559,340,544) 宽 513.38MiB；
             区C [559,340,544, 1,097,653,248) 宽 513.38MiB（互不重叠恰好铺满 ✓）
```

（nb 是池容量，与请求无关；OOM 修复前那轮真机 nb 曾掉到 1,115——插件路径的肥大临时
峰值被计入"激活峰值"，KV 预算被吃 ~2GB，prefill 分块修复后预期回升，见 TASK_STATUS。）

### 2.4 使用：一条 32K 请求（TP4，单卡一池）

请求消耗：FULL `cdiv(32,768+3, 768)=43` 块（lookahead [V] scheduler.py:462）；
GDN 3 组 × 记账 5 块（align 模式 `2+num_spec(3)`，[V] kv_cache_interface.py:629-631）
= 记账 15 / **驻留 12**（每组块表 46 项中仅最后 `1+3=4` 个真实块，其余为共享 null_block
占位不占块号——[V] single_type_kv_cache_manager.py:1142-1165）。合计 **58 块号 =
4.24%**；本池占用 = 43×786,432 + 15×408,576 ≈ 39.9MB；全卡 16 池 ≈ 609MiB。

```text
═══ 统一主轴 = 物理块号（0…1,368）═══  本请求领走 0–42（FULL）+ 43–57（GDN）

区A  conv 尺 [0, 21,027,840)  每格 15,360B
  块号 →   0 …… 42      │ 43…47  │ 48…52  │ 53…57  │   58 …… 1368
        ┌───────────────┬─────────┬─────────┬─────────┬────────────────┐
        │  死 ×43 格      │组0·第i层 │组1·第i层 │组2·第i层 │    空 ×1,311 格  │
        │ （FULL 块）     │ conv×5格 │ conv×5格 │ conv×5格 │                │
        └───────────────┴─────────┴─────────┴─────────┴────────────────┘

区B  K/ssm 共区 [21,027,840, 559,340,544)  每格 393,216B
  块号 →   0 …………………… 42（FULL 领）      43…47   48…52   53…57（GDN 领）  58 … 1368
        ┌───────────────────────────────┬────────┬────────┬────────┬──────────────┐
        │  K 视图每格再切 6 份：            │组0·第i层│组1·第i层│组2·第i层│     空 ×1,311   │
        │  块0→K[0..5] … 块42→K[252..257]  │ ssm×5格 │ ssm×5格 │ ssm×5格 │      格        │
        └───────────────────────────────┴────────┴────────┴────────┴──────────────┘
  每层 5 格 = 1 提交态 + 3 投机态（MTP draft 前推，拒绝回滚用）+ 1 过渡缓冲；稳态实块 4
  读数：K 视图 (8,214=1,369×6, 128, 1, 256)bf16 → 43 格×6 = 258 逻辑块×65,536B = 16.9MB
        ssm 视图 (1,369, 12, 128, 128)bf16 → 3 层×5 格×393,216B = 5.9MB

  ★ 跨池正交方向：组0 的块号 43–47 同时存在于 16 个池（池 i 装组0第 i 层的状态）——
    "×3 组"是池内方向，"×16 池"是跨池方向，3×16=48 个 GDN 层全覆盖。

区C  V 尺 [559,340,544, 1,097,653,248)  每格再切 6（与 K 同构）
  块号 →   0 …… 42（258 逻辑块=16.9MB）  43 …… 57（死×15）   58 …… 1368（空×1,311）
```

**GDN 的 ssm 是什么**：DeltaNet 递归状态矩阵 S ∈ R^{Hv×dv×dk} = (12,128,128)/rank——
线性注意力的全部历史**压缩进这一个固定矩阵**（不随序列增长；这也是 chunked prefill
必须按块对齐的原因：状态只能在整块边界 checkpoint）。conv = 因果短卷积最后 3 个抽头。

**token 寻址链**（t=20,000，设块表[26]=物理 7）：
`t//768=26 → 块表[26]=7 → 页内 32 → 逻辑块 L=7×6+0=42 → K 字节 = 21,027,840 +
42×65,536 + 32×512 = 23,796,736；V = 559,340,544 + 同偏移；slot_mapping = 42×128+32 = 5,408`
（逻辑块=phys×6+i：[VA] block_table.py:288-303；slot=逻辑块×128+偏移：:210-229）。

**关键数字问答**：
- "43×6" 的 6 = `spec.block_size(768) ÷ kernel_block(128)`（[VA] :4562）——分配按
  768 记账、内核按 128 翻页；
- 128 是后端声明的 kernel block（attention_v1.py:142-144），= 块表分页粒度，**与 NPU
  片上 UB 大小无已证关系**（aclnn 内部 tile 是算子细节）；
- K 视图 (8,214,128,1,256)：dim0=逻辑块数，dim2=**Hk/rank=1**（TP4，真机 heads=1）。

### 2.5 真实字节布局总图（五放大窗，偏移全部实算）

```text
━━ 总览 ━━ 0 ─── 21,027,840 ─── 559,340,544 ─── 1,097,653,248（三区，铺满无空洞）
窗① 张量头部（区A）: conv[0]=[0,15,360) … conv[42]=[645,120,660,480) 死×43格 │
    conv[43..57] 活×15 格（3 个 GDN 层各 5 格）│ conv[58..1368] 空×1,311 格
窗② 区A|区B 交界 @21,027,840: conv[1368] 尾 = K[0] 头 = ssm[0] 头（三合一）
窗③ 区B 双标签段: K 视角 K[252..263]… / ssm 视角 ssm[42..57]——同一字节
    [37,936,128, 38,329,344) 上行读作 K[258..263]、下行读作组0 的 ssm 状态（逐字节重合）
窗④ 区B|区C 交界 @559,340,544: ssm[1368] 尾 = K[8213] 尾 = V[0] 头
窗⑤ 区C: V[0..257] 活×258 块 │ V[258..347] 死×90 块（GDN 块的 V 格）│ V[348..8213] 空
```

读图三规则：**地址 0→1,097,653,248 连续排满无空洞；区B 同一段字节两行标签（65,536B
切 vs 393,216B 整格）逐字节重合；死/活只跟块号归属走**。佐证：官方 NOTE 原文
"in order to keep all tensor contiguous, we align ssm and kv block with same page
size"（:4688-4689）——稠密三区是 Ascend 连续性约束选中的；我们 probe 的
`assert k8.is_contiguous()`（store_kernel.py:63）在运行时守护。

### 2.6 总账与取舍

三个聪明处：① ssm 搭乘 K（页宽省 1/3）；② 4 组共享 16 池（免逐组 padding）；
③ 单一块号空间（调度/前缀缓存/MTP lookahead 统一）。代价：死格（统一记账的学费；
32K 请求 FULL 死格 ≈10.1MiB/卡、GDN 死格 ≈89.8MiB/卡）。

---

## 3. OSCAR 的写入：从比特到槽

### 3.1 基础账

bit = 0/1；Byte = 8 bit = **内存最小刻度**。类型 = 每值花几 bit：bf16=16bit、
int8=8bit（1B/值）、**int2=2bit（0.25B/值，仅 4 档）**。int2 的 0.25B 靠 **4 值拼 1
字节**落地（`byte = idx₀|idx₁<<2|idx₂<<4|idx₃<<6`，取回 `(byte>>2k)&3`；torch-npu
的 `>>` 不支持广播，参考实现用 `p%4、p//4%4…` 算术链——format.py:99-110）。

### 3.2 原生 vs OSCAR（一个 V 向量，256 维）

```text
原生 = 照片：256 个 bf16 原值 × 2B = 512B 一字排开，直存直读，信息率 100%
OSCAR = 编号+图例：
  ① 量化   idx = round((v − zero)/scale) ∈ {0,1,2,3}
  ② 图例   V scale + V zero（fp16 各 2B，共 4B）
  ③ 打包   256 维 ÷ 4 值/字节 = 64B
  读回     v ≈ idx × scale + zero（每维 4 种取值）
账本：512B vs 64B+4B = 68B ≈ 7.5 压缩
```

误差三件套（为什么 4 档也敢用）：**旋转**（存前乘正交矩阵摊平能量）、**裁剪**
（分位数 K 0.96/V 0.92 砍离群）、**sink/recent 窗口**（头尾关键 token 另存 BF16）。

### 3.3 槽内字节布局（packed×2，每 token·head；审计版）

```text
区B 的 K 槽（256B）：
  [+0..+8)   元数据：K scale/zero ∥ V scale/zero（各 fp16 2B 小端；V 的图例住 K 槽）
  [+8..+32)  垫 24B（编号区凑 32B 对齐，K_IDX_OFF=32）
  [+32..+96) K 编号 64B
  [+96..+256) 闲置 160B（group16 预留）
区C 的 V 槽（256B）：
  [+0..+64)  V 编号 64B；[+64..+256) 闲置 192B（group16 预留）
逻辑槽 160B = K 槽前 96B ⊕ V 槽前 64B（字节序 = N-01 契约，format.py:19-33；
probe make_slot_bytes 对账字节差=0）
```

**审计表**（数字 ↔ 代码/日志三重验证）：

| 数字 | 出处 | 结果 |
|---|---|---|
| K 跨度 96B（8+24+64）、V 64B、逻辑槽 160B；**信息 136B**（24B 为垫） | format.py:19-33 | ✓ |
| K 槽写 +0..8 元数据（含 V 图例@+4..8）、+32..96 编号；V 槽 +0..64 | store_kernel.py:82-95（index_put_ 逐偏移） | ✓ |
| 槽宽 512/256 = D×dtype 字节 | backend.py:164-172 几何对账断言；真机日志"★ 几何对账: K 槽 256B" | ✓ |
| 每格 768/1,536 token、格宽 393,216B 两算同值 | 768×512=1,536×256=393,216；真机影子池 shape 13,380=nb×12→block 1,536 | ✓ |
| 写入率：跨度 37.5%（K 格）/25%（V 格）、信息 28.1%、token·head 31.25% | 复算 147,456/98,304/110,592/393,216 与 160/512 | ✓ |

---

## 4. packed×2：从假压缩到真节省

**记账机制**：引擎只按**槽宽度**记账（发块号/算 usage/定并发）。最初版槽宽 512B 没变、
槽内放 64B——记账员不知道 ⇒ **0 节省**（假压缩）。packed×2 经 `--kv-cache-dtype
int8_per_token_head` 把槽宽真改成 256B（视图 int8）⇒ 记账员按 256B/token 发牌 ⇒
同预算容量 ×2（**真节省**：KV usage 减半、同池并发翻倍；三区/页宽/nb 全部不变，
变的只有"每格装 1,536 token 而非 768"——32K 请求 FULL 侧 43→22 块号）。

```text
━━ 格级对比（▓信息 ░垫 空=闲置；同一个 K 格 393,216B）━━
原生 bf16：一格 768 token，每 token 512B 全信息
    ┌─────────────┬─────────────┬─────┬─────────────────────┬─────────────┐
    │ token 0     │ token 1     │  …  │                     │ token 767   │
    │▓▓▓▓▓▓▓▓▓▓▓▓▓│▓▓▓▓▓▓▓▓▓▓▓▓▓│     │                     │▓▓▓▓▓▓▓▓▓▓▓▓▓│
    └─────────────┴─────────────┴─────┴─────────────────────┴─────────────┘
    768×512 = 393,216 ✓ · 信息率 100% · token·head K+V = 1,024B
OSCAR packed×2：同一格 1,536 token，每 token 256B（写 96B 跨度 = 72 信息 + 24 垫）
    ┌────────┬────────┬─────┬──────────────────────────────┬────────┐
    │token 0 │token 1 │  …  │                              │token1535│
    │▓▓░░▓▓▓▓│▓▓░░▓▓▓▓│     │                              │▓▓░░▓▓▓▓│
    └────────┴────────┴─────┴──────────────────────────────┴────────┘
    1,536×256 = 393,216 ✓（格子一字未变）· token·head K+V = 512B
    跨度写入 147,456B（37.5%）· 信息 110,592B（28.1%）

━━ token 槽内放大 ━━
原生 K/V 槽（各 512B）= 256 个 bf16 原值满排
OSCAR K 槽（256B）= [8B 元数据▓│24B 垫░│64B K 编号▓│160B 闲置（group16 预留）]
OSCAR V 槽（256B）= [64B V 编号▓│192B 闲置（group16 预留）]
```

| 口径（每 token·head，K+V） | 原生 | OSCAR packed×2 |
|---|---|---|
| 引擎记账宽度 | 1,024B | **512B** ← ×2 密度来源 |
| OSCAR 写入跨度 | — | 160B（K 96 + V 64） |
| 其中信息 | 1,024B | **136B**（8 元数据 + 64 K 编号 + 64 V 编号） |

**四代阶梯**：

| 阶段 | V 槽宽 | 实际写入 | 闲置 | 密度 |
|---|---|---|---|---|
| 原生 bf16 | 512B | 512B 原值 | 0 | 1× |
| OSCAR 最初版（假压缩） | 512B | 64B | 448B 全浪费 | **1×（0 节省）** |
| **packed×2（现状）** | **256B** | 64B | 192B | **2×（真节省）** |
| PR 紧凑布局（需源码 PR） | 无格子，68B 连排 | 68B | ~0 | **7.5×** |

**MTP 草稿层（packed×2 的配套）**：原生路径在 int8 池上无 per_token_head 算子 →
[W] `mtp_shadow.py` 把 MTP 层 kv_cache 入参替换为页外 BF16 影子池（形状取自原生逻辑
视图、块号一一对应，原生 forward :1527 自行重绑定 → 写读 100% 原生算子）。≈2.15GiB/rank
（nb=1,369）懒分配，花 utilization 之外的 10% 余量（3.2GiB ≥ 2.15GiB ✓）。

**int4 / 128B 可行性账**（为什么止步 256B）：
- 词汇层：槽宽 = head_size × dtype 字节数，torch 无 0.5 字节 dtype → **256B 是零改动
  通道的数学地板**；int4 字面量不存在（STR_DTYPE 表 KeyError）。
- 槽数学：int4 的 K 载荷 = 8B 元数据 + 128B 编号 = **136B > 128B 装不下**——int4 槽
  天然是 136B 级"非整宽"（TQ 槽 230B 同理），必须 slot_size_aligned（源码级）。
- 读 int8 池本身几乎无代价：2 值/字节不跨字节、解包 1 模 1 除（比 int2 便宜）、带宽
  更省、原生读者税已由 impl 接管付掉；真正代价 = 混合 block_size（FULL 3,072 vs
  GDN 1,536）在 vllm-ascend 未验证。

---

## 5. 读路径与窗口（sink / recent / staging）

**政策 vs 仓库**：sink=128 / recent=256 是**政策**（哪些 token 配得上 BF16——开头锚点
与结尾依赖段）；**staging=8192 是仓库容量**（页外 per-layer 环形 arena，64 行×128
token，装窗口 token 的 BF16 副本，≈8.4MB/层 ×16 ≈ 134MB/rank）。窗口 token **双份
存储**：池内 INT2 照写（回退）+ arena BF16（优先）。32K 静态快照下仓库里有用的就是
128+256=384 个 token 的副本，8,192 是给滑窗滚动留的余量。

**环 + owner tag**：行号 = 逻辑块号 % 64，新块覆盖同号旧行（零记账淘汰）；tag 对得上
才用 BF16、对不上回退 INT2（永不出错，只掉精度）。**容量只影响精度命中率**：下限
6 行（sink 1 + 尾窗 3 + 余量 2）；8,192 是 PR 默认经验值（[PR] config.py:81，env
`VLLM_OSCAR_STAGING_TOKENS`；我们 serve 把 sink 64→128，因为 kernel 块 128 会让
sink_eff=(64//128)×128=0 静默失效）。

**三条读路径的生效状态（本部署）**：

| 路径 | 位置 | 状态 |
|---|---|---|
| staging 写（窗口 BF16 双写） | backend.py:220-223 | ✅ 每 forward 执行 |
| splice 读（verify/prefill 分支，tag 命中行换 BF16） | :312-313 | ✅ 窗口唯一实际生效的读取路径 |
| 三段窗口 decode + LSE 合并（~130 行） | :251-252 | ❌ 死代码（MTP 下 attn_state 恒 SpecDecoding，[VA] model_runner_v1.py:1477-1505；真机 grep ★读路径=0 可自证） |

**环冲突**：活跃 seq 每跨 8,192 token（64 行×128），新块必撞 sink 行 → **sink 的 BF16
保护静默失效**（16K 上下文从首个 verify 步起：新 token 落逻辑块 128 → 行 0 顶掉 sink
块 0 的 tag）；recent 存活到块 ~190（seq>24,320）。

---

## 6. 三模式流程对比（①插件版 / ②OSCAR PR / ③TQ）

```text
① 插件版（现状 = packed×2）
  config 期页几何定死 → load_model 手术(plugin.py:111) → torch.zeros×16 int8(:4124)
  → 三区视图(:4569-4714；packed 下 FULL 视图 int8/256B) → 写: 旋转+sort裁剪+INT2 打包
  散写(backend.py:108-146) + staging 双写 → 读: SpecDecoding 全落 prefill 分支
  (:202-205→253-293) 全前缀反量化物化+逆旋转+SDPA（110s 主因，ANALYSIS-C §6）

② OSCAR PR（dense；hybrid ✗ config.py:178-192）
  kv_cache_dtype="oscar_int2" → get_kv_cache_shape 第4维=压缩槽 (nb,bs,Hk,slot)
  uint8 无前导2维（oscar_attn.py:87-103；D=128→72B≈7.1×，slot公式 config.py:120-128，
  effective_head_size=slot//2 记账）→ 写: 独立 custom op do_kv_cache_update(:338-345)
  → 读: decode 压缩域 split-KV + 三段窗口 LSE 合并(:618-750)；续读物化(:515-518)
  → 仅 eager(NEVER :134-141)、spec 不放宽(:141)

③ TurboQuant（upstream；hybrid 已解 #39931）
  4 preset(config.py:20-41) → dtype 淘汰制选中 TQ backend(cuda.py:131-147，FA 只认
  auto/fp16/bf16 被淘汰) → TQFullAttentionSpec 页公式 real_page=bs×Hk×slot
  (kv_cache_interface.py:327-349) uint8 池 (nb,bs,Hk,slot) → 写: do_kv_cache_update
  (:363-386) cuBLAS 旋转+单内核写满槽(store.py:412-447) → 读: 压缩域 split-KV 32 路
  centroids 查表(decode.py:552-611)；首 chunk 原生 varlen；★小续算(≤128 新 token)
  直接复用 decode 内核不物化(:69,:666-692)；大续算才物化(:712-834)
  → hybrid: TQ spec 注册进 FullAttentionManager(single:1383-1387) + lcm 页协调 +
  mamba 页 pad(interface.py:573-699)；无窗口；cudagraph UNIFORM_BATCH(:199)
```

---

## 7. Role Model 判决与落地映射

### 7.1 对照表

| 维度 | ① 插件版 | ② OSCAR PR | ③ TQ |
|---|---|---|---|
| 分配期缩池 | **是**（dtype 通道，256B 槽 ×2；任意槽宽 ✗） | 是（槽宽进 shape，7.5×） | 是（Spec 页公式，4.3-4.9×） |
| 写路径 | forward 内联；旋转+sort+散写+staging 双写 | 独立 custom op + 单内核写满槽 | 同②（旋转在 cuBLAS） |
| decode 读 | **死代码**（MTP 路由） | 压缩域 split-KV + 三段窗口 | 压缩域 split-KV（无窗口） |
| 续读 | 每步全前缀物化（10-15× 放大） | 全量物化 | **≤128 新 token 复用 decode 内核不物化** |
| hybrid | 能跑（因为布局兼容） | **未处理** | **已解**（#39931） |
| MTP | 草稿排除+影子池 ✓ | 显式不支持 | 同样 False；但小续算天然覆盖 verify 形态 |
| 在 vllm-ascend | 是（本插件） | 否 | **否**（官方 grep 零命中） |

### 7.2 判决：TQ 工程蓝图 + OSCAR PR 后端本体，缺一不可

- **TQ 赢在三件我们最缺的**：分配期缩池的官方先例；hybrid（16 full + 48 GDN）共存
  修法（attention 页自己定义 + mamba 页 pad 上来——破 `512×768==393,216` 对齐锁的
  参考答案）；小续算 ≤128 复用 decode 内核（恰好命中 MTP verify=4 token 热点）。
- **PR 给后端本体**：classvar/独立写缝/内核/窗口（我们已移植 ~90%）；但 PR 无 hybrid
  作业可抄，必须借 TQ 的接线。
- **零侵入插件期内的可兑现行**：写缝（已做）+ **SpecDecoding 分支接压缩域 decode**
  （下一件最值得做：把"每步全前缀物化"换掉，直接砍 110s 主因）；其余（CacheDType/
  Spec 页公式/lcm 页协调/任意槽宽）需 vllm-ascend 上游 PR（#39931 即模板）。

---

## 8. 证据索引

**[VA] vllm-ascend@19e436985**：model_runner_v1.py:1477-1505（attn_state 判定）,
4124（池 zeros int8）, 4559-4577/4597-4602/4628/4637-4642（FULL 视图构造）,
4679-4680, 4685, 4688-4689/4696-4714（GDN 切分+官方注释 :4701-4704）；patch_mamba_
config.py:53-97/101-117（页几何+断言）；block_table.py:47-68/210-229/288-303；
attention_v1.py:104-112/142-144/456-475/1527。
**[V] vllm@0fc695fc**：kv_cache_utils.py:584/911/967/1142-1195/1318-1326；
kv_cache_interface.py:629-631/830；single_type_kv_cache_manager.py:1142-1165/1383-1387；
scheduler.py:462/293-338；mamba_utils.py:218-233；attention.py:225/276-278/563-606；
config/cache.py:46；torch_utils.py:32-49；platforms/interface.py:573-699；
platforms/cuda.py:131-147；qwen3_5_mtp.py:25,100。
**[PR] oscar-vllm-pr46774@57286d5d**：oscar_attn.py:54-60/87-103/134-141/253-345/
515-518/618-750；config.py:25/81/120-128/154-192；triton_oscar_store.py:99-190；
triton_oscar_decode.py:21/58/61-63/267-335；tests/quantization/test_oscar.py:71。
**[V] TQ**：turboquant_attn.py:69/93-121/139-167/199/363-386/537-557/576-588/666-692/
712-834；turboquant/config.py:20-41/128-174/193-202；centroids.py:82-86；
triton_turboquant_store.py:278-447；triton_turboquant_decode.py:178-306/519-611。
**[W] 本仓**：format.py:19-33/99-110；store_kernel.py:35-95；backend.py:108-146/
148-211/220-223/253-313/324-481；plugin.py:88-146；mtp_shadow.py；serve_oscar.sh:32-46；
config.py:84。
**真机日志锚点**：`GPU KV cache size: 16.36 GiB`（nb=1,369 复原）；`★ 几何对账: K 槽
256B`；`★ MTP 影子池已建立 shape=(13380,...)`（=1,115×12 → block 1,536 实证）；
`heads=1`（Hk/rank）；ArgSort AiCpu warning（clip sort 索引跑 AiCpu）。

*重构版写于 2026-09-05；v1 的过程性内容（勘误记录/重复图）已并入正文或删除，
历史见 git log（2ba7b0d 之前）。*
