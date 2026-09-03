# PLAN-1 — OSCAR INT2 KV 缓存量化适配 vllm-ascend 0.23.0（混合模型·零侵入 Triton 插件）

> 读者：第一次接触本项目的人。写作日期：2026-09-03。状态：DRAFT。
> 范围：Qwen3.5-27B-w8a8-mtp（GDN 线性注意力 + FULL 全注意力混合）在 vLLM Ascend 0.23.0
> （vllm-ascend `skill-ref-023` = v0.23.0 官方 tag `5cb98caa` + PR#12607 `16e0ef69`，
> 本树以 git 合并态 `19e436985` 为基准）上接入 OSCAR INT2 KV 量化；**不修改
> vllm/vllm-ascend 任何原始代码文件**；全部算子为 Triton（NPU 运行时走 triton-ascend），
> 不使用自研 AscendC 算子；所有量化/反量化/旋转/打包在 NPU 上完成。

---

## 0. 一页速览：看这一段就明白 80%

- **要做什么**：把 OSCAR vLLM PR（`references/oscar-vllm-pr46774`，快照 `57286d5d`）的
  INT2 KV 缓存量化（旋转 + 裁剪 + per-vector 非对称 INT2 + Sink/Recent BF16 窗口）以
  **纯外部插件**形式接入 vllm-ascend 的混合模型路径：FULL 层 K/V 以 160B/槽 INT2 存储
  （写在原生 1024B/槽的 K+V 字节区内，**不动任何分配/页表/偏移几何**），GDN 层 conv/ssm
  原样不动。**成功 = 可观察现象**：`vllm serve` 用 `--quantization ascend` 起同一模型，
  插件日志打印 `[oscar-ascend] FULL layers=[...] slot=160B`,单个 token 数值 probe 显示
  decode 输出与 BF16 参考差 ≤1e-4（fp32），且 `npu-smi` 显存中 KV 占用按 160/1024 比例下降。

- **端到端全景**（分段图：A 源码 → B 交付 → C 运行期接线 → D 执行/验证）：

```text
A 插件源码（本工作区 oscar_ascend/ 包）
   ├─ plugin.py: load_plugin()（无参；vllm.general_plugins 入口点调用）
   │    ├─ 检测 vllm-ascend 0.23.0 / AscendAttentionBackendImpl / is_hybrid 模型
   │    ├─ 包裹 AscendAttentionBackendImpl.__init__（fail-soft：任何异常回退原生）
   │    └─ 注册 Triton kernel 参数表（HAS_TRITON 门禁 + 数值 probe 桩）
   └─ kernels/ (triton) + backend.py（AscendOscarAttentionBackendImpl 子类）
        ↓ pip install -e . + VLLM_PLUGINS=oscar_ascend
B 一键交付（delivery/install_and_launch.sh，用户唯一动作）
   ├─ git pull → 安装 wheel → 打印 HEAD/sha256/插件心跳 → 直起 serve
   └─ 门禁：HAS_TRITON 必须 True；版本断言；否则 FAIL 输出修复指引（阻塞 serve）
        ↓
C vllm-ascend 运行期接线（全部为运行期 monkeypatch，零源码改动）
   ├─ _should_oscar(impl) 命中（hybrid + full_attention 层）
   ├─ impl.__class__ = AscendOscarAttentionBackendImpl（照抄 kv_c8.py:130 的类外科手术先例）
   ├─ impl.forward + do_kv_cache_update 重写：INT2 store / dequant / fused decode
   └─ 其余（backend 类、metadata builder、分配器、页表、GDN 路径、MTP 框架）100% 原生
        ↓
D 执行/验证（三个通道）
   ├─ CPU 数值镜像：store 字节差=0；dequant ≤1e-5；decode/prefill ≤1e-4（同真机判据）
   ├─ 真机 probe（delivery/probe_oscar.py，serve 前阻塞）：样例 token 级 check
   └─ serve + 显存/吞吐对比（INT2 vs BF16）
```

- **关键文件速查表**：

| 层 | 文件 | 一句话职责 |
|---|---|---|
| 插件入口 | `oscar_ascend/plugin.py` | 检测+包裹 `AscendAttentionBackendImpl.__init__`，fail-soft |
| 算子后端 | `oscar_ascend/backend.py` | `AscendOscarAttentionBackendImpl(AscendAttentionBackendImpl)`：forward/do_kv_cache_update/窗口 |
| 配置 | `oscar_ascend/config.py` | 从 `OSCAR_ASCEND_*` 环境变量构造 `OscarAscendConfig`（D=256, sink/recent, 旋转路径…） |
| 数值格式 | `oscar_ascend/format.py` | 160B 槽布局常量 + 量化/打包/解包纯函数（CPU 镜像唯一权威） |
| Triton store | `oscar_ascend/kernels/store_kernel.py` | rotate 已在外部 matmul 完成 → per-vector INT2 + 打包 + 散写 160B 槽 |
| Triton decode | `oscar_ascend/kernels/decode_kernel.py` | fused INT2 dequant + 分块 softmax（port PR `_oscar_decode_stage1/2`） |
| Triton dequant | `oscar_ascend/kernels/dequant_kernel.py` | 前缀反量化到 TND 缓冲区（prefill continuation 用） |
| 旋转库 | `oscar_ascend/rotation.py` | 加载 per-layer `[D,D]` 正交旋转（port PR rotation.py） |
| 交付 | `delivery/install_and_launch.sh` | 一键：安装+心跳+门禁+serve |
| 数值 probe | `delivery/probe_oscar.py` | 真机同判据：store/dequant/decode 三查 |

- **一句话总结**：不重构、不改源码、不动几何——把"K/V 字节区"重新解释为
  "160B INT2 槽区"，用类外科手术把 FULL 层的 attention impl 换成 OSCAR 子类，
  其余全部复用原生 vllm-ascend（含 GDN 三大条分配、页表、FIA 前缀路径、MTP）。

---

## 1. 背景与动机（为什么现在做这件事）

- **现状痛点**：目标模型 262,144 上下文 + 4096 hidden（D=256），原生 bf16 KV 每 token
  K+V = 2×256×2 = 1024B；`--max-model-len 262144` 下 KV 占显存约
  `262144 × 1024B × num_kv_heads(×TP 分片)`，是既定瓶颈。OSCAR INT2 把 K/V 压到
  160B/槽（约 15.6% 字节），配合旋转可保精度（OSCAR 论文/PR 语义，见 §6 E1-E5）。
- **为什么是这个时机**：官方 OSCAR vLLM PR 已给出（a）INT2 槽数值契约（store/decode
  Triton 内核 `triton_oscar_store.py`、`triton_oscar_decode.py`），（b）Sink/Recent 窗口
  与 staging 机制（`oscar_attn.py:245-336, 618-750`），（c）TurboQuant 式
  `get_kv_cache_shape` 集成范式；vllm-ascend 参考树自带 **C8 KV 量化的"impl 类外科手术"**
  先例（`kv_c8.py:119-130` 把 `layer.impl.__class__` 换成 `AscendC8AttentionBackendImpl`），
  且自带 qwen3.5 混合模型运行路径（`patch_qwen3_5.py` + `model_runner_v1.py:4493-4708`）。
- **不做会怎样**：KV 带宽/显存维持原样，262k 上下文部署成本不变；INT2 收益完全落空。
- **本方案范围**：FULL 层 K/V 的存储与读写路径替换；GDN 层 conv/ssm 不动；权重量化
  （`--quantization ascend` W8A8）不动；MTP/投机解码框架不动。
- **非目标**：不做 MLA（本模型 dense 非 MLA，`patch_qwen3_5.py:41-87` 可证）；不做
  INT2→持久 fp32 反量化缓存（违背 N-09）；不做页表/块大小重构；不做 CUDA Graph 捕捉
  （OSCAR 窗口路径仅 eager 验证过，`oscar_attn.py:134-141` `_cudagraph_support=NEVER`）。

---

## 2. 端到端链路（每一层：入口 → 做什么 → 输入输出 → 谁消费 → 失败协议）

```text
L0 真机一键（用户唯一动作：git pull && bash delivery/install_and_launch.sh）
   ├─ 阶段1 自检：python -c "import vllm_ascend" 版本=="0.23.0"+PR12607 ⊇ probe；
   │           HAS_TRITON==True（来自 vllm.triton_utils.importing.HAS_TRITON，
   │           references/vllm/vllm/triton_utils/importing.py:14-52：triton-ascend 驱动必须恰好 1 个 active）
   │           否则 FAIL：打印"triton-ascend 未加载/多驱动"修复指引，拒绝起服务
   ├─ 阶段2 git pull + pip install -e ./oscar_ascend（纯 Python 包，无编译）
   ├─ 阶段3 指纹：HEAD commit / 包 sha256 / 插件心跳（plugin.py 全局计数）
   ├─ 阶段4 真机数值 probe（delivery/probe_oscar.py，阻塞 serve；全 PASS 才继续）
   └─ 阶段5 Qwen3.5 启动（serve_oscar.sh：用户目标命令 + VLLM_PLUGINS=oscar_ascend + 环境）
       → 失败协议：任何阶段非零 → 打印阶段+日志尾部+回退指引（FLAG OSCAR_SKIP=1 仅诊断）

L1 本地门禁链（开发机）
   ├─ CPU 数值镜像（format.py ↔ kernels 的 torch 参考实现）：store 字节差=0、dequant≤1e-5
   ├─ 契约表（skill sandbox contract/tables/*.yaml）：槽 160B / 打包位序 / fp16 meta 小端
   ├─ 静态契约：页面跨距=每物理块字节数；算子只认 flat uint8 + 显式 stride（N-06）
   └─ 设备偏离表命中检查：D-03/D-04/D-05（位运算/移位在部分路径不可用）→ Triton 侧等价检查
       （Triton 是 JIT 语义，非 AscendC；命中项以"最小真机 probe"验证，见 §8 R3）

L2 引擎模拟层（skill 沙盒：contract/ir 解释器 + engine trace）
   ├─ 验证：hybrid 组序（bs=1536/128 的 mid-prefill 视图契约）、slot 偏移公式
   └─ 产出：TraceEvent JSON；诚实边界：不模拟真实性能/图模式（sandbox README）

L3 数值权威层（sandbox/l3_numeric/format.py 为唯一权威）
   ├─ store 判据：|packed − reference| == 0；dequant ≤1e-5；decode/prefill/LSE ≤1e-4
   └─ 同一判据函数本地/真机通用（skill §2 铁律3）

L4 虚拟 CANN 层（仅虚构/检查位运算语义；Triton 不经过 CANN 编译，本层降级为
   "Triton 语义对照表"——无成熟先例声明见 R3）

L5 算子栈（python → triton JIT → triton-ascend backend → NPU）
   ├─ 每层入口：python 函数签名 / @triton.jit grid / backend 编译 / 设备 kernel
   ├─ 失败协议：triton 编译异常 → 打印 kernel 名+行号 → 回退纯 torch NPU 路径
       （rotate=matmul，量化=min/max/round/bitwise 全为成熟算子，见 R2）
   └─ 无 AscendC/pybind/aclnn 环节（本方案不用自研算子，铁律1的豁免前提）
```

### 一条完整示例数据流（挑"decode 第 1 步，写一个 D=256 的 INT2 槽，再读回")

1. 引擎调度 decode step；worker `model_runner_v1` 组装 `AscendMetadata`
   （`slot_mapping`、`block_tables`、`seq_lens` 等，`attention_v1.py:155-215`、builder `:217+`）。
2. FULL 层 `Attention.forward`（`attention.py:512-528` 前提 `forward_includes_kv_cache_update=True`,
   base `backend.py:66`）→ `impl.forward(layer, q, k, v, kv_cache, meta, output)`；
   kv_cache 为 `(k_cache, v_cache)` 元组（hybrid 分配路径 `model_runner_v1.py:4501`）。
3. `AscendOscarAttentionBackendImpl.forward`：
   a. `_ensure_rotations`：按 `layer_name` 取 `R_k/R_v`（`rotation.py` port；缺文件→单位阵）。
   b. `k_rot = k @ R_k`、`v_rot = v @ R_v`（bf16→fp32 matmul，NPU aclnn，non-clip 默认）。
   c. `oscar_store(k_rot, v_rot, combined_cache_view, slot_mapping, …)`：
      grid=(N×H,)，slot 基址 = `comb_base + (t*Hk + h)*1024` + `page_stride*blk`；
      写 N-01 布局：meta(0-7)、pad(8-31)、K idx(32-95)、V idx(96-159)。
   d. decode 注意力：`q_rot = q @ R_k` → `oscar_decode_attention(q_rot, combined, block_table,
      seq_lens, …)`（Triton fused stage1 + stage2，PR `triton_oscar_decode.py:24-150/153-250`）
      → `out_true = out_rot @ R_v^T`。
4. 数值验证（probe/沙盒）：对同一 (token, head)，把 160B 槽读回反量化，与 `(k/v)@R` 比：
   store 字节差=0、dequant ≤1e-5。

---

## 3. 方案设计（为什么不选别的）

候选清单：候选 1 = A（原生视图+impl 外科手术）、候选 2 = B（页内三区分区）、
候选 3 = C（改源码整后端）、候选 4 = D（全量 dequant scratch）、候选 5 = E（独立 OSCAR 后端）。

| 候选 | 机制一句话 | 优点 | 缺点 | 结论 |
|---|---|---|---|---|
| **候选 1（A·选中）** | 原生 K/V 双视图不动；实现层"组合 160B 槽视图"（k/v 在组张量内物理连续，`model_runner_v1.py:4569-4577` 切分保证）+ `AscendOscarAttentionBackendImpl` 类外科手术（照 kv_c8.py:130 先例） | 零分配/页表/块大小改动；只换 impl；机制全部有成熟先例（C8 手术 + OSCAR PR 内核 + TurboQuant 槽） | 160B 槽嵌在 1024B 原生字节窗内（只用到约 16%）；INT2 收益靠"读写走 160B 字节"而非"少分页"；decode 必须新 Triton 内核 | ✓ |
| B | 页内 Sink/History/Recent 三区重排（用户 §5 初步构思：改 `page_size` / 每页 token 数） | 每页字节利用率高分 | **必须先改 `AttentionSpec.page_size_bytes`/`get_kv_cache_shape` → 触发 vllm-ascend hybrid 切分矩阵（`model_runner_v1.py:4569-4577` 的 `*2` 与 4493 分支顺序）→ 无成熟先例的分配手术；且三区物理布局与 PR 的 staging 窗口机制（`:253-336`）不一致** | ✗ 被否决：无先例 + 破坏纯零侵入；三区语义由 A 的 staging 仓库实现（同 PR） |
| C | 直接改 vllm/vllm-ascend 源码（加 oscar 后端进 selector/platform） | 最贴近 PR 原始集成 | 违反用户"不允许修改原始代码文件"+ skill 铁律2 零侵入 | ✗ 被否决 |
| D | decode 每次全上下文 dequant 到 scratch 再走原生 paged attention | 复用 `torch_npu._npu_paged_attention`（`attention_v1.py:1363-1382`） | 每步 O(seq_len×D) 带宽+scratch 分配；262k 上下文不可行；N-09 禁止持久 fp32 缓存 | ✗ 被否决：性能/违反 N-09；仅作 A 的 fallback 分支（前缀 prefill dequant 是 TND 一次性，可接受） |
| E | 全新独立 OSCAR AttentionBackend（如 PR 的 `oscar_attn.py` 整后端，走 platform `get_attn_backend_cls`） | 结构上最像 PR | 需要（1）变 `get_kv_cache_shape`→破坏 hybrid 切分（同 B）；（2）重造 metadata builder（vllm-ascend 的 `AscendMetadata` 已被 GDN/MTP/cudagraph 消费）；（3）改 `CacheDType` 校验（`selector.py:68-73` 硬断言）——均无成熟先例 | ✗ 被否决但**部分采纳**：A 的内核/窗口/旋转机制照抄 PR 代码结构 |

- **选型理由（A 的正面论证）**：
  1. 混合模型 K/V 分配几何**完全不变**：FULL 层 `kv_cache` 仍为原生切好的 `(k_cache, v_cache)`
     （`model_runner_v1.py:4493-4577, 4680`）；大小、页跨距、`page_size_padded`
     （`model_runner_v1.py:5025-5034` = mamba 页对齐）全不动。
  2. k/v 在组张量内**物理连续**（`raw_kv_tensor = raw_k_tensor[conv_pad:]` 后一刀切两半，
     `:4575-4577`）→ 可无拷贝构造 `(nb, bs, hk, 1024)` uint8 组合视图，160B 槽正好落位，
     且满足已固化契约 N-06（算子只认 flat uint8 + 显式 page_stride）。
  3. impl 类外科手术 = 现成先例（`kv_c8.py:126-130`）；无需改 backend/metadata/分配器。
  4. 数值契约（量化公式、打包位序、fp16 meta、旋转/逆旋转、判据）全部来自 OSCAR PR 官方内核
     （`triton_oscar_store.py:41-77`、`triton_oscar_decode.py:104-141`）——reference-first 达标。
- **无先例声明**：
  - **无成熟先例①：Triton（triton-ascend）上运行 PR 的 INT2 打包/位提取内核**；PR 内核
    （`triton_oscar_store.py`/`triton_oscar_decode.py`）只为 CUDA Triton 编写。验证方案：
    L3 判据 + 本地 triton-ascend 单 kernel 冒烟（vm/ 回路）+ 真机 probe（全 PASS 才 serve）。
    未验证前该机制标 UNKNOWN（§8 R3）。**若不支持位运算**：降级为"纯 torch NPU 位打包"
    （`view(ushort).view`、移位、`&3` → 全为 torch 成熟算子），内核只剩量化/打包两步，仍纯 NPU。
  - **无成熟先例②：混合模型里"原生 K/V 双视图 + 组合 160B 槽"组合视图**；机制相似物 =
    TurboQuant packed-slot 单张量（`TQFullAttentionSpec`, `kv_cache_interface.py:327-349`）+
    C8 类外科手术（`kv_c8.py:130`），但二者组合无 direct 先例。验证：L2 沙盒回放 + probe
    字节级对账（每个物理块首尾 16B 与偏移公式对照）。
  - **无成熟先例③：OSCAR 窗口 staging 在 Ascend 上的 LSE 合并 decode**
    （`oscar_attn.py:618-750` 是 CUDA 张量代码，无 NPU 依赖，可直接 port；但
    `torch.cumprod/flip` 等在 NPU 的行为需 probe）→ 标 UNKNOWN，验证见 §8 R5。

---

## 4. 文件地图（每个文件为什么存在）

| 文件 | 职责/边界（一句话 + 为什么在它这里） | 谁消费它 | 新建/修改 |
|---|---|---|---|
| `plan/PLAN-1-oscar-int2-vllm-ascend.md` | 本方案（设计决策 + 证据） | 实现者 | 新建 |
| `oscar_ascend/__init__.py` | 包标记 + 版本 | pip | 新建 |
| `oscar_ascend/plugin.py` | `load_plugin()`（vllm.general_plugins 入口）：环境检测 + 包裹 `AscendAttentionBackendImpl.__init__` + 心跳计数；fail-soft | vllm 启动（`arg_utils.py:745-747`） | 新建 |
| `oscar_ascend/config.py` | `OscarAscendConfig`：head_dim/量化位宽/sink/recent/staging/旋转路径/clip（env `OSCAR_ASCEND_*`，避免改 vllm envs）| backend | 新建 |
| `oscar_ascend/format.py` | **CPU 数值唯一权威**：160B 槽布局常量 + 量化/打包/解包纯函数（与 sandbox `l3_numeric/format.py` 漂移检测对账） | kernels 参考实现、probe、测试 | 新建 |
| `oscar_ascend/rotation.py` | 加载 per-layer `[D,D]` 旋转（port `oscar/rotation.py:56-119`；`torch.load` 一次 + lru_cache；缺层→单位阵） | backend | 新建 |
| `oscar_ascend/backend.py` | `AscendOscarAttentionBackendImpl(AscendAttentionBackendImpl)`：forward（窗口/INT2/dequant 三态）、do_kv_cache_update、_ensure_staging/_staging_write/_stage_splice/_decode_attention_windowed（port oscar_attn.py） | 运行期类外科手术 | 新建 |
| `oscar_ascend/kernels/store_kernel.py` | `@triton.jit` 量化打包散写（160B 槽；grid=(N×H,)） | backend.forward | 新建 |
| `oscar_ascend/kernels/decode_kernel.py` | `@triton.jit` fused INT2 decode（stage1 + stage2 复用 vllm `_fwd_kernel_stage2`，port `triton_oscar_decode.py:24-150/253-354`） | backend.forward | 新建 |
| `oscar_ascend/kernels/dequant_kernel.py` | `@triton.jit` 前缀反量化 → `[cached_len, Hk, D]` fp16（rotated space，port `:357-411`） | backend._prefill_attention | 新建 |
| `oscar_ascend/tests/test_numeric.py` | CPU 镜像：store 字节差=0 / dequant≤1e-5 / decode≤1e-4 | CI / 本地门禁 | 新建 |
| `delivery/install_and_launch.sh` | 一键：自检→安装 wheel→指纹心跳→probe（阻塞）→serve | 用户唯一命令 | 新建 |
| `delivery/serve_oscar.sh` | 用户目标启动命令 + `VLLM_PLUGINS=oscar_ascend` + 环境固化 | install_and_launch.sh 阶段5 | 新建 |
| `delivery/probe_oscar.py` | 真机同判据数值 probe（store/dequant/decode 三查 + 指纹） | install_and_launch.sh 阶段4 | 新建 |
| `pyproject.toml` | `[project.entry-points."vllm.general_plugins"] oscar_ascend = "oscar_ascend.plugin:load_plugin"`（插件加载语义 `vllm/plugins/__init__.py:58-80`：`load()` 返回 callable 并执行） | pip | 新建 |

---

## 5. 关键机制/重要方法（做什么 + 为什么这样设计 + 参考）

### 5.1 槽布局（160B，meta-first，N-01 契约；仅 FULL 层 K/V）

```
每 (block b, token t, head h) 组合槽 = 1024B 原生窗口（K 512B + V 512B，物理连续）
逻辑布局（槽窗口偏移，单位 B）：
  [0..1]   K scale  (fp16 LE, 量化时先 fp16 舍入)
  [2..3]   K zero   (fp16 LE, == vmin)
  [4..5]   V scale  (fp16 LE)
  [6..7]   V zero   (fp16 LE)
  [8..31]  pad 24B（对齐 32B；槽窗 1024B，pad 只占逻辑槽；N-01 注明 160=5×32 天然 32B 对齐）
  [32..95] K idx 64B（D=256 INT2, 4 值/字节, 字节 b=k idx[4b]|k idx[4b+1]<<2|...）
  [96..159] V idx 64B（同打包）
  [160..1023] 未使用（原生窗剩余；保证 stride/几何不变时留白）
```

物理地址（全部字节、uint8 视图）：

```
comb_block_stride = k_cache.stride(0)*2            # 字节；hybrid 时= padded 页跨距（N-06）
comb_pos_stride   = k_cache.stride(1)*2            # = Hk*1024（bs 可能被 128 重新切片，取实际 stride）
comb_head_stride  = k_cache.stride(2)*2            # = 1024
slot_base(b,t,h)  = b*comb_block_stride + t*comb_pos_stride + h*comb_head_stride
```

判定不变量：`comb_block_stride == page_size_bytes(每物理块 K+V 字节)`；
`slot_base 偏移 == 原生 bf16 K/V 写入同一 (b,t,h) 的偏移` → 页表/前缀命中/重写全部天然兼容。

### 5.2 组合视图构造（k/v 连续性的来源）

```python
k8 = k_cache.view(torch.uint8)                 # (nb, bs, hk, 512)
v8 = v_cache.view(torch.uint8)                 # (nb, bs, hk, 512)；与 k8 在组张量内连续
comb = k8.as_strided(size=(nb, bs, hk, 1024),
                     stride=(k8.stride(0), k8.stride(1), k8.stride(2), 1))
# 依据：hybrid 切分 raw_kv_tensor = raw[conv_pad:] 后 k=v 前段、v=后段（model_runner_v1.py:4575-4577）
# 注意：若后续版本 K/V 切分不再连续（或 MLA/稀疏路径），此 as_strided 会越界——
#       实现时对 k8.data_ptr()+k8.numel() 与 v8.data_ptr() 做连续性断言（probe 校验）
```

### 5.3 量化/打包（Triton store 伪代码，port PR `triton_oscar_store.py:41-77`，槽位序按 5.1）

```python
@triton.jit
def _store_int2_vec(rot_ptr, comb_ptr, base, slot_base, d_offs, d_mask,
                    D: tl.constexpr, LEVELS: tl.constexpr, DATA_BYTES: tl.constexpr,
                    BLOCK_D: tl.constexpr, BLOCK_PACK: tl.constexpr):
    vec = tl.load(rot_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    vmin = tl.min(tl.where(d_mask, vec, float("inf")), axis=0)
    vmax = tl.max(tl.where(d_mask, vec, -float("inf")), axis=0)
    scale = (vmax - vmin) / (LEVELS - 1)          # INT2: /3
    scale = tl.where(scale > 1e-8, scale, 1e-8)
    scale_f16 = scale.to(tl.float16)              # 先 fp16 舍入（N-02，舍入前后可差 1 电平）
    zero_f16  = vmin.to(tl.float16)
    q = tl.minimum(tl.maximum(((vec - zero_f16.to(tl.float32)) / scale_f16.to(tl.float32) + 0.5).to(tl.int32), 0), 3)
    # 打包 4 值/字节：q[4b]|q[4b+1]<<2|q[4b+2]<<4|q[4b+3]<<6 (N-03)
    packed = 计算 packed(DATA_BYTES)
    # 写 K 区：meta @ slot_base+0..3（LE uint16），idx @ slot_base+32..95
    # 写 V 区：meta @ slot_base+4..7，idx @ slot_base+96..159
    #    （V 区的量化调用同一 helper，rot_ptr 指向 v_rot 基址）
```

Python 入口：`oscar_store(k_rot[N,H,D], v_rot[N,H,D], comb, slot_mapping, config)`；
grid=(N*H,)，slot 基址由 `slot_mapping`（`(block,pos)`）与 `comb` stride 计算（同 PR `:99-112`）。

### 5.4 读取路径（三种状态）

**decode（fused INT2）**：port `triton_oscar_decode.py:24-150`（stage1：INT2 解包+按块 table
分块打分+KV 线程拆分 LSE）+ `:153-250` stage2 复用 vllm `_fwd_kernel_stage2`；Q 已旋转
`Q@R_k`，输出在 rotated-V 空间，回乘 `R_v^T`。组合槽解包：meta 在 `slot_base+0..7`
（K 0-3, V 4-7），idx 在 `+32..95`/`+96..159`。

**prefill（首块）**：不解缓存；对原始 q/k/v 走原生 `npu_fused_infer_attention_score` /
`npu_fusion_attention`（TND，`attention_v1.py:1302-1316/1397-1406` 的 API 形态——
TP 是 vllm-ascend 内部路径；本方案按同一 TND 入口调用，命名/参数以 probe 校准，标
[HYPOTHESIS]）。

**prefill continuation（前缀命中/分块续写）**：`oscar_full_dequant_kv`（`:357-411` port）
把 `cached_len` 前缀反量化（rotated space）→ matmul `R_k^T/R_v^T` 回原空间 → 与当前 chunk
concat → TND attention（D 分支的"一次性 dequant"仅在此发生，每步至多一次 per layer）。

**窗口（sink/recent BF16）**：port `oscar_attn.py:253-336`（staging 双写+owner tag）+
`:618-750`（LSE 合并 decode）；sink 64 / recent 256 默认（env 可调）。staging 为 per-layer
BF16 环形 arena；INT2 副本仍写（回退安全）。

### 5.5 impl 接入（类外科手术）

```python
# plugin.py（fail-soft：除 ValueError 型"不支持"外任何异常 → 恢复原类并 warning）
_ORIG_INIT = AscendAttentionBackendImpl.__init__
def _patched_init(self, *a, **kw):
    _ORIG_INIT(self, *a, **kw)
    if _should_oscar(self):                     # hybrid 且 full_attention 层
        self.__class__ = AscendOscarAttentionBackendImpl
        self._oscar_setup()                     # config + 旋转标识 + staging 懒分配
# _should_oscar 判据：cfg = get_current_vllm_config(); cfg.model_config.is_hybrid
#   and self.attn_type == DECODER and self.sliding_window is None
#   （FULL 层由 layer_types[idx]=="full_attention" 或 MTP 层硬编码 full_attention 决定，
#    qwen3_5_mtp.py:102；不量化 MambaBase——GDN 不走 Attention impl，天然排除）
```

**为什么这么做**：唯一"运行期可替换 impl"的先例就是 C8（`kv_c8.py:127-130`）；
不改 `get_attn_backend_cls`/`CacheDType`/metadata builder，规避 B/E 的全部无先例点。

### 5.6 与 PR 的差异（诚实声明的偏差）

| 点 | PR（vllm 原生） | 本方案（vllm-ascend） | 理由 |
|---|---|---|---|
| 缓存形状 | `(nb,bs,hk,slot)` 单张量 | `(k,v)` 双张量 + 组合视图 | hybrid 分配几何不可动（§3 A） |
| 更新时机 | `forward_includes_kv_cache_update=False` + `do_kv_cache_update` | 保持 `True`（backend 原生未改），在 `forward` 内做 store | 不动 backend 类 |
| 预填充注意力 | flash_attn_varlen | `npu_fused_infer_attention_score`/`npu_fusion_attention`（TND） | Ascend 原生命令（`attention_v1.py:1302-1316/1397-1406`） |
| 槽字节序 | PR 每区 [data\|meta]，D=256 136B | N-01 meta-first 160B（含 pad） | 用户规定 tq_slot_size=160 + 沙盒数值契约唯一权威（N-01） |

### 5.7 配置项（env，全部纯插件，不动 vllm envs）

| env | 默认 | 语义 |
|---|---|---|
| `OSCAR_ASCEND_K_RATIO/V_RATIO` | 0.0（不裁剪） | clip 分位数（>0 时 `thr=quantile`；**torch.quantile 在 NPU 支持未验证**，若不可用→用 `torch.topk` 近似或保持 0，见 R4） |
| `OSCAR_ASCEND_K_ROT_PATH/V_ROT_PATH` | ""（单位阵） | 旋转检查点（`{layers:{i:{rotation:[D,D]}}}`） |
| `OSCAR_ASCEND_SINK/OSCAR_ASCEND_RECENT` | 64 / 256 | BF16 窗口 |
| `OSCAR_ASCEND_STAGING` | 8192 | staging arena token 数 |
| `OSCAR_ASCEND_SLOT_PAD` | 0 | 逻辑 160B 之外的自由 pad（默认不动） |

---

## 6. 证据与参考（reference 表）

| # | 断言 | 证据 | 置信 |
|---|---|---|---|
| E1 | OSCAR INT2 槽 = per-vector 非对称量化；scale=(vmax-vmin)/3 且先 fp16 舍入；q=clamp(round((x-z)/s),0,3) | `references/oscar-vllm-pr46774/vllm/v1/attention/ops/triton_oscar_store.py:41-57` | CONFIRMED |
| E2 | 打包位序 `q[4b]<<2k`；fp16 meta 小端 | 同上 `:58-77`；解码端 `triton_oscar_decode.py:104-141` | CONFIRMED |
| E3 | 旋转：K@R_k、V@R_v、Q@R_k、out@R_v^T；旋转正交 → 分数不变 | `oscar_attn.py:44-50, 594-616`；`rotation.py:56-119` | CONFIRMED |
| E4 | Sink/Recent BF16 staging 机制（owner tag + LSE 合并） | `oscar_attn.py:245-336, 618-750` | CONFIRMED（语义）；NPU 行为 UNKNOWN（R5） |
| E5 | 混合模型禁用边界层跳过 | `config.py:178-198`（is_hybrid → []） | CONFIRMED |
| E6 | vllm-ascend hybrid FULL K/V 由组张量连续切出；k/v 物理相邻 | `vllm-ascend/vllm_ascend/worker/model_runner_v1.py:4493-4577`（尤 :4575-4577） | CONFIRMED（参考树）；部署 fork 需 probe 复核（R1） |
| E7 | FULL 层 K/V 规格/页大小走后端 `get_kv_cache_shape` + spec.page_size；mamba 页对齐 → page_size_padded | `model_runner_v1.py:5025-5034`；`vllm/vllm/v1/kv_cache_interface.py:159-200, 607-625` | CONFIRMED |
| E8 | impl 类外科手术先例 | `vllm-ascend/vllm_ascend/quantization/methods/kv_c8.py:119-139`（尤 :130） | CONFIRMED |
| E9 | impl.forward 内含 K/V 写入（reshape_and_cache）+ FIA TND / paged 口径 | `attention_v1.py:1479-1545`（:1532）、`:1363-1382`、`:1302-1316/1397-1406` | CONFIRMED |
| E10 | 插件加载机制（entry point group `vllm.general_plugins`，`VLLM_PLUGINS`，load 后执行 callable） | `vllm/vllm/plugins/__init__.py:31-80`；`engine/arg_utils.py:745-747` | CONFIRMED |
| E11 | Triton 可用性：HAS_TRITON 需恰一 active 驱动 | `vllm/vllm/triton_utils/importing.py:14-52` | CONFIRMED（本机状态→probe） |
| E12 | 槽 160B meta-first 布局 = 本项目数值契约唯一权威 | skill `knowledge/numerical-contract.md` N-01..N-06 | 项目契约（用户点名） |
| E13 | 用户目标命令里的模型 D=256（KV 512B/token）、TP4、262k 上下文、MTP3 | 用户 §1 启动命令 + `Qwen3.5-27B-w8a8-mtp` | 输入事实 |
| E14 | 原生页面字节数 P=801,792（conv 15,360 + K/ssm 393,216 + V 393,216）与 512B/token 步长 | 用户 §1 描述；**参考树未含该常数**（搜索 801792/393216 无命中） | HYPOTHESIS——以部署环境 `model_runner_v1` 实际 `kv_cache_tensor.size`/spec 打印为准（R1） |
| E15 | Triton（triton-ascend）可编译 PR 内核（位运算/uint16 bitcast/二维 mask 广播） | 无官方树证据；vllm-ascend 自身 Triton 用量见 `patch/worker/patch_triton.py:1-18` | UNKNOWN → R3 验证方案 |

---

## 7. TODO 与验收（每条 = 一个可观察结果）

| # | TODO | 完成判定（可观察现象/命令/判据） | 状态 |
|---|---|---|---|
| 1 | `format.py` + 纯函数参考实现（160B 槽、量化、打包、解包） | `python tests/test_numeric.py` → store 字节差=0、dequant≤1e-5（与 skill sandbox l3_numeric 对账） | [ ] |
| 2 | `rotation.py` + config（env 解析、单位阵回退） | 单测：构造 {i: eye} 检查点 → `get_layer_rotation` 返回 `[D,D]`；缺层→单位阵 | [ ] |
| 3 | Triton store kernel + 组合视图 | `test_numeric` 增加 kernel 路径（`HAS_TRITON` 下）→ 同判据；**无 Triton 时用 torch 参考路径** | [ ] |
| 4 | Triton decode（stage1/2 + 反旋）+ dequant kernel | CPU 数值镜像 decode≤1e-4；`triton` 冒烟编译无异常 | [ ] |
| 5 | `AscendOscarAttentionBackendImpl` + plugin.py（类外科手术 + 心跳 + fail-soft） | 构造 fake impl `__class__` 生效；非 hybrid 模型回退原生；插件心跳计数 ≥1 | [ ] |
| 6 | 窗口 staging（sink/recent + LSE 合并）port | CPU 镜像：窗口外 token 与 INT2 一致、窗口内等于 BF16；恢复 | [ ] |
| 7 | `delivery/install_and_launch.sh` + `serve_oscar.sh` + `probe_oscar.py` | 一条命令跑通（自检→安装→probe PASS→serve 起来）；`VLLM_PLUGINS=oscar_ascend` 心跳在日志 | [ ] |
| 8 | 真机 single-token 数值 probe（D=256） | probe：store 字节差=0、dequant≤1e-5、decode vs bf16 参考 ≤1e-4 全 PASS；**FAIL 阻断 serve** | [ ] |
| 9 | 精度/显存回归（用户目标命令 serve） | `--max-model-len 262144` 拉起；greedy 采样与 bf16 基线在测试集差异 ≤ 阈值；npu-smi KV 占用下降≈84% | [ ] |
| 10 | `handoff_hygiene`/CI 类收尾（如项目要求）+ 文档标记 REVIEWED | 门禁命令全绿；plan README 状态改 REVIEWED | [ ] |

## 8. 风险与未知

| # | 风险 | 层 | 影响 | 验证方式 | 未知标记 |
|---|---|---|---|---|---|
| R1 | 部署 vllm-ascend fork 与参考树的 hybrid 切分/页常数（P=801,792 等）不一致 | L2/L5 | 偏移计算错误→越界写/张量撕裂 | probe 打印 `kv_cache_tensor.size`、`k_cache.stride`、`page_size_padded` 与公式对账 | [HYPOTHESIS] P 值未经参考树证实 |
| R2 | triton-ascend 位运算/位提取不支持 | L5 | store/decode 内核无法编译 | R3 方案：纯 torch NPU 位打包降级（`view(uint16)`/移位/`&3` 全成熟），数值判据不变 | [UNKNOWN] |
| R3 | PR 内核移植后数值语义偏移（fp16 舍入序、打包位序、LSE 合并） | L3/L5 | 精度劣化 | 与沙盒 l3_numeric 同判据（store=0/dequant 1e-5/decode 1e-4）；逐对齐 N-01..N-08 | [UNKNOWN→TODO3-4 后关闭] |
| R4 | `torch.quantile`（裁剪）在 NPU 不可用/慢 | L5 | clip 路径失败 | 默认 clip=0 绕过；>0 时 probe `torch.quantile`，失败→`torch.topk` 近似 | [UNKNOWN] |
| R5 | 窗口 staging 的 `cumprod/flip/gather` NPU 语义 | L5 | LSE 合并出错 | 沙盒镜像 + 窗口 probe（比对 BF16 全窗口 vs 混合） | [UNKNOWN] |
| R6 | CUDA Graph：`_cudagraph_support=NEVER`（PR 语义）→ 用户命令里的 FULL_DECODE_ONLY 捕捉集对 FULL 层无效 | L0 | 性能（MTP enforce_eager 下主要走 eager，可接受） | 日志确认 cudagraph_mode 回落；显式传 `--enforce-eager` 兜底 | CONFIRMED（PR `oscar_attn.py:134-141`）+ 行为影响 |
| R7 | 指纹/心跳缺失导致"修复未上机" | L0 | 误判 | 三防线：HEAD + 源码状态断言 + sha256（skill §9） | 设计内 |
| R8 | 混合模型 prefix 缓存/重写时 INT2 槽与原生偏移不一致 | L2 | 读到未写槽 | 全量 zero 初始化（`needs_kv_cache_zeroing`，`kv_cache_interface.py:878`）+ 只读已写段；probe 校验 | 设计内 |

## 9. 常见误区与踩坑（本主题已证实/相关）

| 误区/坑 | 根因 | 正确做法 | 证据 |
|---|---|---|---|
| "改 page_size 就能省显存" | hybrid 组张量按 mamba 页对齐（page_size_padded），改 attention 页大小会触发 4493/4569 分支重算切分 | 不动几何；160B 槽嵌原生 1024B 窗，省的是带宽/读写字节，不是页表 | `model_runner_v1.py:5025-5034`；§3 B 否决 |
| "量化公式先取 scale 再舍零" | 舍入顺序影响 bin 边界（1 电平差） | 严格 N-02：scale/zero 先 fp16 舍入，再量化 | `triton_oscar_store.py:50-56`；N-02 |
| "反量化用 1/scale 乘" | 1 ulp 差可在 0.5 边界翻转 | 直除 | N-08 |
| "PR 的 get_kv_cache_shape 直接搬" | Ascend hybrid 需要 5 维 `(2,nb,bs,hk,d)` 原生形状才能走通切分 | 保持原生 backend 形状；组合槽在 impl 层 as_strided | `attention_v1.py:104-112`；§3 A |
| "位打包一定能在 Triton-Ascend 跑" | triton-ascend 后端覆盖不全 | 门禁 HAS_TRITON + 冒烟 + torch 降级 | R2/R3 |
| "Sink/Recent 分区做在页内" | 页内分区需每块动态三段边界，页表/前缀命中全部牵动 | 采用 PR staging arena（页外 NPU 缓冲区 + owner tag） | `oscar_attn.py:245-336` |

## 10. 术语表

| 名词 | 一句话解释 |
|---|---|
| FULL 层 | Qwen3.5 混合模型中执行全注意力（self_attn）的层；GDN 层执行线性注意力（linear_attn） |
| 三大条 | vllm-ascend hybrid 每个组张量内的三连续区：条A conv、条B (ssm或K)、条C V（用户 §1 术语，本文沿用） |
| 组合槽视图 | 把原生 K 视图与 V 视图（物理连续）合并成 `(nb,bs,hk,1024)` uint8 的 as_strided 视图 |
| staging arena | OSCAR 的 BF16 Sink/Recent 环形缓冲（带 owner tag 防错配），页外分配 |
| LSE 合并 | 两个部分注意力（INT2 中段 + BF16 窗口）按 log-sum-exp 权重合并 |
| tq_slot_size | TurboQuant 的"每 token 每 KV 头字节数"语义；本方案逻辑槽 = 160B |
| 类外科手术 | 运行期替换 `impl.__class__`（不动 backend/allocator），先例 kv_c8.py |
| page_size_padded | FULL 页被对齐到 mamba 页大小的 pad 后页字节数（hybrid 特有） |

## 11. 当前状态与更新记录

- 状态：DRAFT；下一步 = TODO 1（format.py + 纯函数参考实现与 sandbox 对账）。
- 更新记录：

| 日期 | 变化 | 证据/commit |
|---|---|---|
| 2026-09-03 | 初稿：候选 A 选定 + 数值契约/旋转/窗口/接缝全表 | 本文件 + references/ 树 file:line（§6） |
