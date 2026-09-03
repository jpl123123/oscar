# oscar-ascend — OSCAR INT2 KV 缓存量化 · vllm-ascend 0.23.0 零侵入插件

针对 Qwen3.5-27B-w8a8-mtp（GDN 线性注意力 + FULL 全注意力混合模型）在 vLLM Ascend
0.23.0 上启用 OSCAR INT2 KV 缓存量化：**不修改 vllm/vllm-ascend 任何源码**，通过
`vllm.general_plugins` 入口 + 运行期 `impl.__class__` 类外科手术接入；全部算子
Triton（triton-ascend，回退纯 torch NPU 路径），量化/反量化/旋转/打包全在 NPU。

方案全文：`plan/PLAN-1-oscar-int2-vllm-ascend.md`（design gate PASS）。

## 真机一键（用户唯一动作）

```bash
git pull && bash delivery/install_and_launch.sh
```

流程：自检（vllm-ascend 0.23.0 + HAS_TRITON）→ 安装插件 wheel → 指纹（HEAD/sha/入口点）
→ 生成旋转检查点 `oscar_rotations.pt`（`tools/gen_rotations.py`，离线校准）→ 数值 probe
（ref + triton 双模式，**任意 FAIL 拒绝 serve**）→ 以用户目标命令拉起 `vllm serve`
（`delivery/serve_oscar.sh`，端口 8989，TP4，MTP3，262144 上下文，原命令逐项保留）。

生成 PT / 仅起服务等变体：

```bash
OSCAR_ASCEND_GEN_ROTATIONS=1 bash delivery/install_and_launch.sh   # 强制重新生成旋转 pt
OSCAR_ASCEND_GEN_PROMPTS=8 OSCAR_ASCEND_GEN_MAXLEN=256 bash delivery/install_and_launch.sh
OSCAR_EXTRA_ARGS="--enforce-eager" bash delivery/serve_oscar.sh    # 建议：OSCAR 窗口仅 eager 验证
```

## 本地门禁（开发机，无 NPU）

```bash
python3 tests/test_numeric.py    # CPU 镜像：store 字节差=0 / dequant≤1e-5 / decode≤1e-4
```

## 环境变量（插件私有 `OSCAR_ASCEND_*`，默认=OSCAR PR 默认）

| 变量 | 默认 | 说明 |
|---|---|---|
| `OSCAR_ASCEND_ENABLE` | `auto` | `0`=禁用；`auto`=仅 hybrid 模型注入 |
| `OSCAR_ASCEND_K/V_ROTATION_PATH` | `oscar_rotations.pt` | 旋转检查点（K/V 用 `rotation`/`rotation_v` 字段；缺省→单位阵） |
| `OSCAR_ASCEND_K/V_CLIP_RATIO` | `0.0` | 裁剪分位数（>0 走 `torch.quantile`，NPU 未正式验证 → 默认关） |
| `OSCAR_ASCEND_SINK_TOKENS` / `RECENT_TOKENS` / `STAGING_TOKENS` | `64` / `256` / `8192` | BF16 Sink/Recent 窗口与 staging 容量 |
| `OSCAR_ASCEND_FORCE_TORCH` | `0` | `1`=强制 torch 参考路径（跳过 Triton，调试用） |

## 诚实边界（务必阅读）

- **数值**：本地 CPU 镜像 6/6 PASS（store=0、dequant≤1e-5、decode≤1e-4、旋转不变性）。
  Triton 内核尚未在任何 NPU 编译运行——真机 probe 的 `--mode triton` 会**逐字节对比**
  Triton 与参考实现（`ref`），FAIL 即拒绝 serve（R3）。
- **页常数**：你提供的 P=801,792 等为 HYPOTHESIS（参考树未含）；服务启动日志与 probe
  会打印 `kv_cache_tensor.size` / `k_cache.stride` / `page_size_padded`，与偏移公式对账
  （plan R1）。
- **窗口路径**：OSCAR 的 Sink/Recent 窗口仅 eager 验证（PR `_cudagraph_support=NEVER`，
  plan R6）——建议 `OSCAR_EXTRA_ARGS="--enforce-eager"`。
- **CPU 边界**：唯一 CPU 动作是 `gen_rotations.py` 的一次性离线校准（且默认 NPU eigh、
  仅回退时 CPU）；推理热路径零 CPU 搬运。

## 结构

```
oscar_ascend/
  plugin.py      vllm.general_plugins 入口：impl 类外科手术 + fail-soft + 心跳
  config.py      OSCAR_ASCEND_* 配置
  format.py      160B 槽数值契约唯一权威（N-01..N-09）
  rotation.py    per-layer 正交旋转加载（缺层→单位阵）
  backend.py     AscendOscarAttentionBackendImpl（store/decode/prefill/窗口三态）
  kernels/       Triton + torch 双路径（store / decode / dequant）
delivery/        一键安装&启动 + probe（真机门禁）
tools/           旋转检查点生成（离线校准）
tests/           CPU 数值镜像
plan/            方案与设计 gate 记录
```
