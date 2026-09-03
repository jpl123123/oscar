# oscar-ascend — OSCAR INT2 KV 缓存量化 · vllm-ascend 0.23.0 零侵入插件

针对 Qwen3.5-27B-w8a8-mtp（GDN 线性注意力 + FULL 全注意力混合模型）在 vLLM Ascend
0.23.0 上启用 OSCAR INT2 KV 缓存量化：**不修改 vllm/vllm-ascend 任何源码**，通过
`vllm.general_plugins` 入口 + 运行期 `impl.__class__` 类外科手术接入；全部算子
Triton（triton-ascend，回退纯 torch NPU 路径），量化/反量化/旋转/打包全在 NPU。

方案全文：`plan/PLAN-1-oscar-int2-vllm-ascend.md`（design gate PASS）。

## 真机一键（用户唯一动作）

Docker 环境（vllm / vllm-ascend / torch-npu / triton-ascend 已预装）内，仓库挂载后：

```bash
cd /workspace/new_oscar_triton && git pull && bash delivery/install_and_launch.sh
```

### 容器启动示例（NPU 设备 + 驱动 + 仓库 + 模型目录挂载）

```bash
docker run -it --rm \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v "$(pwd)":/workspace/new_oscar_triton \
  -v /softwarePlatform:/softwarePlatform \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  <vllm-ascend 镜像> bash
```

> 挂载路径/设备数按你现网 docker 启动方式为准；脚本只依赖：容器内 `python3` 可导入
> `vllm/vllm_ascend/torch_npu`、`torch.npu.is_available()` 为 True、仓库与模型路径已挂载。

### 脚本流程（已按 Docker 预装环境裁剪）

自检（版本 / NPU 可见性 / `HAS_TRITON` / 插件入口点）→ 安装插件（**`pip install
--no-deps --no-build-isolation -e .`，只装自身、不解析/升级预装依赖**；入口点已存在则自动跳过，
`OSCAR_SKIP_INSTALL=1` 强制跳过）→ 指纹（HEAD/sha256/入口点）→ 生成旋转检查点
`oscar_rotations.pt`（`tools/gen_rotations.py`，离线校准）→ 数值 probe（ref + triton 双模式，
**任意 FAIL 拒绝 serve**）→ 以用户目标命令拉起 `vllm serve`（`delivery/serve_oscar.sh`，
端口 8989，TP4，MTP3，262144 上下文，原命令逐项保留）。

变体：

```bash
OSCAR_ASCEND_GEN_ROTATIONS=1 bash delivery/install_and_launch.sh   # 强制重新生成旋转 pt
OSCAR_ASCEND_GEN_PROMPTS=8 OSCAR_ASCEND_GEN_MAXLEN=256 bash delivery/install_and_launch.sh
OSCAR_ASCEND_GEN_LLM_ARGS='{"tensor_parallel_size":4,"gpu_memory_utilization":0.9}' \
  bash delivery/install_and_launch.sh   # 校准默认即 TP4（与 serve 一致，27B 单卡必 OOM）
OSCAR_EXTRA_ARGS="--enforce-eager" bash delivery/serve_oscar.sh    # 建议：OSCAR 窗口仅 eager 验证
OSCAR_SKIP_INSTALL=1 bash delivery/install_and_launch.sh           # 复用容器内已装插件
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

- **数值**：本地 CPU 镜像 7/7 PASS（store 字节差=0、dequant≤1e-5、decode≤1e-4、旋转不变性、旋转加载/缺层回退）。
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
