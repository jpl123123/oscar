#!/usr/bin/env bash
# delivery/serve_oscar.sh — 用户目标启动命令（OSCAR 插件环境注入）
#
# 与用户给定命令逐项一致；仅追加环境：
#   * VLLM_PLUGINS=oscar_ascend                      （插件入口）
#   * OSCAR_ASCEND_ENABLE=auto                       （hybrid 自动启用）
#   * OSCAR_ASCEND_K/V_ROTATION_PATH（默认 oscar_rotations.pt，可被外层传入覆盖）
#   * OSCAR_ASCEND_SINK/RECENT/STAGING               （BF16 窗口默认 64/256/8192，同 PR）
# 说明：OSCAR 窗口路径仅 eager 验证（plan R6）→ 建议追加 --enforce-eager：
#   OSCAR_EXTRA_ARGS="--enforce-eager" bash delivery/serve_oscar.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp}"

export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend,oscar_ascend}"
# 真机实测（2026-09-03 12:58）：TP4 多进程 worker 若用 fork 在多线程父进程下触发
# PyTorch "ParallelOpenMP.cpp:64 Invalid thread pool" 崩溃 → 默认 spawn。
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
# CPU 压力控制：限制 OpenMP/torch 线程数（默认 8；按需覆盖）
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"


# 显式选择 0-3 号卡：npu-smi 显示 4-7 卡被其它作业占满（各 ~26GB），TP4 必须落到空闲卡
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"

export OSCAR_ASCEND_ENABLE="${OSCAR_ASCEND_ENABLE:-auto}"
export OSCAR_ASCEND_K_ROTATION_PATH="${OSCAR_ASCEND_K_ROTATION_PATH:-$REPO_ROOT/oscar_rotations.pt}"
export OSCAR_ASCEND_V_ROTATION_PATH="${OSCAR_ASCEND_V_ROTATION_PATH:-$REPO_ROOT/oscar_rotations.pt}"
# 裁剪 must be on：per-vector INT2 的 min/max 被尾部离群拉大 → 无裁剪时 bulk 分量塌缩
# （PR oscar_gpqa_eval.py:71-72 的 0.96/0.92；论文内核测试 0.95/0.90 同量级）。
export OSCAR_ASCEND_K_CLIP_RATIO="${OSCAR_ASCEND_K_CLIP_RATIO:-0.96}"
export OSCAR_ASCEND_V_CLIP_RATIO="${OSCAR_ASCEND_V_CLIP_RATIO:-0.92}"
# sink 必须 ≥ block_size(128) 才产生实际页：64<128 → sink_eff=0，BF16 sink 窗口静默失效
export OSCAR_ASCEND_SINK_TOKENS="${OSCAR_ASCEND_SINK_TOKENS:-128}"
export OSCAR_ASCEND_RECENT_TOKENS="${OSCAR_ASCEND_RECENT_TOKENS:-256}"
export OSCAR_ASCEND_STAGING_TOKENS="${OSCAR_ASCEND_STAGING_TOKENS:-8192}"
# Triton 路径默认启用（store 单核散写 / dequant fused 反量化；2026-09-04 起一键默认）。
# 门禁契约：install_and_launch.sh 阶段5 的 triton probe 默认硬门禁（REQUIRE_TRITON=1），
# probe 覆盖 store 字节 + dequant/decode 数值对照（与 serve 相同的 Hk=1/Hq=8 特化）；
# 观察模式（REQUIRE_TRITON=0）下 probe 失败时启动器会显式注入 USE_TRITON=0 覆盖此默认。
# ⚠️ 直接运行本脚本不经过 probe 门禁：若 triton-ascend 编译异常，用
#    OSCAR_ASCEND_USE_TRITON=0 bash delivery/serve_oscar.sh 回退 torch 参考路径。
export OSCAR_ASCEND_USE_TRITON="${OSCAR_ASCEND_USE_TRITON:-1}"

for rotation_path in "$OSCAR_ASCEND_K_ROTATION_PATH" "$OSCAR_ASCEND_V_ROTATION_PATH"; do
  if [ ! -f "$rotation_path" ]; then
    echo "❌ 旋转检查点不存在: $rotation_path；请先运行 install_and_launch.sh" >&2
    exit 1
  fi
done

# R6 硬性：OSCAR impl 含运行时宿主控制流（.item()/.tolist()/逐层首写日志），
# 仅 eager 验证（参考 PR _cudagraph_support=NEVER）→ 字面 --enforce-eager 防图捕获破坏。
# 恢复 graph：手动删除下面一行的 --enforce-eager（不推荐，未验证）。

# ---------- 方案 E（plan/DESIGN-20260904-E）：×2 packed 槽，默认开启 ----------
# 机制：--kv-cache-dtype int8_per_token_head（合法字面量→torch.int8）使 patch_mamba_config
# 页几何自动变为 256B/token·head × 1536 token/页；assert 256×1536==393216 成立；
# 页字节 801,792B/三大条/16 池/GDN 全部不变 → FULL token 密度 ×2（真显存收益）。
# OSCAR 槽内偏移（96/64B < 256B）零内核改动；MTP 草稿层由插件影子池保持 BF16
# （OSCAR_ASCEND_PACKED=1 时自动启用，见 plugin.py / mtp_shadow.py）。
# 回退：OSCAR_ASCEND_PACKED=0 bash delivery/serve_oscar.sh（同时回到 bf16 几何）。
if [ "${OSCAR_ASCEND_PACKED:-1}" == "1" ]; then
  export OSCAR_ASCEND_PACKED=1
  PACKED_KV_DTYPE_ARGS="--kv-cache-dtype int8_per_token_head"
  # 1536 的 mamba 对齐切分要求预算为其倍数：16384 不是（16384/1536=10.67）→ 用 15360=1536×10。
  # （DESIGN-E §5.2 中"16128"为笔误：16128/1536=10.5 非整倍数。）
  BATCHED_TOKENS="${OSCAR_ASCEND_BATCHED_TOKENS:-15360}"
  echo "🧩 [oscar-ascend] packed×2 模式: kv-cache-dtype=int8_per_token_head, max-num-batched-tokens=$BATCHED_TOKENS（OSCAR_ASCEND_PACKED=0 可回退）"
else
  PACKED_KV_DTYPE_ARGS=""
  BATCHED_TOKENS="${OSCAR_ASCEND_BATCHED_TOKENS:-16384}"
fi

exec vllm serve "$MODEL_PATH" \
    --served-model-name "qwen3.5" \
    --host 0.0.0.0 \
    --port 8989 \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --max-num-batched-tokens "$BATCHED_TOKENS" \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.9 \
    --compilation-config '{"cudagraph_capture_sizes":[1,4,8,12,16,24,32,48,56,64,72,84,96,108,112,128,160,172,196,200,212,232,272,288,312,328,344,360,384,400,416,432,448,480,512], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
    --trust-remote-code \
    --async-scheduling \
    --allowed-local-media-path / \
    --quantization ascend \
    --mm-processor-cache-gb 0 \
    --additional-config '{"enable_cpu_binding":true}' \
    --mamba-cache-dtype bfloat16 \
    --mamba-ssm-cache-dtype bfloat16 \
    --enforce-eager \
    --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' \
    $PACKED_KV_DTYPE_ARGS \
    ${OSCAR_EXTRA_ARGS:-}
