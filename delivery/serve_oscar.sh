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

export VLLM_PLUGINS="${VLLM_PLUGINS:-oscar_ascend}"
# 真机实测（2026-09-03 12:58）：TP4 多进程 worker 若用 fork 在多线程父进程下触发
# PyTorch "ParallelOpenMP.cpp:64 Invalid thread pool" 崩溃 → 默认 spawn。
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export OSCAR_ASCEND_ENABLE="${OSCAR_ASCEND_ENABLE:-auto}"
export OSCAR_ASCEND_K_ROTATION_PATH="${OSCAR_ASCEND_K_ROTATION_PATH:-$REPO_ROOT/oscar_rotations.pt}"
export OSCAR_ASCEND_V_ROTATION_PATH="${OSCAR_ASCEND_V_ROTATION_PATH:-$REPO_ROOT/oscar_rotations.pt}"
export OSCAR_ASCEND_K_CLIP_RATIO="${OSCAR_ASCEND_K_CLIP_RATIO:-0.0}"
export OSCAR_ASCEND_V_CLIP_RATIO="${OSCAR_ASCEND_V_CLIP_RATIO:-0.0}"
export OSCAR_ASCEND_SINK_TOKENS="${OSCAR_ASCEND_SINK_TOKENS:-64}"
export OSCAR_ASCEND_RECENT_TOKENS="${OSCAR_ASCEND_RECENT_TOKENS:-256}"
export OSCAR_ASCEND_STAGING_TOKENS="${OSCAR_ASCEND_STAGING_TOKENS:-8192}"

if [ ! -f "$OSCAR_ASCEND_K_ROTATION_PATH" ]; then
  echo "⚠️ [oscar-ascend] 旋转检查点不存在: $OSCAR_ASCEND_K_ROTATION_PATH（将以单位阵降级运行；"
  echo "   可用 OSCAR_ASCEND_GEN_ROTATIONS=1 bash delivery/install_and_launch.sh 重新生成）"
fi

exec vllm serve "$MODEL_PATH" \
    --served-model-name "qwen3.5" \
    --host 0.0.0.0 \
    --port 8989 \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --max-num-batched-tokens 16384 \
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
    --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' \
    $OSCAR_EXTRA_ARGS
