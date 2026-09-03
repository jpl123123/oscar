#!/usr/bin/env bash
# delivery/install_and_launch.sh — OSCAR INT2 插件一键安装&启动（真机唯一入口）
#
# 用户唯一动作：git pull && bash delivery/install_and_launch.sh
#
# 阶段：
#   1) 自检     —— vllm-ascend 0.23.0 存在 + triton 可用(HAS_TRITON) + 模型目录存在
#   2) 安装     —— pip install -e ./oscar_ascend（纯 Python，无编译）
#   3) 指纹     —— HEAD + 包 sha256 + 插件心跳
#   4) 数值probe —— delivery/probe_oscar.py（阻塞 serve；任意 FAIL 即终止）
#   5) serve    —— delivery/serve_oscar.sh（用户目标启动命令 + 插件环境）
#
# 失败协议：任一阶段非零 → 打印阶段名/日志尾部/修复指引后退出（不静默继续）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
LOG_DIR="${OSCAR_LOG_DIR:-/tmp/oscar_ascend_logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
SERVE_LOG="$LOG_DIR/serve_$STAMP.log"

MODEL_PATH="${MODEL_PATH:-/softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-oscar_ascend}"

fail() { echo "❌ [oscar-ascend] $1" >&2; echo "   日志: $LOG_DIR/*$STAMP*" >&2; exit 1; }
step() { echo "==> [oscar-ascend] $1"; }

# ---------- 阶段1 自检 ----------
step "自检: 环境 / vllm-ascend / triton-ascend / 模型目录"
$PYTHON - <<'PY' || fail "自检失败"
import importlib.metadata as md, os, sys
for name in ("vllm", "vllm-ascend", "triton"):
    try:
        v = md.version(name)
    except Exception:
        v = None
    print(f"  {name}: {v}")
try:
    from vllm.triton_utils import HAS_TRITON
except Exception as e:
    HAS_TRITON = False
    print(f"  triton_utils import failed: {e}")
print(f"  HAS_TRITON: {HAS_TRITON}")
if not HAS_TRITON:
    sys.exit(3)
PY
[ -d "$MODEL_PATH" ] || fail "模型目录不存在: $MODEL_PATH"
[ -f "oscar_ascend/plugin.py" ] || fail "插件源码缺失: oscar_ascend/plugin.py（先实现 plan TODO 1-6）"

# ---------- 阶段2 安装 ----------
step "安装插件 wheel（纯 Python）"
$PYTHON -m pip install -e "$REPO_ROOT" 2>&1 | tee "$LOG_DIR/pip_$STAMP.log"

# ---------- 阶段3 指纹 ----------
step "指纹: HEAD / 包 sha256 / 插件心跳"
HEAD="$(git rev-parse --short HEAD 2>/dev/null || echo no-git)"
WHEEL_SHA="$(sha256sum oscar_ascend/plugin.py | cut -d' ' -f1)"
PLUGIN_HB=""
if $PYTHON -c "import oscar_ascend.plugin as p" 2>/dev/null; then
    PLUGIN_HB="$($PYTHON - <<'PY'
import vllm.plugins as P
# 模拟加载逻辑：能 import 即注册
import oscar_ascend.plugin
print("plugin import OK")
PY
)"
fi
echo "  HEAD      : $HEAD"
echo "  plugin sha: $WHEEL_SHA"
echo "  plugin    : $PLUGIN_HB"
[ -n "$PLUGIN_HB" ] || fail "插件不可导入（pip 安装失败或 entry point 未生效）"

# ---------- 阶段4 数值 probe（阻塞 serve） ----------
step "真机数值 probe（store/dequant/decode 三查；FAIL 阻断 serve）"
if [ "${OSCAR_SKIP_PROBES:-0}" != "1" ]; then
    "$PYTHON" delivery/probe_oscar.py --head-dim 256 2>&1 | tee "$LOG_DIR/probe_$STAMP.log" \
        || fail "数值 probe FAIL（= 修复未上机/数值未对齐），拒绝 serve"
else
    echo "  OSCAR_SKIP_PROBES=1 → 跳过 probe（仅诊断用，禁止用于正式交付验收）"
fi

# ---------- 阶段5 serve ----------
step "启动 vllm serve（目标命令 + 插件环境）"
export OSCAR_ASCEND_LOG_DIR="$LOG_DIR"
bash delivery/serve_oscar.sh "$@"
