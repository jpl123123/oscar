#!/usr/bin/env bash
# delivery/install_and_launch.sh — OSCAR INT2 插件一键安装&启动（真机唯一入口）
#
# 用户唯一动作：git pull && bash delivery/install_and_launch.sh
#
# 阶段：
#   1) 自检     —— vllm-ascend 0.23.0 + triton(HAS_TRITON) 探测 + 模型目录 + 插件源码
#   2) 安装     —— pip install -e .（纯 Python 插件，无编译）
#   3) 指纹     —— HEAD / 包 sha256 / 插件心跳
#   4) 生成 pt  —— tools/gen_rotations.py（OSCAR 旋转检查点；已存在或 env 已给路径则跳过）
#   5) 数值probe —— delivery/probe_oscar.py（ref + triton 双模式，阻塞 serve）
#   6) serve    —— delivery/serve_oscar.sh（用户目标启动命令 + 插件环境）
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
$PYTHON - <<'PY' > "$LOG_DIR/selfcheck_$STAMP.log" 2>&1 || fail "自检失败（见日志）"
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
    print("  ⚠️ HAS_TRITON=False → probe 将只验证 torch 参考路径；"
          "服务端正式性能路径需 triton-ascend（见 README）")
PY
cat "$LOG_DIR/selfcheck_$STAMP.log"
[ -d "$MODEL_PATH" ] || fail "模型目录不存在: $MODEL_PATH"
[ -f "oscar_ascend/plugin.py" ] || fail "插件源码缺失: oscar_ascend/plugin.py"

# ---------- 阶段2 安装 ----------
step "安装插件 wheel（纯 Python）"
$PYTHON -m pip install -e "$REPO_ROOT" 2>&1 | tee "$LOG_DIR/pip_$STAMP.log"

# ---------- 阶段3 指纹 ----------
step "指纹: HEAD / plugin sha / entry point"
HEAD="$(git rev-parse --short HEAD 2>/dev/null || echo no-git)"
WHEEL_SHA="$(sha256sum oscar_ascend/plugin.py | cut -d' ' -f1)"
echo "  HEAD      : $HEAD"
echo "  plugin sha: $WHEEL_SHA"
$PYTHON - <<'PY' || fail "插件不可导入（pip 安装失败或 entry point 未生效）"
import importlib.metadata as md
eps = [e for e in md.entry_points(group="vllm.general_plugins") if e.name == "oscar_ascend"]
assert eps, "未发现 vllm.general_plugins entry point 'oscar_ascend'"
print("  entry point:", eps[0].value)
import oscar_ascend
print("  oscar_ascend import OK, version", oscar_ascend.__version__)
PY

# ---------- 阶段4 生成 pt（OSCAR 旋转检查点） ----------
ROT_DEFAULT="$REPO_ROOT/oscar_rotations.pt"
if [ -z "${OSCAR_ASCEND_K_ROTATION_PATH:-}" ] && [ -z "${OSCAR_ASCEND_V_ROTATION_PATH:-}" ]; then
    if [ "${OSCAR_ASCEND_GEN_ROTATIONS:-0}" == "1" ] || [ ! -e "$ROT_DEFAULT" ]; then
        step "生成 OSCAR 旋转检查点（离线校准，一次性）"
        $PYTHON tools/gen_rotations.py --model "$MODEL_PATH" --save "$ROT_DEFAULT" \
            --prompts "${OSCAR_ASCEND_GEN_PROMPTS:-4}" --max-len "${OSCAR_ASCEND_GEN_MAXLEN:-128}" \
            2>&1 | tee "$LOG_DIR/genrot_$STAMP.log" || fail "旋转检查点生成失败"
    else
        step "复用旋转检查点: $ROT_DEFAULT"
    fi
    export OSCAR_ASCEND_K_ROTATION_PATH="$ROT_DEFAULT"
    export OSCAR_ASCEND_V_ROTATION_PATH="$ROT_DEFAULT"
    echo "  OSCAR_ASCEND_K/V_ROTATION_PATH=$ROT_DEFAULT"
fi

# ---------- 阶段5 数值 probe（阻塞 serve） ----------
step "真机数值 probe（ref + triton；FAIL 阻断 serve）"
if [ "${OSCAR_SKIP_PROBES:-0}" != "1" ]; then
    "$PYTHON" delivery/probe_oscar.py --mode ref \
        || fail "数值 probe(ref) FAIL —— 拒绝 serve"
    if $PYTHON -c "from vllm.triton_utils import HAS_TRITON; import sys; sys.exit(0 if HAS_TRITON else 1)"; then
        "$PYTHON" delivery/probe_oscar.py --mode triton \
            || fail "数值 probe(triton) FAIL —— 拒绝 serve"
    else
        echo "  HAS_TRITON=False → 跳过 triton probe（服务将走 torch 参考路径，性能未达最优）"
    fi
else
    echo "  OSCAR_SKIP_PROBES=1 → 跳过 probe（仅诊断用，禁止用于正式交付验收）"
fi

# ---------- 阶段6 serve ----------
step "启动 vllm serve（目标命令 + 插件环境）"
export OSCAR_ASCEND_LOG_DIR="$LOG_DIR"
bash delivery/serve_oscar.sh "$@" 2>&1 | tee "$SERVE_LOG"
