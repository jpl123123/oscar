#!/usr/bin/env bash
# delivery/install_and_launch.sh — OSCAR INT2 插件一键安装&启动（真机 Docker 环境）
#
# 前提（用户实测环境）：Docker 容器内已预装 vllm / vllm-ascend / torch-npu / triton-ascend，
# 本脚本**不触碰**预装环境：插件以 `pip install --no-deps -e .` 只装自身、不解析/升级依赖。
# 仓库挂载进容器后，在容器内执行：
#   git pull && bash delivery/install_and_launch.sh
#
# 阶段：
#   1) 自检     —— 版本/NPU 可见性/HAS_TRITON/入口点（只报告，严重项才 FAIL）
#   2) 安装     —— 入口点已存在则跳过；否则 pip install --no-deps --no-build-isolation -e .
#   3) 指纹     —— HEAD / plugin sha256 / entry point / import
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
# serve 日志：固定名 + 每次启动原地覆盖（用户要求：不要时间戳命名、必须覆盖）
SERVE_LOG="$LOG_DIR/serve.log"
: > "$SERVE_LOG"   # 前置截断（setsid nohup > 亦会覆盖，双保险）

MODEL_PATH="${MODEL_PATH:-/softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp}"
# VLLM_PLUGINS 为跨组白名单（vllm envs.py:1041 逗号分隔、精确匹配）：必须同时包含
# platform 插件 "ascend" 与我们的 general 插件 "oscar_ascend"——只写后者会把
# vllm_ascend:register 过滤掉 → 平台未激活（真机 diag [3] 石锤）。
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend,oscar_ascend}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
# CPU 压力控制：限制 OpenMP/torch 线程数（默认 8；按需覆盖）
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# vendor vllm 跨进程 RPC 传函数需 pickle 回退（官方提示的出口；gen 校准用）
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"

# 显式选择 0-3 号卡：npu-smi 显示 4-7 卡被其它作业占满（各 ~26GB），TP4 必须落到空闲卡
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"


fail() { echo "❌ [oscar-ascend] $1" >&2; echo "   日志: $LOG_DIR/*$STAMP*" >&2; exit 1; }
step() { echo "==> [oscar-ascend] $1"; }

# ---------- 阶段1 自检（Docker 预装环境） ----------
step "自检: vllm/vllm-ascend/triton/NPU 可见性/插件入口点"
$PYTHON - <<'PY' > "$LOG_DIR/selfcheck_$STAMP.log" 2>&1 || { cat "$LOG_DIR/selfcheck_$STAMP.log"; fail "自检失败（Python 环境异常）"; }
import importlib.metadata as md, os, sys
print("  CWD:", os.getcwd())
for name in ("vllm", "vllm-ascend", "triton", "torch", "torch-npu", "torch_npu"):
    try:
        print(f"  {name}: {md.version(name)}")
    except Exception:
        print(f"  {name}: NOT INSTALLED")
try:
    import torch, torch_npu  # noqa: F401
    import torch_npu  # noqa: F401
    print("  torch.npu.is_available():", torch.npu.is_available())
    print("  device_count:", torch.npu.device_count())
    print("  ASCEND_RT_VISIBLE_DEVICES:", os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "<unset>"))
    if not torch.npu.is_available():
        print("  ❌ torch.npu 不可用（容器需挂载 NPU 设备 / ASCEND_RT_VISIBLE_DEVICES）")
        sys.exit(4)
except Exception as e:
    print("  ❌ torch/torch_npu 导入失败:", e)
    sys.exit(4)
try:
    from vllm.triton_utils import HAS_TRITON
    print("  HAS_TRITON:", HAS_TRITON)
    if not HAS_TRITON:
        print("  ⚠️ HAS_TRITON=False → probe 只验证 torch 参考路径；性能路径需 triton-ascend")
except Exception as e:
    print("  HAS_TRITON: import failed:", e)
eps = [e for e in md.entry_points(group="vllm.general_plugins") if e.name == "oscar_ascend"]
print("  oscar_ascend entry point:", eps[0].value if eps else "MISSING")
PY
cat "$LOG_DIR/selfcheck_$STAMP.log"
[ -d "$MODEL_PATH" ] || fail "模型目录不存在（容器内挂载检查）: $MODEL_PATH"
[ -f "oscar_ascend/plugin.py" ] || fail "插件源码缺失（仓库挂载检查）: oscar_ascend/plugin.py"

# Diag：platform 注册/device_type 现场取证（vendor fork 差异定位；只报告不阻断）
step "平台注册诊断（tools/diag_platform.py，异常时请回传输出）"
$PYTHON tools/diag_platform.py 2>&1 | tee "$LOG_DIR/diag_platform_$STAMP.log" || true

# ---------- 阶段2 安装（只装插件，--no-deps 不触碰预装环境） ----------
_ep_check() {
    $PYTHON -c "import importlib.metadata as md; eps=[e for e in md.entry_points(group='vllm.general_plugins') if e.name=='oscar_ascend']; import sys; sys.exit(0 if eps else 1)"
}
if [ "${OSCAR_SKIP_INSTALL:-auto}" == "1" ]; then
    step "OSCAR_SKIP_INSTALL=1 → 跳过安装（复用已装插件）"
elif _ep_check; then
    step "插件入口点已存在 → 跳过安装（复用已装插件）"
else
    step "安装插件 wheel（pip install --no-deps --no-build-isolation -e .）"
    $PYTHON -m pip install --no-deps --no-build-isolation -e "$REPO_ROOT" \
        2>&1 | tee "$LOG_DIR/pip_$STAMP.log" || fail "pip 安装失败"
fi

# ---------- 阶段3 指纹 ----------
step "指纹: HEAD / plugin sha / entry point / import"
HEAD="$(git rev-parse --short HEAD 2>/dev/null || echo no-git-in-mount)"
WHEEL_SHA="$(sha256sum oscar_ascend/plugin.py | cut -d' ' -f1)"
echo "  HEAD      : $HEAD"
echo "  plugin sha: $WHEEL_SHA"
$PYTHON - <<'PY' || fail "插件不可导入（入口点未生效）"
import importlib.metadata as md
eps = [e for e in md.entry_points(group="vllm.general_plugins") if e.name == "oscar_ascend"]
assert eps, "未发现 vllm.general_plugins entry point 'oscar_ascend'"
print("  entry point:", eps[0].value)
import oscar_ascend
print("  oscar_ascend import OK, version", oscar_ascend.__version__)
PY

# ---------- 阶段3.5 NPU 显存/残留进程预检 ----------
step "NPU 显存预检 + 残留 vllm 进程清理（仅限本模型进程）"
if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info 2>&1 | tee "$LOG_DIR/npu_smi_$STAMP.log" || true
else
    echo "  npu-smi 不在 PATH（Docker 未挂载）——跳过，仅凭日志判断显存"
fi
# 上一轮崩溃的服务/校准进程可能仍占 NPU 显存（W8A8 27B 单卡 29.49GiB 极紧）
pkill -f "$MODEL_PATH" 2>/dev/null || true
sleep 3 || true

# ---------- 阶段3.9 启动前预检（env/插件/入口点；不通过即 fail） ----------
step "启动前预检（delivery/check_oscar_active.sh --preflight）"
bash delivery/check_oscar_active.sh --preflight || fail "预检未通过（见上方 ❌ 项）"

# ---------- 阶段4 生成 pt（OSCAR 旋转检查点；校准默认 TP4 与 serve 一致） ----------
ROT_DEFAULT="$REPO_ROOT/oscar_rotations.pt"
if [ -z "${OSCAR_ASCEND_K_ROTATION_PATH:-}" ] && [ -z "${OSCAR_ASCEND_V_ROTATION_PATH:-}" ]; then
    if [ "${OSCAR_ASCEND_GEN_ROTATIONS:-0}" == "1" ] || [ ! -e "$ROT_DEFAULT" ]; then
        step "生成 OSCAR 旋转检查点（离线校准，一次性；Docker 内执行）"
        # 校准进程关闭插件（OSCAR_ASCEND_ENABLE=0）：BF16 原路径采集 K/V，避免与注入路径互扰
        OSCAR_ASCEND_ENABLE=0 $PYTHON tools/gen_rotations.py --model "$MODEL_PATH" --save "$ROT_DEFAULT" \
            --max-len "${OSCAR_ASCEND_GEN_MAXLEN:-128}" \
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
        # Triton-ascend 尚未上机验收（torch-npu 位运算已有缺陷先例）；默认观察不阻断，
        # OSCAR_ASCEND_REQUIRE_TRITON=1 时才作为硬门禁
        if [ "${OSCAR_ASCEND_REQUIRE_TRITON:-0}" == "1" ]; then
            "$PYTHON" delivery/probe_oscar.py --mode triton \
                || fail "数值 probe(triton) FAIL（REQUIRE_TRITON=1）—— 拒绝 serve"
        else
            "$PYTHON" delivery/probe_oscar.py --mode triton || echo "  ⚠️ triton probe 未通过（观察模式，服务走 torch 参考路径）"
        fi
    else
        echo "  HAS_TRITON=False → 跳过 triton probe（服务将走 torch 参考路径）"
    fi
else
    echo "  OSCAR_SKIP_PROBES=1 → 跳过 probe（仅诊断用，禁止用于正式交付验收）"
fi

# ---------- 阶段6 serve（前台实时输出 + tee 落盘；后台观察者自动判定；不代发请求） ----------
step "启动 vllm serve（前台实时显示；日志固定 /tmp/oscar_ascend_logs/serve.log 覆盖写入）"
export OSCAR_ASCEND_LOG_DIR="$LOG_DIR"
# 后台观察者：等 /health → 自动跑激活判定（打印到同一终端，不打断 serve 前台）
(
    READY=0
    for i in $(seq 1 180); do
        sleep 5
        if python3 - <<'PYCHECK' 2>/dev/null
import urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:8989/health", timeout=2)
except Exception:
    raise SystemExit(1)
PYCHECK
        then READY=1; break; fi
        # serve 提前退出（端口进程消失且日志显示启动失败）→ 直接报
        if grep -q "EngineCore failed to start\|WorkerProc failed to start\|NPUModelRunner failed" "$SERVE_LOG" 2>/dev/null; then
            echo "🚨 [oscar-watch] serve 启动失败（见上方/日志 $SERVE_LOG）"; exit 1
        fi
    done
    if [ "$READY" -eq 1 ]; then
        echo ""
        echo "✅ [oscar-watch] serve 就绪（http://0.0.0.0:8989/health）—— 自动激活判定："
        bash delivery/check_oscar_active.sh "$SERVE_LOG"
    else
        echo "🚨 [oscar-watch] /health 900s 内未就绪（见 $SERVE_LOG）"
    fi
) &
WATCH_PID=$!
# 前台 serve：实时输出 + tee 记录（同一行既给终端也进 serve.log）
bash delivery/serve_oscar.sh "$@" 2>&1 | tee "$SERVE_LOG"
kill "$WATCH_PID" 2>/dev/null || true
