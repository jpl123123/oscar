#!/usr/bin/env bash
# delivery/check_oscar_active.sh — 判定 OSCAR 插件是否真实进入 serve 实例（一键自动化配套）
#
# 用法：
#   bash delivery/check_oscar_active.sh --preflight        # 启动前预检（env/插件/旋转pt；无需日志）
#   bash delivery/check_oscar_active.sh                    # 自动取最新 serve_*.log 判定
#   bash delivery/check_oscar_active.sh <log 文件>
#
# 判定（★ 自证日志，cb4fc15+）：
#   [1] plugin 注入 OK             —— 必须
#   [2] ★ 类外科手术生效 计数 ≥1    —— 必须（Qwen3.5 应 =16）
#   [3] ★ OSCAR 配置生效 计数 ≥1    —— 必须
#   [4] ★ INT2 写路径首次执行 ≥1    —— 信息项：首个请求（您跑 ais_bench）后出现
#   [5] ★ INT2 读路径(decode) ≥1    —— 信息项：首个 decode 请求后出现
# 核心项缺失 → VERDICT=NOT-ACTIVE（install_and_launch.sh 会自动停服并报错）。
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "${1:-}" = "--preflight" ]; then
  echo "==> 预检（serve 启动前）"
  OK=1
  [ "${VLLM_PLUGINS:-}" = "${VLLM_PLUGINS}" ] || true
  case "${VLLM_PLUGINS:-}" in
    *oscar_ascend*|*ascend,oscar_ascend*) echo "  ✅ VLLM_PLUGINS=$VLLM_PLUGINS" ;;
    *) echo "  ❌ VLLM_PLUGINS=${VLLM_PLUGINS:-<未设置>}（需含 ascend 与 oscar_ascend）"; OK=0 ;;
  esac
  [ "${OSCAR_ASCEND_ENABLE:-auto}" != "0" ] && echo "  ✅ OSCAR_ASCEND_ENABLE=${OSCAR_ASCEND_ENABLE:-auto}" \
    || { echo "  ❌ OSCAR_ASCEND_ENABLE=0 会禁用注入"; OK=0; }
  if python3 -c "import oscar_ascend, importlib.metadata as m; eps=[e for e in m.entry_points(group='vllm.general_plugins') if e.name=='oscar_ascend']; assert eps" 2>/dev/null; then
    echo "  ✅ oscar_ascend 可导入 + 入口点存在"
  else
    echo "  ❌ oscar_ascend 不可导入（先 pip install -e .）"; OK=0
  fi
  [ -f "$REPO_ROOT/oscar_rotations.pt" ] && echo "  ✅ oscar_rotations.pt 存在" \
    || { echo "  ⚠️ oscar_rotations.pt 不存在（有单位阵降级路径，属可配置项）"; }
  [ "$OK" -eq 1 ] && echo "🎉 PREFLIGHT PASS" || echo "🚨 PREFLIGHT FAIL"
  exit $((1 - OK))
fi

LOG="${1:-$(ls -t /tmp/oscar_ascend_logs/serve_*.log 2>/dev/null | head -1)}"
[ -n "${LOG}" ] && [ -f "${LOG}" ] || { echo "❌ 未找到 serve 日志（先跑 bash delivery/install_and_launch.sh）"; exit 2; }

echo "==> 检查日志: $LOG"
C_INJECT=$(grep -c "\[oscar-ascend\] plugin 注入 OK" "$LOG" || true)
C_SURG=$(grep -c "★ 类外科手术生效" "$LOG" || true)
C_CFG=$(grep -c "★ OSCAR 配置生效" "$LOG" || true)
C_WRITE=$(grep -c "★ INT2 写路径首次执行" "$LOG" || true)
C_READ=$(grep -c "★ INT2 读路径(decode) 首次执行" "$LOG" || true)
C_SKIP=$(grep -c "\[oscar-ascend\]\[SKIP\]" "$LOG" || true)
C_NOEAGER=$(grep -c "enforce_eager=False" "$LOG" || true)
echo "  [1] plugin 注入 OK        : $C_INJECT 次（要求 ≥1）"
echo "  [2] ★ 类外科手术生效      : $C_SURG 次（要求 ≥1；Qwen3.5 应为 16）"
echo "  [3] ★ OSCAR 配置生效      : $C_CFG 次（要求 ≥1）"
echo "  [4] ★ INT2 写路径首次执行 : $C_WRITE 次（信息项：首发请求后出现——您跑 ais_bench 后可见）"
echo "  [5] ★ INT2 读路径(decode) : $C_READ 次（信息项：首发 decode 后出现）
  [6] enforce_eager=False 行 : $C_NOEAGER 次（警告项：OSCAR 仅 eager 验证；>0 说明未真正生效，见 serve_oscar.sh）"
[ "$C_SKIP" -gt 0 ] && echo "  ⚠️ 存在 [SKIP] 行（$C_SKIP 条）——请查看拒绝原因" || true

OK=1
[ "$C_INJECT" -ge 1 ] || { OK=0; echo "  ❌ [1] 插件未加载（env/入口点问题）"; }
[ "$C_SURG" -ge 1 ] || { OK=0; echo "  ❌ [2] 无类外科手术（hybrid 判定失败或 enable=0）"; }
[ "$C_CFG" -ge 1 ] || { OK=0; echo "  ❌ [3] 无配置生效日志"; }

if [ "$OK" -eq 1 ]; then
  echo "🎉 VERDICT: OSCAR ACTIVE —— 插件已真实进入 serve 实例"
  echo "   （写/读路径 ★ 将在您跑首个请求（ais_bench）后自动出现；届时可重跑本命令复核）"
else
  echo "🚨 VERDICT: NOT-ACTIVE —— 当前实例未运行 OSCAR；请检查上方 ❌ 项与 [SKIP] 行"
fi
exit $((1 - OK))
