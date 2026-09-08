#!/usr/bin/env bash
# delivery/check_oscar_active.sh — 判定 OSCAR 插件是否真实进入 serve 实例（一键自动化配套）
#
# 用法：
#   bash delivery/check_oscar_active.sh --preflight        # 启动前预检（env/插件/旋转pt；无需日志）
#   bash delivery/check_oscar_active.sh                    # 自动取最新 serve_*.log 判定
#   bash delivery/check_oscar_active.sh <log 文件>
#
# 日志计数仅作诊断；最终结果由 check_ready.py 核验每rank的READY manifest、
# 完整层集合和源码指纹。此检查不替代数值/吞吐验收，也不负责停止服务。
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "${1:-}" = "--preflight" ]; then
  echo "==> 预检（serve 启动前）"
  OK=1
  case ",${VLLM_PLUGINS:-}," in
    *,oscar_ascend,*) echo "  ✅ VLLM_PLUGINS=$VLLM_PLUGINS" ;;
    *) echo "  ❌ VLLM_PLUGINS=${VLLM_PLUGINS:-<未设置>}（需含 ascend 与 oscar_ascend）"; OK=0 ;;
  esac
  case ",${VLLM_PLUGINS:-}," in
    *,ascend,*) : ;;
    *) echo "  ❌ 缺少 ascend platform 插件"; OK=0 ;;
  esac
  [ "${OSCAR_ASCEND_ENABLE:-auto}" != "0" ] && echo "  ✅ OSCAR_ASCEND_ENABLE=${OSCAR_ASCEND_ENABLE:-auto}" \
    || { echo "  ❌ OSCAR_ASCEND_ENABLE=0 会禁用注入"; OK=0; }
  if "${PYTHON:-python3}" -c "import oscar_ascend, importlib.metadata as m; eps=[e for e in m.entry_points(group='vllm.general_plugins') if e.name=='oscar_ascend']; assert eps" 2>/dev/null; then
    echo "  ✅ oscar_ascend 可导入 + 入口点存在"
  else
    echo "  ❌ oscar_ascend 不可导入（先 pip install -e .）"; OK=0
  fi
  [ -f "$REPO_ROOT/oscar_rotations.pt" ] && echo "  ✅ oscar_rotations.pt 存在" \
    || { echo "  ⚠️ oscar_rotations.pt 不存在（后续阶段需生成或提供；serve要求文件存在）"; }
  [ "$OK" -eq 1 ] && echo "🎉 PREFLIGHT PASS" || echo "🚨 PREFLIGHT FAIL"
  exit $((1 - OK))
fi

LOG="${1:-$(ls -t /tmp/oscar_ascend_logs/serve.log 2>/dev/null | head -1)}"
[ -f "$LOG" ] || LOG="$(ls -t /tmp/oscar_ascend_logs/serve_*.log 2>/dev/null | head -1)"
[ -n "${LOG}" ] && [ -f "${LOG}" ] || { echo "❌ 未找到 serve 日志（先跑 bash delivery/install_and_launch.sh）"; exit 2; }

echo "==> 检查日志: $LOG"
C_INJECT=$(grep -c "\[oscar-ascend\] plugin 注入 OK" "$LOG" || true)
C_SURG=$(grep -c "★ 类外科手术生效" "$LOG" || true)
C_CFG=$(grep -c "★ OSCAR 配置生效" "$LOG" || true)
C_WRITE=$(grep -c "★ INT2 写路径首次执行" "$LOG" || true)
C_READ=$(grep -c "★ INT2 读路径(decode) 首次执行" "$LOG" || true)
C_SKIP=$(grep -c "\[oscar-ascend\]\[SKIP\]" "$LOG" || true)
C_NOEAGER=$(grep -c "enforce_eager=False" "$LOG" || true)
C_TRITON=$(grep -c "★ OSCAR 配置生效.*triton=启用" "$LOG" || true)
C_GEO=$(grep -c "★ 几何对账: K 槽 256B" "$LOG" || true)
C_MTPSH=$(grep -c "★ MTP 影子池" "$LOG" || true)
echo "  [1] plugin 注入 OK        : $C_INJECT 次（要求 ≥1）"
echo "  [2] ★ 类外科手术生效      : $C_SURG 次（要求 ≥1；Qwen3.5 应为 16）"
echo "  [3] ★ OSCAR 配置生效      : $C_CFG 次（要求 ≥1）"
echo "  [4] ★ INT2 写路径首次执行 : $C_WRITE 次（信息项：首发请求后出现——您跑 ais_bench 后可见）"
echo "  [5] ★ INT2 读路径(decode) : $C_READ 次（信息项：首发 decode 后出现）
  [6] enforce_eager=False 行 : $C_NOEAGER 次（警告项：OSCAR 仅 eager 验证；>0 说明未真正生效，见 serve_oscar.sh）
  [7] triton 路径            : $C_TRITON 条 'triton=启用'（=0 → serve 全 torch 参考路径——检查阶段5 门禁/降级，见 install_and_launch.sh）
  [8] packed×2 几何/MTP 影子池: 几何对账 $C_GEO 条(256B 槽) / 影子池 $C_MTPSH 条（OSCAR_ASCEND_PACKED=1 时应 >0；=0 → legacy 几何或未启用，见 DESIGN-E）"
[ "$C_SKIP" -gt 0 ] && echo "  ⚠️ 存在 [SKIP] 行（$C_SKIP 条）——请查看拒绝原因" || true

"${PYTHON:-python3}" "$REPO_ROOT/delivery/check_ready.py" "$LOG"
exit $?
