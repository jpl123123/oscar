#!/usr/bin/env bash
# delivery/check_oscar_active.sh — 判定 OSCAR 插件是否真的进入了当前 serve 实例
#
# 用法：
#   bash delivery/check_oscar_active.sh                     # 自动取最新 serve_*.log
#   bash delivery/check_oscar_active.sh <log 文件>
#
# 判定（与插件内置 ★ 自证日志配套，cb4fc15+）：
#   [1] plugin 注入 OK          （load_plugin 在 process0/engine/worker 均执行）
#   [2] ★ 类外科手术生效 计数    （必须 = FULL 层数，Qwen3.5 应为 16）
#   [3] ★ OSCAR 配置生效 计数    （= 16）
#   [4] ★ INT2 写路径首次执行 计数（≥1；采样后就地出现）
#   [5] ★ INT2 读路径(decode) 首次执行（≥1，decode 后出现）
# 任一核心项缺失 → VERDICT=NOT-ACTIVE，并给出修复指引（用一键脚本/ serve_oscar.sh 重启）。
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${1:-$(ls -t /tmp/oscar_ascend_logs/serve_*.log 2>/dev/null | head -1)}"
[ -n "${LOG}" ] && [ -f "${LOG}" ] || { echo "❌ 未找到 serve 日志（先跑 bash delivery/install_and_launch.sh）"; exit 2; }

echo "==> 检查日志: $LOG"
C_INJECT=$(grep -c "\[oscar-ascend\] plugin 注入 OK" "$LOG" || true)
C_SURG=$(grep -c "★ 类外科手术生效" "$LOG" || true)
C_CFG=$(grep -c "★ OSCAR 配置生效" "$LOG" || true)
C_WRITE=$(grep -c "★ INT2 写路径首次执行" "$LOG" || true)
C_READ=$(grep -c "★ INT2 读路径(decode) 首次执行" "$LOG" || true)
echo "  [1] plugin 注入 OK        : $C_INJECT 次"
echo "  [2] ★ 类外科手术生效      : $C_SURG 次（Qwen3.5 FULL 层应=16）"
echo "  [3] ★ OSCAR 配置生效      : $C_CFG 次"
echo "  [4] ★ INT2 写路径首次执行 : $C_WRITE 次（≥1 = 已写入缓存）"
echo "  [5] ★ INT2 读路径(decode) : $C_READ 次（≥1 = 解码走 INT2）"

OK=1
[ "$C_INJECT" -ge 1 ] || { OK=0; echo "  ❌ [1] 插件未加载：请用一键脚本或 serve_oscar.sh 重启（保证 VLLM_PLUGINS=ascend,oscar_ascend）"; }
[ "$C_SURG" -ge 1 ] || { OK=0; echo "  ❌ [2] 无类外科手术：插件被加载但没命中 hybrid full-attn 层（检查 OSCAR_ASCEND_ENABLE、模型 is_hybrid）"; }
[ "$C_CFG" -ge 1 ] || { OK=0; echo "  ❌ [3] 无配置生效日志"; }
[ "$C_WRITE" -ge 1 ] || { OK=0; echo "  ❌ [4] 尚未观测到 INT2 写入（须发生一次完整请求后才出现）"; }

if [ "$OK" -eq 1 ]; then
  echo "🎉 VERDICT: OSCAR ACTIVE（INT2 存储/读取已在 serve 实例内生效）"
else
  echo "🚨 VERDICT: NOT-ACTIVE —— 当前实例未运行 OSCAR；请："
  echo "   pkill -f '/softwarePlatform/c00879303' ; git pull && bash delivery/install_and_launch.sh"
fi
exit $((1 - OK))
