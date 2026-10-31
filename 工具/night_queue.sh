#!/usr/bin/env bash
# 无人值守队列：逐项重启服务并运行赛方脚本。一项失败只记录，继续下一项。
#
# 每一项写成一个字符串："名字 模式 环境变量..."，环境变量会传给 serve.sh。
# 模式：perf（全量性能）、accuracy（IFBench，默认 batch 2）、accuracy_b8（IFBench，batch 8，只用于噪声底）
#
# 例子（放进 tmux 或 nohup，先在交互终端跑过一次 perf 和 accuracy，确保评测工具环境已安装）：
#   bash 工具/night_queue.sh \
#     "fallback perf MAX_SEQS=8 BUDGET=512" \
#     "plan_a perf MAX_SEQS=8 BUDGET=512 LINEAR_BACKEND=triton" \
#     "plan_a accuracy MAX_SEQS=8 BUDGET=512 LINEAR_BACKEND=triton"
#
# 结果目录：$WORK/eval/<名字>_<模式>，进度：$WORK/eval/queue_<时间>.log。
# DRY_RUN=1 只打印命令。IFB_TIMEOUT 是 IFBench 的客户端超时（秒），默认 3600，见 accuracy_run.py。
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
WORK="${WORK:-$HOME/uclode}"
DEFAULT_MODEL="${MODEL_DIR:-/model/ModelScope/Qwen/Qwen3.8-27B-FP8}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-27b-fp8}"
PORT="${PORT:-8088}"
mkdir -p "$WORK/eval"
LOG="$WORK/eval/queue_$(date +%Y%m%d_%H%M%S).log"
# 正式计时不能用诊断日志
unset VLLM_LOGGING_LEVEL

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run() {
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '  %q' "$@" | tee -a "$LOG"
    echo | tee -a "$LOG"
    return 0
  fi
  "$@" >>"$LOG" 2>&1
}

(( $# > 0 )) || { echo "用法见脚本开头。" >&2; exit 2; }

failed=0
for item in "$@"; do
  read -r -a parts <<<"$item"
  name="${parts[0]}"
  mode="${parts[1]}"
  settings=("${parts[@]:2}")
  model="$DEFAULT_MODEL"
  for pair in "${settings[@]}"; do
    [[ "$pair" == MODEL_DIR=* ]] && model="${pair#MODEL_DIR=}"
  done
  case "$mode" in
    perf) args=(--mode perf --tokenizer "$model") ;;
    accuracy) args=(--mode accuracy) ; settings+=(LOG_REQ=1) ;;
    accuracy_b8) args=(--mode accuracy --eval-batch-size 8) ; settings+=(LOG_REQ=1) ;;
    *) say "不认识的模式：$mode，跳过 $item"; failed=$((failed + 1)); continue ;;
  esac
  out="$WORK/eval/${name}_${mode}"
  say "开始：$item"
  run bash "$SCRIPT_DIR/serve.sh" stop
  if ! run env "WORK=$WORK" "NAME=${name}_${mode}" "${settings[@]}" bash "$SCRIPT_DIR/serve.sh" start; then
    say "服务启动失败：$item"
    failed=$((failed + 1))
    continue
  fi
  if [[ "$mode" == perf ]]; then
    evaluate=(bash "$ROOT/参赛资料/eval/run.sh")
  else
    evaluate=(python3 "$SCRIPT_DIR/accuracy_run.py" --timeout "${IFB_TIMEOUT:-3600}")
  fi
  if run "${evaluate[@]}" "${args[@]}" --base-url "http://127.0.0.1:$PORT/v1" \
      --model "$SERVED_NAME" --output-dir "$out"; then
    say "完成：$item，结果在 $out"
  else
    say "评测报错：$item，结果在 $out，详情看本日志"
    failed=$((failed + 1))
  fi
done
say "队列结束，失败 $failed 项。服务保持最后一项的配置。"
exit "$failed"
