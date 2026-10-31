#!/usr/bin/env bash
# 夜间队列之后的 MTP 实验：探针 A、B，深度 3 净收益，死循环复现。只产出分析，不进最终配置。
#
#   bash 工具/mtp_tail.sh
#
# 到 DEADLINE（默认北京时间 11-01 06:45）停止并释放 GPU。一步失败只记录，继续下一步。
# 结果：$WORK/mtp、$WORK/scans、$WORK/eval/loop_mtp3，进度：$WORK/mtp/tail_<时间>.log。DRY_RUN=1 只打印命令。
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")" || exit 2
WORK="${WORK:-$HOME/uclode}"
OUT="$WORK/mtp"
MODEL="${MODEL_DIR:-/model/ModelScope/Qwen/Qwen3.8-27B-FP8}"
DEADLINE="${DEADLINE:-2026-11-01T06:45:00+08:00}"
# 带时区算，服务器时区是 UTC
END=$(python3 -c "import datetime,sys;print(int(datetime.datetime.fromisoformat(sys.argv[1]).timestamp()))" "$DEADLINE")
export PATH="$WORK/venv-vllm/bin:$WORK/venv-eval-bootstrap/bin:$PATH"
unset VLLM_LOGGING_LEVEL
mkdir -p "$OUT"
LOG="$OUT/tail_$(date +%Y%m%d_%H%M%S).log"
SERVE="bash 工具/serve.sh"
PROBE="python3 工具/probe_mtp.py"

say() { echo "[$(date -u '+%m-%d %H:%M:%S') UTC] $*" | tee -a "$LOG"; }

step() {
  local left=$(( END - $(date +%s) ))
  if (( left <= 0 )); then
    say "到点，未运行：$1"
    return 1
  fi
  say "开始：$1（剩余 ${left} 秒）"
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "  $2" | tee -a "$LOG"
    return 0
  fi
  timeout "$left" bash -c "$2" >>"$LOG" 2>&1
  say "结束：$1，退出码 $?"
}

step "探针 A 基线" "$SERVE stop; NAME=base_probe $SERVE start && \
$PROBE sample --model-dir $MODEL --n 1000 --positions 7 --out $OUT/a7_base1.jsonl && \
$PROBE sample --model-dir $MODEL --n 1000 --positions 7 --out $OUT/a7_base2.jsonl && \
$PROBE entropy $OUT/a7_base1.jsonl --require && $PROBE entropy $OUT/a7_base2.jsonl --require"
step "探针 A 深度 3" "$SERVE stop; NAME=mtp3 MTP_DEPTH=3 $SERVE start && \
$PROBE sample --model-dir $MODEL --n 1000 --positions 7 --out $OUT/a7_mtp3.jsonl && \
$PROBE compare $OUT/a7_base1.jsonl $OUT/a7_base2.jsonl $OUT/a7_mtp3.jsonl --positions 7"
step "探针 B 深度 3" "$PROBE greedy --model-dir $MODEL --out $OUT/b_mtp3.jsonl && \
$PROBE diverge $OUT/b_base1.jsonl $OUT/b_base2.jsonl && $PROBE diverge $OUT/b_base1.jsonl $OUT/b_mtp3.jsonl"
step "深度 3 净收益" "$SERVE stop; python3 工具/scan.py --tag mtp --concurrency 1 --concurrency 8 --requests 24 --log-iterations 0 --profile s=2048,seqs=8,mtp=3"
step "死循环复现" "$SERVE stop; NAME=loop_mtp3 MTP_DEPTH=3 MAX_SEQS=8 $SERVE start && \
python3 工具/ifbench_tool.py run --limit 100 --max-tokens 8192 --eval-batch-size 8 --out $WORK/eval/loop_mtp3"

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  # timeout 只杀到它的进程组，评测子进程在自己的会话里
  pkill -f "runner.py _accuracy" ; pkill -f "aiperf profile"
  $SERVE stop >>"$LOG" 2>&1
fi
say "MTP 实验结束，GPU 已释放。"
