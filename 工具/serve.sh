#!/usr/bin/env bash
# 启动或停止一套 vLLM 服务，并把命令、版本、逐秒 GPU 状态和服务日志存到独立目录。
#
# 用法：
#   NAME=baseline bash 工具/serve.sh start
#   bash 工具/serve.sh stop
#   bash 工具/serve.sh status
#
# 配置（环境变量）：
#   WORK=$HOME/uclode          工作目录，含 venv-vllm/ 和 runs/
#   MODEL_DIR=/model/ModelScope/Qwen/Qwen3.8-27B-FP8
#   SERVED_NAME=qwen3.8-27b-fp8
#   HOST=127.0.0.1  PORT=8088  默认只监听本机，要给外部访问时显式改 HOST
#   MAX_SEQS=4                 官方值；放开并发时设 8
#   BUDGET=                    --max-num-batched-tokens；空表示用默认值
#   MAX_MODEL_LEN=40960        官方 32768 过不了脚本的精度阶段检查
#   UTIL=0.90
#   MTP_DEPTH=0                大于 0 时开启 MTP
#   ADAPTIVE=0                 1 表示开启上游自适应验证（需要 MTP_DEPTH>0）
#   LOG_ITER=0                 1 表示记录每一步的 token 数和耗时
#   LOG_REQ=0                  1 表示记录服务端解析后的采样参数
#   NO_PREFIX=0                1 表示关闭前缀缓存
#   KV_DTYPE=fp8
#   LINEAR_BACKEND=            --linear-backend；空表示自动选择，如 triton、cutlass
#   EXTRA=                     其他参数，原样追加
#   START_TIMEOUT=1200         等待服务就绪的秒数
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$HOME/uclode}"
VENV="$WORK/venv-vllm"
export PATH="$VENV/bin:$PATH"
RUNS="$WORK/runs"
CURRENT="$RUNS/current"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8088}"
HEALTH_HOST="$HOST"
[[ "$HOST" == "0.0.0.0" ]] && HEALTH_HOST="127.0.0.1"
HEALTH="http://$HEALTH_HOST:$PORT/health"

is_up() { curl --silent --fail --max-time 3 "$HEALTH" >/dev/null 2>&1; }

kill_tree() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] || return 0
  local pid
  pid="$(cat "$pidfile")"
  # 进程组 ID 等于 setsid 启动的进程 ID
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-$pid" 2>/dev/null || true
  rm -f "$pidfile"
}

start() {
  if is_up; then
    echo "端口 $PORT 上已有服务，先停止它：bash 工具/serve.sh stop" >&2
    exit 2
  fi
  local model="${MODEL_DIR:-/model/ModelScope/Qwen/Qwen3.8-27B-FP8}"
  [[ -f "$model/config.json" ]] || { echo "找不到模型：$model" >&2; exit 2; }
  [[ -x "$VENV/bin/vllm" ]] || { echo "找不到 $VENV/bin/vllm，先运行 setup_env.sh" >&2; exit 2; }

  local name="${NAME:-run}"
  local dir
  dir="$RUNS/$(date +%Y%m%d_%H%M%S)_$name"
  mkdir -p "$dir"
  ln -sfn "$dir" "$CURRENT"

  local args=(serve "$model" --host "$HOST" --port "$PORT"
    --tensor-parallel-size 1 --dtype auto
    --gpu-memory-utilization "${UTIL:-0.90}"
    --max-model-len "${MAX_MODEL_LEN:-40960}"
    --max-num-seqs "${MAX_SEQS:-4}"
    --kv-cache-dtype "${KV_DTYPE:-fp8}"
    --reasoning-parser qwen3
    --served-model-name "${SERVED_NAME:-qwen3.8-27b-fp8}")
  [[ -n "${BUDGET:-}" ]] && args+=(--max-num-batched-tokens "$BUDGET")
  [[ "${LOG_ITER:-0}" == 1 ]] && args+=(--enable-logging-iteration-details)
  [[ "${LOG_REQ:-0}" == 1 ]] && args+=(--enable-log-requests)
  [[ "${NO_PREFIX:-0}" == 1 ]] && args+=(--no-enable-prefix-caching)
  [[ -n "${LINEAR_BACKEND:-}" ]] && args+=(--linear-backend "$LINEAR_BACKEND")
  if [[ "${MTP_DEPTH:-0}" -gt 0 ]]; then
    local spec="{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_DEPTH"
    [[ "${ADAPTIVE:-0}" == 1 ]] && spec+=",\"enable_adaptive_verification\":true"
    args+=(--speculative-config "$spec}")
  fi
  # shellcheck disable=SC2206
  [[ -n "${EXTRA:-}" ]] && args+=(${EXTRA})

  {
    echo "时间：$(date -Is)"
    echo "命令：$VENV/bin/vllm ${args[*]}"
    echo "模型目录：$model"
    echo "日志级别：${VLLM_LOGGING_LEVEL:-默认 INFO}"
    "$VENV/bin/python" -c "import vllm,torch,transformers;print('vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__)"
    nvidia-smi --query-gpu=name,driver_version,memory.total,pstate --format=csv,noheader
    (cd "$model" && sha256sum config.json generation_config.json 2>/dev/null) || true
  } > "$dir/meta.txt"
  printf '%s\n' "${args[@]}" > "$dir/cmd.args"

  # 逐秒记录频率、温度、功耗和节流原因，用来标记降频的运行。节流字段的名字因驱动版本而异。
  local fields=timestamp,clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw,utilization.gpu,utilization.memory,memory.used
  local reason
  for reason in clocks_event_reasons.active clocks_throttle_reasons.active; do
    if nvidia-smi --query-gpu="$reason" --format=csv,noheader >/dev/null 2>&1; then
      fields="$fields,$reason"
      break
    fi
  done
  setsid nohup nvidia-smi --query-gpu="$fields" --format=csv -l 1 > "$dir/gpu.csv" 2>"$dir/gpu.err" &
  echo $! > "$RUNS/gpu.pid"

  local t0
  t0=$(date +%s)
  VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1 \
    setsid nohup "$VENV/bin/vllm" "${args[@]}" > "$dir/server.log" 2>&1 &
  echo $! > "$RUNS/server.pid"

  local timeout="${START_TIMEOUT:-1200}"
  until is_up; do
    if ! kill -0 "$(cat "$RUNS/server.pid")" 2>/dev/null; then
      echo "服务启动失败，日志最后 40 行：" >&2
      tail -n 40 "$dir/server.log" >&2
      kill_tree "$RUNS/gpu.pid"
      exit 1
    fi
    if (( $(date +%s) - t0 > timeout )); then
      echo "等待 ${timeout}s 仍未就绪，日志：$dir/server.log" >&2
      exit 1
    fi
    sleep 3
  done
  echo "启动耗时（秒）：$(( $(date +%s) - t0 ))" | tee -a "$dir/meta.txt"
  python3 "$SCRIPT_DIR/log_evidence.py" startup "$dir/server.log" | tee "$dir/startup_evidence.txt"
  echo "运行目录：$dir"
}

stop() {
  kill_tree "$RUNS/server.pid"
  kill_tree "$RUNS/gpu.pid"
  echo "已停止。"
}

status() {
  if is_up; then echo "服务在线：$HEALTH，运行目录：$(readlink "$CURRENT" 2>/dev/null || echo 未知)"
  else echo "服务不在线"; fi
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "用法：bash 工具/serve.sh start|stop|status" >&2; exit 2 ;;
esac
