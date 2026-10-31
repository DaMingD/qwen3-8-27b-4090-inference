#!/usr/bin/env bash
# 复验自查，标准见 交付/6_复验说明.md。这是我们的自查，不是赛方标准。
#
#   bash 工具/repro_check.sh build
#       新目录从零装环境、重新转换权重并比对哈希。隐藏所有 GPU，不碰显卡，放在不计时的阶段跑
#   FINAL=cutlass REF=4=76.31,8=116.70 bash 工具/repro_check.sh verify
#       GPU 算子验收，再用 build 的目录启动最终配置并验收，约 15 分钟。独占 GPU，结束时服务已停止
#
# build 的 passed 不含 GPU 算子，两级都通过才算完成。依赖有差异时 passed 也是 false，差异写在 requirements.diff。
#
# 配置：REPRO=$HOME/repro  ORIGIN=$HOME/uclode  BF16  FP8  DRY_RUN=1 只打印命令
# 结果：$REPRO/repro_build.json、$REPRO/repro_verify.json，日志在同目录。
# 评测工具环境由赛方 run.sh 安装并共用，这里不重做，材料里要写明。
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")" || exit 2
REPRO="${REPRO:-$HOME/repro}"
ORIGIN="${ORIGIN:-$HOME/uclode}"
BF16="${BF16:-/model/ModelScope/Qwen/Qwen3.8-27B}"
FP8="${FP8:-/model/ModelScope/Qwen/Qwen3.8-27B-FP8}"
CHANNEL="models/Qwen3.8-27B-FP8-channel"
mkdir -p "$REPRO"
LOG="$REPRO/repro_${1:-none}.log"

run() {
  echo "+ $*" | tee -a "$LOG"
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    return 0
  fi
  "$@" >>"$LOG" 2>&1
}

# 不计时的阶段也让路给正在跑的评测
low() { nice -n 15 "$@"; }

build() {
  local setup convert hashed=1 mismatched=-1 lock_diff=-1
  if [[ ! -s "$ORIGIN/requirements.lock" ]]; then
    echo "缺少原环境依赖清单：$ORIGIN/requirements.lock；未开始构建。" >&2
    return 2
  fi
  export CUDA_VISIBLE_DEVICES=""  # 显卡可能正在跑评测，这一级全程看不到 GPU
  run env "WORK=$REPRO" GPU_CHECK=0 "REQUIREMENTS_LOCK=$ORIGIN/requirements.lock" nice -n 15 bash 工具/setup_env.sh
  setup=$?
  (( setup == 0 )) || return "$setup"
  run nice -n 15 "$REPRO/venv-vllm/bin/python" 工具/convert_fp8.py --bf16 "$BF16" --fp8 "$FP8" --out "$REPRO/$CHANNEL"
  convert=$?
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    if [[ -f "$ORIGIN/requirements.lock" && -f "$REPRO/requirements.lock" ]]; then
      diff "$ORIGIN/requirements.lock" "$REPRO/requirements.lock" > "$REPRO/requirements.diff"
      lock_diff=$(grep -c '^[<>]' "$REPRO/requirements.diff")
    fi
    (cd "$ORIGIN/$CHANNEL" && low sha256sum ./*.safetensors) > "$REPRO/weights_origin.sha256" && \
    (cd "$REPRO/$CHANNEL" && low sha256sum ./*.safetensors) > "$REPRO/weights_repro.sha256" && hashed=0
    # 两边都要数：复现目录少分片时，差异只出现在 < 一侧
    mismatched=$(diff "$REPRO/weights_origin.sha256" "$REPRO/weights_repro.sha256" | grep -c '^[<>]')
    (cd "$BF16" && sha256sum config.json model.safetensors.index.json; cd "$FP8" && sha256sum config.json model.safetensors.index.json) \
      > "$REPRO/source_models.sha256" 2>>"$LOG"
  fi
  python3 - "$REPRO/repro_build.json" "$setup" "$convert" "$hashed" "$mismatched" "$lock_diff" <<'PY'
import json, sys, time
path, setup, convert, hashed, mismatched, lock_diff = sys.argv[1], *map(int, sys.argv[2:])
checks = {"环境安装和版本检查": setup == 0, "权重转换": convert == 0, "权重哈希计算": hashed == 0,
          "权重分片哈希一致": mismatched == 0, "依赖清单一致": lock_diff == 0}
result = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), setup_exit=setup, convert_exit=convert, hash_exit=hashed,
              weight_files_different=mismatched, requirements_lines_different=lock_diff, gpu_operator_check="未做，由 verify 完成",
              failed=[name for name, ok in checks.items() if not ok], passed=all(checks.values()))
print(json.dumps(result, ensure_ascii=False, indent=2))
open(path, "w", encoding="utf-8").write(json.dumps(result, ensure_ascii=False, indent=2))
PY
}

verify() {
  local final="${FINAL:?要设置 FINAL=cutlass 或 marlin}" model profile kernel backend="" scan_dir smoke=1 gpu_check
  if [[ "$final" == cutlass ]]; then
    model="$REPRO/$CHANNEL"; profile="s=2048,seqs=8,backend=cutlass"; kernel=CutlassFP8ScaledMMLinearKernel; backend=cutlass
  else
    model="$FP8"; profile="s=2048,seqs=8"; kernel=MarlinFP8ScaledMMLinearKernel
  fi
  export PATH="$REPRO/venv-vllm/bin:$ORIGIN/venv-eval-bootstrap/bin:$PATH"
  unset VLLM_LOGGING_LEVEL
  scan_dir="$REPRO/scans/verify_$(date +%H%M%S)"
  run env "WORK=$ORIGIN" bash 工具/serve.sh stop
  run env "WORK=$REPRO" bash 工具/serve.sh stop
  run "$REPRO/venv-vllm/bin/python" -P 工具/env_check.py --work "$REPRO"
  gpu_check=$?
  run python3 工具/scan.py --work "$REPRO" --model-dir "$model" --tag repro --concurrency 4 --concurrency 8 \
    --requests 24 --log-iterations 0 --profile "$profile" --output-dir "$scan_dir"
  local kernel_found=0
  [[ "${DRY_RUN:-0}" != 1 ]] && kernel_found=$(grep -c "$kernel" "$REPRO/runs/current/server.log")
  if run env "WORK=$REPRO" NAME=repro_smoke "MODEL_DIR=$model" MAX_SEQS=8 "LINEAR_BACKEND=$backend" bash 工具/serve.sh start; then
    run python3 工具/ifbench_tool.py run --limit 4 --max-tokens 2048 --out "$scan_dir/smoke"
    smoke=$?
  fi
  run env "WORK=$REPRO" bash 工具/serve.sh stop
  python3 - "$REPRO/repro_verify.json" "$final" "$kernel" "$kernel_found" "$smoke" "$scan_dir/summary.json" "${REF:-}" "$gpu_check" <<'PY'
import json, os, sys, time
path, final, kernel, found, smoke, summary, reference, gpu_check = sys.argv[1:]
reference = dict(pair.split("=") for pair in reference.split(",") if pair)
rows = json.load(open(summary)) if os.path.exists(summary) else []
perf = []
for row in rows:
    expected = float(reference.get(str(row["concurrency"]), "nan"))
    change = row["tok_s"] / expected - 1 if row.get("tok_s") else None
    perf.append(dict(concurrency=row["concurrency"], status=row["status"], tok_s=row.get("tok_s"), reference=expected,
                     change=change, within_5_percent=change is not None and abs(change) <= 0.05))
checks = {"GPU 算子验收": int(gpu_check) == 0, "实际 kernel": int(found) > 0, "赛方流程冒烟": int(smoke) == 0,
          "吞吐相差不超过 5%": len(perf) == 2 and all(r["within_5_percent"] and r["status"] == "完整" for r in perf)}
result = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), final=final, gpu_check_exit=int(gpu_check), expected_kernel=kernel,
              kernel_lines=int(found), smoke_exit=int(smoke), perf=perf, summary=summary,
              failed=[name for name, ok in checks.items() if not ok], passed=all(checks.values()))
print(json.dumps(result, ensure_ascii=False, indent=2))
open(path, "w", encoding="utf-8").write(json.dumps(result, ensure_ascii=False, indent=2))
PY
}

case "${1:-}" in
  build) build ;;
  verify) verify ;;
  *) echo "用法：bash 工具/repro_check.sh build|verify" >&2; exit 2 ;;
esac
