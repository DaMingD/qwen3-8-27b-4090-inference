#!/usr/bin/env bash
# 建立独立的推理环境。不碰已有 conda 环境，不用 sudo，不下载模型。
#
# 服务器现有 torch 是 2.13.0+cu132。vLLM 0.32.0 要求 torch==2.13.0，
# 但 PyPI 上的轮子对应 cu129（docs/getting_started/installation/gpu.cuda.inc.md:32），
# C++ 扩展和 torch 的 CUDA 版本混用有风险，所以单独建环境。
#
# 用法：WORK=$HOME/uclode bash 工具/setup_env.sh
# 可选：PYTHON=python3.12  TORCH_BACKEND=cu129  PIP_INDEX_URL=<镜像>  TORCH_INDEX=<镜像>
#       GPU_CHECK=0 不碰显卡，GPU 算子验收之后用 env_check.py 单独做
#       REQUIREMENTS_LOCK 指定依赖清单，默认使用仓库交付/环境/requirements.lock
set -euo pipefail

WORK="${WORK:-$HOME/uclode}"
VENV="$WORK/venv-vllm"
PYTHON="${PYTHON:-python3.12}"
VLLM_VERSION="0.32.0"
TORCH_BACKEND="${TORCH_BACKEND:-cu129}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/$TORCH_BACKEND}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_LOCK="${REQUIREMENTS_LOCK:-$SCRIPT_DIR/../交付/环境/requirements.lock}"
if [[ ! -s "$REQUIREMENTS_LOCK" ]]; then
  echo "缺少依赖锁定文件：${REQUIREMENTS_LOCK}；未执行安装。" >&2
  exit 2
fi
# 保存输入快照，允许输入清单与输出 requirements.lock 是同一个文件。
LOCK_SNAPSHOT=$(mktemp)
trap 'rm -f "$LOCK_SNAPSHOT"' EXIT
cp "$REQUIREMENTS_LOCK" "$LOCK_SNAPSHOT"

mkdir -p "$WORK"
free_gb=$(df -BG --output=avail "$WORK" | tail -n 1 | tr -dc '0-9')
if (( free_gb < 20 )); then
  echo "$WORK 所在磁盘只剩 ${free_gb}GB，torch 和 CUDA 依赖约需 10GB，日志和评测结果也要空间。换一个 WORK 目录。" >&2
  exit 2
fi

if [[ -e "$VENV" ]]; then
  echo "环境已存在：$VENV。要重建请先自行确认后删除它。" >&2
else
  "$PYTHON" -m venv "$VENV"
fi
export PATH="$VENV/bin:$PATH"

"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install -r "$LOCK_SNAPSHOT" --extra-index-url "$TORCH_INDEX"
"$VENV/bin/python" -m pip check

no_gpu=""
[[ "${GPU_CHECK:-1}" == 0 ]] && no_gpu="--no-gpu"
"$VENV/bin/python" -P "$(dirname -- "${BASH_SOURCE[0]}")/env_check.py" --work "$WORK" $no_gpu

"$VENV/bin/python" -m pip freeze > "$WORK/requirements.lock"
if ! diff -u "$LOCK_SNAPSHOT" "$WORK/requirements.lock" > "$WORK/requirements_install.diff"; then
  echo "安装后的依赖与锁定文件不一致，见 $WORK/requirements_install.diff" >&2
  exit 1
fi
echo "锁定文件：$WORK/requirements.lock"
