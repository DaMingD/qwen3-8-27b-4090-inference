# Qwen3.8-27B 单卡 RTX 4090 推理优化

> 私有初版，非正式提交。下方生成内容为已有结果快照，仍含 24 请求配对值及暂定退路；夜间 100 请求结果、质量判定、复验结果和最终配置将在验收后统一更新。PPT、视频及原始结果 Release 尚未上传。

UCloud AI 黑客松 Track A。不换模型、不改 vLLM 源码，用可证伪的成本模型定位瓶颈，逐项检验 kernel、调度和投机解码，并按预设门槛验收质量。

## 结果

**质量判定：未完成。** CUTLASS 通过了性能条件（并发 8 为 116.7 tok/s，24 个请求），但 IFBench 没有得到完整结果，所以不下质量结论，也不采用。基线为 IFBench 79/100、小测试集 29/30。

**最终配置：** Marlin 官方 FP8 + max-num-seqs=8 + 预算 2048。并发 8 吞吐 92.2 tok/s（官方配置 67.4，+36.8%），并发 4 为 67.3（+0.0%），并发 1 为 26.8（+0.0%）。这个配置只调整调度，不改权重和量化格式，质量依据是官方配置下的基线结果；批次组成变化对数值的影响没有单独完整验证。

噪声底未完成。

MTP 分析未完成，不下结论，最终配置不开 MTP。

**复验自查：** 一级（从零安装、重新转换权重并比对权重哈希和依赖清单，不含 GPU 算子）：未完成；二级（GPU 算子验收、上机启动、kernel、吞吐相差不超过 5%、赛方流程冒烟）：未完成。这是我们的自查，不是赛方标准；评测工具环境没有重新安装。


完整数字见 [结果表](交付/结果表.md)，所有材料的数字都由 `工具/build_results.py` 从原始结果生成。

## 快速复现

硬件：1 张 RTX 4090（49140 MiB，SM89）。软件：Python 3.12、vLLM 0.32.0（commit `5df63a720`）、Torch 2.13.0+cu129、Transformers 5.19.0。完整步骤、通过标准和与赛方流程的差异见 [复验说明](交付/6_复验说明.md)。

赛方评测脚本不在本仓库内，把它放到 `参赛资料/eval/`（含 `run.sh`、`runner.py`）。模型路径可用环境变量修改，默认是 `/model/ModelScope/Qwen/`。

```bash
WORK=$HOME/uclode bash 工具/setup_env.sh
```

最终配置是 CUTLASS 时，先转换权重（只用 CPU，约 5 分钟，约 29 GiB）：

```bash
$HOME/uclode/venv-vllm/bin/python 工具/convert_fp8.py --bf16 /model/ModelScope/Qwen/Qwen3.8-27B --fp8 /model/ModelScope/Qwen/Qwen3.8-27B-FP8 --out $HOME/uclode/models/Qwen3.8-27B-FP8-channel
```

启动命令按最终配置二选一，本次提交使用：Marlin 官方 FP8 + max-num-seqs=8 + 预算 2048。

```bash
NAME=final MAX_SEQS=8 LINEAR_BACKEND=cutlass MODEL_DIR=$HOME/uclode/models/Qwen3.8-27B-FP8-channel bash 工具/serve.sh start
```

```bash
NAME=final MAX_SEQS=8 bash 工具/serve.sh start
```

性能测试走赛方脚本，精度测试走只加长客户端超时的 `accuracy_run.py`（原因见复验说明）：

```bash
bash 参赛资料/eval/run.sh --mode perf --tokenizer /model/ModelScope/Qwen/Qwen3.8-27B-FP8 --base-url http://127.0.0.1:8088/v1 --model qwen3.8-27b-fp8 --output-dir $HOME/uclode/eval/perf
```

```bash
python3 工具/accuracy_run.py --timeout 3600 --mode accuracy --base-url http://127.0.0.1:8088/v1 --model qwen3.8-27b-fp8 --output-dir $HOME/uclode/eval/accuracy
```

## 目录

| 赛方要求 | 位置 |
|---|---|
| 模型选型说明 | [交付/1_模型选型说明.md](交付/1_模型选型说明.md) |
| 部署与配置说明 | [交付/2_部署与配置说明.md](交付/2_部署与配置说明.md) |
| Benchmark 测试报告 | [交付/3_Benchmark测试报告.md](交付/3_Benchmark测试报告.md)、[结果表](交付/结果表.md) |
| 优化方案说明 | [交付/4_优化方案说明.md](交付/4_优化方案说明.md) |
| 性能优化分析 | [交付/5_性能优化分析.md](交付/5_性能优化分析.md) |
| 完整复验材料包 | [交付/6_复验说明.md](交付/6_复验说明.md)、`工具/`、[依赖清单](交付/环境/requirements.lock) |
| 答辩 PPT | 随提交邮件发送 |
| 演示 | 现场真实运行，流程见 [Demo 脚本](交付/Demo脚本.md)；备用录屏在 GitHub Release |
| 原始结果 | GitHub Release，见 [证据索引](交付/证据索引.md) |

全部脚本的用法见 [工具/README.md](工具/README.md)，设计依据、预测和门槛修订见 [优化方案说明](交付/4_优化方案说明.md)。

原始测试结果约 160MB，单独打包在 GitHub Release，文件名和 SHA-256 见 [证据索引](交付/证据索引.md)。仓库不含模型权重、真实密钥和赛方资料。
