# Qwen3.8-27B 单卡 RTX 4090 推理优化

UCloud AI 黑客松 Track A。不换模型、不改 vLLM 源码，用可证伪的成本模型定位瓶颈，逐项检验 kernel、调度和投机解码，并按预设门槛验收质量。

## 结果

**质量判定：通过。** 方案 A 的 IFBench 为 80/100，基线 79/100；小测试集 29/30，基线 29/30。结果满足预设质量门槛，这不等于质量没有下降，100 题的样本证明不了后者。

小测试集新增 1 条失败：`code_lru` 第 2 次，嵌套类 `Node` 写成了 `Node()`，运行时 `NameError`。基线的 `code_brackets` 第 2 次也失败过，这题在 CUTLASS 上两次都通过，但总分持平不抵消新增失败。解释标准是看到这条失败之后、加测之前写定并提交的（提交说明 `Predeclare LRU follow-up acceptance rule`，北京时间 11 月 1 日 07:06）：两种配置各跑 50 次 `code_lru`，CUTLASS 的失败次数不超过 Marlin 加 5 次才算已解释。结果 CUTLASS 49/50，Marlin 47/50，都没有截断和空回答；同样的 `NameError` 在 Marlin 上出现 2 次，在 CUTLASS 上出现 1 次。这说明这类错误不是 CUTLASS 特有的；50 次只能发现较大的差异，不能证明 CUTLASS 不会引入错误。

**最终配置：** CUTLASS per-channel FP8 + max-num-seqs=8 + 预算 2048。并发 8 吞吐 115.5 tok/s（官方配置 67.4，+71.4%），并发 4 为 76.6（+13.7%），并发 1 为 26.0（-3.0%）。低并发下解码更慢，这是用高并发吞吐和首 token 延迟换来的代价。

基线重复组（batch 8，和 batch 2 的基线条件不同，基线还有 3 题经过超时重采）得分 80/100，对→错 4 题、错→对 5 题。它只说明这些条件下的翻转范围，不是严格的同配置随机波动。

MTP 深度 3 只做了分析，最终配置不开。性能有收益：Marlin 加 MTP 并发 1 为 49.3 tok/s、并发 8 为 110.3 tok/s（各 24 个请求，同条件不开 MTP 为 26.74、92.22），并发 1 端到端 P95 为 12.0 秒，低于 13 秒的目标。质量没有完成验收：100 题循环实验把输出限制在 8192 token，22 题达到上限，所以严格提示词 67/100 不能和基线比（不开 MTP 的三次完整运行有 14–21 题超过 8192）；未截断的 78 题配对为 65 → 66，这是事后选出的子集，不作结论。尾部重复 0 题，官方表的死循环在这个上限内没有复现，8192 之后的循环看不到。分布探针没有发现超过检出限（7.8–11.9 个百分点）的差异；贪心探针 3/3 提示词出现分歧，不开 MTP 的重复对照为 0/3，分歧处 top-2 差不超过 0.125，原因没有查明。

**复验自查：** 一级（从零安装、重新转换权重并比对权重哈希和依赖清单，不含 GPU 算子）：首次未通过（权重哈希一致，依赖清单有版本漂移），按锁定清单修复后通过，这不是第二次从零构建；修复后的安装脚本在空目录重新安装一次：通过；二级（GPU 算子验收、上机启动、kernel、吞吐相差不超过 5%、赛方流程冒烟）：通过。这是我们的自查，不是赛方标准；评测工具环境没有重新安装。


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

启动命令按最终配置二选一，本次提交使用：CUTLASS per-channel FP8 + max-num-seqs=8 + 预算 2048。

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
