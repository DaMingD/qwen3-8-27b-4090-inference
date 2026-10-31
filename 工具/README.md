# 上机脚本

对应 [最终思路.md](../最终思路.md)。把 `工具/` 和 `参赛资料/eval/` 传到服务器的同一个目录下，在这个目录里运行。

脚本默认工作目录是 `~/uclode`，模型目录是 `/model/ModelScope/Qwen/Qwen3.8-27B-FP8`。如果不一样，用环境变量 `WORK`、`MODEL_DIR` 改。服务默认只监听 `127.0.0.1`。

| 文件 | 作用 |
|---|---|
| `setup_env.sh` | 建独立的 vLLM 0.32.0 环境，不碰已有 conda 环境。`GPU_CHECK=0` 时不碰显卡 |
| `env_check.py` | 检查版本和 C++ 扩展，默认还做 GPU 算子验收，`--no-gpu` 时隐藏所有 GPU |
| `serve.sh` | 启动和停止服务，保存命令、版本、逐秒 GPU 状态和日志 |
| `log_evidence.py` | 从服务日志提取 kernel、block、KV 容量、采样参数、批次组成和 future 等待时间 |
| `kernel_bench.py` | kernel A/B：prefill、解码步耗时和真实文本的 logprob，换算成预计吞吐 |
| `convert_fp8.py` | 从 BF16 转换 per-channel FP8，只用 CPU |
| `scan.py` | 用赛方脚本的同一套 AIPerf 参数做小样本扫描 |
| `ctx_slope.py` | 验证预测 P1（上下文与解码斜率） |
| `probe_mtp.py` | MTP 探针 A、B |
| `ifbench_tool.py` | IFBench 小样本运行（死循环复现）、输出体检、配对比较 |
| `task_suite.py` | 三类任务小测试集 |
| `accuracy_run.py` | 赛方精度流程，只加长客户端超时，可以续跑 |
| `mtp_tail.sh` | 夜间队列之后的 MTP 实验，到点停止 |
| `night_queue.sh` | 无人值守队列 |
| `repro_check.sh` | 复验自查：`build` 不碰显卡，新目录从零装环境、重转权重、比对权重和依赖；`verify` 做 GPU 算子验收，上机复测吞吐和赛方流程 |
| `build_results.py` | 只在本地运行：从原始结果生成结果表，给出判定，渲染 `交付/` 下的文档 |
| `demo.py` | 现场演示：启动证据、并发 8 实时一轮、三类任务。流程见 [Demo 脚本](../交付/Demo脚本.md) |
| `after_baseline.py` | 等待完整基线成功，串行执行同条件 kernel 配对、调度粗筛和 P1 |
| `ttft_window.py` | 把完整性能结果截到官方样本量比较 TTFT，检验 P5 |
| `step_sim.py` | 成本模型和预测，不需要 GPU |

**验证状态。** 2026-10-31 已在真实服务器完成独立环境和 API 检查、AIPerf 四档冒烟、EvalScope 请求参数核对、75 项上游采样器测试、不开 MTP 的两轮探针，以及 Marlin/Triton/CUTLASS 三方对比。per-channel 权重已转换，vLLM 确认加载 CUTLASS 并完成推理。证据见 原始结果包的 `实测结果/20261031_基线启动` 和 原始结果包的 `实测结果/20261031_kernel对比`。完整基线、配对、粗筛和九格实验已完成，证据见 原始结果包的 `实测结果/20261031_完整基线与配对`。基线 IFBench 和小测试集已在真实服务上跑完，证据见 原始结果包的 `实测结果/20261031_基线质量完整`。`repro_check.sh`、`build_results.py` 只做过干跑和合成数据测试。

**计时口径已修正。** vLLM 的 `iteration elapsed time` 只是 `future.result()` 的等待时间。kernel 和上下文斜率脚本改用客户端请求/流式区间计时，保留事件记录；`log_evidence.py` 不再用日志耗时拟合 GPU 成本或共享系数 λ。日志仍用于核对 token 数与批次组成，P2 需要另做 profiler/CUDA event 测量。

**决策。** G2 已定：`max-num-seqs=8`、预算 2048。Triton 淘汰；CUTLASS 通过性能条件，是否采用由方案 A 的 IFBench 决定，退路是同配置的 Marlin。基线质量：IFBench 79/100，小测试集 29/30（[附录 D.7、D.8](../最终思路.md)）。方案 A 不开 MTP。最终配置和质量判定以 [结果表](../交付/结果表.md) 为准。不得把成本模型的预计吞吐写成 AIPerf 实测结果。

FlashInfer 首次 JIT 编译需要 CUDA Toolkit、C++ 编译器（`g++`）和 `ninja`。这台服务器已经补齐；其他机器要先检查。

核对 block 时用 `VLLM_LOGGING_LEVEL=DEBUG`，正式计时要用 INFO。`serve.sh` 会把日志级别写进运行目录的 `meta.txt`。下面所有命令默认在脚本目录的上一层运行，并且已经设置：

```bash
MODEL=/model/ModelScope/Qwen/Qwen3.8-27B-FP8
```

## 第一步：用现有服务做准备（约 30 分钟）

现有服务是 DEBUG 日志、`LOG_REQ=1`，不能用来计时，但可以做这些事。

```bash
mkdir -p ~/uclode/mtp ~/uclode/kernel ~/uclode/tasks ~/uclode/models ~/uclode/eval
```

安装性能工具环境并冒烟（要在交互终端确认）：

```bash
bash 参赛资料/eval/run.sh --mode perf --base-url http://127.0.0.1:8088/v1 --model qwen3.8-27b-fp8 --tokenizer $MODEL --perf-requests 2 --output-dir ~/uclode/eval/smoke_perf
```

安装精度工具环境，核对 IFBench 的真实请求链路。启动几分钟后，在另一个终端运行第二条命令，确认后按 Ctrl-C 停止第一条：

```bash
bash 参赛资料/eval/run.sh --mode accuracy --base-url http://127.0.0.1:8088/v1 --model qwen3.8-27b-fp8 --output-dir ~/uclode/eval/smoke_acc
```

```bash
python3 工具/log_evidence.py sampling ~/uclode/runs/current/server.log --where max_tokens=32768 --expect temperature=1.0,top_k=20,top_p=0.95
```

上游采样器测试（需要 v0.32.0 源码里的 `tests/`）：

```bash
git clone --depth 1 --branch v0.32.0 https://github.com/vllm-project/vllm.git ~/uclode/vllm-src
```

```bash
~/uclode/venv-vllm/bin/python -m pip install pytest tblib
cd ~/uclode
~/uclode/venv-vllm/bin/python -c 'import vllm,pytest,sys; print(vllm.__file__,flush=True); sys.exit(pytest.main(["vllm-src/tests/v1/sample/test_rejection_sampler.py","-q"]))'
```

先导入已安装的 `vllm`，避免源码目录遮蔽轮子中的 C++ 扩展。本次实际使用 tag 压缩包，源码目录为 `~/uclode/vllm-0.32.0`，因此将上面的 `vllm-src` 换成该目录名。

MTP 探针的基线组（用法见 `probe_mtp.py` 开头）：

```bash
python3 工具/probe_mtp.py sample --model-dir $MODEL --n 2000 --out ~/uclode/mtp/a_base1.jsonl
```

```bash
python3 工具/probe_mtp.py sample --model-dir $MODEL --n 2000 --out ~/uclode/mtp/a_base2.jsonl
```

```bash
python3 工具/probe_mtp.py greedy --model-dir $MODEL --out ~/uclode/mtp/b_base1.jsonl
```

```bash
python3 工具/probe_mtp.py greedy --model-dir $MODEL --out ~/uclode/mtp/b_base2.jsonl
```

同时在后台启动权重转换，不占 GPU。先用 `--dry-run` 看权重名和磁盘空间：

```bash
~/uclode/venv-vllm/bin/python 工具/convert_fp8.py --bf16 /model/ModelScope/Qwen/Qwen3.8-27B --fp8 $MODEL --out ~/uclode/models/Qwen3.8-27B-FP8-channel --dry-run
```

```bash
nohup ~/uclode/venv-vllm/bin/python 工具/convert_fp8.py --bf16 /model/ModelScope/Qwen/Qwen3.8-27B --fp8 $MODEL --out ~/uclode/models/Qwen3.8-27B-FP8-channel > ~/uclode/convert.log 2>&1 &
```

## 第二步：kernel A/B（每条路径约 8 分钟）

先停掉现有服务，确认 shell 里没有 `VLLM_LOGGING_LEVEL=DEBUG`：

```bash
bash 工具/serve.sh stop; unset VLLM_LOGGING_LEVEL
```

Marlin（现状）：

```bash
NAME=kb_marlin LOG_ITER=1 MAX_SEQS=8 bash 工具/serve.sh start && python3 工具/kernel_bench.py run --server-log ~/uclode/runs/current/server.log --out ~/uclode/kernel/marlin.json; bash 工具/serve.sh stop
```

Triton：

```bash
NAME=kb_triton LOG_ITER=1 MAX_SEQS=8 LINEAR_BACKEND=triton bash 工具/serve.sh start && python3 工具/kernel_bench.py run --server-log ~/uclode/runs/current/server.log --out ~/uclode/kernel/triton.json; bash 工具/serve.sh stop
```

CUTLASS per-channel（转换完成后）。启动证据里要出现 `CutlassFP8ScaledMMLinearKernel`：

```bash
NAME=kb_cutlass LOG_ITER=1 MAX_SEQS=8 LINEAR_BACKEND=cutlass MODEL_DIR=$HOME/uclode/models/Qwen3.8-27B-FP8-channel bash 工具/serve.sh start && python3 工具/kernel_bench.py run --server-log ~/uclode/runs/current/server.log --out ~/uclode/kernel/cutlass.json; bash 工具/serve.sh stop
```

比较：

```bash
python3 工具/kernel_bench.py compare ~/uclode/kernel/marlin.json ~/uclode/kernel/triton.json
```

```bash
python3 工具/kernel_bench.py compare ~/uclode/kernel/marlin.json ~/uclode/kernel/cutlass.json
```

按 [第 4.4 节](../最终思路.md) 的规则决定。达标的路径用 AIPerf 小样本确认：

```bash
python3 工具/scan.py --tag kernel --concurrency 8 --requests 24 --profile "seqs=8" --profile "seqs=8,backend=triton"
```

## 第三步：基线全量性能（约 85 分钟）

官方配置，只改 `--max-model-len`，INFO 日志：

```bash
NAME=baseline_perf bash 工具/serve.sh start
```

```bash
bash 参赛资料/eval/run.sh --mode perf --base-url http://127.0.0.1:8088/v1 --model qwen3.8-27b-fp8 --tokenizer $MODEL --output-dir ~/uclode/eval/baseline_perf
```

对照预测 P5：官方配置下 TTFT P95 预计约为 3.5、4.1、4.2、34.8 秒（并发 1、2、4、8）。

## 第四步：kernel 配对确认、调度粗筛和上下文斜率（预算约 75–90 分钟）

本次用 `after_baseline.py` 接续：只有完整基线退出码为 0，且四档各 100 个正式请求、8 个预热请求通过原生记录校验，才会停止基线服务。它依次运行下述配对、三档调度扫描和九格 P1；任一测量失败即停止，不自动选择最终配置或启动 IFBench。先 `--dry-run` 核对命令；实际运行命令为 `python3 工具/after_baseline.py`。Marlin 对照异常时队列停止；CUTLASS 和粗筛的异常只记录，看 `complete.txt` 的异常步骤。队列运行时不要另外执行本节命令占用同一张卡。

**配对确认必须补 Marlin 对照。** 两侧均为 24 个正式请求、8 个预热请求、`max-num-seqs=8`、预算 2048、INFO、请求及逐步日志关闭，同种子且数据集逐字节相同。100 请求的原基线只用于最终整体对比，不直接用作本次 kernel 门槛的分母。补对照预计增加约 15 分钟，实际以运行时长为准。

手动执行时，先确认基线已经完整结束，再运行 `bash 工具/serve.sh stop`，然后依次运行：

```bash
python3 工具/scan.py --tag marlin_pair --concurrency 1 --concurrency 4 --concurrency 8 --requests 24 --log-iterations 0 --profile "s=2048,seqs=8"
```

CUTLASS 的 AIPerf 确认（约 21 分钟）。三档均和上述同条件 Marlin 对照比较；原门槛未通过的记录不改，使用 §4.4 中预先固定的新门槛：

```bash
python3 工具/scan.py --tag cutlass --model-dir $HOME/uclode/models/Qwen3.8-27B-FP8-channel --concurrency 1 --concurrency 4 --concurrency 8 --requests 24 --log-iterations 0 --profile "s=2048,seqs=8,backend=cutlass"
```

完成 kernel 配对后做调度粗筛，预算不要选在 1568 到 1568+并发数之间：

```bash
python3 工具/scan.py --tag coarse --concurrency 8 --requests 24 --profile "s=2048,seqs=8" --profile "s=512,seqs=8" --profile "s=128,seqs=8"
```

如果粗筛最优预算比默认高 5% 以上，并且 CUTLASS 通过上面的确认，再用同样的命令测一次 `s=最优,seqs=8,backend=cutlass`，只跑并发 8。

```bash
NAME=slope LOG_ITER=1 MAX_SEQS=8 bash 工具/serve.sh start && python3 工具/ctx_slope.py --server-log ~/uclode/runs/current/server.log --out ~/uclode/ctx_slope.json; bash 工具/serve.sh stop
```

## 第五步：MTP 开启组和净收益（已写进 `mtp_tail.sh`，接在夜间队列后面，下面是各步的命令）

第一步的 `a_base1`、`a_base2` 只采了前 3 个位置，分布检验的信息不足。先在不开 MTP 的服务上重采 7 个位置，确认每条提示词至少有一个位置同时满足熵 ≥ 0.5 bit、条件样本 ≥ 500。`entropy --require` 不满足时返回非零，此时先调整提示词或增加样本，不直接运行 MTP 开启组。500 个条件样本时，两组各同样本量的 p=0.5 检出限约为 11 个百分点，不能套用 950 样本时的约 8 个百分点。熵不是草稿拒绝率，拒绝/接受情况仍要另外观测。

```bash
NAME=base_probe bash 工具/serve.sh start
```

```bash
python3 工具/probe_mtp.py sample --model-dir $MODEL --n 1000 --positions 7 --out ~/uclode/mtp/a7_base1.jsonl && python3 工具/probe_mtp.py sample --model-dir $MODEL --n 1000 --positions 7 --out ~/uclode/mtp/a7_base2.jsonl && python3 工具/probe_mtp.py entropy ~/uclode/mtp/a7_base1.jsonl --require && python3 工具/probe_mtp.py entropy ~/uclode/mtp/a7_base2.jsonl --require
```

开启 MTP：

```bash
bash 工具/serve.sh stop; NAME=mtp3 MTP_DEPTH=3 bash 工具/serve.sh start
```

```bash
python3 工具/probe_mtp.py sample --model-dir $MODEL --n 1000 --positions 7 --out ~/uclode/mtp/a7_mtp3.jsonl && python3 工具/probe_mtp.py compare ~/uclode/mtp/a7_base1.jsonl ~/uclode/mtp/a7_base2.jsonl ~/uclode/mtp/a7_mtp3.jsonl --positions 7
```

```bash
python3 工具/probe_mtp.py greedy --model-dir $MODEL --out ~/uclode/mtp/b_mtp3.jsonl && python3 工具/probe_mtp.py diverge ~/uclode/mtp/b_base1.jsonl ~/uclode/mtp/b_base2.jsonl && python3 工具/probe_mtp.py diverge ~/uclode/mtp/b_base1.jsonl ~/uclode/mtp/b_mtp3.jsonl
```

合成负载的净收益只测深度 3。无投机对照直接用 `marlin_pair`，条件相同，不用重跑：

```bash
bash 工具/serve.sh stop; python3 工具/scan.py --tag mtp --concurrency 1 --concurrency 8 --requests 24 --log-iterations 0 --profile "s=2048,seqs=8,mtp=3"
```

还有时间才测深度 1（`mtp=1`）和上游自适应验证（`ADAPTIVE=1 MTP_DEPTH=3` 启动，报错就说明它在这个模型上不可用，如实记录）。

## 第六步：基线 IFBench 和小测试集（已完成）

接续队列结束时留着一套开了逐步日志的 Marlin 服务，要先停掉。

```bash
bash 工具/serve.sh stop; unset VLLM_LOGGING_LEVEL; NAME=baseline_acc LOG_REQ=1 MAX_SEQS=4 bash 工具/serve.sh start
```

赛方脚本的 IFBench 客户端超时是 600 秒，改不了（原因见 `accuracy_run.py` 开头和最终思路附录 D.7），所以用 `accuracy_run.py`。2026-10-31 的基线在 87/100 时停下，用下面的命令续跑，服务不用重启。先停掉原来的串行脚本，再停评测进程，不要动 `vllm serve`；备份后再续跑。

```bash
python3 工具/accuracy_run.py --timeout 3600 --resume ~/uclode/eval/baseline_accuracy/accuracy --mode accuracy --base-url http://127.0.0.1:8088/v1 --model qwen3.8-27b-fp8 --output-dir ~/uclode/eval/baseline_accuracy_resume && python3 工具/ifbench_tool.py scan ~/uclode/eval/baseline_accuracy_resume/accuracy && python3 工具/task_suite.py run --out ~/uclode/tasks/baseline.json
```

启动后一分钟内核对：`task_config.json` 里有 `timeout`、`use_cache`、`rerun_review`；`console.log` 里有 `Forcing prediction reuse`，没有 `Cached evaluation identity does not match`；新目录的预测从 87 条往上加，题号不重复。判断进度要同时看预测条数和服务端的请求取消数，只看已保存的预测会漏掉正在重试的请求。

2026-10-31 上机确认该容器禁止 `unshare --net`。`task_suite.py` 会先保存代码题回答，标记为待评分，长文本和推理题照常评分。取回结果后，在本地 macOS（或 `unshare --net` 可用的 Linux）执行下列命令完成代码题测试。结果逐题保存，到点停掉不会丢已完成的题。隔离环境禁止网络连接、文件写入和派生进程，并限制用户目录读取。基线和候选配置使用相同的本地 Python 环境与测试用例，配对比较传入完成评分后的文件。

```bash
python3 工具/task_suite.py grade /path/to/baseline.json --out /path/to/baseline_graded.json
```

基线质量队列会在 IFBench 成功后自动接小测试集，结束时间的分支不自动判断，由值守者处理：IFBench 结束时查 [最终思路第 8 节](../最终思路.md) 的表；到 00:45（IFBench 在 00:33 后才结束时，就在结束时）小测试集还在跑，就停掉它，已完成的题已保存，然后启动第七步的队列。

```bash
pkill -f "task_suite.py run"
```

待评分记录会使 `compare` 返回失败，防止将未执行的代码题计入质量结论。本次基线质量队列于北京时间 20:31 启动，EvalScope 于 20:33 开始正式评测，随后自动运行输出检查和小测试集回答采集。

## 第七步：无人值守队列

配置按决策结果改。基线质量已完成，队列现在就可以启动。下面是 CUTLASS 进入方案 A、退路为 Marlin 默认预算时的队列。

退路只补并发 8：并发 1、2、4 时 `max-num-seqs` 不起作用，用官方配置的基线即可。先核对 `marlin_pair` 的并发 1、4 和基线的吞吐相差不超过 2%；不成立，或者退路改了预算，就用 `night_queue.sh` 跑完整四档（约 82 分钟）。

先用 `DRY_RUN=1` 看命令：

```bash
DRY_RUN=1 bash 工具/night_queue.sh "plan_a perf MAX_SEQS=8 LINEAR_BACKEND=cutlass MODEL_DIR=$HOME/uclode/models/Qwen3.8-27B-FP8-channel" "plan_a accuracy MAX_SEQS=8 LINEAR_BACKEND=cutlass MODEL_DIR=$HOME/uclode/models/Qwen3.8-27B-FP8-channel" "baseline accuracy_b8 MAX_SEQS=8"
```

确认后放进 tmux。`bash 工具/night_queue.sh` 的参数和上面相同，退路的并发 8 排在最前面（约 13 分钟）：

```bash
bash 工具/serve.sh stop; python3 工具/scan.py --tag fallback_c8 --concurrency 8 --requests 100 --log-iterations 0 --profile "s=2048,seqs=8"; bash 工具/night_queue.sh "plan_a perf MAX_SEQS=8 LINEAR_BACKEND=cutlass MODEL_DIR=$HOME/uclode/models/Qwen3.8-27B-FP8-channel" "plan_a accuracy MAX_SEQS=8 LINEAR_BACKEND=cutlass MODEL_DIR=$HOME/uclode/models/Qwen3.8-27B-FP8-channel" "baseline accuracy_b8 MAX_SEQS=8"; bash 工具/mtp_tail.sh
```

队列运行时的检查按阶段触发，时间都是估算。CUTLASS 四档性能结束后确认 IFBench 真正开始：`task_config.json` 里 `timeout=3600`、`eval_batch_size=2`、`limit=100`；`cmd.args` 里有 `--linear-backend cutlass`、`--max-num-seqs 8` 和 channel 模型路径；启动日志有 `CutlassFP8ScaledMMLinearKernel`；首批预测有效。预测条数一时不增加时，看服务端的请求和生成是否在推进。

确认后先查磁盘（约需 50 GB），再运行复验一级。它隐藏所有 GPU，用 `nice` 低优先级，只占 CPU 和磁盘；CUTLASS 性能阶段仍然不要跑，避免 CPU 和磁盘干扰计时。

```bash
bash 工具/repro_check.sh build
```

## 第八步：06:45 人工检查

06:45 是检查点，不是 GPU 释放的保证。只有 `mtp_tail.sh` 内部有截止时间。

| 06:45 时的状态 | 处理 |
|---|---|
| 全部结束 | 确认 GPU 已释放 |
| MTP 在跑 | 脚本到点自停 |
| 噪声底在跑 | 07:00 前结束不了就先停串行脚本再停它，写未完成 |
| CUTLASS IFBench 在跑 | 07:00 按进度估算：08:00 前能完成就继续，这期间只做不占 GPU 的事；否则立刻停，判未完成 |

同时看一级复验。等待任务在 tmux 会话 `repro_wait` 里，IFBench 的启动检查不通过就不会运行 `build`，所以没有结果文件不一定是还没跑完，要看 `~/uclode/eval/repro_build_wait.log`。结果在 `~/repro/repro_build.json`。

一级没有结果时，先运行 `repro_check.sh build`，再运行二级，因为二级要用它装好的环境和转换好的权重。来不及就只报告实际验证到的级别。

## 第九步：早上收尾

GPU 空闲后运行复验二级，它先做一级没做的 GPU 算子验收。`REF` 换成要提交的并发 4、8 吞吐，`FINAL` 是 `cutlass` 或 `marlin`，结束时服务已停（约 15 分钟）。两级的 `passed` 都为 true 才算通过；`failed` 列出没过的项，依赖差异在 `requirements.diff`：

```bash
FINAL=cutlass REF=4=76.31,8=116.70 bash 工具/repro_check.sh verify
```

看 IFBench 结果：

```bash
python3 工具/ifbench_tool.py compare ~/uclode/eval/baseline_accuracy_resume/accuracy ~/uclode/eval/plan_a_accuracy/accuracy ~/uclode/eval/baseline_accuracy_b8/accuracy
```

小测试集在方案 A 的服务上再跑一次，只有 IFBench 通过时才有意义：

```bash
python3 工具/task_suite.py run --out ~/uclode/tasks/plan_a.json
```

取回 `baseline.json` 和 `plan_a.json`，在本地用同一个 Python 评分后比较（未评分的文件会被 `compare` 拒绝）：

```bash
python3 工具/task_suite.py grade baseline.json --out baseline_graded.json && python3 工具/task_suite.py grade plan_a.json --out plan_a_graded.json && python3 工具/task_suite.py compare baseline_graded.json plan_a_graded.json
```

把夜间结果取回 `实测结果/20261101_夜间/`，生成结果表和文档。判定只读原始结果，读不到的一律算未完成：

```bash
python3 工具/build_results.py --night 实测结果/20261101_夜间 --manual 交付/手写结论.json
```

## 死循环复现（已写进 `mtp_tail.sh`，到点前有时间才跑到）

结果不是 IFBench 成绩。这里用并发 8 加速，`max_tokens` 限制在 8192：

```bash
NAME=loop_mtp3 MTP_DEPTH=3 MAX_SEQS=8 bash 工具/serve.sh start && python3 工具/ifbench_tool.py run --limit 100 --max-tokens 8192 --eval-batch-size 8 --out ~/uclode/eval/loop_mtp3
```
