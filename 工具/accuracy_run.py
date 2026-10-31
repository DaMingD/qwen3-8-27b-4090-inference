#!/usr/bin/env python3
"""赛方精度阶段，只多一个客户端超时，可以从已停止的结果续跑。不修改赛方文件。

赛方 accuracy_config 没有设 generation_config.timeout，OpenAI SDK 默认 600 秒读超时。
请求不是流式的，600 秒内没生成完就断开重采，约 18000 token（30 tok/s）以上的回答拿不到。
赛方的 --request-timeout 只作用于性能阶段。

  python3 工具/accuracy_run.py --timeout 3600 --mode accuracy --base-url URL --model NAME --output-dir 新目录
  python3 工具/accuracy_run.py --timeout 3600 --resume 旧目录/accuracy --mode accuracy --base-url URL --model NAME --output-dir 新目录

除 --timeout、--resume 外的参数原样交给 runner.py，结果目录结构和赛方一致。

续跑把旧的 native 复制到新目录，原目录不动。timeout 属于 generation_config，会改变 EvalScope 的
缓存身份，只开 use_cache 会被拒绝，所以同时开 rerun_review：保留已有预测，只生成缺的题，重判所有题。
"""
import argparse
from pathlib import Path
import shutil
import signal
import sys

from ifbench_tool import load_runner


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeout", type=float, required=True, help="单次请求的客户端超时，秒")
    ap.add_argument("--resume", type=lambda p: Path(p).expanduser(), help="旧结果的 accuracy 目录")
    args, rest = ap.parse_known_args()
    if args.resume and not (args.resume / "native/predictions").is_dir():
        raise SystemExit(f"{args.resume} 下没有 native/predictions。")
    runner = load_runner()
    official = runner.accuracy_config

    def accuracy_config(namespace, dest):
        config = official(namespace, dest)
        config["generation_config"]["timeout"] = args.timeout
        if args.resume:
            config.update(use_cache=config["work_dir"], rerun_review=True)
            if Path(dest).is_dir():  # --dry-run 的路径是占位符
                shutil.copytree(args.resume / "native", config["work_dir"])
        return config

    def interrupted(signum, frame):  # 和 runner.py 一样，让停止信号带上评测子进程
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    runner.accuracy_config = accuracy_config
    try:
        return runner.main(rest)
    except (runner.EvaluationError, OSError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
