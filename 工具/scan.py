#!/usr/bin/env python3
"""用赛方脚本的同一套 AIPerf 参数做小样本扫描，每个档位重启一次服务。

用法（先跑过一次 run.sh 的冒烟，确保评测工具环境已安装）：
  python3 工具/scan.py --tag coarse --concurrency 8 --requests 24 \\
      --profile "s=2048,seqs=4" --profile "s=2048,seqs=8" \\
      --profile "s=512,seqs=8" --profile "s=128,seqs=8"

profile 的键：s=预算、seqs=max-num-seqs、mtp=MTP 深度、prefix=0 关闭前缀缓存、backend=线性层 kernel。
命令、数据集和结果解析都复用 参赛资料/eval/runner.py，避免和赛方口径不一致。

每档输出 吞吐、TPOT、TTFT P95，以及去掉窗口开头 c 个请求后的 TTFT P95。
测量窗口开头的请求同时到达，小样本下 P95 主要由它们决定，所以筛选时不看整体 TTFT P95。
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parent / "参赛资料" / "eval" / "runner.py"
sys.path.insert(0, str(HERE))
import log_evidence  # noqa: E402


def load_runner():
    spec = importlib.util.spec_from_file_location("eval_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_runner"] = module
    spec.loader.exec_module(module)
    return module


def parse_profile(text):
    profile = dict(pair.split("=") for pair in text.split(","))
    unknown = set(profile) - {"s", "seqs", "mtp", "prefix", "backend"}
    if unknown:
        raise SystemExit(f"不认识的 profile 键：{unknown}")
    return profile


def server_env(profile, name, args):
    env = dict(os.environ, NAME=name, WORK=args.work, PORT=str(args.port),
               LOG_ITER=str(args.log_iterations), LOG_REQ="0", VLLM_LOGGING_LEVEL="INFO",
               MODEL_DIR=args.model_dir, MTP_DEPTH="0", ADAPTIVE="0", NO_PREFIX="0",
               LINEAR_BACKEND="", BUDGET="2048", MAX_SEQS="4", EXTRA="")
    if "s" in profile:
        env["BUDGET"] = profile["s"]
    if "seqs" in profile:
        env["MAX_SEQS"] = profile["seqs"]
    if "mtp" in profile:
        env["MTP_DEPTH"] = profile["mtp"]
    if profile.get("prefix") == "0":
        env["NO_PREFIX"] = "1"
    if "backend" in profile:
        env["LINEAR_BACKEND"] = profile["backend"]
    return env


def ttft_split(records_path, concurrency, runner):
    """按请求开始时间排序，分出窗口开头同时到达的 c 个请求。"""
    records = [r for r in runner.read_jsonl(records_path)
               if r.get("metadata", {}).get("benchmark_phase") == "profiling"]
    records.sort(key=lambda r: r["metadata"]["request_start_ns"])
    values = [r["metrics"]["time_to_first_token"]["value"] for r in records]
    rest = values[concurrency:] or values
    return {"all_p95_s": runner.percentile(values, 95) / 1e3, "rest_p95_s": runner.percentile(rest, 95) / 1e3,
            "opening_s": [round(v / 1e3, 2) for v in values[:concurrency]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", action="append", required=True)
    ap.add_argument("--concurrency", type=int, action="append", required=True, help="可重复")
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--tag", default="scan")
    ap.add_argument("--log-iterations", type=int, choices=(0, 1), default=1,
                    help="配对性能确认设 0；调度诊断设 1")
    ap.add_argument("--output-dir", help="指定新结果目录；已存在时拒绝覆盖")
    ap.add_argument("--work", default=os.path.expanduser("~/uclode"))
    ap.add_argument("--model-dir", default="/model/ModelScope/Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--model", default="qwen3.8-27b-fp8")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--env-dir", default=os.path.expanduser("~/.cache/ai-infra-eval/envs"))
    args = ap.parse_args()

    runner = load_runner()
    out_root = (Path(args.output_dir) if args.output_dir else
                Path(args.work) / "scans" / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    out_root.mkdir(parents=True, exist_ok=False)
    ns = argparse.Namespace(
        base_url=f"http://127.0.0.1:{args.port}/v1", model=args.model, tokenizer=args.model_dir,
        tokenizer_revision=None, seed=args.seed, perf_requests=args.requests, request_timeout=600,
        env_dir=Path(args.env_dir))
    key = "EMPTY"
    python = runner.tool_python(ns.env_dir, "perf")
    dataset = out_root / "dataset.jsonl"
    # 不同档位用同一份输入，和赛方脚本一样靠 cache_salt 隔离前缀缓存
    child_env = runner.child_environment(ns.env_dir / "nltk_data")
    runner.execute([str(python), str(RUNNER), "_dataset", ns.tokenizer, "", str(dataset),
                    str(args.requests + runner.WARMUP + 1), str(args.seed)],
                   out_root / "dataset.log", key, child_env)

    serve = ["bash", str(HERE / "serve.sh")]
    rows = []
    for text in args.profile:
        profile = parse_profile(text)
        name = f"{args.tag}_" + text.replace(",", "_").replace("=", "")
        env = server_env(profile, name, args)
        subprocess.run(serve + ["start"], env=env, check=True)
        try:
            run_dir = Path(os.readlink(Path(args.work) / "runs" / "current"))
            for concurrency in args.concurrency:
                dest = out_root / name / f"c{concurrency}"
                dest.mkdir(parents=True)
                log = run_dir / "server.log"
                offset = log.stat().st_size
                salt = f"{uuid.uuid4().hex}:{name}:c{concurrency}"
                command = runner.aiperf_command(ns, concurrency, dataset, dest, salt, key)
                runner.write_json(dest / "command.json", command, key)
                print(f"\n== {text} 并发 {concurrency} ==", flush=True)
                row = {"profile": text, "concurrency": concurrency, "artifacts": str(dest)}
                try:
                    runner.execute(command, dest / "console.log", key, child_env)
                    result = runner.parse_performance(dest, args.requests, concurrency,
                                                      warmup_expected=runner.WARMUP, strict_isl=True)
                    row.update(status=result["status"], issues=result["issues"],
                               tok_s=result.get("output_token_throughput"),
                               tpot_p95_ms=result["metrics"]["inter_token_latency"]["p95"],
                               ttft=ttft_split(dest / "profile_export.jsonl", concurrency, runner))
                except (runner.EvaluationError, OSError, ValueError, KeyError, TypeError) as error:
                    row.update(status="异常", issues=[str(error)])
                with open(log, encoding="utf-8", errors="replace") as f:
                    f.seek(offset)
                    analysis = log_evidence.analyze_steps(log_evidence.parse_steps(f.readlines()))
                row["steps"] = analysis
                runner.write_json(dest / "step_analysis.json", analysis, key)
                log_evidence.print_steps(analysis)
                rows.append(row)
                (out_root / "summary.json").write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        finally:
            subprocess.run(serve + ["stop"], env=env, check=False)

    lines = ["| 配置 | 并发 | 状态 | tok/s | TPOT P95 ms | TTFT P95 s | 去掉开头后 TTFT P95 s | 混合步 |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        ttft = row.get("ttft", {})
        steps = row.get("steps", {})
        total = steps.get("total") or 1
        fmt = lambda v, spec: format(v, spec) if isinstance(v, (int, float)) else "-"
        lines.append(f"| {row['profile']} | {row['concurrency']} | {row['status']} | {fmt(row.get('tok_s'), '.1f')} "
                     f"| {fmt(row.get('tpot_p95_ms'), '.1f')} | {fmt(ttft.get('all_p95_s'), '.1f')} "
                     f"| {fmt(ttft.get('rest_p95_s'), '.1f')} | {steps.get('mixed', 0) / total:.0%} |")
    table = "\n".join(lines)
    (out_root / "summary.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table + f"\n\n结果目录：{out_root}")
    if any(row['status'] != '完整' for row in rows):
        raise SystemExit("扫描存在异常结果，请先检查原始记录。")


if __name__ == "__main__":
    main()
