#!/usr/bin/env python3
"""IFBench 小样本运行、输出体检和配对比较。

  run      用赛方脚本的 EvalScope 环境和配置，只改题数和 max_tokens。用来复现死循环
  scan     读预测结果：停止原因、输出长度、截断、尾部重复
  compare  同题配对：对→错、错→对、符号检验。给质量门槛用

  python3 工具/ifbench_tool.py run --limit 100 --max-tokens 8192 --out ~/uclode/eval/loop_mtp3
  python3 工具/ifbench_tool.py scan ~/uclode/eval/baseline_acc1/accuracy
  python3 工具/ifbench_tool.py compare ~/uclode/eval/baseline_acc1/accuracy ~/uclode/eval/baseline_acc2/accuracy

run 的 --limit 和 --max-tokens 都偏离赛方口径，结果不能当 IFBench 成绩。
正式成绩只来自赛方精度流程（run.sh --mode accuracy 或只改超时的 accuracy_run.py），结果目录同样可以用 scan 和 compare 分析。
"""
import argparse
import collections
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parent / "参赛资料" / "eval" / "runner.py"
METRIC = "prompt_level_strict"


def load_runner():
    spec = importlib.util.spec_from_file_location("eval_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_runner"] = module
    spec.loader.exec_module(module)
    return module


def read_records(folder, kind):
    """kind 是 predictions 或 reviews。folder 可以是赛方结果的 accuracy 目录。"""
    root = Path(folder).expanduser() / "native" / kind
    records = {}
    for path in root.rglob("*.jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    records[row["index"]] = row
    if not records:
        raise SystemExit(f"{root} 下没有记录。")
    return records


def flatten_text(node):
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(flatten_text(item) for item in node)
    if isinstance(node, dict):
        return "".join(flatten_text(node.get(key)) for key in ("reasoning", "text", "content") if node.get(key))
    return ""


def tail_period(text, max_period=400, repeats=3):
    """尾部有没有连续重复的片段。返回最短周期，没有就返回 None。"""
    for period in range(5, max_period + 1):
        if len(text) < period * repeats:
            break
        tail = text[-period:]
        if all(text[-period * (k + 1): -period * k or None] == tail for k in range(1, repeats)):
            return period
    return None


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p / 100
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)


def inspect_prediction(row):
    output = row.get("model_output") or {}
    choices = output.get("choices") or [{}]
    message = choices[0].get("message") or {}
    text = flatten_text(message)
    return {"stop": str(choices[0].get("stop_reason")), "tokens": (output.get("usage") or {}).get("output_tokens"),
            "period": tail_period(text), "error": row.get("error"), "chars": len(text)}


def scan(folder, max_tokens=None):
    rows = {index: inspect_prediction(row) for index, row in read_records(folder, "predictions").items()}
    stops = collections.Counter(r["stop"] for r in rows.values())
    tokens = [r["tokens"] for r in rows.values() if isinstance(r["tokens"], (int, float))]
    print(f"## {folder}\n题数 {len(rows)}，停止原因 {dict(stops)}，错误 {sum(bool(r['error']) for r in rows.values())}")
    if tokens:
        print(f"输出 token：中位数 {percentile(tokens, 50):.0f}，P95 {percentile(tokens, 95):.0f}，最大 {max(tokens):.0f}")
    truncated = [i for i, r in rows.items() if r["stop"] in ("length", "max_tokens", "model_length")]
    looping = [i for i, r in rows.items() if r["period"]]
    both = sorted(set(truncated) & set(looping))
    print(f"截断 {len(truncated)} 题：{sorted(truncated)}")
    print(f"尾部重复 {len(looping)} 题：{sorted(looping)}")
    print(f"死循环（达到 token 上限且尾部重复）{len(both)} 题：{both}")
    return rows


def sign_test(a, b):
    """双侧精确符号检验。a、b 是两个方向的翻转数。"""
    n = a + b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(base, other):
    a, b = read_records(base, "reviews"), read_records(other, "reviews")
    common = sorted(set(a) & set(b))
    if set(a) != set(b):
        print(f"注意：题目不完全相同，只比较共同的 {len(common)} 题。")
    score = lambda row: row["sample_score"]["score"]["value"].get(METRIC)
    pairs = [(i, score(a[i]), score(b[i])) for i in common]
    pairs = [(i, x, y) for i, x, y in pairs if x is not None and y is not None]
    lost = [i for i, x, y in pairs if x and not y]
    won = [i for i, x, y in pairs if y and not x]
    total_a, total_b = sum(bool(x) for _, x, _ in pairs), sum(bool(y) for _, _, y in pairs)
    print(f"## {base} → {other}（{METRIC}，{len(pairs)} 题）")
    print(f"通过数：{total_a} → {total_b}（净变化 {total_b - total_a:+d}）")
    print(f"对→错 {len(lost)} 题：{lost}")
    print(f"错→对 {len(won)} 题：{won}")
    print(f"符号检验双侧 p = {sign_test(len(lost), len(won)):.3f}")
    pred_a, pred_b = read_records(base, "predictions"), read_records(other, "predictions")
    if lost or won:
        print("| 题号 | 方向 | 停止原因 前/后 | 输出 token 前/后 | 尾部重复 前/后 |\n|---:|---|---|---|---|")
        for i in lost + won:
            x, y = inspect_prediction(pred_a.get(i, {})), inspect_prediction(pred_b.get(i, {}))
            print(f"| {i} | {'对→错' if i in lost else '错→对'} | {x['stop']} / {y['stop']} | {x['tokens']} / {y['tokens']} "
                  f"| {x['period']} / {y['period']} |")
    return {"lost": lost, "won": won, "total_a": total_a, "total_b": total_b}


def cmd_run(args):
    runner = load_runner()
    out = Path(args.out).expanduser()
    if out.exists():
        raise SystemExit("输出目录已存在，换一个 --out。")
    out.mkdir(parents=True)
    env_dir = Path(args.env_dir).expanduser()
    ns = argparse.Namespace(model=args.model, base_url=args.base_url, api_key_env="OPENAI_API_KEY",
                            eval_batch_size=args.eval_batch_size)
    config = runner.accuracy_config(ns, out)
    config["limit"] = args.limit
    config["generation_config"] = {"max_tokens": args.max_tokens}
    if args.temperature is not None:
        config["generation_config"]["temperature"] = args.temperature
    runner.write_json(out / "task_config.json", config)
    command = [str(runner.tool_python(env_dir, "accuracy")), str(RUNNER), "_accuracy", str(out / "task_config.json")]
    key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    runner.execute(command, out / "console.log", key, runner.child_environment(env_dir / "nltk_data"))
    runner.sanitize_files(out, key)
    scan(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", required=True)
    run.add_argument("--limit", type=int, default=100)
    run.add_argument("--max-tokens", type=int, default=8192)
    run.add_argument("--temperature", type=float, help="不传就用服务端默认值，和正式评测一致")
    run.add_argument("--eval-batch-size", type=int, default=2)
    run.add_argument("--base-url", default="http://127.0.0.1:8088/v1")
    run.add_argument("--model", default="qwen3.8-27b-fp8")
    run.add_argument("--env-dir", default="~/.cache/ai-infra-eval/envs")
    sc = sub.add_parser("scan")
    sc.add_argument("folders", nargs="+")
    co = sub.add_parser("compare")
    co.add_argument("base")
    co.add_argument("others", nargs="+")
    args = ap.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "scan":
        for folder in args.folders:
            scan(folder)
    else:
        for other in args.others:
            compare(args.base, other)


if __name__ == "__main__":
    main()
