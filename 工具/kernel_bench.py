#!/usr/bin/env python3
"""kernel A/B：用同一组请求比较线性层 kernel 的 prefill、解码耗时和数值差异。只依赖标准库。

每个 kernel 单独启动一次服务，都要 LOG_ITER=1、MAX_SEQS=8：
  NAME=kb_marlin LOG_ITER=1 MAX_SEQS=8 bash 工具/serve.sh start
  python3 工具/kernel_bench.py run --server-log ~/uclode/runs/current/server.log --out ~/uclode/kernel/marlin.json
  bash 工具/serve.sh stop
  NAME=kb_triton LOG_ITER=1 MAX_SEQS=8 LINEAR_BACKEND=triton bash 工具/serve.sh start
  python3 工具/kernel_bench.py run --server-log ~/uclode/runs/current/server.log --out ~/uclode/kernel/triton.json
  bash 工具/serve.sh stop
  python3 工具/kernel_bench.py compare ~/uclode/kernel/marlin.json ~/uclode/kernel/triton.json

一次 run 约 5 分钟，测四件事：
  fingerprint  真实文本的逐位置 logprob。比较 top-1 一致率和平均负对数似然（NLL）
  sweep        单次请求耗时随输入 token 数的变化（输出 1 token）
  prefill      8192 token 的 prefill 总耗时 T_p，以及实际 chunk
  decode       并发 1、4、8 全部活跃时的客户端流式 token 间隔，输入长度 8192

compare 把 T_p 和 t(c) 代入成本模型 c*OSL / (c*T_p + 511*t(c))，换算成各并发的预计吞吐。
fingerprint 只是进入 IFBench 前的低成本检查，不能代替质量验收。
iteration elapsed time 只覆盖 future.result() 的等待范围。保留作诊断字段，不用于吞吐预测。
"""
import argparse
import collections
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctx_slope  # noqa: E402
import log_evidence  # noqa: E402
import step_sim  # noqa: E402

ISL, OSL = 8192, 512
SWEEP = (32, 128, 512, 1024, 1536)
REPEATS = 6                  # 每个长度的重复次数，第一次含 JIT，丢弃
CONCURRENCIES = (1, 4, 8)
TEXT_CHARS = 12000           # 每份文本的字符数，约 3k–4k token
HERE = Path(__file__).resolve().parent


def steps_since(log, offset):
    time.sleep(1)   # 日志由引擎进程异步写入
    with open(log, encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        return log_evidence.parse_steps(f.readlines())


def random_prompt(rng, length):
    return [rng.randrange(1000, 200000) for _ in range(length)]


def one_token(url, model, prompt, tag):
    start = time.perf_counter()
    result = ctx_slope.post(url, {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0,
                                  "cache_salt": f"kb-{tag}-{time.time()}"})
    if result["usage"]["prompt_tokens"] != len(prompt):
        raise SystemExit(f"输入 {len(prompt)} 个 token，服务端计 {result['usage']['prompt_tokens']}")
    return time.perf_counter() - start


def run_sweep(url, model, log, rng):
    rows = []
    for length in SWEEP:
        offset = os.path.getsize(log)
        wall_ms = []
        for index in range(REPEATS):
            wall_ms.append(1000 * one_token(url, model, random_prompt(rng, length), f"sweep-{length}-{index}"))
        ms = [s["ms"] for s in steps_since(log, offset) if s["ctx_tok"] == length and s["gen_req"] == 0][1:]
        rows.append({"tokens": length, "steps": len(ms), "median_ms": statistics.median(wall_ms[1:]),
                     "engine_wait_median_ms": statistics.median(ms) if ms else None})
        print(f"sweep {rows[-1]}", flush=True)
    return rows


def run_prefill(url, model, log, rng):
    offset = os.path.getsize(log)
    seconds = [one_token(url, model, random_prompt(rng, ISL), f"prefill-{i}") for i in range(REPEATS)][1:]
    steps = [s for s in steps_since(log, offset) if s["ctx_tok"] > 0]
    by_chunk = collections.defaultdict(list)
    for step in steps:
        by_chunk[step["ctx_tok"]].append(step["ms"])
    result = {"seconds": [round(x, 3) for x in seconds], "T_p": statistics.median(seconds),
              "chunks": {str(k): {"n": len(v), "median_ms": statistics.median(v)} for k, v in sorted(by_chunk.items())}}
    print(f"prefill T_p={result['T_p']:.3f}s，chunk：{result['chunks']}", flush=True)
    return result


def run_decode(url, model, log, rng, plateau):
    ctx_slope.PLATEAU_S = plateau
    rows = []
    for concurrency in CONCURRENCIES:
        offset = os.path.getsize(log)
        event_rows = [None] * concurrency
        output, errors = ctx_slope.run_cell(url, model, ISL, concurrency, rng, event_rows)
        pure = [s["ms"] for s in steps_since(log, offset) if s["ctx_tok"] == 0 and s["gen_req"] == concurrency]
        rates = []
        try:
            rates, _ = ctx_slope.active_stream_stats(event_rows)
        except ValueError as error:
            errors.append(str(error))
        rows.append({"c": concurrency, "output": output, "steps": len(pure), "errors": errors,
                     "median_ms": statistics.median(rates) if len(rates) == concurrency else None,
                     "timing_method": "client_stream_all_active_mean_interval",
                     "per_request_ms": rates,
                     "engine_wait_median_ms": statistics.median(pure) if pure else None})
        print(f"decode {rows[-1]}", flush=True)
        rows[-1]["stream_events"] = event_rows
    return rows


def read_texts(paths):
    # 两次运行之间不要改这些文件，否则 token 序列不同，compare 会拒绝比较
    texts = []
    for path in paths or [HERE / "step_sim.py", HERE / "README.md", HERE / "log_evidence.py"]:
        with open(path, encoding="utf-8", errors="replace") as f:
            texts.append({"source": str(path), "text": f.read()[:TEXT_CHARS]})
    return texts


def fingerprint_one(url, model, text, tag):
    result = ctx_slope.post(url, {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1,
                                  "return_token_ids": True, "cache_salt": f"kb-fp-{tag}-{time.time()}"})
    choice = result["choices"][0]
    ids = choice.get("prompt_token_ids") or result.get("prompt_token_ids")
    table = choice.get("prompt_logprobs") or result.get("prompt_logprobs")
    if not ids or not table or len(ids) != len(table):
        raise SystemExit("响应里没有对齐的 prompt_token_ids 和 prompt_logprobs，请检查 vLLM 版本。")
    rows = []
    for position, entry in enumerate(table):
        if not entry:
            continue
        actual = entry.get(str(ids[position]))
        top = max(entry.items(), key=lambda item: item[1]["logprob"])
        rows.append([position, ids[position], actual["logprob"] if actual else None, int(top[0]), top[1]["logprob"]])
    return rows


def run_fingerprint(url, model, texts):
    out = []
    for index, item in enumerate(texts):
        rows = fingerprint_one(url, model, item["text"], f"{index}-a")
        again = fingerprint_one(url, model, item["text"], f"{index}-b")
        repeat = [abs(a[2] - b[2]) for a, b in zip(rows, again) if a[2] is not None and b[2] is not None]
        out.append({"source": item["source"], "positions": rows,
                    "repeat_max_abs": max(repeat) if repeat else None})
        print(f"fingerprint {item['source']}：{len(rows)} 个位置，同服务重复的最大差 {out[-1]['repeat_max_abs']}", flush=True)
    return out


def startup_kernels(log):
    found = collections.Counter()
    with open(log, encoding="utf-8", errors="replace") as f:
        for line in f:
            match = re.search(r"Selected (\S+) for (.+)", line)
            if match:
                found[f"{match.group(1)} -> {match.group(2).strip()}"] += 1
            elif "Iteration(" in line:
                break
    return dict(found)


def cmd_run(args):
    log = os.path.realpath(os.path.expanduser(args.server_log))
    url = args.base_url.rstrip("/") + "/completions"
    rng = random.Random(42)
    out = {"kernels": startup_kernels(log), "server_log": log, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"已选 kernel：{out['kernels']}", flush=True)
    # fingerprint 不计时，放最前面：接口不对可以马上发现，也顺便完成首次 JIT
    out["fingerprint"] = run_fingerprint(url, args.model, read_texts(args.text_file))
    out["sweep"] = run_sweep(url, args.model, log, rng)
    if any(row["steps"] < REPEATS - 1 for row in out["sweep"]):
        print("没有找到逐步日志，服务要用 LOG_ITER=1 启动。", file=sys.stderr)
        return 1
    out["prefill"] = run_prefill(url, args.model, log, rng)
    out["decode"] = run_decode(url, args.model, log, rng, args.plateau)
    if any(row["steps"] < 50 or row["errors"] or row["median_ms"] is None for row in out["decode"]):
        print("解码测量不完整，每档至少需要 50 个纯解码步且请求无错误。", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"已写入 {args.out}")
    return 0


def decode_fit(run):
    rows = [(r["c"], r["median_ms"]) for r in run["decode"] if r["median_ms"] is not None]
    return log_evidence.lstsq([[1.0, float(c)] for c, _ in rows], [ms for _, ms in rows])


def predicted(run, concurrency):
    if any(row.get("timing_method") != "client_stream_all_active_mean_interval" or row["errors"]
           or row["median_ms"] is None for row in run["decode"]):
        raise SystemExit("吞吐预测需要完整的客户端流式计时结果，请重跑旧版日志计时数据。")
    t0, slope = decode_fit(run)
    return concurrency * OSL / (concurrency * run["prefill"]["T_p"] + 511 * (t0 + slope * concurrency) / 1e3)


def simulate(run, concurrency):
    t0, slope = decode_fit(run)
    step_sim.T1 = run["prefill"]["T_p"]   # prefill_rate() 从模块变量读
    return step_sim.window(concurrency, 100, max_seqs=8, t0=t0 / 1e3, slope=slope / 1e3)


def fingerprint_stats(a, b):
    agree, total, diffs = 0, 0, []
    nll_a, nll_b = [], []
    if len(a["fingerprint"]) != len(b["fingerprint"]):
        raise SystemExit("两次运行的文本数量不同，不能比较。")
    for x, y in zip(a["fingerprint"], b["fingerprint"]):
        if len(x["positions"]) != len(y["positions"]):
            raise SystemExit("两次运行的 token 数量不同，不能比较。")
        for p, q in zip(x["positions"], y["positions"]):
            if p[:2] != q[:2]:
                raise SystemExit("两次运行的 token 序列不同，不能比较。")
            total += 1
            agree += p[3] == q[3]
            if p[2] is not None and q[2] is not None:
                diffs.append(abs(p[2] - q[2]))
                nll_a.append(-p[2])
                nll_b.append(-q[2])
    diffs.sort()
    return {"total": total, "top1_agree": agree / total, "median": statistics.median(diffs),
            "p95": diffs[int(len(diffs) * 0.95)], "nll_a": statistics.fmean(nll_a), "nll_b": statistics.fmean(nll_b)}


def cmd_compare(args):
    runs = [json.load(open(path, encoding="utf-8")) for path in (args.a, args.b)]
    a, b = runs
    for path, run in zip((args.a, args.b), runs):
        print(f"- {path}：{run['kernels']}")
    print("\n## 单次请求耗时（输出 1 token）\n| 输入 token 数 | A ms | B ms | B 相对 A |\n|---:|---:|---:|---:|")
    for x, y in zip(a["sweep"], b["sweep"]):
        if x["median_ms"] and y["median_ms"]:
            print(f"| {x['tokens']} | {x['median_ms']:.1f} | {y['median_ms']:.1f} | {y['median_ms'] / x['median_ms'] - 1:+.1%} |")
    print(f"\n## 8192 token prefill\nT_p：A {a['prefill']['T_p']:.3f}s，B {b['prefill']['T_p']:.3f}s，"
          f"B 相对 A {b['prefill']['T_p'] / a['prefill']['T_p'] - 1:+.1%}")
    print("\n## 全部请求活跃时的流式 token 间隔（输入 8192 token）\n| 并发 | A ms | B ms | B 相对 A |\n|---:|---:|---:|---:|")
    for x, y in zip(a["decode"], b["decode"]):
        if x["median_ms"] and y["median_ms"]:
            print(f"| {x['c']} | {x['median_ms']:.1f} | {y['median_ms']:.1f} | {y['median_ms'] / x['median_ms'] - 1:+.1%} |")
    print("\n## 成本模型换算的预计吞吐（放开 max-num-seqs，默认预算）\n| 并发 | A tok/s | B tok/s | B 相对 A |\n|---:|---:|---:|---:|")
    for concurrency in (1, 2, 4, 8):
        x, y = predicted(a, concurrency), predicted(b, concurrency)
        print(f"| {concurrency} | {x:.1f} | {y:.1f} | {y / x - 1:+.1%} |")
    saved = (a["prefill"]["T_p"] - b["prefill"]["T_p"]) / ISL * 1e3
    print(f"\n## 吞吐打平时的输入/输出长度比（B 每个输入 token 省 {saved:.4f} ms）")
    print("B 的 prefill 更快、解码更慢时，比值高于平衡点 B 胜出。官方负载是 16。\n"
          "| 并发 | B 每步多花 ms | 平衡点 |\n|---:|---:|---:|")
    for x, y in zip(a["decode"], b["decode"]):
        extra = y["median_ms"] - x["median_ms"]
        print(f"| {x['c']} | {extra:+.2f} | {extra / (x['c'] * saved):.1f} |" if saved > 0 else f"| {x['c']} | {extra:+.2f} | - |")
    print("\n## 步级模拟（100 个请求，max-num-seqs=8，预算 2048，相加模型，A -> B）")
    print("| 并发 | tok/s | TTFT P95 s | TPOT P95 ms | 端到端 P95 s |\n|---:|---:|---:|---:|---:|")
    for concurrency in (1, 2, 4, 8):
        x, y = simulate(a, concurrency), simulate(b, concurrency)
        print(f"| {concurrency} | {x['tok_s']:.1f} -> {y['tok_s']:.1f} | {x['ttft']:.1f} -> {y['ttft']:.1f} "
              f"| {x['tpot']:.1f} -> {y['tpot']:.1f} | {x['e2e']:.1f} -> {y['e2e']:.1f} |")
    stats = fingerprint_stats(a, b)
    print(f"\n## 真实文本的逐位置 logprob（{stats['total']} 个位置）")
    print(f"top-1 一致率 {stats['top1_agree']:.2%}；logprob 差绝对值的中位数 {stats['median']:.4f}，P95 {stats['p95']:.4f}")
    print(f"平均 NLL：A {stats['nll_a']:.4f}，B {stats['nll_b']:.4f}（越低越好，这只是低成本检查）")
    repeat = [x["repeat_max_abs"] for run in runs for x in run["fingerprint"] if x["repeat_max_abs"] is not None]
    if repeat:
        print(f"同服务重复运行的最大差：{max(repeat):.6f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--server-log", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:8088/v1")
    run.add_argument("--model", default="qwen3.8-27b-fp8")
    run.add_argument("--plateau", type=float, default=12.0, help="所有序列进入解码后还要跑多少秒")
    run.add_argument("--text-file", action="append", help="fingerprint 用的文本，可重复。不要用评测题目")
    compare = sub.add_parser("compare")
    compare.add_argument("a")
    compare.add_argument("b")
    args = ap.parse_args()
    return cmd_run(args) if args.command == "run" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
