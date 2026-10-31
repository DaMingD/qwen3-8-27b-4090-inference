#!/usr/bin/env python3
"""验证预测 P1：纯解码步耗时是否随上下文和并发按 t(c,L) = t0 + c*(a + b*L) 变化。

需要服务已用 LOG_ITER=1、MAX_SEQS=8 启动。只依赖标准库。
  python3 工具/ctx_slope.py --server-log ~/uclode/runs/current/server.log

每个格子同时发 c 个随机 token ID 请求。prefill 串行，所以输出要足够长，
保证所有序列都进入解码后还有约 20 秒的共同解码区间；用客户端流式 token 间隔计时。
步级日志只核对批次组成，iteration elapsed time 不代表完整 GPU 步长。
"""
import argparse
import json
import math
import os
import random
import statistics
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import log_evidence  # noqa: E402

PREFILL_TOK_S = 2367.0        # 官方基线推算，只用来估算输出长度
STEP_S = 0.04
PLATEAU_S = 20.0
PREDICTED_B = 32.8e3 / 1e12 * 1e3 * 1e3   # ms/序列/千 token，32.8 KB ÷ 1 TB/s


def post(url, body, timeout=1800):
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def run_cell(url, model, length, concurrency, rng, event_rows=None):
    output = max(512, math.ceil(((concurrency - 1) * length / PREFILL_TOK_S + PLATEAU_S) / STEP_S))
    errors = []
    prompts = [[rng.randrange(1000, 200000) for _ in range(length)] for _ in range(concurrency)]

    def worker(index):
        prompt = prompts[index]
        try:
            body = {"model": model, "prompt": prompt, "max_tokens": output, "temperature": 0,
                    "ignore_eos": True, "cache_salt": f"slope-{length}-{concurrency}-{index}-{time.time()}"}
            if event_rows is None:
                result = post(url, body)
            else:
                body.update(stream=True, return_token_ids=True, stream_options={"include_usage": True})
                request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
                events = []
                result = {}
                with urllib.request.urlopen(request, timeout=1800) as response:
                    for line in response:
                        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                            continue
                        chunk = json.loads(line[6:])
                        if chunk.get("usage"):
                            result = chunk
                        count = sum(len(c.get("token_ids") or []) for c in chunk.get("choices", []))
                        if count:
                            events.append([time.perf_counter(), count])
                event_rows[index] = events
                if sum(e[1] for e in events) != output:
                    errors.append(f"请求 {index} 流式 token ID 数量与预期不符")
            if result["usage"]["completion_tokens"] != output:
                errors.append(f"输出 {result['usage']['completion_tokens']} 个 token，期望 {output}")
        except Exception as error:  # 留给调用方统一报告
            errors.append(str(error))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return output, errors


def active_stream_stats(event_rows):
    """用所有请求同时解码的区间统计 token 间隔和平均已生成长度。"""
    if not all(event_rows):
        raise ValueError("存在没有流式 token 记录的请求")
    start = max(events[0][0] for events in event_rows) + 1.0
    end = min(events[-1][0] for events in event_rows) - 0.5
    rates, positions = [], []
    for events in event_rows:
        active = [e for e in events if start <= e[0] <= end]
        if len(active) < 50 or active[-1][0] <= active[0][0]:
            raise ValueError("所有请求同时解码的有效流式区间不足 50 个响应块")
        rates.append(1000 * (active[-1][0] - active[0][0]) / sum(e[1] for e in active[1:]))
        before = sum(e[1] for e in events if e[0] < active[0][0])
        positions.append(before + sum(e[1] for e in active) / 2)
    return rates, statistics.fmean(positions)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8088/v1")
    ap.add_argument("--model", default="qwen3.8-27b-fp8")
    ap.add_argument("--lengths", default="2048,16384,32768")
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--out", help="另存 JSON")
    args = ap.parse_args()

    log = os.path.realpath(os.path.expanduser(args.server_log))
    url = args.base_url.rstrip("/") + "/completions"
    rng = random.Random(42)
    cells = []
    for length in map(int, args.lengths.split(",")):
        for concurrency in map(int, args.concurrency.split(",")):
            offset = os.path.getsize(log)
            started = time.time()
            event_rows = [None] * concurrency
            output, errors = run_cell(url, args.model, length, concurrency, rng, event_rows)
            rates, mean_generated = [], None
            try:
                rates, mean_generated = active_stream_stats(event_rows)
            except ValueError as error:
                errors.append(str(error))
            time.sleep(1)
            with open(log, encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                steps = log_evidence.parse_steps(f.readlines())
            pure = [s["ms"] for s in steps if s["ctx_tok"] == 0 and s["gen_req"] == concurrency]
            cell = {"length": length, "concurrency": concurrency, "output": output, "steps": len(pure),
                    "median_ms": statistics.median(rates) if rates else None,
                    "engine_wait_median_ms": statistics.median(pure) if pure else None,
                    "timing_method": "client_stream_all_active_mean_interval",
                    "mean_ctx": length + mean_generated if mean_generated is not None else None,
                    "seconds": round(time.time() - started), "errors": errors}
            cells.append(cell)
            print(cell, flush=True)
            cell["stream_events"] = event_rows

    usable = [c for c in cells if c["median_ms"] is not None and c["steps"] >= 50 and not c["errors"]]
    print("\n| L | c | 纯解码步数 | 中位数 ms |\n|---:|---:|---:|---:|")
    for c in cells:
        median = f"{c['median_ms']:.1f}" if c["median_ms"] else "-"
        print(f"| {c['length']} | {c['concurrency']} | {c['steps']} | {median} |")
    fit = None
    if len(usable) >= 4:
        rows = [[1.0, float(c["concurrency"]), c["concurrency"] * c["mean_ctx"] / 1e3] for c in usable]
        fit = log_evidence.lstsq(rows, [c["median_ms"] for c in usable])
    if fit:
        t0, a, b = fit
        print(f"\n拟合：t0 = {t0:.1f} ms，a = {a:.3f} ms/序列，b = {b:.4f} ms/序列/千 token")
        print(f"预测 b = {PREDICTED_B:.4f}（判定区间 {PREDICTED_B / 2:.4f} 到 {PREDICTED_B * 2:.4f}）")
        if PREDICTED_B / 2 <= b <= PREDICTED_B * 2:
            print("结论：P1 未被否定。")
        else:
            print("结论：P1 被否定，KV 读取不是斜率的主要来源，或带宽假设不成立。")
        if b > 0:
            print(f"由 b 反推的有效带宽约 {32.8e3 / (b / 1e6) / 1e9:.0f} GB/s（只在 KV 读取假设成立时有意义）")
    else:
        print("\n可用格子不足 4 个，无法拟合。")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"cells": cells, "fit": fit, "predicted_b": PREDICTED_B}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
