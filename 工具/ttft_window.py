#!/usr/bin/env python3
"""检验官方表的 TTFT P95 是否来自窗口开头和小样本。对应预测 P5，见 交付/4_优化方案说明.md。

用法：python3 工具/ttft_window.py ~/uclode/eval/baseline_perf/performance

把 100 个请求按开始时间排序，只取前 n 个算 P95，和官方同样本量的值比较。
再列出窗口开头的 TTFT 和到达间隔，看开头之后请求是否仍然扎堆。
"""
import argparse
from pathlib import Path

import scan
import step_sim


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="含 c1、c2、c4、c8 的目录")
    args = ap.parse_args()
    runner = scan.load_runner()
    series = {}
    for path in sorted(args.folder.glob("c[0-9]*")):
        records = [r for r in runner.read_jsonl(path / "profile_export.jsonl")
                   if r.get("metadata", {}).get("benchmark_phase") == "profiling"]
        records.sort(key=lambda r: r["metadata"]["request_start_ns"])
        series[int(path.name[1:])] = ([r["metrics"]["time_to_first_token"]["value"] / 1e3 for r in records],
                                      [r["metadata"]["request_start_ns"] / 1e9 for r in records])

    print("| 并发 | n | 前 n 个 TTFT P95 s | 官方 s | 差 |\n|---:|---:|---:|---:|---:|")
    for (concurrency, n), official in step_sim.OFFICIAL.items():
        if concurrency in series and len(series[concurrency][0]) >= n:
            value = runner.percentile(series[concurrency][0][:n], 95)
            print(f"| {concurrency} | {n} | {value:.2f} | {official[1]:.2f} | {value / official[1] - 1:+.1%} |")

    print("\n| 并发 | 全部 P95 s | 去掉开头后 P95 s | 中位数 s | 开头 c 个 s | 到达间隔 s |\n|---:|---:|---:|---:|---|---|")
    for concurrency, (ttft, starts) in sorted(series.items()):
        rest = ttft[concurrency:]
        gaps = [round(b - a, 1) for a, b in zip(starts, starts[1:])][concurrency - 1:concurrency - 1 + 2 * concurrency]
        print(f"| {concurrency} | {runner.percentile(ttft, 95):.2f} | {runner.percentile(rest, 95):.2f} "
              f"| {runner.percentile(ttft, 50):.2f} | {[round(x, 1) for x in ttft[:concurrency]]} | {gaps} |")


if __name__ == "__main__":
    main()
