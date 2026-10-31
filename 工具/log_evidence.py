#!/usr/bin/env python3
"""从 vLLM 服务日志里提取证据。只依赖标准库，只读日志。

  startup LOG    版本、选中的 kernel、block 大小、KV 容量、状态缓存模式、默认采样参数
  sampling LOG   服务端解析后的采样参数（需要 LOG_REQ=1）
  steps LOG      每一步的 token 数和 future 等待时间（需要 LOG_ITER=1）：context token 总数、纯解码和混合步计数、等待时间

日志格式对应 vLLM v0.32.0。没找到的项目会明确写“未找到”，不当作通过。
"""
import argparse
import collections
import json
import re
import statistics
import sys

ITER_RE = re.compile(
    r"Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, "
    r"(\d+) generation requests, (\d+) generation tokens, "
    r"iteration elapsed time: ([\d.]+) ms( \(dummy\))?")
REQ_RE = re.compile(r"Received request (\S+): params: (.*?), lora_request:")
PARAM_NAMES = ("temperature", "top_p", "top_k", "min_p", "seed", "max_tokens",
               "repetition_penalty", "presence_penalty", "frequency_penalty", "ignore_eos")


def read_lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.readlines()


def startup(path):
    lines = read_lines(path)
    kernels = collections.Counter()
    found = collections.defaultdict(list)
    patterns = {
        "版本": r"Initializing a V1 LLM engine \(v([^)]+)\)",
        "block（各组）": r"kv cache group sizes (.*)",
        "block（调度用）": r"kv lcm block sizes (\d+)",
        "KV 容量": r"KV cache size: (.*)",
        "状态缓存模式": r"Mamba cache mode is set to '(\w+)'",
        "默认采样参数": r"Default vLLM sampling parameters have been overridden by (.*?)If this is",
    }
    for line in lines:
        match = re.search(r"Selected (\S+) for (.+)", line)
        if match:
            kernels[(match.group(1).strip(), re.sub(r"\s+", " ", match.group(2)).strip())] += 1
        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match and match.group(1).strip() not in found[key]:
                found[key].append(match.group(1).strip())
        gdn = re.search(r"Using .*(?:GDN|gated.?delta).*(?:backend|kernel).*", line, re.I)
        if gdn:
            text = gdn.group(0).strip()
            if text not in found["线性注意力后端"]:
                found["线性注意力后端"].append(text)
        if re.search(r"\badaptive verification\b", line, re.I):
            text = line.strip()[-160:]
            if text not in found["自适应验证"]:
                found["自适应验证"].append(text)
    print("## 启动证据")
    keys = list(patterns) + ["线性注意力后端"]
    for key in keys:
        print(f"- {key}：{' | '.join(found[key]) if found[key] else '未找到'}")
    if found["自适应验证"]:
        print(f"- 自适应验证：{' | '.join(found['自适应验证'][:3])}")
    print("- 选中的 kernel：" + ("未找到" if not kernels else ""))
    for (kernel, target), count in sorted(kernels.items(), key=lambda x: -x[1]):
        print(f"  - {kernel} 用于 {target}（{count} 次）")
    errors = [l.strip() for l in lines if re.search(r"\b(ERROR|CRITICAL|Traceback)\b", l)]
    print(f"- 错误行数：{len(errors)}")
    for line in errors[:5]:
        print(f"  - {line[:200]}")


def parse_params(text):
    out = {}
    for name in PARAM_NAMES:
        match = re.search(rf"\b{name}=([^,)\s]+)", text)
        out[name] = match.group(1) if match else "未出现"
    return out


def sampling(path, expect, where):
    combos = collections.Counter()
    key, _, value = (where or "").partition("=")
    for line in read_lines(path):
        match = REQ_RE.search(line)
        if match:
            params = parse_params(match.group(2))
            if not where or params.get(key) == value:
                combos[tuple(sorted(params.items()))] += 1
    print("## 服务端解析后的采样参数" + (f"（只看 {where}）" if where else ""))
    if not combos:
        print("未找到符合条件的 Received request 行。启动时要设 LOG_REQ=1，而且要等请求发进来。")
        return 1
    print(f"请求数：{sum(combos.values())}，不同组合：{len(combos)}")
    status = 0
    for combo, count in combos.most_common():
        params = dict(combo)
        print(f"- {count} 个请求：" + "，".join(f"{k}={v}" for k, v in params.items() if v != "未出现"))
        for pair in filter(None, (expect or "").split(",")):
            key, value = pair.split("=")
            if key not in params or params[key] == "未出现":
                print(f"  无法核对 {key}：日志里没有这个字段")
                status = 1
            elif float(params[key]) != float(value):
                print(f"  与预期不符：{key}={params[key]}，预期 {value}")
                status = 1
    if expect and status == 0:
        print(f"全部符合预期：{expect}")
    return status


def parse_steps(lines):
    steps = []
    for line in lines:
        match = ITER_RE.search(line)
        if match and not match.group(7):
            steps.append({"ctx_req": int(match.group(2)), "ctx_tok": int(match.group(3)),
                          "gen_req": int(match.group(4)), "gen_tok": int(match.group(5)),
                          "ms": float(match.group(6))})
    return steps


def lstsq(rows, ys):
    """最小二乘（正规方程 + 高斯消元），避免依赖 numpy。"""
    n = len(rows[0])
    a = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(n)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda k: abs(a[k][i]))
        if abs(a[pivot][i]) < 1e-12:
            return None
        a[i], a[pivot] = a[pivot], a[i]
        b[i], b[pivot] = b[pivot], b[i]
        for k in range(i + 1, n):
            factor = a[k][i] / a[i][i]
            for j in range(i, n):
                a[k][j] -= factor * a[i][j]
            b[k] -= factor * b[i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(a[i][j] * x[j] for j in range(i + 1, n))) / a[i][i]
    return x


def analyze_steps(steps):
    """返回步数、token 数和日志等待时间，不从等待时间推断 GPU 步长。"""
    result = {"total": len(steps)}
    if not steps:
        return result
    decode = [s for s in steps if s["ctx_tok"] == 0 and s["gen_req"] > 0]
    prefill = [s for s in steps if s["ctx_tok"] > 0 and s["gen_req"] == 0]
    mixed = [s for s in steps if s["ctx_tok"] > 0 and s["gen_req"] > 0]
    result.update(decode=len(decode), prefill=len(prefill), mixed=len(mixed))
    result["chunks"] = collections.Counter(s["ctx_tok"] for s in steps if s["ctx_tok"] > 0).most_common(8)

    groups = collections.defaultdict(list)
    for s in decode:
        groups[s["gen_req"]].append(s["ms"])
    result["decode_medians"] = {k: statistics.median(v) for k, v in sorted(groups.items()) if len(v) >= 5}
    result["timing_scope"] = "future_result_wait_only"
    result["timing_note"] = "日志只覆盖 future.result() 等待区间，不能据此拟合完整步长或共享系数。"
    return result


def print_steps(result):
    print("## 逐步日志")
    if not result["total"]:
        print("未找到 Iteration 行。启动时要设 LOG_ITER=1。")
        return 1
    print(f"步数：{result['total']}（纯解码 {result['decode']}，纯 prefill {result['prefill']}，混合 {result['mixed']}）")
    print("每步 context token 总数（token 数：次数）：" + "，".join(f"{k}：{v}" for k, v in result["chunks"]))
    if result.get("decode_medians"):
        print("纯解码日志等待时间中位数（解码序列数：ms）：" + "，".join(f"{k}：{v:.1f}" for k, v in result["decode_medians"].items()))
    print(result.get("timing_note", "请核对计时区间后解释耗时。"))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("startup", "sampling", "steps"))
    ap.add_argument("log")
    ap.add_argument("--expect", help="sampling：逐项核对，如 temperature=1.0,top_k=20,top_p=0.95")
    ap.add_argument("--where", help="sampling：只看某类请求，如 max_tokens=32768（IFBench）")
    ap.add_argument("--json", help="steps：另存解析结果")
    args = ap.parse_args()
    if args.command == "startup":
        startup(args.log)
        return 0
    if args.command == "sampling":
        return sampling(args.log, args.expect, args.where)
    result = analyze_steps(parse_steps(read_lines(args.log)))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    return print_steps(result)


if __name__ == "__main__":
    sys.exit(main())
