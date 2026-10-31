#!/usr/bin/env python3
"""MTP 准入探针。这些结果只用来定位异常，不能证明无偏或认定回滚有错。

探针 A（分布检验，默认采样参数）：
  python3 工具/probe_mtp.py sample --model-dir $MODEL --n 2000 --out a_base1.jsonl   # MTP 关
  python3 工具/probe_mtp.py sample --model-dir $MODEL --n 2000 --out a_base2.jsonl   # MTP 关，测噪声底
  python3 工具/probe_mtp.py sample --model-dir $MODEL --n 2000 --out a_mtp3.jsonl    # MTP 开
  python3 工具/probe_mtp.py compare a_base1.jsonl a_base2.jsonl a_mtp3.jsonl

探针 B（确定性比对，T=0、并发 1）：
  python3 工具/probe_mtp.py greedy --model-dir $MODEL --out b_base1.jsonl    # MTP 关，跑两次
  python3 工具/probe_mtp.py greedy --model-dir $MODEL --out b_mtp3.jsonl     # MTP 开，跑两次
  python3 工具/probe_mtp.py diverge b_base1.jsonl b_base2.jsonl              # 噪声底
  python3 工具/probe_mtp.py diverge b_base1.jsonl b_mtp3.jsonl
  python3 工具/probe_mtp.py recheck b_mtp3.jsonl --out b_mtp3_recheck.jsonl  # 用 prompt_logprobs 从头重算
  python3 工具/probe_mtp.py diverge b_mtp3.jsonl b_mtp3_recheck.jsonl

流式模式下每个响应块对应一步，块长小于「深度 + 1」说明这一步发生了拒绝，
下一块的 token 就是状态恢复之后产生的。这样不用改 vLLM 也能按拒绝分组。
"""
import argparse
import collections
import concurrent.futures
import json
import math
import statistics
import urllib.request

# 自己写的问题，和评测题目无关
DEFAULT_QUESTIONS = [
    "一个水池有两根进水管，单开甲管 6 小时注满，单开乙管 9 小时注满。两管同开多久能注满？写出推理过程。",
    "Explain why quicksort has O(n log n) average complexity but O(n^2) worst-case complexity.",
    "写一个 Python 函数，判断一个字符串是否是合法的括号序列，并给出三个测试用例。",
    "Summarize the main trade-offs between write-ahead logging and shadow paging in databases.",
    "有 5 个人排队，甲不能排第一个，乙必须紧挨着丙，一共有多少种排法？",
    "Describe how a TCP connection is established and closed, including what happens if a packet is lost.",
    "比较 B+ 树和 LSM 树在写入放大、读取放大和空间放大上的差异。",
    "A train leaves at 9:00 at 80 km/h and another leaves the same station at 10:00 at 100 km/h. When does the second catch up?",
]


def load_prompts(args):
    questions = DEFAULT_QUESTIONS[: args.num_prompts]
    if args.raw:
        return [{"q": q, "prompt": q} for q in questions]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    return [{"q": q, "prompt": tokenizer.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)} for q in questions]


def post(url, body, timeout=1800):
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def stream(url, body, timeout=1800):
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                yield json.loads(line[6:])


def token_id(text):
    return int(text.rsplit(":", 1)[1])


# ---------- 探针 A ----------

def cmd_sample(args):
    """不传温度等参数，让服务端用模型默认值，和 IFBench 的请求一致。"""
    prompts = load_prompts(args)
    url = args.base_url.rstrip("/") + "/completions"

    def draw(job):
        prompt_index, draw_index = job
        result = post(url, {"model": args.model, "prompt": prompts[prompt_index]["prompt"], "max_tokens": args.positions,
                            "ignore_eos": True, "return_token_ids": True})
        return {"p": prompt_index, "i": draw_index, "t": result["choices"][0]["token_ids"]}

    jobs = [(p, i) for p in range(len(prompts)) for i in range(args.n)]
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool, open(args.out, "w") as f:
        for done, row in enumerate(pool.map(draw, jobs), 1):
            f.write(json.dumps(row) + "\n")
            if done % 500 == 0:
                print(f"{done}/{len(jobs)}", flush=True)
    print(f"已写入 {args.out}")


def conditional(rows, position, prefix):
    """给定前面的 token 等于 prefix，第 position 个 token（从 0 数）的分布。"""
    counts = collections.Counter(tuple(r["t"][:position + 1])[-1] for r in rows
                                 if len(r["t"]) > position and tuple(r["t"][:position]) == prefix)
    return counts


def compare_pair(a, b):
    """逐 token 比较两组频率，返回最大差、总变差和超过 3.5σ 的 token 数。"""
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return None
    worst, tv, flagged = 0.0, 0.0, []
    for token in set(a) | set(b):
        pa, pb = a[token] / na, b[token] / nb
        pooled = (a[token] + b[token]) / (na + nb)
        se = math.sqrt(max(pooled * (1 - pooled), 1e-12) * (1 / na + 1 / nb))
        diff = abs(pa - pb)
        tv += diff / 2
        worst = max(worst, diff)
        if pooled > 0 and diff > 3.5 * se:
            flagged.append((token, round(pa, 4), round(pb, 4)))
    half = 3.5 * math.sqrt(0.25 * (1 / na + 1 / nb))
    return {"n": (na, nb), "max_diff": worst, "tv": tv, "flagged": flagged, "detect_p05": half}


def cmd_compare(args):
    data = [[json.loads(l) for l in open(path)] for path in (args.base1, args.base2, args.test)]
    print("检出限是 p=0.5 的 token 差异在 3.5σ 下的值。小于检出限的偏差、其他上下文的偏差都看不见。")
    print("| 提示词 | 位置 | 条件样本（基线1/基线2/测试） | 噪声底 TV | 噪声底超限 | 测试 TV | 测试超限 | 检出限 |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    alarms = 0
    for prompt in sorted({r["p"] for r in data[0]}):
        rows = [[r for r in d if r["p"] == prompt] for d in data]
        prefix = ()
        for position in range(1, args.positions):
            # 第 0 个 token 来自 prefill，不经过验证。按基线1的众数固定前缀，比较后面的位置。
            prefix = tuple(collections.Counter(tuple(r["t"][:position]) for r in rows[0]).most_common(1)[0][0])
            dists = [conditional(r, position, prefix) for r in rows]
            noise = compare_pair(dists[0], dists[1])
            test = compare_pair(dists[0], dists[2])
            if not noise or not test:
                continue
            alarms += bool(test["flagged"]) and not noise["flagged"]
            print(f"| {prompt} | {position} | {noise['n'][0]}/{noise['n'][1]}/{test['n'][1]} | {noise['tv']:.3f} "
                  f"| {len(noise['flagged'])} | {test['tv']:.3f} | {len(test['flagged'])} | {test['detect_p05'] * 100:.1f}pp |")
            if test["flagged"]:
                print(f"  超限 token（基线概率，测试概率）：{test['flagged'][:5]}")
    print(f"\n超限且噪声底未超限的格子数：{alarms}")
    print("为 0 只表示没有发现超过检出限的差异；大于 0 时先增加样本复测，再定位。")


def cmd_entropy(args):
    """检查分布检验的信息量；熵不等于草稿拒绝率，也不证明重采样路径覆盖。"""
    data = [json.loads(l) for l in open(args.input)]
    if not data or len({len(r['t']) for r in data}) != 1 or len(data[0]['t']) < 2:
        raise SystemExit("输入为空、输出长度不一致或不足两个位置，无法检查熵。")
    qualified = set()
    print("| 提示词 | 位置 | 条件样本 | 支持集 | 众数概率 | 熵（bit） | 两组同样本量检出限 p=0.5 |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for prompt in sorted({r["p"] for r in data}):
        rows = [r for r in data if r["p"] == prompt]
        for position in range(1, len(rows[0]["t"])):
            prefix = tuple(collections.Counter(tuple(r["t"][:position]) for r in rows).most_common(1)[0][0])
            counts = conditional(rows, position, prefix)
            total = sum(counts.values())
            entropy = -sum(v / total * math.log2(v / total) for v in counts.values())
            detection = 3.5 * math.sqrt(0.5 / total)
            print(f"| {prompt} | {position} | {total} | {len(counts)} | {max(counts.values()) / total:.3f} | {entropy:.2f} | {detection:.1%} |")
            if entropy >= args.min_entropy and total >= args.min_samples:
                qualified.add(prompt)
    missing = sorted({r['p'] for r in data} - qualified)
    print(f"未满足熵 ≥ {args.min_entropy} bit 且条件样本 ≥ {args.min_samples} 的提示词：{missing}")
    print("此检查不证明拒绝率或重采样路径覆盖；正式比较仍要检查各组的条件样本数。")
    if args.require and missing:
        raise SystemExit(2)


# ---------- 探针 B ----------

def cmd_greedy(args):
    prompts = load_prompts(args)
    url = args.base_url.rstrip("/") + "/completions"
    with open(args.out, "w") as f:
        for index, item in enumerate(prompts):
            row = {"p": index, "t": [], "lp": [], "gap": [], "chunks": [], "prompt_ids": None}
            for chunk in stream(url, {"model": args.model, "prompt": item["prompt"], "max_tokens": args.max_tokens,
                                      "temperature": 0, "stream": True, "logprobs": 2,
                                      "return_tokens_as_token_ids": True, "return_token_ids": True,
                                      "ignore_eos": True}):
                if not chunk.get("choices"):
                    continue
                choice = chunk["choices"][0]
                if choice.get("prompt_token_ids"):
                    row["prompt_ids"] = choice["prompt_token_ids"]
                logprobs = choice.get("logprobs")
                if not logprobs or not logprobs.get("tokens"):
                    continue
                row["chunks"].append(len(logprobs["tokens"]))
                for tok, lp, top in zip(logprobs["tokens"], logprobs["token_logprobs"], logprobs["top_logprobs"]):
                    row["t"].append(token_id(tok))
                    row["lp"].append(lp)
                    values = sorted(top.values(), reverse=True)
                    row["gap"].append(values[0] - values[1] if len(values) > 1 else None)
            f.write(json.dumps(row) + "\n")
            print(f"提示词 {index}：{len(row['t'])} token，{len(row['chunks'])} 步，平均每步 "
                  f"{len(row['t']) / max(len(row['chunks']), 1):.2f}", flush=True)
    print(f"已写入 {args.out}")


def after_rejection_flags(chunks):
    """每个 token 是否位于“上一步发生拒绝”之后的那一块。块长小于最大块长视为发生了拒绝。"""
    if not chunks:
        return []
    full = max(chunks)
    flags, previous_short = [False] * chunks[0], False   # 第一块来自 prefill，不算拒绝
    for size in chunks[1:]:
        flags.extend([previous_short] * size)
        previous_short = size < full
    return flags


def cmd_diverge(args):
    left = {r["p"]: r for r in map(json.loads, open(args.left))}
    right = {r["p"]: r for r in map(json.loads, open(args.right))}
    groups = collections.defaultdict(list)
    print("| 提示词 | 长度 | 第一分歧位置 | 分歧处 top-2 差（左/右） | 共同前缀 |Δlogprob| 中位数 | 最大 |")
    print("|---:|---:|---:|---:|---:|---:|")
    diverged = 0
    for p in sorted(set(left) & set(right)):
        a, b = left[p], right[p]
        n = min(len(a["t"]), len(b["t"]))
        first = next((i for i in range(n) if a["t"][i] != b["t"][i]), None)
        end = first if first is not None else n
        chunks = a["chunks"] if len(set(a["chunks"])) > 1 else b["chunks"]
        flags = after_rejection_flags(chunks)
        deltas = []
        for i in range(end):
            if a["lp"][i] is None or b["lp"][i] is None:
                continue
            delta = abs(a["lp"][i] - b["lp"][i])
            deltas.append(delta)
            key = "拒绝之后" if i < len(flags) and flags[i] else "其他位置"
            groups[key].append(delta)
        gap = "-"
        if first is not None:
            diverged += 1
            gap = f"{a['gap'][first]} / {b['gap'][first]}" if a.get("gap") and b.get("gap") else "-"
        median = f"{statistics.median(deltas):.4f}" if deltas else "-"
        worst = f"{max(deltas):.4f}" if deltas else "-"
        print(f"| {p} | {n} | {first if first is not None else '无'} | {gap} | {median} | {worst} |")
    print(f"\n有分歧的提示词：{diverged}/{len(set(left) & set(right))}")
    for key, values in sorted(groups.items()):
        values.sort()
        print(f"{key}：n={len(values)}，|Δlogprob| 中位数 {statistics.median(values):.4f}，"
              f"P95 {values[int(0.95 * (len(values) - 1))]:.4f}")
    print("要和噪声底（同配置重复、prefill 重算）一起看。偏差明显超过对照且集中在拒绝之后，"
          "优先排查状态恢复与数值执行路径，这不足以认定回滚有错。分歧处 top-2 差很小，更可能只是数值抖动。")


def cmd_recheck(args):
    url = args.base_url.rstrip("/") + "/completions"
    with open(args.out, "w") as f:
        for row in map(json.loads, open(args.input)):
            if not row["prompt_ids"]:
                raise SystemExit("输入里没有 prompt_ids，先用 greedy 重新生成。")
            prompt = row["prompt_ids"] + row["t"]
            result = post(url, {"model": args.model, "prompt": prompt, "max_tokens": 1, "temperature": 0,
                                "prompt_logprobs": 1})
            table = result["choices"][0]["prompt_logprobs"]
            start = len(row["prompt_ids"])
            logprobs = []
            for offset, token in enumerate(row["t"]):
                entry = table[start + offset] or {}
                item = entry.get(str(token))
                logprobs.append(item["logprob"] if item else None)
            f.write(json.dumps({"p": row["p"], "t": row["t"], "lp": logprobs, "gap": [], "chunks": row["chunks"],
                                "prompt_ids": row["prompt_ids"]}) + "\n")
            print(f"提示词 {row['p']} 重算完成", flush=True)
    print(f"已写入 {args.out}")


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base-url", default="http://127.0.0.1:8088/v1")
    common.add_argument("--model", default="qwen3.8-27b-fp8")
    prompts = argparse.ArgumentParser(add_help=False)
    prompts.add_argument("--model-dir", default="/model/ModelScope/Qwen/Qwen3.8-27B-FP8")
    prompts.add_argument("--raw", action="store_true", help="不套聊天模板，只用于测试脚本")
    prompts.add_argument("--num-prompts", type=int, default=3)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample", parents=[common, prompts])
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--positions", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_sample)
    p = sub.add_parser("compare")
    p.add_argument("base1")
    p.add_argument("base2")
    p.add_argument("test")
    p.add_argument("--positions", type=int, default=3)
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("entropy")
    p.add_argument("input")
    p.add_argument("--min-entropy", type=float, default=0.5)
    p.add_argument("--min-samples", type=int, default=500)
    p.add_argument("--require", action="store_true", help="任一提示词不满足信息量门槛时返回非零")
    p.set_defaults(func=cmd_entropy)
    p = sub.add_parser("greedy", parents=[common, prompts])
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_greedy)
    p = sub.add_parser("diverge")
    p.add_argument("left")
    p.add_argument("right")
    p.set_defaults(func=cmd_diverge)
    p = sub.add_parser("recheck", parents=[common])
    p.add_argument("input")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_recheck)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
