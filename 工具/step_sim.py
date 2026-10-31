#!/usr/bin/env python3
"""步级闭环模拟：解释官方基线，并在上机前写下可被否定的预测。

用法：python3 工具/step_sim.py [--json 输出路径]
只依赖标准库，不接触模型、不发请求。

负载（与官方脚本一致）：闭环 c 个客户端，每个请求输入 8192、输出 512，
一个请求结束后客户端立即发下一个。

参数来源（均来自官方基线表，属于样本内拟合）：
  T1 = 8192 token 的 prefill 总耗时 3.46 s，取自并发 1 的 TTFT
  t0, slope = 每步解码耗时 t(c) = t0 + slope*c 的截距和斜率，取自 TPOT
并发 2 以上的 TTFT 没有参与拟合。

两种步长模型，真实步长预计落在两者之间，这正是要测的问题：
  sum：步长 = 解码部分 + prefill 部分（混合执行没有共享收益，悲观）
  max：步长 = max(解码部分, prefill 部分)（读取可完全共享，乐观）

调度近似 vLLM v0.32.0：运行中的解码先占预算，余量分给 prefill。
前缀缓存开启时（默认），linear_attn 状态按 block 对齐，chunk 被切到块边界，
见 aligned_chunk()。BLOCK 是实测值，换配置后看启动日志
`kv lcm block sizes`；block=None 表示关闭前缀缓存，不做对齐。

两种统计口径：
  window()：测量窗口。c 个请求在 t=0 同时到达，总共发 n 个，P95 用线性插值。
            这和官方基线、赛方脚本的做法相同，用于和它们比较。
  steady()：稳态。丢弃前 1/4 已完成请求，用于看预算 S 的趋势。

已知局限：
  - 官方数据的 T1、t0、slope 既用于拟合又用于比较，吻合不证明机制正确。
  - 没有模拟 MTP，也没有模拟 KV 容量不足时的抢占。
"""
import argparse
import json

ISL, OSL = 8192, 512
T1 = 3.46                    # s，并发 1 下 prefill 总耗时
T0, SLOPE = 0.0306, 0.00079  # s，t(c) = T0 + SLOPE*c
DEFAULT_S = 2048
BLOCK = 1568                 # token，实测：启动日志 kv lcm block sizes（2026-10-31，基线启动）

# 官方基线（Qwen3.8-27B基准测试.md）：(并发, 请求数) -> (时长 s, TTFT P95 s, TPOT P95 ms, 端到端 P95 s)
OFFICIAL = {(1, 12): (234.32, 3.481, 31.441, 19.527), (2, 24): (286.01, 5.023, 37.909, 23.826),
            (3, 24): (220.13, 7.499, 44.974, 28.865), (4, 32): (249.08, 9.395, 51.992, 34.914),
            (8, 24): (186.76, 41.822, 53.106, 67.417), (2, 100): (1191.12, 4.608, 37.905, 23.818)}

KV_BYTES_PER_TOKEN = 32.8e3  # FP8 KV，16 层全注意力合计
BANDWIDTH = 1e12             # B/s，假设值，按峰值约 1 TB/s 估算，不是实测有效带宽
CTX_REF = ISL + OSL / 2      # 拟合官方数据时的平均上下文


def aligned_chunk(start, n, isl, block, s):
    """vllm/v1/core/sched/scheduler.py::_mamba_block_aligned_split 的简化版。

    预算达到 block 时，chunk 向下取整到块边界，预算不足一个 block 时这一步可能分不到；
    预算小于 block 时按子块推进，并在下一个块边界停一次。
    """
    if block is None or n <= 0:
        return n
    end = start + n
    if end < isl:
        aligned = end // block * block
        if aligned > start or block <= s:
            end = aligned
    next_boundary = (start // block + 1) * block
    stops = [next_boundary if start % block else 0, (isl - 1) // block * block]
    inner = [x for x in stops if start < x < end]
    if inner:
        end = min(inner)
    return max(end - start, 0)


def num_prefill_steps(s, block, isl=ISL):
    start, steps = 0, 0
    while start < isl:
        start += aligned_chunk(start, min(s, isl - start), isl, block, s)
        steps += 1
    return steps


def prefill_rate(model, block, t0, s=DEFAULT_S):
    """由并发 1 的 T1 反推 prefill 速率（tok/s）。"""
    if model == "max":
        return ISL / T1
    return ISL / (T1 - num_prefill_steps(s, block) * t0)


def step_time(model, n_dec, n_pf, rate, t0, slope):
    if model == "max":
        return max(t0 + slope * max(n_dec, 1), n_pf / rate)
    if model == "sum":
        return t0 + slope * n_dec + n_pf / rate
    raise ValueError(model)


def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p / 100
    low = int(pos)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (pos - low)


def _run(c, s, max_seqs, model, block, t0, slope, finish, issue_limit=None):
    """跑到完成 finish 个请求。返回 ([(到达序号, 端到端, TTFT, 完成时刻)], 总时长, 混合步占比)。"""
    rate = prefill_rate(model, block, t0)
    issued = c
    waiting = [{"id": i, "pf": 0, "out": OSL, "t0": 0.0, "tf": None} for i in range(c)]
    running, done = [], []
    t, mixed_steps, steps = 0.0, 0, 0
    while len(done) < finish:
        while waiting and len(running) < max_seqs:
            running.append(waiting.pop(0))
        budget, n_dec, n_pf = s, 0, 0
        for r in running:
            r["n_dec"] = 0
            if r["pf"] >= ISL and budget > 0:
                r["n_dec"] = 1
                n_dec += 1
                budget -= 1
        for r in running:
            r["n_pf"] = 0
            if r["pf"] < ISL and budget > 0:
                want = min(budget, ISL - r["pf"])
                r["n_pf"] = aligned_chunk(r["pf"], want, ISL, block, s)
                n_pf += r["n_pf"]
                budget -= r["n_pf"]
        t += step_time(model, n_dec, n_pf, rate, t0, slope)
        steps += 1
        mixed_steps += bool(n_dec and n_pf)
        for r in list(running):
            if r["n_pf"]:
                r["pf"] += r["n_pf"]
                if r["pf"] >= ISL:
                    r["tf"] = t
                    r["out"] -= 1
            elif r["n_dec"]:
                r["out"] -= 1
            if r["pf"] >= ISL and r["out"] <= 0:
                running.remove(r)
                done.append((r["id"], t - r["t0"], r["tf"] - r["t0"], t))
                if issue_limit is None or issued < issue_limit:
                    waiting.append({"id": issued, "pf": 0, "out": OSL, "t0": t, "tf": None})
                    issued += 1
    return done, t, mixed_steps / steps


def window(c, n, s=DEFAULT_S, max_seqs=4, model="sum", block=BLOCK, t0=T0, slope=SLOPE):
    """测量窗口口径。ttft_rest 去掉窗口开头同时到达的 c 个请求。"""
    done, duration, _ = _run(c, s, max_seqs, model, block, t0, slope, n, issue_limit=n)
    rest = [x[2] for x in done if x[0] >= c] or [x[2] for x in done]
    return {"tok_s": n * OSL / duration, "duration": duration,
            "ttft": percentile([x[2] for x in done], 95), "ttft_rest": percentile(rest, 95),
            "tpot": percentile([(x[1] - x[2]) / (OSL - 1) for x in done], 95) * 1e3,
            "e2e": percentile([x[1] for x in done], 95)}


def steady(c, s, max_seqs, model, block=BLOCK, t0=T0, slope=SLOPE, n=400):
    """稳态口径，返回 (吞吐 tok/s, TTFT P95 s, 混合步占比)。"""
    done, _, mix = _run(c, s, max_seqs, model, block, t0, slope, n)
    rec = done[len(done) // 4:]
    throughput = (len(rec) - 1) * OSL / (rec[-1][3] - rec[0][3])
    return throughput, percentile([x[2] for x in rec], 95), mix


def slope_at(ctx):
    """把斜率拆成“与上下文无关”和“KV 读取”两部分，后者与上下文成正比。

    b 不由官方数据拟合，由 KV 字节数和带宽假设推出，因此是样本外预测。
    """
    b = KV_BYTES_PER_TOKEN / BANDWIDTH
    a = SLOPE - b * CTX_REF
    return a + b * ctx


def reproduce_official():
    print("## 复现官方基线（窗口口径，max_seqs=4、预算 2048，样本数和官方相同）")
    print("| 并发 | 请求数 | 时长 模拟/官方 s | TTFT P95 模拟/官方 s | 误差 | TPOT P95 模拟/官方 ms | 端到端 P95 模拟/官方 s |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    rows = []
    for (c, n), (off_d, off_ttft, off_tpot, off_e2e) in OFFICIAL.items():
        w = window(c, n)
        print(f"| {c} | {n} | {w['duration']:.0f} / {off_d:.0f} | {w['ttft']:.2f} / {off_ttft:.2f} "
              f"| {(w['ttft'] / off_ttft - 1) * 100:+.0f}% | {w['tpot']:.1f} / {off_tpot:.1f} "
              f"| {w['e2e']:.1f} / {off_e2e:.1f} |")
        rows.append({"c": c, "n": n, **w})
    return rows


def predict_script():
    print("\n## 样本外预测：赛方脚本口径（100 个正式请求），相加模型")
    print("| 配置 | 并发 | tok/s | TTFT P95 s | 去掉窗口开头后 TTFT P95 s |")
    print("|---|---:|---:|---:|---:|")
    rows = []
    for label, seqs in (("官方（max_seqs=4）", 4), ("放开并发（max_seqs=8）", 8)):
        for c in (1, 2, 4, 8):
            if seqs == 8 and c < 8:
                continue
            w = window(c, 100, max_seqs=seqs)
            print(f"| {label} | {c} | {w['tok_s']:.1f} | {w['ttft']:.1f} | {w['ttft_rest']:.1f} |")
            rows.append({"label": label, "c": c, **w})
    return rows


def predict_chunk():
    print("\n## 样本外预测：放开并发上限后，预算 S 对吞吐的影响（稳态，tok/s）")
    print(f"block={BLOCK}（实测值）。")
    print("| 并发 | S | sum（悲观） | max（乐观） | 混合步占比（max） | 稳态 TTFT P95 s（sum） |")
    print("|---:|---:|---:|---:|---:|---:|")
    rows = []
    for c in (2, 4, 8):
        for s in (64, 128, 256, 512, 1024, 2048, 4096, 8192):
            lo, ttft, _ = steady(c, s, 64, "sum")
            hi, _, mix = steady(c, s, 64, "max")
            print(f"| {c} | {s} | {lo:.1f} | {hi:.1f} | {mix * 100:.0f}% | {ttft:.1f} |")
            rows.append({"c": c, "s": s, "sum": lo, "max": hi, "mix": mix, "ttft": ttft})
    return rows


def predict_alignment():
    print("\n## 对齐的影响：并发 8、放开上限（稳态 tok/s，sum / max）")
    print("| S | 对齐 prefill 步数 | 开启前缀缓存（对齐） | 关闭前缀缓存（不对齐） |")
    print("|---:|---:|---:|---:|")
    rows = []
    for s in (128, 512, 1024, 2048, 4096):
        al = (steady(8, s, 64, "sum")[0], steady(8, s, 64, "max")[0])
        no = (steady(8, s, 64, "sum", block=None)[0], steady(8, s, 64, "max", block=None)[0])
        steps = num_prefill_steps(s, BLOCK)
        print(f"| {s} | {steps} | {al[0]:.1f} / {al[1]:.1f} | {no[0]:.1f} / {no[1]:.1f} |")
        rows.append({"s": s, "steps": steps, "align": al, "none": no})
    # 预算落在 [B, B+c) 时，扣除解码后不够一个 block，prefill 可能被卡住
    trap = BLOCK + 3
    print(f"\n陷阱检查：c=8、S={trap}（在 B 到 B+c 之间）"
          f" sum 吞吐 {steady(8, trap, 64, 'sum')[0]:.1f} tok/s，"
          f"S=2048 为 {steady(8, 2048, 64, 'sum')[0]:.1f} tok/s")
    return rows


def predict_ctx():
    print("\n## 样本外预测：纯解码步耗时随上下文和并发的变化")
    print("公式：t(c, L) = t0 + c * (a + b * L)，b = 32.8 KB / 1 TB/s（带宽是假设值）")
    print("| 平均上下文 L | 斜率 ms/序列 | c=1 ms | c=4 ms | c=8 ms |")
    print("|---:|---:|---:|---:|---:|")
    rows = []
    for ctx in (2048, 8448, 16384, 32768):
        sl = slope_at(ctx)
        vals = [(T0 + c * sl) * 1e3 for c in (1, 4, 8)]
        print(f"| {ctx} | {sl * 1e3:.2f} | {vals[0]:.1f} | {vals[1]:.1f} | {vals[2]:.1f} |")
        rows.append({"ctx": ctx, "slope_ms": sl * 1e3, "t_ms": vals})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="把结果另存为 JSON")
    args = ap.parse_args()
    out = {"official": reproduce_official(), "script": predict_script(), "chunk": predict_chunk(),
           "alignment": predict_alignment(), "ctx": predict_ctx()}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
