#!/usr/bin/env python3
"""从原始结果生成唯一的结果表和结论，所有交付材料的数字都从这里来。

  python3 工具/build_results.py --night 实测结果/20261101_夜间

读不到的结果一律标成未完成，不沿用估算。交付/模板 里的 {{键}} 替换后写到 交付/（README 写到仓库根目录），未知的键直接报错。
需要赛方脚本和原始结果包，所以只在本地运行，干净克隆里要先把它们放好。
质量判定的规则见 交付/4_优化方案说明.md，在方案 A 的 IFBench 出结果之前固定。
MTP 的结论要人看过才能写，放在 --manual 指定的 JSON 里，没有就写未完成。

夜间目录的结构（和服务器 ~/uclode 下的名字相同）：
  eval/plan_a_perf  eval/plan_a_accuracy  eval/baseline_accuracy_b8  scans/*fallback_c8*
  tasks/plan_a_graded.json  repro/repro_build.json  repro/repro_verify.json
"""
import argparse
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ifbench_tool  # noqa: E402
import scan  # noqa: E402

ROOT = HERE.parent
BASE = ROOT / "实测结果/20261031_完整基线与配对"
QUALITY = ROOT / "实测结果/20261031_基线质量完整"
RETRIED = [35, 81, 87]  # 基线里经过超时重采的题，见 交付/3_Benchmark测试报告.md
LIMITS = {1: -0.06, 4: 0.05, 8: 0.10}
NET_LIMIT = -3
UNFINISHED = "未完成"
if not scan.RUNNER.exists():
    raise SystemExit("需要赛方脚本（参赛资料/eval/）和原始结果包（实测结果/），见 交付/证据索引.md。")
runner = scan.load_runner()


def perf_cell(folder, concurrency, requests):
    if not (folder / "profile_export_aiperf.json").exists():
        return None
    result = runner.parse_performance(folder, requests, concurrency, warmup_expected=runner.WARMUP, strict_isl=True)
    ttft = scan.ttft_split(folder / "profile_export.jsonl", concurrency, runner)
    metrics = result["metrics"]
    return dict(status=result["status"], requests=requests, tok_s=result["output_token_throughput"],
                ttft_p95=ttft["all_p95_s"], ttft_rest_p95=ttft["rest_p95_s"],
                tpot_p95=metrics["inter_token_latency"]["p95"], e2e_p95=metrics["request_latency"]["p95"] / 1e3)


def performance(night):
    pair = next(BASE.glob("scans/*post_baseline"))
    official = BASE / "eval/baseline_perf/performance"
    marlin = pair / "marlin_pair/marlin_pair_s2048_seqs8"
    cutlass = pair / "cutlass_pair/cutlass_pair_s2048_seqs8_backendcutlass"
    table = {"baseline": {}, "fallback": {}, "plan_a": {}}
    for c in (1, 2, 4, 8):
        table["baseline"][c] = perf_cell(official / f"c{c}", c, 100)
        # 并发不超过 4 时 max-num-seqs 不起作用，配对实验里和基线只差 0.2% 以内，所以退路只补并发 8
        table["fallback"][c] = table["baseline"][c] if c < 8 else perf_cell(marlin / "c8", 8, 24)
        full = perf_cell(night / f"eval/plan_a_perf/performance/c{c}", c, 100) if night else None
        table["plan_a"][c] = full or perf_cell(cutlass / f"c{c}", c, 24)
    if night:
        for folder in sorted(night.glob("scans/*fallback_c8*/*/c8")):
            table["fallback"][8] = perf_cell(folder, 8, 100) or table["fallback"][8]
    return table


def fmt(value, spec):
    return UNFINISHED if value is None else format(value, spec)


def perf_markdown(table):
    names = {"baseline": "官方配置（Marlin，seqs=4）", "fallback": "退路（Marlin，seqs=8）", "plan_a": "方案 A（CUTLASS，seqs=8）"}
    lines = ["| 配置 | 并发 | 请求数 | tok/s | TTFT P95（秒） | 去掉开头后（秒） | TPOT P95（ms） | 端到端 P95（秒） | 状态 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for key, name in names.items():
        for c, cell in table[key].items():
            cell = cell or {}
            lines.append(f"| {name} | {c} | {cell.get('requests', '—')} | {fmt(cell.get('tok_s'), '.2f')} | {fmt(cell.get('ttft_p95'), '.2f')} "
                         f"| {fmt(cell.get('ttft_rest_p95'), '.2f')} | {fmt(cell.get('tpot_p95'), '.1f')} | {fmt(cell.get('e2e_p95'), '.1f')} "
                         f"| {cell.get('status', UNFINISHED)} |")
    return "\n".join(lines)


def scores(folder):
    try:
        reviews = ifbench_tool.read_records(folder, "reviews")
    except SystemExit:
        return None
    values = {i: row["sample_score"]["score"]["value"].get(ifbench_tool.METRIC) for i, row in reviews.items()}
    return {i: bool(v) for i, v in values.items() if v is not None}


def paired(base, other, exclude=()):
    common = sorted((set(base) & set(other)) - set(exclude))
    lost = [i for i in common if base[i] and not other[i]]
    won = [i for i in common if other[i] and not base[i]]
    return dict(questions=len(common), base=sum(base[i] for i in common), other=sum(other[i] for i in common),
                lost=lost, won=won, p=ifbench_tool.sign_test(len(lost), len(won)))


def accuracy_summary(folder):
    path = folder / "accuracy/summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def tasks_passed(path):
    if not path or not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    if any(row.get("passed") is None for row in rows):
        return None
    result = {}
    for row in rows:
        entry = result.setdefault(row["task"], [0, 0])
        entry[0] += bool(row["passed"])
        entry[1] += 1
    return result


def judge(table, night, explained):
    """返回判定和每个条件。IFBench 没有完整结果就是未完成，不做质量结论。"""
    base_dir = QUALITY / "eval/baseline_accuracy_resume"
    base_scores = scores(base_dir / "accuracy")
    summary = accuracy_summary(night / "eval/plan_a_accuracy") if night else None
    info = dict(base_score=sum(base_scores.values()), plan_score=None, pair=None, pair_excluded=None, conditions={}, truncated=None, loops=None)
    if not summary or summary.get("status") != "完整":
        return UNFINISHED, info
    folder = night / "eval/plan_a_accuracy/accuracy"
    plan_scores = scores(folder)
    if not plan_scores or len(plan_scores) != 100:
        return UNFINISHED, info
    pair = paired(base_scores, plan_scores)
    rows = [ifbench_tool.inspect_prediction(r) for r in ifbench_tool.read_records(folder, "predictions").values()]
    truncated = sum(r["stop"] in ("length", "max_tokens", "model_length") for r in rows)
    loops = sum(bool(r["period"]) for r in rows)
    info["plan_loose"] = summary["metrics"]["prompt_level_loose"]["score"]
    info.update(plan_score=pair["other"], pair=pair, pair_excluded=paired(base_scores, plan_scores, RETRIED), truncated=truncated, loops=loops)
    conditions = {"净退化不超过 3 题": pair["other"] - pair["base"] >= NET_LIMIT,
                  "对→错没有显著多于错→对": not (len(pair["lost"]) > len(pair["won"]) and pair["p"] < 0.05),
                  "没有新增截断": truncated == 0, "没有新增死循环": loops == 0}
    for c, limit in LIMITS.items():
        base = table["fallback"][c]["tok_s"]
        cell = table["plan_a"][c]
        conditions[f"并发 {c} 吞吐变化不低于 {limit:+.0%}"] = cell["requests"] == 100 and cell["tok_s"] / base - 1 >= limit
    base_tasks = tasks_passed(QUALITY / "tasks/baseline_graded.json")
    plan_tasks = tasks_passed(night / "tasks/plan_a_graded.json")
    info["tasks"] = (sum(v[0] for v in plan_tasks.values()), sum(v[1] for v in plan_tasks.values())) if plan_tasks else None
    if plan_tasks is None:
        info["conditions"] = conditions
        return ("未通过" if not all(conditions.values()) else "待小测试集"), info
    worse = [n for n, v in base_tasks.items() if n in plan_tasks and plan_tasks[n][0] / plan_tasks[n][1] < v[0] / v[1] and n not in explained]
    info["new_task_failures"] = worse
    conditions["小测试集没有未解释的新增失败"] = not worse
    info["conditions"] = conditions
    return ("通过" if all(conditions.values()) else "未通过"), info


def noise(night):
    folder = night / "eval/baseline_accuracy_b8" if night else None
    summary = accuracy_summary(folder) if folder else None
    if not summary or summary.get("status") != "完整":
        return "噪声底未完成。"
    base = scores(QUALITY / "eval/baseline_accuracy_resume/accuracy")
    pair = paired(base, scores(folder / "accuracy"))
    return (f"基线重复组（batch 8，和 batch 2 的基线条件不同，基线还有 3 题经过超时重采）得分 {pair['other']}/100，"
            f"对→错 {len(pair['lost'])} 题、错→对 {len(pair['won'])} 题。它只说明这些条件下的翻转范围，不是严格的同配置随机波动。")


def repro(night):
    def load(name):
        path = night / "repro" / name if night else None
        return json.loads(path.read_text()) if path and path.exists() else None
    def state(result):
        if not result:
            return UNFINISHED
        if not result["passed"]:
            return f"未通过（{'、'.join(result.get('failed', []))}）"
        if result.get("recovery"):
            return "首次未通过（权重哈希一致，依赖清单有版本漂移），按锁定清单修复后通过，这不是第二次从零构建"
        return "通过"
    def clean_install():
        path = night / "repro/clean_install.exit" if night else None
        if not path or not path.exists():
            return ""
        return "；修复后的安装脚本在空目录重新安装一次：" + ("通过" if path.read_text().strip() == "0" else "未通过")
    build, verify = load("repro_build.json"), load("repro_verify.json")
    parts = ["一级（从零安装、重新转换权重并比对权重哈希和依赖清单，不含 GPU 算子）：" + state(build) + clean_install(),
             "二级（GPU 算子验收、上机启动、kernel、吞吐相差不超过 5%、赛方流程冒烟）：" + state(verify)]
    return "；".join(parts) + "。这是我们的自查，不是赛方标准；评测工具环境没有重新安装。"


def quality_markdown(verdict, info, base_tasks_text, base_summary):
    metrics = base_summary["metrics"]
    pair, excluded = info["pair"], info["pair_excluded"]
    lines = ["| 项目 | 基线 | 方案 A |", "|---|---:|---:|",
             f"| IFBench 严格提示词（比赛指标） | {info['base_score']}/100 | {info['plan_score'] if info['plan_score'] is not None else UNFINISHED}{'/100' if info['plan_score'] is not None else ''} |",
             f"| IFBench 宽松提示词 | {metrics['prompt_level_loose']['score'] * 100:.0f}/100 | {UNFINISHED if pair is None else f"{info['plan_loose'] * 100:.0f}/100"} |",
             f"| 小测试集 | {base_tasks_text} | {f'{info['tasks'][0]}/{info['tasks'][1]}' if info.get('tasks') else UNFINISHED} |"]
    if pair:
        lines += ["", f"配对：对→错 {len(pair['lost'])} 题 {pair['lost']}，错→对 {len(pair['won'])} 题 {pair['won']}，双侧符号检验 p = {pair['p']:.3f}。",
                  f"排除基线超时重采的 3 题后：{excluded['base']}/{excluded['questions']} → {excluded['other']}/{excluded['questions']}，对→错 {len(excluded['lost'])}，错→对 {len(excluded['won'])}，p = {excluded['p']:.3f}。",
                  f"方案 A 截断 {info['truncated']} 题，尾部重复 {info['loops']} 题。"]
    if info["conditions"]:
        lines += ["", "| 条件 | 结果 |", "|---|---|"] + [f"| {k} | {'满足' if v else '不满足'} |" for k, v in info["conditions"].items()]
    return "\n".join(lines)


def render(text, values, name):
    def replace(match):
        if match.group(1) not in values:
            raise SystemExit(f"{name} 里有未知的键：{match.group(1)}")
        return str(values[match.group(1)])
    return re.sub(r"\{\{(\w+)\}\}", replace, text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--night", type=Path)
    ap.add_argument("--manual", type=Path, default=ROOT / "交付/手写结论.json")
    ap.add_argument("--explained", default="", help="已逐条解释的小测试集题目，逗号分隔")
    ap.add_argument("--out", type=Path, default=ROOT / "交付")
    args = ap.parse_args()
    night = args.night if args.night and args.night.exists() else None
    explained = set(filter(None, args.explained.split(",")))
    table = performance(night)
    verdict, info = judge(table, night, explained)
    base_summary = accuracy_summary(QUALITY / "eval/baseline_accuracy_resume")
    base_tasks = tasks_passed(QUALITY / "tasks/baseline_graded.json")
    base_tasks_text = f"{sum(v[0] for v in base_tasks.values())}/{sum(v[1] for v in base_tasks.values())}"
    manual = json.loads(args.manual.read_text(encoding="utf-8")) if args.manual.exists() else {}
    final = table["plan_a"] if verdict == "通过" else table["fallback"]
    base = table["baseline"]
    values = {
        "verdict": verdict,
        "final_config": "CUTLASS per-channel FP8 + max-num-seqs=8 + 预算 2048" if verdict == "通过" else "Marlin 官方 FP8 + max-num-seqs=8 + 预算 2048",
        "perf_table": perf_markdown(table),
        "quality_table": quality_markdown(verdict, info, base_tasks_text, base_summary),
        "noise_sentence": noise(night),
        "repro_sentence": repro(night),
        "mtp_sentence": manual.get("mtp_sentence", "MTP 分析未完成，不下结论，最终配置不开 MTP。"),
        "base_score": info["base_score"], "plan_score": info["plan_score"] if info["plan_score"] is not None else UNFINISHED,
        "base_tasks": base_tasks_text, "plan_tasks": f"{info['tasks'][0]}/{info['tasks'][1]}" if info.get("tasks") else UNFINISHED,
    }
    for c in (1, 2, 4, 8):
        for prefix, source in (("base", base), ("fallback", table["fallback"]), ("plan", table["plan_a"]), ("final", final)):
            cell = source[c] or {}
            values[f"{prefix}_c{c}_tps"] = fmt(cell.get("tok_s"), ".1f")
            values[f"{prefix}_c{c}_ttft"] = fmt(cell.get("ttft_p95"), ".1f")
            values[f"{prefix}_c{c}_rest"] = fmt(cell.get("ttft_rest_p95"), ".1f")
            values[f"{prefix}_c{c}_e2e"] = fmt(cell.get("e2e_p95"), ".1f")
            values[f"{prefix}_c{c}_tpot"] = fmt(cell.get("tpot_p95"), ".1f")
            values[f"{prefix}_c{c}_n"] = cell.get("requests", "—")
        values[f"final_c{c}_gain"] = f"{final[c]['tok_s'] / base[c]['tok_s'] - 1:+.1%}" if final[c] else UNFINISHED
        values[f"fallback_c{c}_gain"] = f"{table['fallback'][c]['tok_s'] / base[c]['tok_s'] - 1:+.1%}" if table["fallback"][c] else UNFINISHED
        values[f"plan_c{c}_gain"] = f"{table['plan_a'][c]['tok_s'] / base[c]['tok_s'] - 1:+.1%}" if table["plan_a"][c] else UNFINISHED
    templates = args.out / "模板"
    # 三种结论各一段，这里只选一段，其他材料引用 {{conclusion}}
    key = "未通过" if verdict == "未通过" else ("通过" if verdict == "通过" else UNFINISHED)
    values["unfinished_reason"] = "小测试集" if verdict == "待小测试集" else "IFBench"
    values["conclusion"] = render((templates / f"结论_{key}.md").read_text(encoding="utf-8"), values, key)
    args.out.mkdir(exist_ok=True)
    (args.out / "结果表.json").write_text(json.dumps(dict(values=values, info=info, performance=table), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for path in sorted(templates.glob("*.md")):
        if path.name.startswith("结论_"):
            continue
        # README 的链接按仓库根目录写
        target = args.out.parent if path.name == "README.md" else args.out
        (target / path.name).write_text(render(path.read_text(encoding="utf-8"), values, path.name), encoding="utf-8")
    print(f"判定：{verdict}。结果表和材料已写入 {args.out}")
    for name, passed in info["conditions"].items():
        print(f"  {'满足' if passed else '不满足'}：{name}")


if __name__ == "__main__":
    main()
