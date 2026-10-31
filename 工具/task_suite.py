#!/usr/bin/env python3
"""三类任务小测试集：长文本、逻辑推理、代码生成。题目和判分在开始优化前固定，和评测题目无关。

  python3 工具/task_suite.py run --out ~/uclode/tasks/baseline.json
  python3 工具/task_suite.py compare ~/uclode/tasks/baseline.json ~/uclode/tasks/plan_a.json
  python3 工具/task_suite.py grade baseline.json --out baseline_graded.json

  长文本：确定性生成的虚构档案，约 8k 和 30k token，用可核对的事实问答判分
  推理：答案唯一的题目，比较最后一行的答案
  代码：隔离执行测试用例。没有可用的隔离方式时先保存回答，标记待评分，再到有隔离的机器上 grade

默认采样参数和 thinking 都跟随服务端，和 IFBench 一致。采样是随机的，所以每题重复 --repeats 次。
这个测试集只用来发现新增失败，题目太少，不能用来比较模型能力。只依赖标准库。
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

SHORT_RECORDS, LONG_RECORDS = 180, 680   # 每条约 45 token
CITIES = ["青州", "临川", "云港", "北屿", "南浔", "西陵", "东临", "平原"]
STATES = ["已完成", "进行中", "暂停"]
SURNAMES = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨"
GIVEN = ["明远", "书瑶", "景行", "若溪", "承泽", "知秋", "怀瑾", "清和", "子衿", "望舒"]
ANSWER_FORMAT = "请在回答的最后一行严格按“答案：<内容>”的格式给出结果，不要添加其他文字。"
CODE_FORMAT = "只输出一个 Python 代码块，不要写测试和使用示例，只能用 Python 标准库。"


def make_records(count, seed):
    rng = random.Random(seed)
    records, ids = [], rng.sample(range(100000, 999999), count)
    budgets = rng.sample(range(100, 99999), count)
    for number, budget in zip(ids, budgets):
        records.append({"id": f"K-{number}", "owner": rng.choice(SURNAMES) + rng.choice(GIVEN), "budget": budget,
                        "city": rng.choice(CITIES), "state": rng.choice(STATES),
                        "date": f"20{rng.randrange(10, 26)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}"})
    return records


def render(records):
    return "\n".join(f"档案 {r['id']}：负责人{r['owner']}，地点{r['city']}，预算{r['budget']}万元，"
                     f"状态{r['state']}，立项日期{r['date']}。" for r in records)


def long_tasks():
    short = make_records(SHORT_RECORDS, 20261031)
    long = make_records(LONG_RECORDS, 20261101)
    top = max(short, key=lambda r: r["budget"])
    city, state = short[7]["city"], short[7]["state"]
    count = sum(r["city"] == city and r["state"] == state for r in short)
    early, mid_a, mid_b = long[5], long[LONG_RECORDS // 2], long[LONG_RECORDS // 2 + 40]
    gap = abs(mid_a["budget"] - mid_b["budget"])
    def ask(records, question):
        return f"下面是一份虚构的档案清单。\n\n{render(records)}\n\n问题：{question}\n{ANSWER_FORMAT}"
    return [
        ("long_lookup_8k", ask(short, f"档案 {short[90]['id']} 的负责人是谁？"), short[90]["owner"]),
        ("long_max_8k", ask(short, "预算最高的档案编号是什么？"), top["id"]),
        ("long_count_8k", ask(short, f"地点是{city}且状态是{state}的档案一共有几条？只回答数字。"), str(count)),
        ("long_early_30k", ask(long, f"档案 {early['id']} 的立项日期是哪一天？按 YYYY-MM-DD 回答。"), early["date"]),
        ("long_gap_30k", ask(long, f"档案 {mid_a['id']} 和档案 {mid_b['id']} 的预算相差多少万元？只回答数字。"), str(gap)),
    ]


LOGIC_TASKS = [
    ("logic_queue", "有 5 个人排成一列，甲不能排第一个，乙必须紧挨着丙。一共有多少种排法？只回答数字。", "36"),
    ("logic_train", "一列火车 9:00 从车站出发，时速 80 公里。另一列 10:00 从同一车站沿同一方向出发，时速 100 公里。"
                    "第二列在什么时刻追上第一列？按 HH:MM 回答。", "14:00"),
    ("logic_liars", "A、B、C 三人中，每人要么总说真话，要么总说假话。A 说：“B 说假话。”B 说：“A 和 C 是同一类人。”"
                    "C 说：“A 说真话。”哪些人说真话？只回答字母，多个字母按字母顺序连写。", "B"),
    ("logic_mod", "7 的 2026 次方除以 13 的余数是多少？只回答数字。", "4"),
    ("logic_prob", "A bag has 3 red balls and 5 blue balls. Two balls are drawn without replacement. "
                   "What is the probability that they have the same color? Reply with a reduced fraction like a/b.", "13/28"),
]

CODE_TASKS = [
    ("code_brackets", "写一个函数 is_valid(s: str) -> bool，判断只含 ()[]{} 的字符串括号是否匹配，空串返回 True。",
     "assert is_valid('')\nassert is_valid('([]{})')\nassert not is_valid('(]')\nassert not is_valid('((')\nassert not is_valid('}')\n"),
    ("code_intervals", "写一个函数 merge_intervals(items: list[list[int]]) -> list[list[int]]，合并重叠或相接的闭区间，"
                       "返回按起点升序的结果，不要修改输入。",
     "x=[[8,10],[1,3],[2,6],[15,18],[18,20]]\ny=[r[:] for r in x]\nassert merge_intervals(x)==[[1,6],[8,10],[15,20]]\nassert x==y\n"
     "assert merge_intervals([])==[]\nassert merge_intervals([[1,1]])==[[1,1]]\n"),
    ("code_lru", "写一个类 LRUCache(capacity: int)，有 get(key) 和 put(key, value) 两个方法，都是 O(1)。"
                 "key 不存在时 get 返回 -1，超过容量时淘汰最久没有使用的 key，get 和 put 都算使用。",
     "c=LRUCache(2)\nc.put(1,1)\nc.put(2,2)\nassert c.get(1)==1\nc.put(3,3)\nassert c.get(2)==-1\nc.put(1,9)\nc.put(4,4)\n"
     "assert c.get(3)==-1\nassert c.get(1)==9\nassert c.get(4)==4\n"),
    ("code_topo", "写一个函数 topo_sort(n: int, edges: list[tuple[int, int]]) -> list[int]。节点是 0 到 n-1，边 (a, b) 表示 a 要排在 b 前面。"
                  "返回任意一个合法拓扑序；有环时返回空列表。",
     "o=topo_sort(4,[(0,1),(1,2),(0,3)])\nassert sorted(o)==[0,1,2,3]\nfor a,b in [(0,1),(1,2),(0,3)]:\n    assert o.index(a)<o.index(b)\n"
     "assert topo_sort(2,[(0,1),(1,0)])==[]\nassert sorted(topo_sort(3,[]))==[0,1,2]\n"),
    ("code_edit", "写一个函数 edit_distance(a: str, b: str) -> int，返回两个字符串的 Levenshtein 编辑距离。",
     "assert edit_distance('','')==0\nassert edit_distance('kitten','sitting')==3\nassert edit_distance('abc','')==3\n"
     "assert edit_distance('flaw','lawn')==2\nassert edit_distance('intention','execution')==5\n"),
]


def all_tasks():
    tasks = [{"name": n, "kind": "long", "prompt": p, "answer": a} for n, p, a in long_tasks()]
    tasks += [{"name": n, "kind": "logic", "prompt": f"{p}\n{ANSWER_FORMAT}", "answer": a} for n, p, a in LOGIC_TASKS]
    tasks += [{"name": n, "kind": "code", "prompt": f"{p}\n{CODE_FORMAT}", "tests": t} for n, p, t in CODE_TASKS]
    return tasks


def judge_answer(content, expected):
    lines = [line.strip() for line in (content or "").strip().splitlines() if line.strip()]
    if not lines:
        return False, "没有最终回答"
    match = re.search(r"答案\s*[：:]\s*(.+)$", lines[-1])
    if not match:
        return False, "最后一行不是规定格式"
    answer = match.group(1).strip().strip("`*。.").replace(" ", "")
    return answer == expected, answer


def limit_child():
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    if sys.platform == "linux":
        resource.setrlimit(resource.RLIMIT_AS, (2**31, 2**31))


def isolation():
    """返回这台机器可用的隔离方式，没有就返回 None。"""
    if sys.platform == "darwin":
        return "sandbox-exec" if shutil.which("sandbox-exec") else None
    if shutil.which("unshare") and subprocess.run(["unshare", "--net", "true"], capture_output=True).returncode == 0:
        return "unshare --net"
    return None


def judge_code(content, tests, sandbox):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", content or "", re.S)
    if not blocks:
        return False, "没有代码块"
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "solution.py"
        path.write_text(blocks[-1] + "\n" + tests, encoding="utf-8")
        command = [sys.executable, "-I", "-S", str(path)]
        if sandbox:
            if sys.platform == "darwin":
                # 本地评分禁止网络、文件写入及派生进程，并限制用户目录读取。
                prefix = json.dumps(str(Path(sys.base_prefix).resolve()))
                home = json.dumps(str(Path.home().resolve()))
                temporary = json.dumps(str(Path(folder).resolve()))
                profile = ('(version 1)(allow default)(deny network*)(deny file-write*)(deny process-fork)'
                           f'(deny file-read-data (require-all (subpath {home}) (require-not (subpath {prefix}))))'
                           '(deny file-read-data (require-all (subpath "/private/var/folders") '
                           f'(require-not (subpath {temporary}))))')
                command = ["sandbox-exec", "-p", profile] + command
            else:
                command = ["unshare", "--net"] + command
        try:
            result = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=20,
                                    preexec_fn=limit_child, env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return False, "超时"
    if result.returncode == 0:
        return True, "测试通过"
    return False, (result.stderr.strip().splitlines() or ["退出码非 0"])[-1][:200]


def judge(task, content, sandbox):
    if task["kind"] == "code":
        return judge_code(content, task["tests"], sandbox)
    return judge_answer(content, task["answer"])


def ask(url, model, task, repeat, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": task["prompt"]}], "max_tokens": max_tokens}
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            data = json.loads(response.read())
    except Exception as error:  # 记录成失败，不中断其他题
        return {"task": task["name"], "repeat": repeat, "error": str(error), "seconds": round(time.time() - started, 1)}
    choice = data["choices"][0]
    return {"task": task["name"], "repeat": repeat, "finish": choice.get("finish_reason"),
            "content": choice["message"].get("content"), "reasoning": choice["message"].get("reasoning"),
            "usage": data.get("usage"), "seconds": round(time.time() - started, 1)}


def cmd_run(args):
    sandbox = not args.no_sandbox
    if sandbox and not args.defer_code and not isolation():
        args.defer_code = True
        print("没有可用的隔离方式，代码题保存原始回答并标记待评分，之后用 grade 完成。", flush=True)
    tasks = {t["name"]: t for t in all_tasks()}
    if args.tasks:
        names = args.tasks.split(",")
        if set(names) - set(tasks):
            raise SystemExit(f"没有这些题目：{sorted(set(names) - set(tasks))}")
        tasks = {name: tasks[name] for name in names}
    url = args.base_url.rstrip("/") + "/chat/completions"
    jobs = [(task, repeat) for task in tasks.values() for repeat in range(args.repeats)]
    rows = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def save():
        # 每题写一次：时间不够要停掉时，已完成的题不会丢
        with open(args.out + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": args.model, "repeats": args.repeats,
                       "max_tokens": args.max_tokens, "rows": rows}, f, ensure_ascii=False, indent=1)
        os.replace(args.out + ".tmp", args.out)

    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        for row in pool.map(lambda job: ask(url, args.model, job[0], job[1], args.max_tokens), jobs):
            task = tasks[row["task"]]
            if row.get("error"):
                row.update(passed=False, detail=row["error"])
            elif task["kind"] == "code" and args.defer_code:
                row.update(passed=None, detail="待隔离环境评分，已保存原始输出")
            else:
                row["passed"], row["detail"] = judge(task, row["content"], sandbox)
            rows.append(row)
            save()
            status = "待评分" if row["passed"] is None else ("通过" if row["passed"] else "失败")
            print(f"{row['task']} #{row['repeat']}：{status}，{row['detail']}，"
                  f"结束原因 {row.get('finish')}，{row['seconds']}s", flush=True)
    print(f"通过 {sum(r['passed'] is True for r in rows)}/{len(rows)}，待评分 {sum(r['passed'] is None for r in rows)}，已写入 {args.out}")


def cmd_grade(args):
    method = isolation()
    if not method:
        raise SystemExit("这台机器没有可用的隔离方式（macOS sandbox-exec 或 unshare --net），不能执行模型生成的代码。")
    result = json.loads(Path(args.input).read_text())
    tasks = {t["name"]: t for t in all_tasks()}
    for row in result["rows"]:
        if row.get("passed") is None:
            task = tasks[row["task"]]
            if task["kind"] != "code":
                raise SystemExit("发现非代码题待评分，请检查记录。")
            row["passed"], row["detail"] = judge_code(row.get("content"), task["tests"], True)
            print(f"{row['task']} #{row['repeat']}：{row['passed']}，{row['detail']}")
    result["code_grading"] = {"platform": sys.platform, "python": sys.version, "isolation": method}
    with open(args.out, "x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)


def summarize(path):
    result = {}
    with open(path, encoding="utf-8") as f:
        for row in json.load(f)["rows"]:
            if row.get("passed") is None:
                raise SystemExit("存在待评分记录，请先用 grade 完成代码题评分。")
            entry = result.setdefault(row["task"], {"passed": 0, "total": 0, "truncated": 0})
            entry["passed"] += bool(row["passed"])
            entry["total"] += 1
            entry["truncated"] += row.get("finish") == "length"
    return result


def cmd_compare(args):
    a, b = summarize(args.a), summarize(args.b)
    print("| 题目 | A 通过 | B 通过 | B 截断 |\n|---|---:|---:|---:|")
    worse = []
    for name in a:
        x, y = a[name], b.get(name, {"passed": 0, "total": 0, "truncated": 0})
        print(f"| {name} | {x['passed']}/{x['total']} | {y['passed']}/{y['total']} | {y['truncated']} |")
        if y["passed"] / max(y["total"], 1) < x["passed"] / x["total"]:
            worse.append(name)
    if set(b) - set(a):
        print(f"\nA 没有这些题，未比较：{sorted(set(b) - set(a))}")
    print(f"\n合计：A {sum(v['passed'] for v in a.values())}，B {sum(b[n]['passed'] for n in a if n in b)}（只算 A 有的题）")
    print(f"B 通过率更低的题目：{worse or '无'}（每题样本很少，这些题要逐条看输出）")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:8088/v1")
    run.add_argument("--model", default="qwen3.8-27b-fp8")
    run.add_argument("--tasks", help="只跑这些题目，逗号分隔，默认全部")
    run.add_argument("--repeats", type=int, default=2)
    run.add_argument("--concurrency", type=int, default=2)
    run.add_argument("--max-tokens", type=int, default=8192)
    run.add_argument("--no-sandbox", action="store_true", help="不断网执行模型生成的代码，只在确认风险后使用")
    run.add_argument("--defer-code", action="store_true", help="保存代码题回答，稍后在本地隔离环境评分")
    grade = sub.add_parser("grade")
    grade.add_argument("input")
    grade.add_argument("--out", required=True, help="新文件，保留原始输出")
    compare = sub.add_parser("compare")
    compare.add_argument("a")
    compare.add_argument("b")
    args = ap.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "grade":
        cmd_grade(args)
    else:
        cmd_compare(args)


if __name__ == "__main__":
    main()
