#!/usr/bin/env python3
"""现场演示：连接正在运行的服务，只发请求，不改配置。

用法（在笔记本上，先把服务端口转到本机）：
  ssh -N -L 8088:127.0.0.1:8088 <服务器>
  python3 工具/demo.py all --remote "ssh <服务器>"

evidence  GPU、启动参数、实际加载的 kernel
load      官方负载形状跑一轮并发 8（输入 8192、输出 512），实时显示每个请求，和成本模型的预计值比较。
          一轮约 40 秒，不能代替 100 请求的正式结果
ask       长文本、逻辑推理、代码生成各一题，流式输出后判定。代码在隔离环境里跑测试；
          服务器上没有隔离环境，在那里运行时只生成不执行
results   打印优化前后的正式结果（交付/结果表.json）

长文本题第一次要做完整 prefill。重复运行会命中 prefix cache，耗时明显变短，这不是优化效果。
"""
import argparse
import json
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel_bench  # noqa: E402
import task_suite  # noqa: E402

CONCURRENCY = 8
DEMO_TASKS = ("long_gap_30k", "logic_train", "code_lru")
TASK_LABEL = {"long": "长文本理解", "logic": "逻辑推理", "code": "代码生成"}
# kernel 对比得到的并发 8 预计吞吐（最终思路附录 C），在 AIPerf 之前算出，不是实测
EXPECTED = {"Cutlass": 111.3, "Marlin": 90.1}
DIM, BOLD, RESET = "\x1b[2m", "\x1b[1m", "\x1b[0m"


def stream(url, body, timeout=300):
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                yield json.loads(line[6:])


def shell(args, command):
    argv = args.remote + [command] if args.remote else ["bash", "-c", command]
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=20,
                              stdin=subprocess.DEVNULL).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def loaded_kernels(args):
    return shell(args, f"grep -o '[A-Za-z0-9]*ScaledMMLinearKernel' {args.work}/runs/current/server.log | sort -u")


class Gpu:
    """后台读 nvidia-smi。拿不到时只是不显示，不影响演示。"""

    def __init__(self, args):
        self.last = None
        self.peak = 0
        query = "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
        command = ["nvidia-smi", query, "--format=csv,noheader,nounits", "-l", "1"]
        command = args.remote + [" ".join(command)] if args.remote else command
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                            stdin=subprocess.DEVNULL, text=True)
        except OSError:
            self.process = None
            return
        threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        for line in self.process.stdout:
            try:
                self.last = [float(value) for value in line.split(",")]
            except ValueError:
                continue
            self.peak = max(self.peak, self.last[1])

    def line(self):
        if not self.last:
            return "GPU 无读数"
        util, used, total, power, temperature = self.last
        return f"GPU {util:3.0f}%  显存 {used / 1024:.1f}/{total / 1024:.1f} GB  {power:.0f} W  {temperature:.0f}°C"

    def stop(self):
        if self.process:
            self.process.terminate()


def cmd_evidence(args):
    with urllib.request.urlopen(args.base_url + "/models", timeout=10) as response:
        model = json.loads(response.read())["data"][0]
    print(f"{BOLD}服务{RESET}  {model['id']}  上下文上限 {model.get('max_model_len', '未知')}")
    print(f"{BOLD}硬件{RESET}  " + (shell(args, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
                                    or "读不到，确认 --remote"))
    command = shell(args, f"cat {args.work}/runs/current/cmd.args")
    print(f"{BOLD}启动参数{RESET}\n  vllm " + (" ".join(command.split()) or "读不到"))
    print(f"{BOLD}实际 kernel{RESET}  " + (loaded_kernels(args).replace("\n", ", ") or "读不到"))


def run_request(row, url, model, prompt):
    body = {"model": model, "prompt": prompt, "max_tokens": kernel_bench.OSL, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    try:
        for chunk in stream(url, body):
            if chunk.get("choices") and row["first"] is None:
                row["first"] = time.time()
            usage = chunk.get("usage")
            row["tokens"] = usage["completion_tokens"] if usage else row["tokens"] + 1
    except (urllib.error.URLError, OSError, ValueError) as error:
        row["error"] = str(error)
    row["done"] = time.time()


def draw(rows, started, gpu, repaint):
    width = max(10, min(shutil.get_terminal_size().columns - 40, 50))
    lines = []
    for index, row in enumerate(rows, 1):
        fill = width * row["tokens"] // kernel_bench.OSL
        if row["error"]:
            state = "失败      "
        elif row["first"] is None:
            state = "等待首字  "
        else:
            state = f"TTFT {row['first'] - started:4.1f}s"
        lines.append(f"请求 {index}  {state}  {'█' * fill}{'░' * (width - fill)}  {row['tokens']:>3}/{kernel_bench.OSL}")
    total = sum(row["tokens"] for row in rows)
    elapsed = time.time() - started
    lines.append(f"{BOLD}合计 {total} token  {elapsed:5.1f} 秒  {total / max(elapsed, 1e-3):6.1f} tok/s{RESET}")
    lines.append(gpu.line())
    if repaint:
        print(f"\x1b[{len(lines)}A", end="")
    print("\n".join("\x1b[2K" + line for line in lines), flush=True)


def cmd_load(args):
    kernels = loaded_kernels(args)
    expected = next((value for name, value in EXPECTED.items() if name in kernels), None)
    print(f"{BOLD}并发 {CONCURRENCY}，每个请求输入 {kernel_bench.ISL}、输出 {kernel_bench.OSL} token，一起发出{RESET}")
    rng = random.Random()  # 每次不同的随机输入，不会命中 prefix cache
    rows = [{"first": None, "tokens": 0, "done": None, "error": None} for _ in range(CONCURRENCY)]
    prompts = [kernel_bench.random_prompt(rng, kernel_bench.ISL) for _ in rows]
    gpu = Gpu(args)
    started = time.time()
    threads = [threading.Thread(target=run_request, daemon=True,
                                args=(row, args.base_url + "/completions", args.model, prompt))
               for row, prompt in zip(rows, prompts)]
    for thread in threads:
        thread.start()
    live = sys.stdout.isatty()
    drawn = False
    while any(thread.is_alive() for thread in threads):
        if live:
            draw(rows, started, gpu, drawn)
            drawn = True
        time.sleep(0.25)
    draw(rows, started, gpu, drawn)
    gpu.stop()

    failed = [row["error"] for row in rows if row["error"]]
    if failed:
        print(f"{len(failed)} 个请求失败，这一轮不算数：{failed[0][:120]}")
        return
    elapsed = max(row["done"] for row in rows) - started
    total = sum(row["tokens"] for row in rows)
    first = sorted(row["first"] - started for row in rows)
    print(f"\n这一轮      {total / elapsed:6.1f} tok/s（{elapsed:.1f} 秒，只有一轮，不是正式结果）")
    if expected:
        print(f"成本模型    {expected:6.1f} tok/s（压测之前从 prefill 和解码步耗时算出）")
    values = read_results().get("values", {})
    if values.get("final_c8_tps"):
        print(f"正式结果    {values['final_c8_tps']:>6} tok/s（{values.get('final_c8_n', '?')} 个请求）")
    print("首字时间    " + " ".join(f"{value:.1f}" for value in first) + " 秒：prefill 排队逐个完成，这是高并发 TTFT 的主要来源")
    if gpu.peak:
        print(f"显存峰值    {gpu.peak / 1024:.1f} GB")


def show_prompt(task):
    prompt = task["prompt"]
    if len(prompt) <= 400:
        return prompt
    lines = prompt.splitlines()
    return "\n".join(lines[:2] + [f"……（共 {len(prompt)} 字符）……"] + lines[-2:])


def run_task(task, args, sandbox):
    print(f"\n{BOLD}【{TASK_LABEL[task['kind']]}】{RESET}\n{show_prompt(task)}\n")
    body = {"model": args.model, "messages": [{"role": "user", "content": task["prompt"]}], "max_tokens": 8192,
            "stream": True, "stream_options": {"include_usage": True}}
    started = time.time()
    first, content, usage = None, [], {}
    for chunk in stream(args.base_url + "/chat/completions", body):
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            text = delta.get("content")
            if first is None and (reasoning or text):
                first = time.time()
            if reasoning:
                print(DIM + reasoning + RESET, end="", flush=True)
            if text:
                content.append(text)
                print(text, end="", flush=True)
        if time.time() - started > args.time_limit:
            print(f"\n到 {args.time_limit} 秒上限，停止这一题，不判定。")
            return
    elapsed = time.time() - started
    answer = "".join(content)
    if task["kind"] == "code" and not sandbox:
        verdict = "这台机器没有隔离环境，代码没有执行"
    else:
        passed, detail = task_suite.judge(task, answer, sandbox)
        verdict = ("通过" if passed else "未通过") + f"（{detail}）"
    print(f"\n\n{BOLD}判定：{verdict}{RESET}")
    print(f"{DIM}输入 {usage.get('prompt_tokens', '?')}，输出 {usage.get('completion_tokens', '?')} token，"
          f"首字 {(first or time.time()) - started:.1f} 秒，总计 {elapsed:.1f} 秒{RESET}")


def cmd_ask(args):
    tasks = {task["name"]: task for task in task_suite.all_tasks()}
    sandbox = task_suite.isolation()
    for name in args.tasks.split(","):
        run_task(tasks[name], args, sandbox)
        pause(args, name != args.tasks.split(",")[-1])


def read_results():
    path = HERE.parent / "交付" / "结果表.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def cmd_results(args):
    results = read_results()
    values = results.get("values")
    if not values:
        print("没有 交付/结果表.json，先在本地运行 build_results.py。")
        return
    print(f"{BOLD}最终配置{RESET}  {values['final_config']}\n")
    print("并发  优化前 tok/s  最终 tok/s    变化    TTFT P95 前→后（秒）")
    for c in (1, 2, 4, 8):
        print(f"{c:>3}  {values[f'base_c{c}_tps']:>11}  {values[f'final_c{c}_tps']:>10}  {values[f'final_c{c}_gain']:>7}"
              f"    {values[f'base_c{c}_ttft']} → {values[f'final_c{c}_ttft']}")
    print(f"\nIFBench  {values['base_score']} → {values['plan_score'] if values['plan_score'] is not None else '未完成'}")
    print(f"判定     {values['verdict']}")


def pause(args, more=True):
    if more and not args.no_pause and sys.stdin.isatty():
        input(f"{DIM}（回车继续）{RESET}")


def cmd_all(args):
    for step in (cmd_evidence, cmd_load, cmd_ask, cmd_results):
        step(args)
        print()
        pause(args, step is not cmd_results)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("all", "evidence", "load", "ask", "results"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8088/v1")
    ap.add_argument("--model", default="qwen3.8-27b-fp8")
    ap.add_argument("--remote", default="", help="读 GPU 和服务日志用的命令前缀，例如 \"ssh 服务器\"；在服务器上运行时不填")
    ap.add_argument("--work", default="~/uclode")
    ap.add_argument("--tasks", default=",".join(DEMO_TASKS))
    ap.add_argument("--time-limit", type=int, default=60, help="每题最多多少秒")
    ap.add_argument("--no-pause", action="store_true")
    args = ap.parse_args()
    args.remote = shlex.split(args.remote)
    try:
        {"all": cmd_all, "evidence": cmd_evidence, "load": cmd_load, "ask": cmd_ask, "results": cmd_results}[args.command](args)
    except urllib.error.URLError as error:
        raise SystemExit(f"连不上服务 {args.base_url}：{error}。检查服务是否启动、端口转发是否还在。")


if __name__ == "__main__":
    main()
