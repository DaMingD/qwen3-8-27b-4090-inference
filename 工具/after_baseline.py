#!/usr/bin/env python3
"""等待完整基线成功后，串行运行 CUTLASS 配对确认、调度粗筛和 P1。

python3 工具/after_baseline.py --dry-run
不自动选择最终配置、不启动 IFBench。基线或 Marlin 对照异常时停止队列；CUTLASS 和粗筛是实验对象，异常只记录。
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import scan

OPTIONAL = ('cutlass_pair', 'coarse')


def validate_pair(control, candidate):
    """只用同一输入及完整原生记录判断事先固定的性能门槛。"""
    if (control / 'dataset.jsonl').read_bytes() != (candidate / 'dataset.jsonl').read_bytes():
        raise ValueError('配对输入不同，拒绝比较')
    groups = []
    for path in (control, candidate):
        rows = json.loads((path / 'summary.json').read_text())
        if len(rows) != 3 or {r['concurrency'] for r in rows} != {1, 4, 8}:
            raise ValueError('配对档位不完整')
        if any(r['status'] != '完整' or r.get('issues') or not r.get('tok_s') for r in rows):
            raise ValueError('配对结果含异常')
        groups.append({r['concurrency']: r for r in rows})
    limits = {1: -0.06, 4: 0.05, 8: 0.10}
    comparisons = []
    for c, limit in limits.items():
        a, b = groups[0][c], groups[1][c]
        ratio = b['tok_s'] / a['tok_s'] - 1
        comparisons.append(dict(concurrency=c, marlin_tok_s=a['tok_s'], cutlass_tok_s=b['tok_s'],
                                relative_change=ratio, minimum_change=limit,
                                passed=ratio >= limit or math.isclose(ratio, limit, abs_tol=1e-12)))
    return dict(comparisons=comparisons, performance_passed=all(r['passed'] for r in comparisons),
                quality='未验收；性能通过也不自动选为最终配置',
                dataset_sha256=hashlib.sha256((control / 'dataset.jsonl').read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', default=os.path.expanduser('~/uclode'))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--wait-seconds', type=int, default=7200)
    args = ap.parse_args()
    work = Path(args.work).resolve()
    here = Path(__file__).resolve().parent
    python = str(work / 'venv-vllm/bin/python')
    official = '/model/ModelScope/Qwen/Qwen3.8-27B-FP8'
    channel = str(work / 'models/Qwen3.8-27B-FP8-channel')
    root = work / 'scans' / (time.strftime('%Y%m%d_%H%M%S') + '_post_baseline')
    env = dict(os.environ, WORK=str(work), MODEL_DIR=official, MAX_SEQS='8', BUDGET='2048',
               MTP_DEPTH='0', ADAPTIVE='0', NO_PREFIX='0', EXTRA='', LINEAR_BACKEND='',
               LOG_ITER='0', LOG_REQ='0', VLLM_LOGGING_LEVEL='INFO')
    env['PATH'] = f"{work}/venv-vllm/bin:{work}/venv-eval-bootstrap/bin:" + env['PATH']
    serve = ['bash', str(here / 'serve.sh')]
    common = [python, str(here / 'scan.py'), '--work', str(work), '--requests', '24']
    paired = ['--concurrency', '1', '--concurrency', '4', '--concurrency', '8', '--log-iterations', '0']
    jobs = [
        ('marlin_pair', common + paired + ['--tag', 'marlin_pair', '--model-dir', official,
          '--profile', 's=2048,seqs=8', '--output-dir', str(root / 'marlin_pair')]),
        ('cutlass_pair', common + paired + ['--tag', 'cutlass_pair', '--model-dir', channel,
          '--profile', 's=2048,seqs=8,backend=cutlass', '--output-dir', str(root / 'cutlass_pair')]),
        ('coarse', common + ['--tag', 'coarse', '--model-dir', official, '--concurrency', '8',
          '--log-iterations', '1', '--profile', 's=2048,seqs=8', '--profile', 's=512,seqs=8',
          '--profile', 's=128,seqs=8', '--output-dir', str(root / 'coarse')]),
        ('slope', [python, str(here / 'ctx_slope.py'), '--server-log', str(work / 'runs/current/server.log'),
          '--out', str(root / 'ctx_slope.json')]),
    ]
    plan = dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                baseline_exit=str(work / 'eval/baseline_perf.exit'),
                revised_cutlass_thresholds={'1': -0.06, '4': 0.05, '8': 0.10},
                matched_settings='24 正式 + 8 预热，同种子同输入，seqs=8，预算2048，INFO，LOG_ITER=0，LOG_REQ=0',
                jobs=jobs)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    lock = open(work / 'eval/post_baseline.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    (root / 'script_hashes.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                       for p in here.glob('*.py')}, indent=2))
    (work / 'eval/post_baseline_location.txt').write_text(str(root) + '\n')
    print(f'等待基线成功完成，队列结果：{root}', flush=True)
    deadline = time.monotonic() + args.wait_seconds
    marker = Path(plan['baseline_exit'])
    while not marker.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('等待基线超时；没有停止服务或发起新请求')
        time.sleep(10)
    if marker.read_text().strip() != '0':
        raise RuntimeError('基线失败；保留现场，不执行后续实验')
    runner = scan.load_runner()
    for c in (1, 2, 4, 8):
        result = runner.parse_performance(work / f'eval/baseline_perf/performance/c{c}', 100, c,
                                          warmup_expected=8, strict_isl=True)
        if result['status'] != '完整':
            raise RuntimeError(f'基线 c{c} 不完整：{result["issues"]}')
    print('基线四档完整性检查通过，开始后续实验', flush=True)
    subprocess.run(serve + ['stop'], env=env, check=True)
    failed = []
    for name, command in jobs:
        print(f'开始 {name}：{time.strftime("%H:%M:%S")}', flush=True)
        if name == 'slope':
            slope_env = dict(env, NAME='slope', LOG_ITER='1')
            subprocess.run(serve + ['start'], env=slope_env, check=True)
        with open(root / f'{name}.log', 'w') as log:
            # 跑不通也是结果，不能挡住后面的实验
            done = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=name not in OPTIONAL)
        if done.returncode:
            failed.append(name)
            print(f'{name} 有异常结果，详情看 {name}.log 和 summary.json；继续后面的实验', flush=True)
            subprocess.run(serve + ['stop'], env=env, check=True)
            continue
        if name == 'cutlass_pair':
            try:
                result = validate_pair(root / 'marlin_pair', root / 'cutlass_pair')
            except ValueError as error:
                failed.append(name)
                print(f'配对无法判定：{error}；继续后面的实验', flush=True)
                continue
            (root / 'cutlass_gate.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(json.dumps(result, ensure_ascii=False), flush=True)
        if name == 'slope':
            result = json.loads((root / 'ctx_slope.json').read_text())
            if len(result['cells']) != 9 or any(r['errors'] or r['median_ms'] is None for r in result['cells']):
                raise RuntimeError('P1 格子不完整，请检查 ctx_slope.json')
        print(f'完成 {name}', flush=True)
    (root / 'complete.txt').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z') + f'\n异常步骤：{failed}\n')
    print(f'队列完成，异常步骤：{failed}；服务保留 Marlin、seqs=8、预算2048、LOG_ITER=1；正式计时需重新关闭逐步日志。', flush=True)


if __name__ == '__main__':
    main()
