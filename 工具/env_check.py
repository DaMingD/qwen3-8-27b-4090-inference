#!/usr/bin/env python3
"""检查独立环境的版本和 vLLM C++ 扩展，结果写到 <work>/env_info.json，有问题时退出码为 1。

  python -P 工具/env_check.py --work ~/uclode
  python -P 工具/env_check.py --work ~/repro --no-gpu

--no-gpu 在 import torch 之前隐藏所有 GPU，不分配显存也不执行算子，用在显卡正在跑评测的时候。
这样只确认了版本和扩展能加载，之后要不带这个参数再跑一次，才算完成 GPU 算子验收。
"""
import argparse
import importlib
import json
import os
import pathlib
import platform
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()
    if args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    import transformers
    import vllm

    info = {
        "python": platform.python_version(),
        "vllm": vllm.__version__,
        "vllm_commit": importlib.import_module("vllm._version").__commit_id__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu_rms_norm_check": "skipped" if args.no_gpu else "failed",
    }
    if not args.no_gpu:
        info["cuda_available"] = torch.cuda.is_available()
    if not args.no_gpu and info["cuda_available"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["device_capability"] = list(torch.cuda.get_device_capability(0))
        info["memory_mib"] = torch.cuda.get_device_properties(0).total_memory // 2**20
    try:
        importlib.import_module("vllm._C_stable_libtorch")
        if not args.no_gpu:
            from vllm import _custom_ops as ops
            x = torch.randn(2, 128, device="cuda", dtype=torch.float16)
            y = torch.empty_like(x)
            ops.rms_norm(y, x, torch.ones(128, device="cuda", dtype=x.dtype), 1e-6)
            reference = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
            torch.testing.assert_close(y.float(), reference, rtol=2e-3, atol=2e-3)
            torch.cuda.synchronize()
            info["gpu_rms_norm_check"] = "passed"
        info["vllm_C_extension"] = "ok"
    except Exception as error:  # 记录下来，让人看到，不中断
        info["vllm_C_extension"] = f"失败：{error}"
    print(json.dumps(info, ensure_ascii=False, indent=2))
    pathlib.Path(args.work, "env_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))

    problems = []
    if info["vllm"] != "0.32.0":
        problems.append(f"vLLM 版本是 {info['vllm']}")
    if not info["torch_cuda"]:
        problems.append("torch 不是 CUDA 版本")
    if not args.no_gpu:
        if not info["cuda_available"]:
            problems.append("CUDA 不可用")
        elif info.get("device_capability") != [8, 9]:
            problems.append(f"架构是 {info.get('device_capability')}，方案按 SM89 写的")
    if info["vllm_C_extension"] != "ok":
        problems.append("vLLM C++ 扩展加载或 GPU 算子验证失败，详见 vllm_C_extension")
    if problems:
        print("需要处理：" + "；".join(problems), file=sys.stderr)
        sys.exit(1)
    print("环境检查通过（未做 GPU 算子验收）。" if args.no_gpu else "环境检查通过。")


if __name__ == "__main__":
    main()
