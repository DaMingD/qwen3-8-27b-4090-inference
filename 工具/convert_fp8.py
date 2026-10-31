#!/usr/bin/env python3
"""把 BF16 权重转换成 per-channel FP8，让 SM89 走 CUTLASS 的真实 FP8 计算。只用 CPU。

官方 FP8 是 block-128，CUTLASS 的 block kernel 要求 SM90+，所以在 4090 上回退到 Marlin。
per-channel FP8 没有这个限制，但要从 BF16 重新量化，不能从 block FP8 转过来。

哪些权重要量化，完全照官方 FP8 版本：带 weight_scale_inv 的才量化，其他原样复制。
所以线性注意力的投影也会量化，lm_head、embedding、视觉模块、norm、门控等保持原精度。

用法（在 vLLM 环境里运行，逐分片处理，不需要一次把权重全部读入内存）：
  ~/uclode/venv-vllm/bin/python 工具/convert_fp8.py \\
      --bf16 /model/ModelScope/Qwen/Qwen3.8-27B \\
      --fp8 /model/ModelScope/Qwen/Qwen3.8-27B-FP8 \\
      --out ~/uclode/models/Qwen3.8-27B-FP8-channel

启动和验收：
  NAME=kb_cutlass LOG_ITER=1 MAX_SEQS=8 MODEL_DIR=~/uclode/models/Qwen3.8-27B-FP8-channel \\
      LINEAR_BACKEND=cutlass bash 工具/serve.sh start
启动证据里应出现 CutlassFP8ScaledMMLinearKernel。

每个权重都会记录量化误差 ||W - W_hat|| / ||W||。隔 --compare-every 个权重，
同时算官方 block-128 的误差，结果写入 conversion_meta.json。
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import time

from safetensors import safe_open
from safetensors.torch import save_file
import torch

FP8_MAX = 448.0     # float8_e4m3fn 的最大有限值
SCALE_SUFFIX = ".weight_scale_inv"
INDEX = "model.safetensors.index.json"


def load_index(folder):
    path = folder / INDEX
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["weight_map"]
    with safe_open(folder / "model.safetensors", framework="pt") as f:
        return {key: "model.safetensors" for key in f.keys()}


def quantize_channel(weight):
    """每个输出通道（行）一个 scale，返回 (fp8 权重, scale，形状为 [输出通道数, 1], 相对误差)。"""
    if weight.dim() != 2:
        raise ValueError(f"只支持 2 维权重，实际形状 {tuple(weight.shape)}")
    real = weight.float()
    peak = real.abs().amax(dim=1, keepdim=True)
    scale = torch.where(peak > 0, peak / FP8_MAX, torch.ones_like(peak))
    quantized = (real / scale).to(torch.float8_e4m3fn)
    restored = quantized.float() * scale
    if not torch.isfinite(restored).all():
        raise ValueError("量化后出现非有限值")
    return quantized, scale, float((real - restored).norm() / real.norm().clamp_min(1e-12))


def block_error(weight, fp8_dir, fp8_map, name):
    """官方 block FP8 相对同一份 BF16 权重的误差。"""
    with safe_open(fp8_dir / fp8_map[f"{name}.weight"], framework="pt") as f:
        quantized = f.get_tensor(f"{name}.weight")
    with safe_open(fp8_dir / fp8_map[name + SCALE_SUFFIX], framework="pt") as f:
        scale = f.get_tensor(name + SCALE_SUFFIX)
    rows, cols = quantized.shape
    block_n, block_k = (rows + scale.shape[0] - 1) // scale.shape[0], (cols + scale.shape[1] - 1) // scale.shape[1]
    grid = scale.float().repeat_interleave(block_n, 0).repeat_interleave(block_k, 1)[:rows, :cols]
    real = weight.float()
    return float((real - quantized.float() * grid).norm() / real.norm().clamp_min(1e-12))


def build_quant_config(ignore):
    return {"quant_method": "compressed-tensors", "format": "float-quantized", "quantization_status": "compressed",
            "config_groups": {"group_0": {
                "targets": ["Linear"],
                "weights": {"num_bits": 8, "type": "float", "strategy": "channel", "dynamic": False, "symmetric": True},
                "input_activations": {"num_bits": 8, "type": "float", "strategy": "token", "dynamic": True, "symmetric": True},
            }},
            "ignore": ignore}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bf16", type=Path, required=True)
    ap.add_argument("--fp8", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare-every", type=int, default=20, help="每隔多少个权重比较一次官方 block FP8 的误差，0 表示不比较")
    ap.add_argument("--dry-run", action="store_true", help="只核对权重名和磁盘空间，不写文件")
    args = ap.parse_args()

    bf16_map, fp8_map = load_index(args.bf16), load_index(args.fp8)
    quantized = {key[: -len(SCALE_SUFFIX)] for key in fp8_map if key.endswith(SCALE_SUFFIX)}
    if not quantized:
        raise SystemExit(f"{args.fp8} 里没有 {SCALE_SUFFIX}，不是预期的 block FP8 版本。")
    missing = [name for name in sorted(quantized) if f"{name}.weight" not in bf16_map]
    if missing:
        raise SystemExit(f"BF16 版本缺少 {len(missing)} 个权重，例如 {missing[:3]}")
    extra = {key for key in fp8_map if not key.endswith(SCALE_SUFFIX)} ^ set(bf16_map)
    if extra:
        raise SystemExit(f"两个版本的权重名不一致（{len(extra)} 个），例如 {sorted(extra)[:3]}")
    # compressed-tensors 用 ignore 列表按模块名排除。这里按模块名精确列出，不依赖官方的匹配模式
    ignore = sorted({key[: -len(".weight")] for key in bf16_map if key.endswith(".weight")} - quantized)
    config = json.loads((args.fp8 / "config.json").read_text(encoding="utf-8"))
    if "quantization_config" not in config:
        raise SystemExit("config.json 顶层没有 quantization_config。")
    print(f"权重 {len(bf16_map)} 个，要量化 {len(quantized)} 个，保持原精度的模块 {len(ignore)} 个")
    print(f"官方配置：{json.dumps({k: v for k, v in config['quantization_config'].items() if k != 'ignored_layers'})}")
    free_gb = shutil.disk_usage(args.out.parent if args.out.parent.exists() else Path.home()).free / 2**30
    print(f"磁盘可用 {free_gb:.0f} GB，转换后约等于官方 FP8 版本的大小")
    if args.dry_run:
        return
    args.out.mkdir(parents=True, exist_ok=False)

    started = time.time()
    errors, comparisons, weight_map = {}, [], {}
    count = 0
    for shard in sorted(set(bf16_map.values())):
        out = {}
        with safe_open(args.bf16 / shard, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                name = key[: -len(".weight")] if key.endswith(".weight") else None
                if name in quantized:
                    fp8_weight, scale, error = quantize_channel(tensor)
                    out[key], out[f"{name}.weight_scale"] = fp8_weight, scale
                    weight_map[f"{name}.weight_scale"] = shard
                    errors[name] = error
                    count += 1
                    if args.compare_every and count % args.compare_every == 0:
                        comparisons.append({"name": name, "channel": error,
                                            "block": block_error(tensor, args.fp8, fp8_map, name)})
                else:
                    out[key] = tensor
                weight_map[key] = shard
        save_file(out, args.out / shard, metadata={"format": "pt"})
        print(f"{shard} 完成，已量化 {count}/{len(quantized)}，{time.time() - started:.0f}s", flush=True)

    total = sum(os.path.getsize(args.out / shard) for shard in set(bf16_map.values()))
    (args.out / INDEX).write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2),
                                  encoding="utf-8")
    original = config["quantization_config"]
    config["quantization_config"] = build_quant_config(ignore)
    (args.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in args.fp8.iterdir():
        if item.is_file() and item.suffix != ".safetensors" and item.name not in (INDEX, "config.json"):
            target = args.out / item.name
            if not target.exists():
                os.symlink(item, target)
    values = sorted(errors.values())
    meta = {"source_bf16": str(args.bf16), "source_fp8": str(args.fp8), "original_quantization_config": original,
            "quantized": len(errors), "ignored": len(ignore), "bytes": total, "seconds": round(time.time() - started),
            "channel_error": {"median": statistics.median(values), "max": values[-1]},
            "comparison_with_block": comparisons}
    (args.out / "conversion_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{total / 2**30:.1f} GiB，per-channel 相对误差中位数 {meta['channel_error']['median']:.5f}")
    if comparisons:
        print(f"抽样 {len(comparisons)} 个权重：per-channel 中位数 {statistics.median(c['channel'] for c in comparisons):.5f}，"
              f"官方 block 中位数 {statistics.median(c['block'] for c in comparisons):.5f}")


if __name__ == "__main__":
    main()
