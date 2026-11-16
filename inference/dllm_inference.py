#!/usr/bin/env python3
"""Block-diffusion multi-turn inference for ChemDiffAgent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dllm_generation import BlockDiffusionGenerator
from rollout import run_dataset
from tool_registry import load_tools

# Use the same model/operator and profiler implementations as SFT.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.sft.train.profiling import create_profiler
from phase_profiling import PhaseProfiler
from contextlib import nullcontext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input .jsonl/.json dataset")
    parser.add_argument("--output", required=True, help="Output rollout .jsonl")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile", action="store_true", help="Record a finite NPU forward window")
    parser.add_argument("--profile-phase", choices=["all", "both", "prefill", "denoise", "cache_commit"], default="all")
    parser.add_argument("--profile-dir", default="prof")
    parser.add_argument("--profile-skip-first", type=int, default=0)
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-active", type=int, default=2)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--num-rolls", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--denoising-steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--remasking-strategy",
        choices=["low_confidence_static", "low_confidence_dynamic", "sequential"],
        default="low_confidence_static",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--tool-timeout-seconds", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-results", help="Original results: continue failed tools and preserve all other rows")
    parser.add_argument("--print-turns", action="store_true")
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    if device.type == "npu":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model(model_path: str, device: torch.device, dtype_name: str):
    dtype = resolve_dtype(dtype_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_class = AutoModelForCausalLM
    if device.type == "npu":
        from training.sft.models.sdar import SDARConfig, SDARForCausalLM
        config = SDARConfig.from_pretrained(model_path)
        if config.model_type != "sdar":
            raise ValueError("NPU inference requires an SDAR checkpoint")
        model_class = SDARForCausalLM
    model = model_class.from_pretrained(
        model_path,
        **({"config": config} if device.type == "npu" else {}),
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    model.config.use_cache = True
    model.config.fuse_cross_entropy = False
    return model, tokenizer


def main() -> None:
    args = parse_args()
    if args.num_rolls < 1:
        raise ValueError("--num-rolls must be >= 1")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    device = setup_device(args.device, args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)

    tools = load_tools()
    model, tokenizer = load_model(args.model_path, device, args.dtype)
    generator = BlockDiffusionGenerator(
        model,
        tokenizer,
        device,
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        remasking_strategy=args.remasking_strategy,
        confidence_threshold=args.confidence_threshold,
        max_context_tokens=args.max_context_tokens,
    )
    settings = {"enabled": args.profile, "output_dir": args.profile_dir,
                "skip_first": args.profile_skip_first, "warmup": args.profile_warmup,
                "active": args.profile_active}
    if args.profile_phase != "all":
        context = nullcontext(PhaseProfiler(settings, Path(args.output).resolve().parent,
                              args.shard_index, device.type, args.num_shards, args.profile_phase))
    else:
        context = create_profiler(settings, Path(args.output).resolve().parent,
                         args.shard_index, device.type,
                         world_size=args.num_shards, schedule_unit="model_forward")
    with context as profiler:
        generator.profiler = profiler
        summary = run_dataset(
            input_path=args.input,
            output_path=args.output,
            generator=generator,
            tools=tools,
            max_samples=args.max_samples,
            sample_id=args.sample_id or None,
            num_rolls=args.num_rolls,
            resume=args.resume,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            max_context_tokens=args.max_context_tokens,
            tool_timeout_seconds=args.tool_timeout_seconds,
            print_turns=args.print_turns,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            retry_results=args.retry_results,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def setup_device(requested: str, seed: int) -> torch.device:
    # Register the private-use device before torch.device parses npu:N.
    if requested.split(":", 1)[0] == "npu":
        import torch_npu  # noqa: F401
        if not torch.npu.is_available():
            raise RuntimeError("Requested NPU is unavailable")
    device = torch.device(requested)
    if device.type == "npu":
        torch.npu.set_device(device)
        torch.npu.manual_seed_all(seed)
    elif device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable; use --device cpu")
    torch.manual_seed(seed)
    return device


if __name__ == "__main__":
    main()
