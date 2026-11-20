#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoRA-Med SLAKE inference efficiency benchmark: A0 / A6 / A7.

No training, no DeepSeek API, no correctness judging, no checkpoint modifications.
Run each variant in a separate Python process on ONE identical GPU.

Measured timing includes: tensor H2D copies, frozen BioMedCLIP inference for
A6/A7, residual adaptation, Qwen3-VL prefill and fixed-length greedy decoding.
Excluded: JPEG disk reads, PIL/CPU image transforms, tokenization, Python
DataLoader/collator and model loading.

Default protocol: 128 fixed SLAKE-test questions; B=1; 32 generated tokens,
4 warm-up batches. `min_new_tokens == max_new_tokens` makes output lengths equal.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from config.slake.stage2_eval_config_slake import Stage2EvalConfig
from datas.slake_datasets import SLAKEDataset
from utils.biomedclip.biomed_clip_loader import load_biomedclip
from utils.data_tools.collator.slake.slake_datasets_eval_collator import SLAKEEvalCollator
from utils.qwen3vl.qwen3_vl_8B_lora_wrapper import Qwen3VLLoraAndVisualAdapterWrapper
from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader

GIB = 1024 ** 3
VARIANTS = ("A0", "A6", "A7")


def get_visual_adapter(model):
    return model.base_model.model.model.visual.res_adapter


def load_selected_checkpoint(cfg, override):
    """Default: exactly the checkpoint selected by validation, never by test."""
    if override:
        checkpoint = Path(override).expanduser().resolve()
        selection = "explicit --weights-path"
    else:
        best_json = Path(cfg.best_checkpoint_path)
        if not best_json.is_file():
            raise FileNotFoundError(
                f"Missing validation selection file:\n{best_json}\n"
                "Use an existing matched run, or pass --weights-path explicitly."
            )
        with best_json.open("r", encoding="utf-8") as f:
            best = json.load(f)
        if best.get("selection_split") not in (None, "validation"):
            raise ValueError("Checkpoint selection must use validation, not test")
        checkpoint = Path(best["selected_checkpoint_path"]).expanduser().resolve()
        selection = f"validation: {best_json}"

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
    required = ["adapter_model.safetensors"]
    if cfg.enable_visual_adapter:
        required.append("visual_adapter.pt")
    for file_name in required:
        if not (checkpoint / file_name).is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {checkpoint / file_name}")
    return checkpoint, selection


def make_model(cfg, weights_path, biomed_extractor):
    """Mirror the project's SLAKE test model loading, without importing LLM Judge."""
    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path,
        processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        torch_dtype=cfg.torch_dtype,
        attn_implementation=cfg.attn_implementation,
        device_map="auto",
    )
    processor = loader.load_processor()
    processor.tokenizer.padding_side = "left"
    base_model = loader.load_model()

    # A CPU/disk-offloaded model would not represent single-GPU inference.
    device_map = getattr(base_model, "hf_device_map", None)
    if isinstance(device_map, dict):
        offloaded = {k: str(v) for k, v in device_map.items()
                     if str(v).lower() in ("cpu", "disk")}
        if offloaded:
            raise RuntimeError(f"CPU/disk offloading detected: {offloaded}")

    wrapper = Qwen3VLLoraAndVisualAdapterWrapper(
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=cfg.lora_target_modules,
        gradient_checkpointing=False,
        visual_adapter_hidden_dim=cfg.visual_adapter_hidden_dim,
        visual_adapter_r=cfg.visual_adapter_r,
        enable_visual_adapter=cfg.enable_visual_adapter,
        biomed_extractor=biomed_extractor,
        use_cross_modal_prior=cfg.use_cross_modal_prior,
        router_hidden_dim=cfg.router_hidden_dim,
        scale_mode=cfg.scale_mode,
        fixed_scale_weights=cfg.fixed_scale_weights,
        gate_mode=cfg.gate_mode,
        fixed_gate=cfg.fixed_gate,
        gate_init=cfg.gate_init,
        lambda_mode=cfg.lambda_mode,
        fixed_lambda=cfg.fixed_lambda,
        lambda_max=cfg.lambda_max,
        lambda_init=cfg.lambda_init,
        use_rms_norm=cfg.use_rms_norm,
        residual_norm_eps=cfg.residual_norm_eps,
        residual_norm_ratio_clip=cfg.residual_norm_ratio_clip,
        fixed_stream_scales=cfg.fixed_stream_scales,
    )
    model = wrapper.wrap(base_model)
    set_peft_model_state_dict(
        model, load_file(str(weights_path / "adapter_model.safetensors"))
    )
    if cfg.enable_visual_adapter:
        adapter_state = torch.load(
            str(weights_path / "visual_adapter.pt"), map_location="cpu",
            weights_only=True,
        )
        get_visual_adapter(model).load_state_dict(adapter_state, strict=True)

    model.requires_grad_(False)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()

    if cfg.enable_visual_adapter:
        # The evaluation script precomputes BioMedCLIP once per batch. During
        # generate(), the wrapper's original patched forward would otherwise
        # clear its cached features at every autoregressive step.
        def patched_generate_forward(self, *args, **kwargs):
            kwargs.pop("biomed_image_tensors", None)
            kwargs.pop("biomed_text_tokens", None)
            return self.original_forward(*args, **kwargs)
        model.forward = types.MethodType(patched_generate_forward, model)

    return model, processor


def run_one_batch(model, batch_inputs, cfg, decode_tokens, device, vision):
    """Same model-side inference sequence as SLAKE Test; no answer decoding."""
    ref_dtype = next(vision.parameters()).dtype
    with torch.no_grad():
        inputs = {
            name: (value.to(device) if torch.is_tensor(value) else value)
            for name, value in batch_inputs.items()
        }
        if cfg.enable_visual_adapter:
            img = inputs.pop("biomed_image_tensors").to(ref_dtype)
            txt = inputs.pop("biomed_text_tokens")
            img_feat, txt_feat = model.biomed_extractor(img, txt)
            vision.current_biomed_img_feat = img_feat
            vision.current_biomed_txt_feat = txt_feat

        try:
            outputs = model.generate(
                **inputs,
                do_sample=False,
                min_new_tokens=decode_tokens,
                max_new_tokens=decode_tokens,
            )
            # Same amount of decode work for all models, independent of the
            # length of their natural-language answers.
            new_tokens = int(outputs.shape[-1] - inputs["input_ids"].shape[-1])
            if new_tokens != decode_tokens:
                raise RuntimeError(
                    f"Expected {decode_tokens} generated tokens, got {new_tokens}"
                )
            count = int(inputs["input_ids"].shape[0])
        finally:
            if cfg.enable_visual_adapter:
                vision.current_biomed_img_feat = None
                vision.current_biomed_txt_feat = None
    return count, count * new_tokens


def pctl(values, percent):
    ordered = sorted(values)
    if not ordered:
        return None
    idx = (len(ordered) - 1) * percent / 100
    lo, hi = int(idx), min(int(idx) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def run_benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This benchmark requires an NVIDIA GPU.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expose exactly one GPU, e.g. export CUDA_VISIBLE_DEVICES=4 "
            "(visible GPU is then cuda:0)."
        )
    if args.sample_count <= 0 or args.batch_size <= 0 or args.decode_tokens <= 0:
        raise ValueError("sample-count, batch-size, decode-tokens must be positive")
    if args.warmup < 1:
        raise ValueError("warmup must be at least 1")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    cfg = Stage2EvalConfig(
        ablation_id=args.variant, seed=args.seed, stage1_seed=args.seed,
    )
    checkpoint, selection = load_selected_checkpoint(cfg, args.weights_path)

    print(f"\nVariant: {args.variant}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Selection: {selection}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("No DeepSeek API / no scoring / no model training.\n")

    biomed_extractor = biomed_transform = biomed_tokenizer = None
    if cfg.enable_visual_adapter:
        biomed_extractor, biomed_transform, biomed_tokenizer = load_biomedclip(
            biomedclip_path=cfg.biomedclip_path, print_rank=cfg.print_rank,
        )

    model, processor = make_model(cfg, checkpoint, biomed_extractor)
    vision = model.base_model.model.model.visual
    device = next(vision.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(f"Qwen vision tower not on CUDA: {device}")

    full_dataset = SLAKEDataset(
        json_path=cfg.slake_test_json_path, image_root=cfg.slake_image_root,
    )
    n = min(args.sample_count, len(full_dataset))
    rng = random.Random(args.sample_seed)
    indices = sorted(rng.sample(range(len(full_dataset)), n))
    index_hash = hashlib.sha256(
        json.dumps(indices, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    selected = Subset(full_dataset, indices)

    collator = SLAKEEvalCollator(
        processor=processor,
        cfg=cfg,
        biomed_transform=biomed_transform,
        biomed_tokenizer=biomed_tokenizer,
    )
    data_loader = DataLoader(
        selected,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=0,
        shuffle=False,
    )

    # Warmup with the first few batches. Data preparation is OUTSIDE timing.
    print(f"Warmup: {args.warmup} batches")
    warm_it = iter(data_loader)
    for i in range(args.warmup):
        try:
            batch_inputs, _ = next(warm_it)
        except StopIteration:
            warm_it = iter(data_loader)
            batch_inputs, _ = next(warm_it)
        run_one_batch(model, batch_inputs, cfg, args.decode_tokens, device, vision)
    torch.cuda.synchronize()

    # Reset peak *after* loading and warmup. Peak includes resident model weights
    # plus transient inference allocations; no per-model memory is subtracted.
    torch.cuda.reset_peak_memory_stats(0)
    resident_gib = torch.cuda.memory_allocated(0) / GIB

    times = []
    total_samples = total_tokens = 0
    print(f"Benchmark: {n} questions, batch={args.batch_size}, "
          f"fixed output={args.decode_tokens} tokens")
    for i, (batch_inputs, _) in enumerate(data_loader, start=1):
        torch.cuda.synchronize()
        start = time.perf_counter()
        count, tokens = run_one_batch(
            model, batch_inputs, cfg, args.decode_tokens, device, vision,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_samples += count
        total_tokens += tokens
        if i % 16 == 0 or i == len(data_loader):
            print(f"  {i}/{len(data_loader)} batches", flush=True)

    total_sec = sum(times)
    result = {
        "variant": args.variant,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "selection": selection,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "SLAKE test",
        "sample_seed": args.sample_seed,
        "sample_indices_sha256": index_hash,
        "measured_samples": total_samples,
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup,
        "decode_tokens": args.decode_tokens,
        "generated_tokens": total_tokens,
        "measurement_scope": (
            "Synchronized device-side pipeline from H2D tensor copies through "
            "BioMedCLIP (A6/A7) and greedy generate(); excludes file I/O, "
            "CPU processor/collator and model loading."
        ),
        "total_inference_seconds": total_sec,
        "average_ms_per_sample": 1000 * total_sec / total_samples,
        "median_batch_latency_ms": 1000 * statistics.median(times),
        "p95_batch_latency_ms": 1000 * pctl(times, 95),
        "samples_per_second": total_samples / total_sec,
        "generated_tokens_per_second": total_tokens / total_sec,
        "resident_gpu_allocated_gib": resident_gib,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(0) / GIB,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved(0) / GIB,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.variant}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out_path}")
    print(f"Average: {result['average_ms_per_sample']:.1f} ms/sample")
    print(f"Throughput: {result['samples_per_second']:.3f} samples/sec")
    print(f"Generated tokens: {result['generated_tokens_per_second']:.2f} tok/sec")
    print(f"Peak PyTorch allocated VRAM: {result['peak_gpu_allocated_gib']:.2f} GiB")


def summarize(args):
    root = Path(args.output_dir)
    rows = []
    for variant in VARIANTS:
        path = root / f"{variant}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; benchmark {variant} first.")
        with path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))

    comparable_fields = (
        "gpu_name", "sample_indices_sha256", "measured_samples", "batch_size",
        "decode_tokens", "sample_seed", "dataset",
    )
    for field in comparable_fields:
        values = {str(r[field]) for r in rows}
        if len(values) != 1:
            raise ValueError(
                f"Non-comparable runs: {field} differs across A0/A6/A7: {values}"
            )

    headings = (
        "variant", "average_ms_per_sample", "median_batch_latency_ms",
        "p95_batch_latency_ms", "samples_per_second",
        "generated_tokens_per_second", "resident_gpu_allocated_gib",
        "peak_gpu_allocated_gib", "peak_gpu_reserved_gib",
    )
    csv_path = root / "efficiency_comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headings)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                h: (round(r[h], 4) if isinstance(r[h], float) else r[h])
                for h in headings
            })

    print("\n=== SLAKE inference efficiency (fixed-token decode) ===")
    print("Variant | ms/sample | samples/s | generated tok/s | peak GPU GiB")
    for r in rows:
        print(f"{r['variant']:>7} | "
              f"{r['average_ms_per_sample']:9.1f} | "
              f"{r['samples_per_second']:9.3f} | "
              f"{r['generated_tokens_per_second']:15.2f} | "
              f"{r['peak_gpu_allocated_gib']:12.2f}")
    print(f"\nSaved comparison: {csv_path}")
    print("Scope: GPU/model-side inference; NOT CPU image preparation or file I/O.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--summary", action="store_true",
                        help="Summarize A0/A6/A7 JSON; no GPU needed.")
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--sample-seed", type=int, default=2048)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--weights-path", type=str, default=None,
                        help="Optional explicit Stage2 checkpoint directory.")
    parser.add_argument("--output-dir", type=str, default="logs/efficiency_slake")
    args = parser.parse_args()

    if args.summary:
        summarize(args)
    else:
        if not args.variant:
            parser.error("--variant A0|A6|A7 is required unless --summary is set")
        run_benchmark(args)


if __name__ == "__main__":
    main()