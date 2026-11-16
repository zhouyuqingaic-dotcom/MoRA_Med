#!/usr/bin/env python3
"""Estimate the four A7 fixed stream multipliers from A6 Stage-1 TRAIN data.

Matches the supplied MoRA_Med.zip codebase.  No A7 coefficients are required.
Only vision_tower + BioMedCLIP are forwarded (no LM text generation/judge).
The RMS function is observed without altering A6's returned residual.

Run from project root with PYTHONPATH set to the project root:
  CUDA_VISIBLE_DEVICES=1 python other/estimate_a7_scales_mimic.py --num-samples 2048
"""
import argparse
import csv
import json
import math
import random
import statistics
import types
from pathlib import Path

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from tqdm import tqdm

from config.stage1_train_config_mimic_cxr import TrainConfig
from datas.mimic_cxr_datasets import MIMICCXRDataset
from utils.biomedclip.biomed_clip_loader import load_biomedclip
from utils.data_tools.collator.mimic_cxr.mimic_cxr_datasets_train_collator import (
    MIMICCXRTrainCollator,
)
from utils.qwen3vl.qwen3_vl_8B_lora_wrapper import (
    Qwen3VLLoraAndVisualAdapterWrapper,
)
from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader
import utils.qwen3vl.qwen3_vl_8B_visual_adapters_fusion as fusion_module

STREAMS = ("pooled", "deepstack_0", "deepstack_1", "deepstack_2")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--sampling-seed", type=int, default=2048)
    parser.add_argument("--a6-seed", type=int, default=2048)
    parser.add_argument(
        "--weights-dir", type=str, default=None,
        help="A6 Stage-1 checkpoint directory containing visual_adapter.pt and adapter_model.safetensors; defaults to A6 TrainConfig output_dir/final_weights",
    )
    parser.add_argument(
        "--output", type=str, default="logs/a7_mimic_train_rms_seed2048.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this on the GPU server")

    random.seed(args.sampling_seed)
    torch.manual_seed(args.sampling_seed)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES chooses the physical GPU.

    # A6 config is explicit; A7_FIXED_STREAM_SCALES may stay None during calibration.
    cfg = TrainConfig(ablation_id="A6", seed=args.a6_seed)
    if not cfg.use_rms_norm or cfg.fixed_stream_scales is not None:
        raise AssertionError("Expected unmodified A6 capped RMS, not A7")
    if cfg.lambda_mode != "learnable" or not cfg.use_discriminative_lr:
        raise AssertionError("A6 configuration differs from learnable-lambda DLR baseline")
    weights = Path(args.weights_dir or (Path(cfg.output_dir) / "final_weights"))
    for filename in ("adapter_model.safetensors", "visual_adapter.pt"):
        if not (weights / filename).is_file():
            raise FileNotFoundError(f"A6 weights missing: {weights / filename}")

    print(f"A6 Stage-1 checkpoint: {weights}", flush=True)
    print(f"MIMIC-CXR cache prefix: {cfg.mimic_cxr_cache_prefix}", flush=True)
    print(f"RMS epsilon={cfg.residual_norm_eps}, clip={cfg.residual_norm_ratio_clip}", flush=True)

    dataset = MIMICCXRDataset(
        csv_path=cfg.mimic_cxr_metadata_csv,
        image_root=cfg.mimic_cxr_image_root,
        report_root=cfg.mimic_cxr_report_root,
        target_section=cfg.mimic_cxr_target_section,
        load_report=cfg.mimic_cxr_load_report,
        allowed_view_positions=cfg.mimic_cxr_view_positions,
        drop_empty_target=cfg.mimic_cxr_drop_empty_target,
        cache_dir=cfg.mimic_cxr_cache_dir,
        use_indices_cache=cfg.mimic_cxr_use_indices_cache,
        rebuild_indices_cache=False,  # never alter the existing training cache
        cache_prefix=cfg.mimic_cxr_cache_prefix,
    )
    if cfg.max_mimic_cxr_train_samples is not None:
        dataset.samples = dataset.samples[:cfg.max_mimic_cxr_train_samples]
    n = min(args.num_samples, len(dataset))
    if n == 0:
        raise ValueError("MIMIC training dataset is empty")
    indices = sorted(random.Random(args.sampling_seed).sample(range(len(dataset)), n))
    print(f"Selected {n} distinct TRAIN samples from {len(dataset)} cached samples", flush=True)

    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path,
        processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        torch_dtype=cfg.torch_dtype,
        attn_implementation=cfg.attn_implementation,
        device_map={"": 0},
    )
    base_model, processor = loader.load()
    processor.tokenizer.padding_side = "right"
    extractor, image_transform, text_tokenizer = load_biomedclip(
        biomedclip_path=cfg.biomedclip_path,
        print_rank=cfg.print_rank,
    )
    wrapper = Qwen3VLLoraAndVisualAdapterWrapper(
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=cfg.lora_target_modules,
        gradient_checkpointing=False,
        visual_adapter_hidden_dim=cfg.visual_adapter_hidden_dim,
        visual_adapter_r=cfg.visual_adapter_r,
        enable_visual_adapter=cfg.enable_visual_adapter,
        biomed_extractor=extractor,
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
        fixed_stream_scales=None,
    )
    model = wrapper.wrap(base_model)

    set_peft_model_state_dict(model, load_file(str(weights / "adapter_model.safetensors")))
    vision_tower = model.base_model.model.model.visual
    adapter = vision_tower.res_adapter
    state = torch.load(str(weights / "visual_adapter.pt"), map_location="cpu", weights_only=True)
    adapter.load_state_dict(state, strict=True)
    if not adapter.use_rms_norm or adapter.fixed_stream_scales is not None:
        raise AssertionError("Loaded visual adapter is not in A6 RMS mode")
    model.requires_grad_(False)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()

    collator = MIMICCXRTrainCollator(
        processor=processor, cfg=cfg,
        biomed_transform=image_transform, biomed_tokenizer=text_tokenizer,
    )

    # Observe EXACTLY the rho used by A6, not a post-Gate or post-lambda scale.
    # The fusion module's original RMS function is still used to compute outputs.
    original_rms = fusion_module.rms_normalize_residual
    original_forward = adapter.forward
    current_stream = [None]
    row_values = [None] * 4
    row_unclipped = [None] * 4
    ratio_clip = cfg.residual_norm_ratio_clip

    def observed_forward(self, *forward_args, **forward_kwargs):
        stream_id = forward_kwargs.get("stream_id", 0)
        if not isinstance(stream_id, int) or stream_id not in range(4):
            raise ValueError(f"Unexpected stream_id: {stream_id}")
        if current_stream[0] is not None:
            raise RuntimeError("Unexpected nested Fusion forward")
        current_stream[0] = stream_id
        try:
            return original_forward(*forward_args, **forward_kwargs)
        finally:
            current_stream[0] = None

    def observed_rms(x, residual, eps=1e-6, ratio_clip=10.0):
        sid = current_stream[0]
        if sid is None:
            raise RuntimeError("RMS called outside observed adapter.forward")
        if row_values[sid] is not None:
            raise RuntimeError(f"Multiple images for one stream; use batch_size=1: {sid}")
        rms_x = torch.sqrt(torch.mean(x.float() ** 2, dim=(-2, -1), keepdim=True) + eps)
        rms_r = torch.sqrt(torch.mean(residual.float() ** 2, dim=(-2, -1), keepdim=True) + eps)
        ratio_raw = float((rms_x / rms_r).item())
        rho = min(ratio_raw, ratio_clip) if ratio_clip is not None else ratio_raw
        if not math.isfinite(rho) or rho <= 0:
            raise RuntimeError(f"Invalid RMS ratio {rho} for stream {sid}")
        row_values[sid] = rho
        row_unclipped[sid] = ratio_raw
        return original_rms(x, residual, eps=eps, ratio_clip=ratio_clip)

    adapter.forward = types.MethodType(observed_forward, adapter)
    fusion_module.rms_normalize_residual = observed_rms
    collected = {name: [] for name in STREAMS}
    records = []
    dtype = next(vision_tower.parameters()).dtype
    real_device = next(vision_tower.parameters()).device
    try:
        with torch.inference_mode():
            for ds_idx in tqdm(indices, desc="A6 RMS on MIMIC TRAIN", dynamic_ncols=True):
                row_values[:] = [None] * 4
                row_unclipped[:] = [None] * 4
                batch = collator([dataset[ds_idx]])
                for key in ("pixel_values", "image_grid_thw", "biomed_image_tensors", "biomed_text_tokens"):
                    if key not in batch:
                        raise KeyError(f"Collator did not return required key: {key}")
                pixels = batch["pixel_values"].to(device=real_device, dtype=dtype)
                grid = batch["image_grid_thw"].to(device=real_device)
                biomed_image = batch["biomed_image_tensors"].to(device=real_device, dtype=dtype)
                biomed_text = batch["biomed_text_tokens"].to(device=real_device)
                img_feat, txt_feat = model.biomed_extractor(biomed_image, biomed_text)
                vision_tower.current_biomed_img_feat = img_feat
                vision_tower.current_biomed_txt_feat = txt_feat
                # The original visual forward automatically runs pooled + deepstack fusion.
                # No language-model decoder / generation / LLM judge is needed.
                vision_tower(pixels, grid_thw=grid)
                if any(v is None for v in row_values):
                    raise RuntimeError(f"Missing stream RMS for dataset index {ds_idx}: {row_values}")
                records.append((ds_idx, *row_values))
                for stream, rho in zip(STREAMS, row_values):
                    collected[stream].append(rho)
                vision_tower.current_biomed_img_feat = None
                vision_tower.current_biomed_txt_feat = None
    finally:
        adapter.forward = original_forward
        fusion_module.rms_normalize_residual = original_rms
        vision_tower.current_biomed_img_feat = None
        vision_tower.current_biomed_txt_feat = None

    if len(records) != n:
        raise RuntimeError(f"Only {len(records)} / {n} samples were collected")

    per_stream = {}
    medians = []
    for stream in STREAMS:
        values = collected[stream]
        midpoint = statistics.median(values)
        medians.append(round(midpoint, 8))
        per_stream[stream] = {
            "n": len(values),
            "median": round(midpoint, 8),
            "mean": round(statistics.fmean(values), 8),
            "min": round(min(values), 8),
            "max": round(max(values), 8),
            "fraction_at_clip": (
                sum(v >= ratio_clip - 1e-6 for v in values) / len(values)
                if ratio_clip is not None else None
            ),
        }

    result = {
        "method": "A6-trained checkpoint / Stage-1 MIMIC-CXR TRAIN median capped-RMS",
        "source_checkpoint": str(weights),
        "train_cache_prefix": cfg.mimic_cxr_cache_prefix,
        "dataset_total_samples": len(dataset),
        "samples_used": n,
        "sample_seed": args.sampling_seed,
        "a6_seed": args.a6_seed,
        "eps": cfg.residual_norm_eps,
        "rms_clip": ratio_clip,
        "stream_order": list(STREAMS),
        "fixed_stream_scales": medians,
        "per_stream": per_stream,
        "note": "Train-only estimates; do not tune using SLAKE test data",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    csvpath = out.with_suffix(".csv")
    with csvpath.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("dataset_index", *STREAMS))
        writer.writerows(records)
    print("\nA7_FIXED_STREAM_SCALES = (", ", ".join(f"{v:.8g}" for v in medians), ")", sep="", flush=True)
    print(f"Wrote summary: {out}", flush=True)
    print(f"Wrote per-sample CSV: {csvpath}", flush=True)


if __name__ == "__main__":
    main()
