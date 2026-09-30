import argparse
import csv
import glob
import json
import math
import os
import types
from collections import Counter
from pathlib import Path

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.LLM_config import LLMAPIConfig
from config.slake.stage2_eval_config_slake import Stage2EvalConfig
from datas.slake_datasets import SLAKEDataset
from LLM_api.deepseek import DeepSeekClient
from LLM_api.prompts.slake_prompt_builder_deepseek import (
    build_llm_judge_user_prompt,
    parse_llm_judge_response,
)
from utils.biomedclip.biomed_clip_loader import load_biomedclip
from utils.data_tools.collator.slake.slake_datasets_eval_collator import (
    SLAKEEvalCollator,
)
from utils.data_tools.prompt_cleaning.slake_answer_cleaning import (
    slake_answer_eval_cleaning,
)
from utils.evaluation.router_shuffle import (
    build_shuffle_map,
    compute_shuffle_map_sha256,
    get_shuffle_entry,
    load_shuffle_map,
    save_shuffle_map,
    validate_shuffle_map,
)
from utils.qwen3vl.qwen3_vl_8B_lora_wrapper import (
    Qwen3VLLoraAndVisualAdapterWrapper,
)
from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader


VALID_CONDITIONS = {"normal", "router_shuffle"}


def get_adapter(model):
    return model.base_model.model.model.visual.res_adapter


def clear_visual_runtime_cache(vision_tower):
    vision_tower.current_biomed_img_feat = None
    vision_tower.current_biomed_txt_feat = None
    vision_tower.current_router_txt_feat = None


def build_model(weights_path, loader, cfg, biomed_extractor):
    base_model = loader.load_model()

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
    )

    model = wrapper.wrap(base_model)

    set_peft_model_state_dict(
        model,
        load_file(
            os.path.join(weights_path, "adapter_model.safetensors")
        ),
    )

    if cfg.enable_visual_adapter:
        state = torch.load(
            os.path.join(weights_path, "visual_adapter.pt"),
            map_location="cpu",
        )
        get_adapter(model).load_state_dict(state, strict=True)

    model.requires_grad_(False)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()

    return model, base_model


def patch_generate_forward(model):
    """
    generate() 内部连续调用 forward。
    BioMedCLIP 特征已在 batch 开始时缓存，因此这里直接调用 Wrapper
    保存的 original_forward，避免重复编码，也避免清掉 router-only cache。
    """
    def forward(self, *args, **kwargs):
        kwargs.pop("biomed_image_tensors", None)
        kwargs.pop("biomed_text_tokens", None)
        return self.original_forward(*args, **kwargs)

    model.forward = types.MethodType(forward, model)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "SLAKE test: normal / Router-only shuffle / RMS diagnostics"
        )
    )
    parser.add_argument("--ablation-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stage1-seed", type=int, default=None)
    parser.add_argument(
        "--test-checkpoint-mode",
        choices=["best_validation", "all"],
        default=None,
    )
    parser.add_argument(
        "--condition",
        choices=sorted(VALID_CONDITIONS),
        default="normal",
    )
    parser.add_argument("--shuffle-map", type=str, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--shuffle-id", type=str, default=None)
    parser.add_argument(
        "--shuffle-language-key",
        type=str,
        default="auto",
        help=(
            "默认 auto，只检测 q_lang/language/lang；"
            "若当前 JSON 已明确为单语言，可传 none。"
        ),
    )
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--collect-rms", action="store_true")
    return parser.parse_args()


def build_eval_config(args):
    kwargs = {}

    if args.ablation_id is not None:
        kwargs["ablation_id"] = args.ablation_id

    if args.seed is not None:
        kwargs["seed"] = args.seed
        if args.stage1_seed is None:
            kwargs["stage1_seed"] = args.seed

    if args.stage1_seed is not None:
        kwargs["stage1_seed"] = args.stage1_seed

    if args.test_checkpoint_mode is not None:
        kwargs["test_checkpoint_mode"] = args.test_checkpoint_mode

    return Stage2EvalConfig(**kwargs)


def validate_run_tag(run_tag: str) -> str:
    run_tag = str(run_tag).strip()
    if not run_tag:
        raise ValueError("run_tag 不能为空。")
    if os.path.basename(run_tag) != run_tag or run_tag in {".", ".."}:
        raise ValueError(f"run_tag 只能是单层目录名：{run_tag!r}")
    return run_tag


def prepare_shuffle_context(args, cfg, dataset):
    if args.condition == "normal":
        if args.shuffle_map is not None or args.shuffle_seed is not None:
            raise ValueError(
                "condition=normal 时不要传 --shuffle-map / --shuffle-seed。"
            )
        return None, {
            "shuffle_id": None,
            "shuffle_seed": None,
            "shuffle_map_path": None,
            "shuffle_map_sha256": None,
            "shuffle_language_key": None,
        }

    if not cfg.enable_visual_adapter:
        raise ValueError("router_shuffle 需要启用 Visual Adapter。")
    if cfg.scale_mode != "learned":
        raise ValueError("router_shuffle 只适用于 scale_mode='learned'。")

    map_path = Path(args.shuffle_map) if args.shuffle_map else None

    if map_path is None:
        if args.shuffle_seed is None:
            raise ValueError(
                "router_shuffle 必须提供 --shuffle-map，"
                "或提供 --shuffle-seed 自动生成固定 mapping。"
            )
        map_path = (
            Path(cfg.test_output_dir)
            / "diagnostics"
            / "shuffle_maps"
            / f"shuffle_seed_{int(args.shuffle_seed)}.json"
        )

    if map_path.is_file():
        payload = load_shuffle_map(map_path)
    else:
        if args.shuffle_seed is None:
            raise FileNotFoundError(
                f"shuffle map 不存在且未提供 --shuffle-seed：{map_path}"
            )

        payload = build_shuffle_map(
            dataset=dataset,
            seed=int(args.shuffle_seed),
            language_key=args.shuffle_language_key,
            dataset_name="SLAKE-test",
        )
        save_shuffle_map(payload, map_path)
        print(f"已生成固定 Router shuffle map：{map_path}")

    validate_shuffle_map(dataset, payload, language_key="auto")

    if (
        args.shuffle_seed is not None
        and payload.get("shuffle_seed") is not None
        and int(payload["shuffle_seed"]) != int(args.shuffle_seed)
    ):
        raise ValueError(
            "CLI --shuffle-seed 与 mapping 内部记录不一致："
            f"{args.shuffle_seed} vs {payload['shuffle_seed']}"
        )

    return payload, {
        "shuffle_id": args.shuffle_id,
        "shuffle_seed": payload.get("shuffle_seed"),
        "shuffle_map_path": str(map_path.resolve()),
        "shuffle_map_sha256": compute_shuffle_map_sha256(payload),
        "shuffle_language_key": payload.get("language_key"),
    }


def derive_run_tag(args, shuffle_payload):
    if args.run_tag:
        return validate_run_tag(args.run_tag)

    if args.condition == "normal":
        return "normal_rms" if args.collect_rms else "normal"

    if args.shuffle_id:
        suffix = str(args.shuffle_id).strip()
    elif shuffle_payload and shuffle_payload.get("shuffle_seed") is not None:
        suffix = str(shuffle_payload["shuffle_seed"])
    elif args.shuffle_map:
        suffix = Path(args.shuffle_map).stem
    else:
        suffix = "custom"

    return validate_run_tag(f"router_shuffle_{suffix}")


def tokenize_router_questions(biomed_tokenizer, questions, device):
    tokens = biomed_tokenizer(questions)

    if not isinstance(tokens, torch.Tensor):
        raise TypeError(
            f"biomed_tokenizer 应返回 Tensor，实际为 {type(tokens)}。"
        )

    if tokens.ndim != 2 or tokens.shape[0] != len(questions):
        raise ValueError(
            "Router-only BioMedCLIP token 形状错误："
            f"{tuple(tokens.shape)}, batch={len(questions)}"
        )

    return tokens.to(device)


def make_router_batch_info(metadata, condition, shuffle_payload):
    info = []

    for meta in metadata:
        source_index = int(meta["index"])

        if condition == "normal":
            info.append(
                {
                    "source_index": source_index,
                    "target_index": source_index,
                    "target_question": meta["question"],
                    "language": None,
                }
            )
        else:
            entry = get_shuffle_entry(shuffle_payload, source_index)
            info.append(
                {
                    "source_index": source_index,
                    "target_index": int(entry["target_index"]),
                    "target_question": str(entry["target_question"]),
                    "language": entry.get("language"),
                }
            )

    return info


def attach_rms_metadata(
    batch_rms_records,
    metadata,
    router_batch_info,
    checkpoint_name,
    condition,
):
    output = []

    for raw in batch_rms_records:
        record = dict(raw)
        visual_item_index = int(record["visual_item_index"])

        if not 0 <= visual_item_index < len(metadata):
            raise RuntimeError(
                "RMS visual_item_index 越界："
                f"{visual_item_index} / batch={len(metadata)}"
            )

        meta = metadata[visual_item_index]
        router_info = router_batch_info[visual_item_index]

        record.update(
            {
                "checkpoint": checkpoint_name,
                "evaluation_condition": condition,
                "index": meta.get("index"),
                "image_path": meta.get("image_path"),
                "question": meta.get("question"),
                "router_target_index": router_info["target_index"],
                "router_question": router_info["target_question"],
            }
        )
        output.append(record)

    return output


def evaluate_checkpoint(
    weights_path,
    loader,
    processor,
    cfg,
    test_loader,
    llm_client,
    llm_cfg,
    biomed_extractor,
    biomed_tokenizer,
    output_root,
    condition,
    shuffle_payload,
    run_provenance,
    collect_rms,
):
    name = os.path.basename(weights_path)
    output_dir = os.path.join(output_root, name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"开始 Test：{name}")
    print(f"Condition：{condition}")
    print(f"Collect RMS：{collect_rms}")
    print(f"输出目录：{output_dir}")
    print("=" * 70)

    model, base_model = build_model(
        weights_path,
        loader,
        cfg,
        biomed_extractor,
    )

    adapter = get_adapter(model) if cfg.enable_visual_adapter else None
    vision_tower = model.base_model.model.model.visual
    ref_param = next(vision_tower.parameters())
    device = ref_param.device
    adapter_dtype = ref_param.dtype

    if cfg.enable_visual_adapter:
        adapter.enable_diagnostics(collect_rms)
        adapter.reset_diagnostics()
        patch_generate_forward(model)

    records = []
    rms_records = []

    # =========================================================
    # Phase 1: local generation
    # =========================================================
    with torch.no_grad():
        for batch_inputs, metadata in tqdm(
            test_loader,
            desc=f"Local Inference [{condition}]",
        ):
            inputs = {
                key: (
                    value.to(device)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in batch_inputs.items()
            }

            router_batch_info = make_router_batch_info(
                metadata,
                condition,
                shuffle_payload,
            )

            if cfg.enable_visual_adapter:
                clear_visual_runtime_cache(vision_tower)

                image = inputs.pop("biomed_image_tensors").to(adapter_dtype)
                text = inputs.pop("biomed_text_tokens")

                image_feat, text_feat = model.biomed_extractor(image, text)

                vision_tower.current_biomed_img_feat = image_feat
                vision_tower.current_biomed_txt_feat = text_feat

                if condition == "router_shuffle":
                    router_questions = [
                        item["target_question"]
                        for item in router_batch_info
                    ]
                    router_tokens = tokenize_router_questions(
                        biomed_tokenizer,
                        router_questions,
                        device,
                    )

                    # 同一批原图，只替换 Router 使用的问题文本。
                    _, router_text_feat = model.biomed_extractor(
                        image,
                        router_tokens,
                    )
                    vision_tower.current_router_txt_feat = router_text_feat
                else:
                    vision_tower.current_router_txt_feat = None

                # 防止误读上一 batch 的 latest_*。
                adapter.latest_routing_weights = None
                adapter.latest_soft_gate = None
                adapter.latest_lambda = None

                if collect_rms:
                    adapter.reset_diagnostics()

            generate_args = {
                "max_new_tokens": cfg.max_new_tokens,
                "do_sample": cfg.do_sample,
            }
            if cfg.do_sample:
                generate_args["temperature"] = cfg.temperature

            try:
                generated = model.generate(**inputs, **generate_args)

                if cfg.enable_visual_adapter:
                    if (
                        adapter.latest_routing_weights is None
                        or adapter.latest_soft_gate is None
                        or adapter.latest_lambda is None
                    ):
                        raise RuntimeError(
                            "当前 batch 没有产生新的 routing/gate/lambda。"
                        )

                    routing = (
                        adapter.latest_routing_weights
                        .detach()
                        .float()
                        .cpu()
                    )
                    gate = (
                        adapter.latest_soft_gate
                        .detach()
                        .float()
                        .reshape(-1)
                        .cpu()
                    )
                    lambda_value = float(
                        adapter.latest_lambda
                        .detach()
                        .float()
                        .cpu()
                        .item()
                    )

                    batch_size = len(metadata)
                    if routing.shape != (batch_size, 3):
                        raise RuntimeError(
                            f"routing shape={tuple(routing.shape)}, "
                            f"expected=({batch_size}, 3)"
                        )
                    if gate.shape != (batch_size,):
                        raise RuntimeError(
                            f"gate shape={tuple(gate.shape)}, "
                            f"expected=({batch_size},)"
                        )

                    if collect_rms:
                        batch_rms_records = adapter.pop_diagnostics()
                        if not batch_rms_records:
                            raise RuntimeError(
                                "collect_rms=True，但当前 batch 没有 RMS 记录。"
                            )

                        rms_records.extend(
                            attach_rms_metadata(
                                batch_rms_records,
                                metadata,
                                router_batch_info,
                                name,
                                condition,
                            )
                        )
                else:
                    routing = None
                    gate = None
                    lambda_value = None

            finally:
                if cfg.enable_visual_adapter:
                    clear_visual_runtime_cache(vision_tower)
                    if collect_rms:
                        adapter.reset_diagnostics()

            generated = [
                output[len(input_ids):]
                for input_ids, output in zip(
                    inputs["input_ids"],
                    generated,
                )
            ]

            predictions = processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for i, prediction in enumerate(predictions):
                meta = metadata[i]
                router_info = router_batch_info[i]

                gt_raw = meta.get(
                    "gt_answer",
                    meta.get("answer", ""),
                )
                gt_norm = slake_answer_eval_cleaning(gt_raw)
                pred_norm = slake_answer_eval_cleaning(prediction)
                matched = pred_norm == gt_norm

                category = (
                    "closed"
                    if meta.get("answer_type", "").strip().upper() == "CLOSED"
                    else "open"
                )

                record = {
                    "index": meta.get("index"),
                    "image_path": meta.get("image_path"),
                    "question": meta["question"],
                    "question_category": category,

                    "evaluation_condition": condition,
                    "router_target_index": router_info["target_index"],
                    "router_question": router_info["target_question"],

                    "gt_raw": gt_raw,
                    "gt_norm": gt_norm,
                    "pred_raw": prediction.strip(),
                    "pred_norm": pred_norm,
                    "is_norm_match": matched,

                    "llm_judge_score": None,
                    "llm_judge_reason": None,
                    "llm_judge_raw_response": None,
                    "final_correct": matched,

                    "routing_f3": None,
                    "routing_f5": None,
                    "routing_f7": None,
                    "residual_gate": None,
                    "lambda_value": None,
                    "effective_residual_scale": None,
                }

                if routing is not None:
                    record["routing_f3"] = float(routing[i, 0])
                    record["routing_f5"] = float(routing[i, 1])
                    record["routing_f7"] = float(routing[i, 2])
                    record["residual_gate"] = float(gate[i])
                    record["lambda_value"] = lambda_value
                    record["effective_residual_scale"] = (
                        lambda_value * float(gate[i])
                    )

                records.append(record)

    # GPU generation 完成后释放显存，再进行远程 judge。
    del model
    del base_model
    torch.cuda.empty_cache()

    # =========================================================
    # Phase 2: LLM Judge
    # =========================================================
    for record in tqdm(records, desc=f"LLM Judging [{condition}]"):
        if (
            record["question_category"] == "open"
            and not record["is_norm_match"]
        ):
            # 始终使用原始 question，而不是 shuffled router question。
            prompt = build_llm_judge_user_prompt(
                question=record["question"],
                gt_raw=record["gt_raw"],
                gt_norm=record["gt_norm"],
                pred_raw=record["pred_raw"],
                pred_norm=record["pred_norm"],
            )

            response = llm_client.ask(
                llm_cfg.medical_vqa_llm_judge_system_prompt,
                prompt,
            )
            judged = parse_llm_judge_response(response)

            record["llm_judge_raw_response"] = response
            record["llm_judge_score"] = judged["score"]
            record["llm_judge_reason"] = (
                judged.get("reasoning")
                or judged.get("reason")
                or judged.get("explanation")
            )
            record["final_correct"] = judged["score"] == "correct"

    if not records:
        raise RuntimeError("Test 没有产生任何样本记录。")

    # =========================================================
    # Phase 3: metrics
    # =========================================================
    closed = [
        item for item in records
        if item["question_category"] == "closed"
    ]
    opened = [
        item for item in records
        if item["question_category"] == "open"
    ]

    if not closed or not opened:
        raise RuntimeError(
            f"SLAKE closed/open 分组异常：closed={len(closed)}, open={len(opened)}"
        )

    closed_correct = sum(bool(item["final_correct"]) for item in closed)
    open_correct = sum(bool(item["final_correct"]) for item in opened)

    result = {
        "checkpoint": name,
        "checkpoint_path": os.path.abspath(weights_path),
        "ablation_id": cfg.ablation_id,
        "seed": cfg.seed,
        "stage1_seed": cfg.stage1_seed,
        "evaluation_condition": condition,
        "collect_rms": bool(collect_rms),
        "use_rms_norm": bool(cfg.use_rms_norm),

        **run_provenance,

        "closed_acc": closed_correct / len(closed),
        "open_strict_acc": open_correct / len(opened),
        "overall_strict_acc": (
            closed_correct + open_correct
        ) / len(records),
    }

    # =========================================================
    # Phase 4: Router / Gate / Lambda summary
    # =========================================================
    if cfg.enable_visual_adapter:
        routing_tensor = torch.tensor(
            [
                [
                    item["routing_f3"],
                    item["routing_f5"],
                    item["routing_f7"],
                ]
                for item in records
            ],
            dtype=torch.float32,
        )
        gates = torch.tensor(
            [item["residual_gate"] for item in records],
            dtype=torch.float32,
        )
        effective = torch.tensor(
            [item["effective_residual_scale"] for item in records],
            dtype=torch.float32,
        )

        safe_routing = routing_tensor.clamp_min(1e-12)
        entropy = -(safe_routing * safe_routing.log()).sum(dim=-1)
        mean_weights = routing_tensor.mean(dim=0)

        gate_q = torch.quantile(
            gates,
            torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32),
        )
        effective_q = torch.quantile(
            effective,
            torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32),
        )

        result["routing"] = {
            "mean_f3": float(mean_weights[0]),
            "mean_f5": float(mean_weights[1]),
            "mean_f7": float(mean_weights[2]),
            "gate_mean": float(gates.mean()),
            "gate_std": float(gates.std(unbiased=False)),
            "lambda": records[0]["lambda_value"],
            "effective_mean": float(effective.mean()),
            "effective_std": float(effective.std(unbiased=False)),
            "entropy_mean": float(entropy.mean()),
            "entropy_normalized": float(entropy.mean() / math.log(3)),
        }

        print("\n" + "=" * 70)
        print(f"三尺度路由诊断 | checkpoint={name} | condition={condition}")
        print("=" * 70)
        print(
            "平均专家权重："
            f"F3={mean_weights[0]:.6f}, "
            f"F5={mean_weights[1]:.6f}, "
            f"F7={mean_weights[2]:.6f}"
        )
        print(
            "residual gate："
            f"mean={gates.mean():.6f}, "
            f"std={gates.std(unbiased=False):.6f}, "
            f"P10={gate_q[0]:.6f}, "
            f"P50={gate_q[1]:.6f}, "
            f"P90={gate_q[2]:.6f}"
        )
        print(f"lambda：{records[0]['lambda_value']:.6f}")
        print(
            "lambda x gate："
            f"mean={effective.mean():.6f}, "
            f"std={effective.std(unbiased=False):.6f}, "
            f"P10={effective_q[0]:.6f}, "
            f"P50={effective_q[1]:.6f}, "
            f"P90={effective_q[2]:.6f}"
        )
        print(
            "路由 entropy："
            f"mean={entropy.mean():.6f}, "
            f"归一化={entropy.mean() / math.log(3):.6f}"
        )
        print("=" * 70)

    # =========================================================
    # Phase 5: RMS summary
    # =========================================================
    if collect_rms:
        if not rms_records:
            raise RuntimeError("collect_rms=True，但整个 Test 没有 RMS 记录。")

        stream_counts = Counter(
            str(item.get("stream_name", "unknown"))
            for item in rms_records
        )
        rho_rows = [
            item for item in rms_records
            if item.get("rho_clipped") is not None
        ]

        result["rms_diagnostics"] = {
            "num_records": len(rms_records),
            "stream_counts": dict(sorted(stream_counts.items())),
            "rho_clipping_rate": (
                sum(bool(item["rho_clipped"]) for item in rho_rows)
                / len(rho_rows)
                if rho_rows
                else None
            ),
        }

    # =========================================================
    # Phase 6: save
    # =========================================================
    jsonl_path = os.path.join(output_dir, "samples.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = os.path.join(output_dir, "samples.csv")
    with open(
        csv_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)

    if collect_rms:
        rms_path = os.path.join(output_dir, "rms_diagnostics.jsonl")
        with open(rms_path, "w", encoding="utf-8") as file:
            for record in rms_records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        rms_path = None

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"逐样本 JSONL：{jsonl_path}")
    print(f"逐样本 CSV：{csv_path}")
    if rms_path is not None:
        print(f"RMS diagnostics：{rms_path}")
    print(f"汇总 JSON：{summary_path}")

    return result


def get_checkpoints(cfg):
    if cfg.test_checkpoint_mode == "best_validation":
        if not os.path.isfile(cfg.best_checkpoint_path):
            raise FileNotFoundError(
                "找不到 Validation 最佳节点记录："
                f"{cfg.best_checkpoint_path}\n"
                "请先运行对应的 stage2_validate_checkpoints_*.py。"
            )

        with open(
            cfg.best_checkpoint_path,
            "r",
            encoding="utf-8",
        ) as file:
            best = json.load(file)

        # 防止拿错 ablation / seed 的 best checkpoint。
        for key, expected in (
            ("ablation_id", cfg.ablation_id),
            ("seed", cfg.seed),
            ("stage1_seed", cfg.stage1_seed),
        ):
            if key in best and best[key] != expected:
                raise ValueError(
                    "best_checkpoint.json 与当前配置不一致："
                    f"{key}: file={best[key]!r}, current={expected!r}"
                )

        return [best["selected_checkpoint_path"]]

    final_weights = os.path.join(cfg.stage2_run_dir, "final_weights")
    checkpoints = glob.glob(
        os.path.join(cfg.stage2_run_dir, "checkpoint-*")
    )
    checkpoints.sort(key=lambda path: int(path.rsplit("-", 1)[-1]))
    checkpoints.append(final_weights)
    return checkpoints


def main():
    args = parse_args()
    cfg = build_eval_config(args)

    if args.collect_rms and not cfg.enable_visual_adapter:
        raise ValueError("--collect-rms 需要启用 Visual Adapter。")

    # =========================================================
    # 1. Qwen loader / processor
    # =========================================================
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

    # =========================================================
    # 2. BioMedCLIP
    # =========================================================
    biomed_extractor = None
    biomed_transform = None
    biomed_tokenizer = None

    if cfg.enable_visual_adapter:
        (
            biomed_extractor,
            biomed_transform,
            biomed_tokenizer,
        ) = load_biomedclip(
            biomedclip_path=cfg.biomedclip_path,
            print_rank=cfg.print_rank,
        )

    # =========================================================
    # 3. SLAKE test
    # =========================================================
    dataset = SLAKEDataset(
        json_path=cfg.slake_test_json_path,
        image_root=cfg.slake_image_root,
    )

    collator = SLAKEEvalCollator(
        processor=processor,
        cfg=cfg,
        biomed_transform=biomed_transform,
        biomed_tokenizer=biomed_tokenizer,
    )

    test_loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_eval_batch_size,
        collate_fn=collator,
        num_workers=cfg.dataloader_num_workers,
        shuffle=False,
    )

    # =========================================================
    # 4. Router shuffle / output identity
    # =========================================================
    shuffle_payload, shuffle_provenance = prepare_shuffle_context(
        args,
        cfg,
        dataset,
    )

    run_tag = derive_run_tag(args, shuffle_payload)
    output_root = os.path.join(
        cfg.test_output_dir,
        "diagnostics",
        run_tag,
    )
    os.makedirs(output_root, exist_ok=True)

    run_provenance = {
        "run_tag": run_tag,
        **shuffle_provenance,
    }

    print("\n" + "=" * 70)
    print("SLAKE Test Diagnostic Run")
    print(
        f"ablation={cfg.ablation_id}, seed={cfg.seed}, "
        f"stage1_seed={cfg.stage1_seed}"
    )
    print(
        f"condition={args.condition}, collect_rms={args.collect_rms}"
    )
    print(f"run_tag={run_tag}")
    print(f"output_root={output_root}")

    if shuffle_payload is not None:
        print(f"shuffle_seed={shuffle_provenance['shuffle_seed']}")
        print(f"shuffle_map={shuffle_provenance['shuffle_map_path']}")
        print(f"shuffle_sha256={shuffle_provenance['shuffle_map_sha256']}")

    print("=" * 70 + "\n")

    # =========================================================
    # 5. Checkpoints
    # =========================================================
    checkpoints = get_checkpoints(cfg)

    print(f"发现 {len(checkpoints)} 个评测节点：")
    for path in checkpoints:
        print(f"  - {os.path.basename(path)}")

    # =========================================================
    # 6. LLM Judge
    # =========================================================
    llm_cfg = LLMAPIConfig()
    llm_client = DeepSeekClient.from_config(llm_cfg)

    print(
        "LLM Judge："
        f"model={llm_cfg.judge_model_name}, "
        "thinking=disabled, "
        f"temperature={llm_cfg.temperature}"
    )

    # =========================================================
    # 7. Evaluation
    # =========================================================
    results = [
        evaluate_checkpoint(
            weights_path=path,
            loader=loader,
            processor=processor,
            cfg=cfg,
            test_loader=test_loader,
            llm_client=llm_client,
            llm_cfg=llm_cfg,
            biomed_extractor=biomed_extractor,
            biomed_tokenizer=biomed_tokenizer,
            output_root=output_root,
            condition=args.condition,
            shuffle_payload=shuffle_payload,
            run_provenance=run_provenance,
            collect_rms=args.collect_rms,
        )
        for path in checkpoints
    ]

    leaderboard_path = os.path.join(output_root, "leaderboard.json")
    with open(leaderboard_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n\n" + "🏆" * 12 + " SLAKE Test " + "🏆" * 12)
    print("| Checkpoint | Closed Acc | Open Acc | Overall Acc |")
    print("| :--- | :---: | :---: | :---: |")

    for result in results:
        print(
            f"| {result['checkpoint']} "
            f"| {result['closed_acc']:.2%} "
            f"| {result['open_strict_acc']:.2%} "
            f"| {result['overall_strict_acc']:.2%} |"
        )

    print(f"\nCondition：{args.condition}")
    print(f"Run tag：{run_tag}")
    print(f"Test checkpoint mode：{cfg.test_checkpoint_mode}")
    print(f"输出目录：{output_root}")
    print(f"排行榜：{leaderboard_path}")


if __name__ == "__main__":
    main()