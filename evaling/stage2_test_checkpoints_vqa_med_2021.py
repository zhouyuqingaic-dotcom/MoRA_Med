import csv
import gc
import glob
import json
import math
import os
import types

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config.LLM_config import LLMAPIConfig
from config.vqa_med_2021.stage2_eval_config_vqa_med_2021 import (
    Stage2EvalConfig,
)
from datas.vqa_med_2021_datasets import (
    VQAMED2021Dataset,
)
from LLM_api.deepseek import DeepSeekClient
from utils.biomedclip.biomed_clip_loader import (
    load_biomedclip,
)
from utils.data_tools.collator.vqa_med_2021.vqa_med_2021_eval_collator import (
    VQAMED2021EvalCollator,
)
from utils.data_tools.prompt_cleaning.vqa_med_2021_answer_cleaning import (
    vqa_med_2021_answer_eval_cleaning,
    vqa_med_2021_references_eval_cleaning,
)
from utils.qwen3vl.qwen3_vl_8B_lora_wrapper import (
    Qwen3VLLoraAndVisualAdapterWrapper,
)
from utils.qwen3vl.qwen3_vl_8B_quant_loader import (
    Qwen3VLQuantizedLoader,
)


def get_adapter(model):
    """
    获取挂载在 Qwen3-VL 视觉塔中的 Visual Adapter。
    """
    return (
        model.base_model
        .model
        .model
        .visual
        .res_adapter
    )


def get_vision_tower(model):
    """获取 Qwen3-VL 视觉塔。"""
    return (
        model.base_model
        .model
        .model
        .visual
    )


def find_lora_weights(weights_path):
    """
    查找当前 checkpoint 的 PEFT LoRA 权重。
    """
    candidates = [
        os.path.join(
            weights_path,
            "adapter_model.safetensors",
        ),
        os.path.join(
            weights_path,
            "adapter_model.bin",
        ),
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "找不到 LoRA 权重：\n"
        + "\n".join(
            f"  {candidate}"
            for candidate in candidates
        )
    )


def load_lora_state(weights_file):
    """读取 safetensors 或 torch bin 格式的 LoRA 权重。"""
    if weights_file.endswith(
        ".safetensors"
    ):
        return load_file(
            weights_file
        )

    return torch.load(
        weights_file,
        map_location="cpu",
    )


def validate_checkpoint(weights_path, cfg):
    """
    检查待评测节点是否包含当前消融所需权重。
    """
    if not os.path.isdir(weights_path):
        raise FileNotFoundError(
            f"评测节点不存在：{weights_path}"
        )

    find_lora_weights(
        weights_path
    )

    if cfg.enable_visual_adapter:
        visual_adapter_path = os.path.join(
            weights_path,
            "visual_adapter.pt",
        )

        if not os.path.isfile(
            visual_adapter_path
        ):
            raise FileNotFoundError(
                "当前消融启用了 Visual Adapter，"
                "但节点中缺少："
                f"{visual_adapter_path}"
            )


def build_model(
    weights_path,
    loader,
    cfg,
    biomed_extractor,
):
    """
    为指定 checkpoint 重建完整模型并加载：
    1. Qwen3-VL 4-bit 底座；
    2. 当前 Stage 2 checkpoint 的 LoRA；
    3. 当前 Stage 2 checkpoint 的 Visual Adapter。
    """
    validate_checkpoint(
        weights_path,
        cfg,
    )

    base_model = loader.load_model()

    wrapper = (
        Qwen3VLLoraAndVisualAdapterWrapper(
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            lora_target_modules=(
                cfg.lora_target_modules
            ),
            gradient_checkpointing=False,

            visual_adapter_hidden_dim=(
                cfg.visual_adapter_hidden_dim
            ),
            visual_adapter_r=(
                cfg.visual_adapter_r
            ),
            enable_visual_adapter=(
                cfg.enable_visual_adapter
            ),

            biomed_extractor=(
                biomed_extractor
            ),
            use_cross_modal_prior=(
                cfg.use_cross_modal_prior
            ),

            router_hidden_dim=(
                cfg.router_hidden_dim
            ),
            scale_mode=cfg.scale_mode,
            fixed_scale_weights=(
                cfg.fixed_scale_weights
            ),

            gate_mode=cfg.gate_mode,
            fixed_gate=cfg.fixed_gate,
            gate_init=cfg.gate_init,

            lambda_mode=cfg.lambda_mode,
            fixed_lambda=cfg.fixed_lambda,
            lambda_max=cfg.lambda_max,
            lambda_init=cfg.lambda_init,

            use_rms_norm=cfg.use_rms_norm,
            residual_norm_eps=(
                cfg.residual_norm_eps
            ),
            residual_norm_ratio_clip=(
                cfg.residual_norm_ratio_clip
            ),
        )
    )

    model = wrapper.wrap(
        base_model
    )

    lora_weights_file = find_lora_weights(
        weights_path
    )

    set_peft_model_state_dict(
        model,
        load_lora_state(
            lora_weights_file
        ),
    )

    if cfg.enable_visual_adapter:
        visual_adapter_path = os.path.join(
            weights_path,
            "visual_adapter.pt",
        )

        state = torch.load(
            visual_adapter_path,
            map_location="cpu",
        )

        get_adapter(model).load_state_dict(
            state,
            strict=True,
        )

    model.requires_grad_(
        False
    )

    if hasattr(
        model,
        "gradient_checkpointing_disable",
    ):
        model.gradient_checkpointing_disable()

    model.config.use_cache = True
    model.eval()

    return model, base_model


def patch_generate_forward(model):
    """
    每个 batch 的 BioMedCLIP 特征已经提前计算并缓存，
    generate() 内不再重复执行 BioMedCLIP。
    """
    def forward(
        self,
        *args,
        **kwargs,
    ):
        kwargs.pop(
            "biomed_image_tensors",
            None,
        )
        kwargs.pop(
            "biomed_text_tokens",
            None,
        )

        return self.original_forward(
            *args,
            **kwargs,
        )

    model.forward = types.MethodType(
        forward,
        model,
    )


def build_llm_judge_user_prompt(
    question,
    references_raw,
    references_norm,
    pred_raw,
    pred_norm,
):
    """
    构造 VQA-Med 2021 多参考答案 LLM Judge Prompt。
    """
    return (
        f"Question: {str(question).strip()}\n"
        "Raw Acceptable Ground Truth Answers: "
        f"{json.dumps(list(references_raw), ensure_ascii=False)}\n"
        "Normalized Acceptable Ground Truth Answers: "
        f"{json.dumps(list(references_norm), ensure_ascii=False)}\n"
        f"Raw Prediction: {str(pred_raw).strip()}\n"
        f"Normalized Prediction: {str(pred_norm).strip()}\n\n"
        "Each item in the Ground Truth list is an alternative acceptable "
        "answer. Judge the Prediction as correct when it is clinically "
        "equivalent to any one acceptable Ground Truth answer. "
        "Return the raw JSON object directly."
    )


def parse_llm_judge_response(
    response_text,
):
    """
    安全解析 LLM Judge 返回的 JSON。
    """
    if response_text == "ERROR":
        return {
            "score": "incorrect",
            "reasoning": (
                "API Request Failed."
            ),
        }

    text = response_text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        result = json.loads(
            text
        )
    except json.JSONDecodeError:
        return {
            "score": "incorrect",
            "reasoning": (
                "JSON Decode Error. "
                f"Raw Output: {text}"
            ),
        }

    valid_scores = {
        "correct",
        "partially_correct",
        "incorrect",
    }

    if result.get("score") not in valid_scores:
        return {
            "score": "incorrect",
            "reasoning": (
                "Invalid Score generated. "
                f"Raw Output: {text}"
            ),
        }

    if "reasoning" not in result:
        result["reasoning"] = (
            result.get("reason")
            or result.get("explanation")
            or "No reasoning provided by model."
        )

    return result


def maybe_limit_test_dataset(
    dataset,
    max_samples,
):
    """
    Smoke Test 时仅评测 Test 前 max_samples 条。
    """
    if max_samples is None:
        return dataset

    max_samples = int(
        max_samples
    )

    if max_samples <= 0:
        raise ValueError(
            "max_vqa_med_2021_test_samples "
            "必须大于 0 或为 None。"
        )

    limit = min(
        max_samples,
        len(dataset),
    )

    print(
        "[Smoke Test] VQA-Med 2021 Test："
        f"使用前 {limit}/{len(dataset)} 条。"
    )

    return Subset(
        dataset,
        list(range(limit)),
    )


def safe_accuracy(
    correct,
    total,
):
    """
    空分组返回 None。

    VQA-Med 2021 Test 当前全部为 OPEN，
    CLOSED 数量为 0 时不能除零。
    """
    if total == 0:
        return None

    return correct / total


def format_metric(
    value,
):
    """终端排行榜中将 None 显示为 N/A。"""
    if value is None:
        return "N/A"

    return f"{value:.2%}"


def csv_safe_record(record):
    """
    将 list/dict 字段转换成 JSON 字符串。
    """
    return {
        key: (
            json.dumps(
                value,
                ensure_ascii=False,
            )
            if isinstance(
                value,
                (list, dict),
            )
            else value
        )
        for key, value
        in record.items()
    }


def evaluate_checkpoint(
    weights_path,
    loader,
    processor,
    cfg,
    test_loader,
    llm_client,
    llm_cfg,
    biomed_extractor,
):
    """
    完成单个 checkpoint 的完整 VQA-Med 2021 Test 评测。
    """
    name = os.path.basename(
        os.path.normpath(
            weights_path
        )
    )

    output_dir = os.path.join(
        cfg.output_dir,
        name,
    )
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print(
        f"开始评测：{name}"
    )
    print(
        f"输出目录：{output_dir}"
    )
    print("=" * 60)

    model, base_model = build_model(
        weights_path,
        loader,
        cfg,
        biomed_extractor,
    )

    adapter = (
        get_adapter(model)
        if cfg.enable_visual_adapter
        else None
    )

    vision_tower = get_vision_tower(
        model
    )

    ref_param = next(
        vision_tower.parameters()
    )
    device = ref_param.device
    adapter_dtype = ref_param.dtype

    if cfg.enable_visual_adapter:
        patch_generate_forward(
            model
        )

    records = []

    # =========================================================
    # Phase 1：本地生成
    # =========================================================
    try:
        with torch.inference_mode():
            for (
                batch_inputs,
                metadata,
            ) in tqdm(
                test_loader,
                desc="Local Inference",
            ):
                inputs = {
                    key: (
                        value.to(
                            device,
                            non_blocking=True,
                        )
                        if isinstance(
                            value,
                            torch.Tensor,
                        )
                        else value
                    )
                    for key, value
                    in batch_inputs.items()
                }

                if cfg.enable_visual_adapter:
                    image = inputs.pop(
                        "biomed_image_tensors"
                    ).to(
                        adapter_dtype
                    )

                    text = inputs.pop(
                        "biomed_text_tokens"
                    )

                    (
                        image_feat,
                        text_feat,
                    ) = model.biomed_extractor(
                        image,
                        text,
                    )

                    vision_tower.current_biomed_img_feat = (
                        image_feat
                    )
                    vision_tower.current_biomed_txt_feat = (
                        text_feat
                    )

                generate_args = {
                    "max_new_tokens": (
                        cfg.max_new_tokens
                    ),
                    "do_sample": (
                        cfg.do_sample
                    ),
                }

                if cfg.do_sample:
                    generate_args[
                        "temperature"
                    ] = cfg.temperature

                generated = model.generate(
                    **inputs,
                    **generate_args,
                )

                if cfg.enable_visual_adapter:
                    routing = (
                        adapter
                        .latest_routing_weights
                        .detach()
                        .float()
                        .cpu()
                    )

                    gate = (
                        adapter
                        .latest_soft_gate
                        .detach()
                        .float()
                        .reshape(-1)
                        .cpu()
                    )

                    lambda_value = float(
                        adapter
                        .latest_lambda
                        .detach()
                        .float()
                        .cpu()
                        .item()
                    )

                    batch_size = len(
                        metadata
                    )

                    if routing.shape != (
                        batch_size,
                        3,
                    ):
                        raise RuntimeError(
                            "routing shape="
                            f"{tuple(routing.shape)}, "
                            f"expected=({batch_size}, 3)"
                        )

                    if gate.shape != (
                        batch_size,
                    ):
                        raise RuntimeError(
                            "gate shape="
                            f"{tuple(gate.shape)}, "
                            f"expected=({batch_size},)"
                        )
                else:
                    routing = None
                    gate = None
                    lambda_value = None

                input_width = inputs[
                    "input_ids"
                ].shape[1]

                answer_token_ids = generated[
                    :,
                    input_width:,
                ]

                predictions = (
                    processor.batch_decode(
                        answer_token_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )

                if len(predictions) != len(
                    metadata
                ):
                    raise RuntimeError(
                        "预测数量与 metadata 数量不一致："
                        f"predictions={len(predictions)}, "
                        f"metadata={len(metadata)}"
                    )

                for i, prediction in enumerate(
                    predictions
                ):
                    meta = metadata[i]

                    references_raw = [
                        str(reference).strip()
                        for reference
                        in meta.get(
                            "references",
                            [],
                        )
                        if str(reference).strip()
                    ]

                    if not references_raw:
                        gt_fallback = str(
                            meta.get(
                                "gt_answer",
                                "",
                            )
                        ).strip()

                        if gt_fallback:
                            references_raw = [
                                gt_fallback
                            ]

                    references_norm = (
                        vqa_med_2021_references_eval_cleaning(
                            references_raw
                        )
                    )

                    if not references_norm:
                        raise ValueError(
                            "参考答案清洗后为空："
                            f"image_id="
                            f"{meta.get('image_id', '')}"
                        )

                    pred_raw = prediction.strip()
                    pred_norm = (
                        vqa_med_2021_answer_eval_cleaning(
                            pred_raw
                        )
                    )

                    matched_reference_index = None

                    for (
                        reference_index,
                        reference_norm,
                    ) in enumerate(
                        references_norm
                    ):
                        if pred_norm == reference_norm:
                            matched_reference_index = (
                                reference_index
                            )
                            break

                    matched = (
                        matched_reference_index
                        is not None
                    )

                    matched_reference_norm = None
                    matched_reference_raw = None

                    if matched:
                        matched_reference_norm = (
                            references_norm[
                                matched_reference_index
                            ]
                        )

                        for raw_reference in references_raw:
                            if (
                                vqa_med_2021_answer_eval_cleaning(
                                    raw_reference
                                )
                                == matched_reference_norm
                            ):
                                matched_reference_raw = (
                                    raw_reference
                                )
                                break

                    answer_type = str(
                        meta.get(
                            "answer_type",
                            "",
                        )
                    ).strip().upper()

                    category = (
                        "closed"
                        if answer_type == "CLOSED"
                        else "open"
                    )

                    record = {
                        "index": meta.get(
                            "index"
                        ),
                        "split": meta.get(
                            "split",
                            "test",
                        ),
                        "image_id": meta.get(
                            "image_id",
                            "",
                        ),
                        "image_name": meta.get(
                            "image_name",
                            "",
                        ),
                        "image_path": meta.get(
                            "image_path",
                            "",
                        ),
                        "question": meta[
                            "question"
                        ],
                        "question_type": meta.get(
                            "question_type",
                            "Abnormality",
                        ),
                        "answer_type": meta.get(
                            "answer_type",
                            "UNKNOWN",
                        ),
                        "question_category": (
                            category
                        ),

                        "gt_answer": meta.get(
                            "gt_answer",
                            "",
                        ),
                        "references_raw": (
                            references_raw
                        ),
                        "references_norm": (
                            references_norm
                        ),

                        "pred_raw": pred_raw,
                        "pred_norm": pred_norm,

                        "is_norm_match": matched,
                        "matched_reference_index": (
                            matched_reference_index
                        ),
                        "matched_reference_raw": (
                            matched_reference_raw
                        ),
                        "matched_reference_norm": (
                            matched_reference_norm
                        ),

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
                        record["routing_f3"] = float(
                            routing[i, 0]
                        )
                        record["routing_f5"] = float(
                            routing[i, 1]
                        )
                        record["routing_f7"] = float(
                            routing[i, 2]
                        )
                        record[
                            "residual_gate"
                        ] = float(
                            gate[i]
                        )
                        record[
                            "lambda_value"
                        ] = lambda_value
                        record[
                            "effective_residual_scale"
                        ] = (
                            lambda_value
                            * float(
                                gate[i]
                            )
                        )

                    records.append(
                        record
                    )

                if cfg.enable_visual_adapter:
                    vision_tower.current_biomed_img_feat = None
                    vision_tower.current_biomed_txt_feat = None

    finally:
        if cfg.enable_visual_adapter:
            vision_tower.current_biomed_img_feat = None
            vision_tower.current_biomed_txt_feat = None

    del model
    del base_model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================
    # Phase 2：LLM Judge
    # =========================================================
    multi_reference_system_prompt = (
        llm_cfg.medical_vqa_llm_judge_system_prompt
        + "\n\nFor VQA-Med 2021, the input may contain multiple "
          "alternative acceptable Ground Truth answers. "
          "Judge the Prediction as correct when it is clinically "
          "equivalent to any one acceptable Ground Truth answer."
    )

    for record in tqdm(
        records,
        desc="LLM Judging",
    ):
        if (
            record["question_category"]
            == "open"
            and not record["is_norm_match"]
        ):
            prompt = build_llm_judge_user_prompt(
                question=(
                    record["question"]
                ),
                references_raw=(
                    record["references_raw"]
                ),
                references_norm=(
                    record["references_norm"]
                ),
                pred_raw=(
                    record["pred_raw"]
                ),
                pred_norm=(
                    record["pred_norm"]
                ),
            )

            response = llm_client.ask(
                multi_reference_system_prompt,
                prompt,
            )

            judged = (
                parse_llm_judge_response(
                    response
                )
            )

            record[
                "llm_judge_raw_response"
            ] = response
            record[
                "llm_judge_score"
            ] = judged["score"]
            record[
                "llm_judge_reason"
            ] = (
                judged.get("reasoning")
                or judged.get("reason")
                or judged.get("explanation")
            )

            record["final_correct"] = (
                judged["score"]
                == "correct"
            )

    if not records:
        raise RuntimeError(
            "没有生成任何评测记录。"
        )

    # =========================================================
    # Phase 3：准确率统计
    # =========================================================
    closed = [
        item
        for item in records
        if item["question_category"]
        == "closed"
    ]

    opened = [
        item
        for item in records
        if item["question_category"]
        == "open"
    ]

    closed_correct = sum(
        int(
            item["final_correct"]
        )
        for item in closed
    )

    open_correct = sum(
        int(
            item["final_correct"]
        )
        for item in opened
    )

    exact_correct = sum(
        int(
            item["is_norm_match"]
        )
        for item in records
    )

    result = {
        "checkpoint": name,
        "checkpoint_path": os.path.abspath(
            weights_path
        ),
        "num_samples": len(
            records
        ),
        "num_closed": len(
            closed
        ),
        "num_open": len(
            opened
        ),

        "closed_acc": safe_accuracy(
            closed_correct,
            len(closed),
        ),
        "open_strict_acc": safe_accuracy(
            open_correct,
            len(opened),
        ),
        "overall_strict_acc": (
            closed_correct
            + open_correct
        ) / len(records),

        "normalized_exact_any_reference": (
            exact_correct
            / len(records)
        ),
    }

    # =========================================================
    # Phase 4：Router / Gate / Lambda 诊断
    # =========================================================
    if cfg.enable_visual_adapter:
        routing = torch.tensor(
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
            [
                item["residual_gate"]
                for item in records
            ],
            dtype=torch.float32,
        )

        effective = torch.tensor(
            [
                item[
                    "effective_residual_scale"
                ]
                for item in records
            ],
            dtype=torch.float32,
        )

        safe_routing = routing.clamp_min(
            1e-12
        )
        entropy = -(
            safe_routing
            * safe_routing.log()
        ).sum(
            dim=-1
        )

        mean_weights = routing.mean(
            dim=0
        )

        quantiles = torch.tensor(
            [0.1, 0.5, 0.9],
            dtype=torch.float32,
        )

        gate_q = torch.quantile(
            gates,
            quantiles,
        )

        effective_q = torch.quantile(
            effective,
            quantiles,
        )

        result["routing"] = {
            "mean_f3": float(
                mean_weights[0]
            ),
            "mean_f5": float(
                mean_weights[1]
            ),
            "mean_f7": float(
                mean_weights[2]
            ),
            "gate_mean": float(
                gates.mean()
            ),
            "gate_std": float(
                gates.std(
                    unbiased=False
                )
            ),
            "gate_p10": float(
                gate_q[0]
            ),
            "gate_p50": float(
                gate_q[1]
            ),
            "gate_p90": float(
                gate_q[2]
            ),
            "lambda": records[0][
                "lambda_value"
            ],
            "effective_mean": float(
                effective.mean()
            ),
            "effective_std": float(
                effective.std(
                    unbiased=False
                )
            ),
            "effective_p10": float(
                effective_q[0]
            ),
            "effective_p50": float(
                effective_q[1]
            ),
            "effective_p90": float(
                effective_q[2]
            ),
            "entropy_mean": float(
                entropy.mean()
            ),
            "entropy_normalized": float(
                entropy.mean()
                / math.log(3)
            ),
        }

        print("\n" + "=" * 70)
        print(
            "三尺度路由诊断 | "
            f"checkpoint={name}"
        )
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

        print(
            f"lambda："
            f"{records[0]['lambda_value']:.6f}"
        )

        print(
            "lambda × gate："
            f"mean={effective.mean():.6f}, "
            f"std={effective.std(unbiased=False):.6f}, "
            f"P10={effective_q[0]:.6f}, "
            f"P50={effective_q[1]:.6f}, "
            f"P90={effective_q[2]:.6f}"
        )

        print(
            "路由 entropy："
            f"mean={entropy.mean():.6f}, "
            f"归一化="
            f"{entropy.mean() / math.log(3):.6f}"
        )

        print("=" * 70)

    # =========================================================
    # Phase 5：保存逐样本和汇总结果
    # =========================================================
    jsonl_path = os.path.join(
        output_dir,
        "samples.jsonl",
    )

    with open(
        jsonl_path,
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    csv_path = os.path.join(
        output_dir,
        "samples.csv",
    )

    with open(
        csv_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                records[0].keys()
            ),
        )
        writer.writeheader()

        for record in records:
            writer.writerow(
                csv_safe_record(
                    record
                )
            )

    summary_path = os.path.join(
        output_dir,
        "summary.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"逐样本 JSONL：{jsonl_path}"
    )
    print(
        f"逐样本 CSV：{csv_path}"
    )
    print(
        f"汇总 JSON：{summary_path}"
    )
    print(
        "Normalized Exact Any Reference："
        f"{result['normalized_exact_any_reference']:.2%}"
    )
    print(
        "Overall Strict Accuracy："
        f"{result['overall_strict_acc']:.2%}"
    )

    return result


def get_checkpoints(cfg):
    """
    与 SLAKE、VQA-Med 2019 相同：

    eval_final_weights_only=True：
        仅返回 final_weights。

    eval_final_weights_only=False：
        按 step 顺序返回 checkpoint-*，
        最后追加 final_weights。
    """
    final_weights = os.path.join(
        cfg.stage2_run_dir,
        "final_weights",
    )

    if cfg.eval_final_weights_only:
        validate_checkpoint(
            final_weights,
            cfg,
        )
        return [
            final_weights
        ]

    checkpoints = [
        path
        for path in glob.glob(
            os.path.join(
                cfg.stage2_run_dir,
                "checkpoint-*",
            )
        )
        if os.path.isdir(path)
    ]

    checkpoints.sort(
        key=lambda path: int(
            path.rsplit(
                "-",
                1,
            )[-1]
        )
    )

    if os.path.isdir(
        final_weights
    ):
        checkpoints.append(
            final_weights
        )

    if not checkpoints:
        raise FileNotFoundError(
            "没有发现可评测节点："
            f"{cfg.stage2_run_dir}"
        )

    for checkpoint in checkpoints:
        validate_checkpoint(
            checkpoint,
            cfg,
        )

    return checkpoints


def main():
    cfg = Stage2EvalConfig()

    os.makedirs(
        cfg.output_dir,
        exist_ok=True,
    )

    checkpoints = get_checkpoints(
        cfg
    )

    print(
        f"发现 {len(checkpoints)} 个评测节点："
    )

    for path in checkpoints:
        print(
            f"  - {os.path.basename(path)}"
        )

    # =========================================================
    # 1. Qwen3-VL Loader 与 Processor
    # =========================================================
    loader = Qwen3VLQuantizedLoader(
        model_path=(
            cfg.model_name_or_path
        ),
        processor_path=(
            cfg.model_name_or_path
        ),
        load_in_4bit=(
            cfg.load_in_4bit
        ),
        bnb_4bit_quant_type=(
            cfg.bnb_4bit_quant_type
        ),
        bnb_4bit_use_double_quant=(
            cfg.bnb_4bit_use_double_quant
        ),
        bnb_4bit_compute_dtype=(
            cfg.bnb_4bit_compute_dtype
        ),
        torch_dtype=(
            cfg.torch_dtype
        ),
        attn_implementation=(
            cfg.attn_implementation
        ),
        device_map="auto",
    )

    processor = loader.load_processor()

    processor.tokenizer.padding_side = (
        "left"
    )

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
            biomedclip_path=(
                cfg.biomedclip_path
            ),
            print_rank=(
                cfg.print_rank
            ),
        )

    # =========================================================
    # 3. VQA-Med 2021 Test DataLoader
    # =========================================================
    dataset = VQAMED2021Dataset(
        jsonl_path=(
            cfg.vqa_med_2021_test_jsonl_path
        ),
        expected_split="test",
        expected_count=(
            cfg.vqa_med_2021_test_expected_count
        ),
        verify_images=(
            cfg.vqa_med_2021_verify_images
        ),
        strict=(
            cfg.vqa_med_2021_dataset_strict
        ),
    )

    dataset = maybe_limit_test_dataset(
        dataset,
        cfg.max_vqa_med_2021_test_samples,
    )

    collator = VQAMED2021EvalCollator(
        processor=processor,
        cfg=cfg,
        biomed_transform=(
            biomed_transform
        ),
        biomed_tokenizer=(
            biomed_tokenizer
        ),
    )

    test_loader = DataLoader(
        dataset,
        batch_size=(
            cfg.per_device_eval_batch_size
        ),
        collate_fn=collator,
        num_workers=(
            cfg.dataloader_num_workers
        ),
        shuffle=False,
        pin_memory=True,
        persistent_workers=(
            cfg.dataloader_num_workers > 0
        ),
    )

    print(
        "VQA-Med 2021 Test 样本数："
        f"{len(dataset)}"
    )

    # =========================================================
    # 4. LLM Judge
    # =========================================================
    llm_cfg = LLMAPIConfig()

    llm_client = (
        DeepSeekClient.from_config(
            llm_cfg
        )
    )

    print(
        "LLM Judge："
        f"model={llm_cfg.judge_model_name}, "
        "thinking=disabled, "
        f"temperature={llm_cfg.temperature}"
    )

    # =========================================================
    # 5. 逐 checkpoint 评测
    # =========================================================
    results = [
        evaluate_checkpoint(
            path,
            loader,
            processor,
            cfg,
            test_loader,
            llm_client,
            llm_cfg,
            biomed_extractor,
        )
        for path in checkpoints
    ]

    leaderboard_path = os.path.join(
        cfg.output_dir,
        "leaderboard.json",
    )

    with open(
        leaderboard_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # =========================================================
    # 6. 终端排行榜
    # =========================================================
    print(
        "\n\n"
        + "🏆" * 20
        + " VQA-Med 2021 评测结果 "
        + "🏆" * 20
    )

    print(
        f"| 评测节点 "
        f"({cfg.ablation_id}) "
        "| Closed Acc "
        "| Open Acc "
        "| Overall Acc "
        "| Exact-Any-Ref |"
    )

    print(
        "| :--- | :---: "
        "| :---: | :---: | :---: |"
    )

    for result in results:
        print(
            f"| {result['checkpoint']} "
            f"| {format_metric(result['closed_acc'])} "
            f"| {format_metric(result['open_strict_acc'])} "
            f"| {format_metric(result['overall_strict_acc'])} "
            f"| {format_metric(result['normalized_exact_any_reference'])} |"
        )

    best = max(
        results,
        key=lambda item: (
            item["overall_strict_acc"]
        ),
    )

    print(
        f"\n最佳节点："
        f"{best['checkpoint']} "
        f"(Overall: "
        f"{best['overall_strict_acc']:.2%})"
    )

    print(
        f"评测输出目录："
        f"{cfg.output_dir}"
    )

    print(
        f"排行榜："
        f"{leaderboard_path}"
    )


if __name__ == "__main__":
    main()
