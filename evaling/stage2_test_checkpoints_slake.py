import glob
import os
import types

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. SLAKE 评测配置与数据集
from config.slake.stage2_eval_config_slake import Stage2EvalConfig
from datas.slake_datasets import SLAKEDataset

# 2. Qwen3-VL 量化加载器与新版训练 Wrapper
from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader
from utils.qwen3vl.qwen3_vl_8B_lora_wrapper import (
    Qwen3VLLoraAndVisualAdapterWrapper,
)

# 3. SLAKE Eval Collator 与答案清洗器
from utils.data_tools.collator.slake.slake_datasets_eval_collator import (
    SLAKEEvalCollator,
)
from utils.data_tools.prompt_cleaning.slake_answer_cleaning import (
    slake_answer_eval_cleaning,
)

# 4. BioMedCLIP
from utils.biomedclip.biomed_clip_loader import load_biomedclip

# 5. LLM Judge
from config.LLM_config import LLMAPIConfig
from LLM_api.gpt_5_mini import GPT5MiniClient
from LLM_api.prompts.slake_prompt_builder_gpt_5_mini import (
    build_llm_judge_user_prompt,
    parse_llm_judge_response,
)


def get_visual_adapter(model):
    """获取当前模型中的 RoMA-Net V2-lite visual adapter。"""
    if hasattr(model, "module"):
        model = model.module

    try:
        return (
            model.base_model
            .model
            .model
            .visual
            .res_adapter
        )
    except AttributeError:
        return None


def build_model_for_checkpoint(
    weights_path,
    loader,
    cfg,
    biomed_extractor,
):
    """
    使用与 Stage 2 训练完全一致的 Wrapper 重建模型，
    然后加载当前 checkpoint 的 LoRA 与 visual adapter。

    这里不再手工创建旧版三专家结构，也不再手工 patch 视觉塔；
    模型结构与视觉塔 patch 全部由新版 Wrapper 负责。
    """
    lora_path = os.path.join(
        weights_path,
        "adapter_model.safetensors",
    )

    if not os.path.isfile(lora_path):
        print(
            f"⚠️ 跳过 {weights_path}："
            f"缺少 LoRA 权重 {lora_path}"
        )
        return None

    # 每个 checkpoint 都从同一个干净的 4-bit 底座重新构建。
    base_model = loader.load_model()

    wrapper = Qwen3VLLoraAndVisualAdapterWrapper(
        # LoRA：必须与 Stage 2 训练配置一致。
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=cfg.lora_target_modules,
        gradient_checkpointing=False,

        # Visual adapter。
        visual_adapter_hidden_dim=cfg.visual_adapter_hidden_dim,
        visual_adapter_r=cfg.visual_adapter_r,
        enable_visual_adapter=cfg.enable_visual_adapter,

        # BioMedCLIP route feature。
        biomed_extractor=biomed_extractor,
        use_cross_modal_prior=cfg.use_cross_modal_prior,

        # Router backbone。
        router_hidden_dim=cfg.router_hidden_dim,

        # 四尺度 routing。
        scale_mode=cfg.scale_mode,
        fixed_scale_weights=cfg.fixed_scale_weights,

        # Soft gate。
        gate_mode=cfg.gate_mode,
        fixed_gate=cfg.fixed_gate,
        gate_init=cfg.gate_init,

        # 有界 lambda。
        lambda_mode=cfg.lambda_mode,
        fixed_lambda=cfg.fixed_lambda,
        lambda_max=cfg.lambda_max,
        lambda_init=cfg.lambda_init,

        # RMS residual normalization。
        use_rms_norm=cfg.use_rms_norm,
        residual_norm_eps=cfg.residual_norm_eps,
        residual_norm_ratio_clip=cfg.residual_norm_ratio_clip,
    )

    model = wrapper.wrap(base_model)

    # ---------------------------------------------------------
    # 加载当前 Stage 2 checkpoint 的 LoRA 权重
    # ---------------------------------------------------------
    lora_state_dict = load_file(lora_path)

    set_peft_model_state_dict(
        model,
        lora_state_dict,
    )

    # ---------------------------------------------------------
    # A1～A5：加载当前 checkpoint 的 visual adapter
    # A0：没有 visual adapter，因此跳过
    # ---------------------------------------------------------
    if cfg.enable_visual_adapter:
        visual_adapter_path = os.path.join(
            weights_path,
            "visual_adapter.pt",
        )

        if not os.path.isfile(visual_adapter_path):
            print(
                f"⚠️ 跳过 {weights_path}："
                f"缺少 visual adapter 权重 {visual_adapter_path}"
            )
            del model
            del base_model
            torch.cuda.empty_cache()
            return None

        adapter_module = get_visual_adapter(model)

        if adapter_module is None:
            raise RuntimeError(
                "模型中没有找到 visual.res_adapter。"
            )

        adapter_state_dict = torch.load(
            visual_adapter_path,
            map_location="cpu",
        )

        # 不考虑旧版兼容；结构不一致时立即报错。
        adapter_module.load_state_dict(
            adapter_state_dict,
            strict=True,
        )

    # ---------------------------------------------------------
    # 评测模式：关闭全部梯度和梯度检查点，开启 KV cache
    # ---------------------------------------------------------
    model.requires_grad_(False)

    if hasattr(
        model,
        "gradient_checkpointing_disable",
    ):
        model.gradient_checkpointing_disable()

    if hasattr(model, "config"):
        model.config.use_cache = True

    model.eval()

    return model, base_model


def evaluate_single_checkpoint(
    weights_path,
    loader,
    processor,
    cfg,
    test_loader,
    llm_client,
    llm_cfg,
    biomed_extractor=None,
):
    """评测单个 Stage 2 checkpoint。"""
    print("\n" + "=" * 60)
    print(
        "🌟 开始评测 Checkpoint: "
        f"{os.path.basename(weights_path)}"
    )
    print("=" * 60)

    built_model = build_model_for_checkpoint(
        weights_path=weights_path,
        loader=loader,
        cfg=cfg,
        biomed_extractor=biomed_extractor,
    )

    if built_model is None:
        return None

    model, base_model = built_model

    vision_tower = (
        model.base_model
        .model
        .model
        .visual
    )

    ref_param = next(
        vision_tower.parameters()
    )

    model_device = ref_param.device

    adapter_dtype = (
        ref_param.dtype
        if ref_param.is_floating_point()
        else torch.bfloat16
    )

    # =========================================================
    # 推理专用 BioMedCLIP 缓存桥
    # =========================================================
    # 新版 Wrapper 已经负责 patch 视觉塔。
    # 训练时，Wrapper 的外层 forward 会从输入中提取 BioMedCLIP
    # 输入并计算特征；但 generate() 会连续执行多次 forward。
    #
    # 评测时在每个 batch 开始前只计算一次 BioMedCLIP 特征，
    # 并缓存到 vision_tower；随后让 generate() 的多次 forward
    # 直接复用这些缓存特征，避免每生成一个 token 都重复编码。
    if cfg.enable_visual_adapter:
        def eval_model_forward(
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

            # original_forward 是 Wrapper 在 wrap() 时保存的
            # PEFT/Qwen 原始 forward；调用它不会重新清空缓存特征。
            return self.original_forward(
                *args,
                **kwargs,
            )

        model.forward = types.MethodType(
            eval_model_forward,
            model,
        )

    # =========================================================
    # Phase 1：本地生成与严格匹配统计
    # =========================================================
    metrics = {
        "total": 0,
        "norm_match": 0,
        "closed_total": 0,
        "closed_correct": 0,
        "open_total": 0,
        "open_correct": 0,
    }
    all_records = []

    with torch.no_grad():
        for batch_inputs, metadata_list in tqdm(
            test_loader,
            desc="Local Inference",
        ):
            inputs = {
                key: (
                    value.to(model_device)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in batch_inputs.items()
            }

            if cfg.enable_visual_adapter:
                biomed_img = inputs.pop(
                    "biomed_image_tensors"
                )
                biomed_txt = inputs.pop(
                    "biomed_text_tokens"
                )

                # BioMedCLIP 图像输入使用模型浮点 dtype；
                # 文本 token 保持原始整数 dtype。
                biomed_img = biomed_img.to(
                    dtype=adapter_dtype
                )

                img_feat, txt_feat = (
                    model.biomed_extractor(
                        biomed_img,
                        biomed_txt,
                    )
                )

                vision_tower.current_biomed_img_feat = (
                    img_feat
                )
                vision_tower.current_biomed_txt_feat = (
                    txt_feat
                )

            generation_kwargs = {
                "max_new_tokens": cfg.max_new_tokens,
                "do_sample": cfg.do_sample,
            }

            # 贪心解码时不传 temperature，避免无效参数警告。
            if cfg.do_sample:
                generation_kwargs["temperature"] = (
                    cfg.temperature
                )

            generated_ids = model.generate(
                **inputs,
                **generation_kwargs,
            )

            if cfg.enable_visual_adapter:
                # 当前 batch 完成后立即清空，防止特征残留到下一批。
                vision_tower.current_biomed_img_feat = None
                vision_tower.current_biomed_txt_feat = None

            generated_ids_trimmed = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(
                    inputs["input_ids"],
                    generated_ids,
                )
            ]

            output_texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for index, raw_pred in enumerate(output_texts):
                meta = metadata_list[index]

                gt_raw = meta.get(
                    "gt_answer",
                    meta.get("answer", ""),
                )

                answer_type = (
                    meta.get("answer_type", "")
                    .strip()
                    .upper()
                )
                is_closed_question = (
                    answer_type == "CLOSED"
                )

                pred_norm = slake_answer_eval_cleaning(
                    raw_pred
                )
                gt_norm = slake_answer_eval_cleaning(
                    gt_raw
                )
                is_norm_match = (
                    pred_norm == gt_norm
                )

                metrics["total"] += 1

                if is_norm_match:
                    metrics["norm_match"] += 1

                if is_closed_question:
                    metrics["closed_total"] += 1
                    if is_norm_match:
                        metrics["closed_correct"] += 1
                else:
                    metrics["open_total"] += 1
                    if is_norm_match:
                        metrics["open_correct"] += 1

                all_records.append(
                    {
                        "question": meta["question"],
                        "gt_raw": gt_raw,
                        "gt_norm": gt_norm,
                        "pred_raw": raw_pred.strip(),
                        "pred_norm": pred_norm,
                        "is_norm_match": is_norm_match,
                        "question_category": (
                            "closed"
                            if is_closed_question
                            else "open"
                        ),
                    }
                )

    # 本地生成完成后释放当前 checkpoint 模型显存。
    del model
    del base_model
    torch.cuda.empty_cache()

    # =========================================================
    # Phase 2：LLM Judge 评估开放题语义正确性
    # =========================================================
    semantic_rescued_strict = 0
    semantic_rescued_relaxed = 0

    for record in tqdm(
        all_records,
        desc="LLM Judging",
    ):
        if (
            record["question_category"] == "open"
            and not record["is_norm_match"]
        ):
            user_prompt = build_llm_judge_user_prompt(
                question=record["question"],
                gt_raw=record["gt_raw"],
                gt_norm=record["gt_norm"],
                pred_raw=record["pred_raw"],
                pred_norm=record["pred_norm"],
            )

            raw_response = llm_client.ask(
                llm_cfg.vqa_rad_llm_judge_system_prompt,
                user_prompt,
                temperature=0.0,
            )

            parsed_result = parse_llm_judge_response(
                raw_response
            )

            if parsed_result["score"] == "correct":
                semantic_rescued_strict += 1
                semantic_rescued_relaxed += 1
            elif parsed_result["score"] == "partially_correct":
                semantic_rescued_relaxed += 1

    open_semantic_strict_correct = (
        metrics["open_correct"]
        + semantic_rescued_strict
    )

    overall_strict_acc = (
        (
            metrics["closed_correct"]
            + open_semantic_strict_correct
        )
        / metrics["total"]
        if metrics["total"]
        else 0.0
    )

    return {
        "checkpoint": os.path.basename(weights_path),
        "closed_acc": (
            metrics["closed_correct"]
            / metrics["closed_total"]
            if metrics["closed_total"]
            else 0.0
        ),
        "open_strict_acc": (
            open_semantic_strict_correct
            / metrics["open_total"]
            if metrics["open_total"]
            else 0.0
        ),
        "overall_strict_acc": overall_strict_acc,
    }


def main():
    cfg = Stage2EvalConfig()

    # =========================================================
    # 1. 扫描当前 Stage 2 run 下的 checkpoint 与 final_weights
    # =========================================================
    base_weight_dir = cfg.stage2_run_dir

    checkpoint_dirs = glob.glob(
        os.path.join(
            base_weight_dir,
            "checkpoint-*",
        )
    )

    checkpoint_dirs.sort(
        key=lambda path: int(
            path.rsplit("-", 1)[-1]
        )
    )

    final_weights_path = os.path.join(
        base_weight_dir,
        "final_weights",
    )

    if os.path.isdir(final_weights_path):
        checkpoint_dirs.append(
            final_weights_path
        )

    if not checkpoint_dirs:
        print(
            f"❌ 在 {base_weight_dir} 下没有找到任何 "
            "checkpoint-* 或 final_weights 目录。"
        )
        return

    print(
        f"🔍 发现 {len(checkpoint_dirs)} 个 SLAKE 评测节点 | "
        f"消融={cfg.ablation_id}"
    )

    for checkpoint_path in checkpoint_dirs:
        print(
            "  - "
            f"{os.path.basename(checkpoint_path)}"
        )

    # =========================================================
    # 2. 初始化底座加载器与 Processor
    # =========================================================
    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path,
        processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=(
            cfg.bnb_4bit_use_double_quant
        ),
        bnb_4bit_compute_dtype=(
            cfg.bnb_4bit_compute_dtype
        ),
        torch_dtype=cfg.torch_dtype,
        attn_implementation=cfg.attn_implementation,
        device_map="auto",
    )

    processor = loader.load_processor()

    # 生成阶段使用左侧 padding，便于批量自回归生成。
    processor.tokenizer.padding_side = "left"

    # =========================================================
    # 3. A1～A5 加载 BioMedCLIP；A0 跳过
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
    # 4. 构造 SLAKE test DataLoader
    # =========================================================
    test_dataset = SLAKEDataset(
        json_path=cfg.slake_test_json_path,
        image_root=cfg.slake_image_root,
    )

    test_collator = SLAKEEvalCollator(
        processor=processor,
        cfg=cfg,
        biomed_transform=biomed_transform,
        biomed_tokenizer=biomed_tokenizer,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.per_device_eval_batch_size,
        collate_fn=test_collator,
        num_workers=cfg.dataloader_num_workers,
        shuffle=False,
    )

    # =========================================================
    # 5. 初始化 LLM Judge
    # =========================================================
    llm_cfg = LLMAPIConfig()

    llm_client = GPT5MiniClient(
        api_key=llm_cfg.gpt_5_mini_key,
        base_url=llm_cfg.base_url,
        model=llm_cfg.judge_model_name,
    )

    # =========================================================
    # 6. 逐 checkpoint 评测
    # =========================================================
    results = []

    for checkpoint_path in checkpoint_dirs:
        result = evaluate_single_checkpoint(
            weights_path=checkpoint_path,
            loader=loader,
            processor=processor,
            cfg=cfg,
            test_loader=test_loader,
            llm_client=llm_client,
            llm_cfg=llm_cfg,
            biomed_extractor=biomed_extractor,
        )

        if result is not None:
            results.append(result)

    if not results:
        print(
            "❌ 没有任何 checkpoint 完成评测，"
            "请检查 LoRA 与 visual_adapter.pt 是否齐全。"
        )
        return

    # =========================================================
    # 7. 输出排行榜
    # =========================================================
    print(
        "\n\n"
        + "🏆" * 20
        + " SLAKE 评测结果 "
        + "🏆" * 20
    )

    print(
        f"| 评测节点 ({cfg.ablation_id}) "
        "| Closed Acc | Open Acc | Overall Acc |"
    )
    print(
        "| :--- | :---: | :---: | :---: |"
    )

    for result in results:
        print(
            f"| {result['checkpoint']} "
            f"| {result['closed_acc']:.2%} "
            f"| {result['open_strict_acc']:.2%} "
            f"| {result['overall_strict_acc']:.2%} |"
        )

    best_overall = max(
        results,
        key=lambda item: item["overall_strict_acc"],
    )

    print(
        "\n🎯 最佳节点："
        f"{best_overall['checkpoint']} "
        f"(Overall: {best_overall['overall_strict_acc']:.2%})"
    )


if __name__ == "__main__":
    main()