import os
import random
import numpy as np
import torch
import torch.distributed as dist
from transformers import Trainer, TrainingArguments

# 1. 导入配置
from config.stage1_train_config_mimic_cxr import TrainConfig

# 2. 导入数据集与 Collator
from datas.mimic_cxr_datasets import MIMICCXRDataset
from utils.data_tools.collator.mimic_cxr.mimic_cxr_datasets_train_collator import (
    MIMICCXRTrainCollator,
)

# 3. 导入模型加载与包装器
from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader
from utils.qwen3vl.qwen3_vl_8B_lora_wrapper import (
    Qwen3VLLoraAndVisualAdapterWrapper,
)
from utils.biomedclip.biomed_clip_loader import load_biomedclip

# 4. 导入 DDP 打印工具
from utils.ddp.ddp_utils import ddp_print
# 导入学习率相关工具
from utils.training.discriminative_optimizer import (
    build_discriminative_adamw,
    format_optimizer_groups,
)

def set_seed(seed: int):
    """
    设置全局随机种子，尽量保证不同运行之间具有可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # ==========================================
    # 0. DDP 环境感知与初始化
    # ==========================================
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank != -1:
        torch.cuda.set_device(local_rank)

    cfg = TrainConfig()
    output_dir = cfg.output_dir

    set_seed(cfg.seed)

    ddp_print("\n" + "=" * 60, print_rank=cfg.print_rank)
    ddp_print(
        f"🚀 [1/6] 启动 RoMA-Net V2-lite Stage 1 分布式训练！"
        f"当前消融: {cfg.ablation_id}",
        print_rank=cfg.print_rank,
    )
    ddp_print(
        f"    enable_visual_adapter={cfg.enable_visual_adapter}, "
        f"scale_mode={cfg.scale_mode}, "
        f"gate_mode={cfg.gate_mode}, "
        f"lambda_mode={cfg.lambda_mode}, "
        f"use_rms_norm={cfg.use_rms_norm}",
        print_rank=cfg.print_rank,
    )
    ddp_print("=" * 60, print_rank=cfg.print_rank)

    if local_rank in [-1, 0]:
        os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # 1. 加载 MIMIC-CXR 数据集
    # ==========================================
    ddp_print(
        "⏳ [2/6] 正在加载 MIMIC-CXR 训练集...",
        print_rank=cfg.print_rank,
    )

    train_dataset = MIMICCXRDataset(
        csv_path=cfg.mimic_cxr_metadata_csv,
        image_root=cfg.mimic_cxr_image_root,
        report_root=cfg.mimic_cxr_report_root,
        target_section=cfg.mimic_cxr_target_section,
        load_report=cfg.mimic_cxr_load_report,
        allowed_view_positions=cfg.mimic_cxr_view_positions,
        drop_empty_target=cfg.mimic_cxr_drop_empty_target,
        cache_dir=cfg.mimic_cxr_cache_dir,
        use_indices_cache=cfg.mimic_cxr_use_indices_cache,
        rebuild_indices_cache=cfg.mimic_cxr_rebuild_indices_cache,
        cache_prefix=cfg.mimic_cxr_cache_prefix,
    )

    if cfg.max_mimic_cxr_train_samples is not None:
        train_dataset.samples = train_dataset.samples[:cfg.max_mimic_cxr_train_samples]

        ddp_print(
            f"⚠️ 已截断数据集用于调试，当前样本量: {len(train_dataset)}",
            print_rank=cfg.print_rank,
        )
    else:
        ddp_print(
            f"✅ 数据集加载完成，共有 {len(train_dataset)} 条训练样本。",
            print_rank=cfg.print_rank,
        )

    # ==========================================
    # 2. 加载 Qwen3-VL 4-bit 底座模型与 Processor
    # ==========================================
    ddp_print(
        "\n⏳ [3/6] 正在加载 Qwen3-VL 4-bit 底座模型...",
        print_rank=cfg.print_rank,
    )

    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path,
        processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        torch_dtype=cfg.torch_dtype,
        attn_implementation=cfg.attn_implementation,
        device_map={"": local_rank} if local_rank != -1 else "auto",
    )

    base_model, processor = loader.load()
    processor.tokenizer.padding_side = "right"

    ddp_print("✅ Qwen3-VL 底座与 Processor 加载完毕。", print_rank=cfg.print_rank)

    # ==========================================
    # 3. 加载 BioMedCLIP
    # ==========================================
    biomed_extractor = None
    biomed_transform = None
    biomed_tokenizer = None

    if cfg.enable_visual_adapter:
        ddp_print(
            "\n⏳ [4/6] 当前启用 visual adapter，正在加载 BioMedCLIP...",
            print_rank=cfg.print_rank,
        )

        biomed_extractor, biomed_transform, biomed_tokenizer = load_biomedclip(
            biomedclip_path=cfg.biomedclip_path,
            print_rank=cfg.print_rank,
        )

        ddp_print("✅ BioMedCLIP 加载完毕。", print_rank=cfg.print_rank)
    else:
        ddp_print(
            "\nℹ️ [4/6] 当前为 A0 / LoRA-only，不加载 BioMedCLIP。",
            print_rank=cfg.print_rank,
        )

    # ==========================================
    # 4. 注入 LoRA 与 RoMA-Net V2-lite Visual Adapter
    # ==========================================
    ddp_print(
        "\n⏳ [5/6] 正在注入 LoRA 与 RoMA-Net V2-lite Visual Adapter...",
        print_rank=cfg.print_rank,
    )

    wrapper = Qwen3VLLoraAndVisualAdapterWrapper(
        # LoRA
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=cfg.lora_target_modules,
        gradient_checkpointing=cfg.gradient_checkpointing,

        # Visual adapter
        visual_adapter_hidden_dim=cfg.visual_adapter_hidden_dim,
        visual_adapter_r=cfg.visual_adapter_r,
        enable_visual_adapter=cfg.enable_visual_adapter,

        # BioMedCLIP route feature
        biomed_extractor=biomed_extractor,
        use_cross_modal_prior=cfg.use_cross_modal_prior,

        # Router
        router_hidden_dim=cfg.router_hidden_dim,

        # Scale routing
        scale_mode=cfg.scale_mode,
        fixed_scale_weights=cfg.fixed_scale_weights,

        # Soft gate
        gate_mode=cfg.gate_mode,
        fixed_gate=cfg.fixed_gate,
        gate_init=cfg.gate_init,

        # Lambda
        lambda_mode=cfg.lambda_mode,
        fixed_lambda=cfg.fixed_lambda,
        lambda_max=cfg.lambda_max,
        lambda_init=cfg.lambda_init,

        # RMS residual norm
        use_rms_norm=cfg.use_rms_norm,
        residual_norm_eps=cfg.residual_norm_eps,
        residual_norm_ratio_clip=cfg.residual_norm_ratio_clip,
    )

    model = wrapper.wrap(base_model)

    ddp_print("✅ 模型接驳完成。", print_rank=cfg.print_rank)

    # ==========================================
    # 5. 初始化 Collator
    # ==========================================
    ddp_print(
        "\n⏳ [6/6] 挂载 MIMIC-CXR 专属 Collator...",
        print_rank=cfg.print_rank,
    )

    collator = MIMICCXRTrainCollator(
        processor=processor,
        cfg=cfg,
        biomed_transform=biomed_transform,
        biomed_tokenizer=biomed_tokenizer,
    )

    # ==========================================
    # 6. 配置 Hugging Face TrainingArguments
    # ==========================================

    ddp_print(
        f"\n🔥 启动 Hugging Face Trainer... 设定轮数: {cfg.num_train_epochs}",
        print_rank=cfg.print_rank,
    )

    ddp_print(
        "    DDP 配置：find_unused_parameters=False",
        print_rank=cfg.print_rank,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,

        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,

        learning_rate=(
            cfg.lora_learning_rate
            if cfg.use_discriminative_lr
            else cfg.learning_rate
        ),
        weight_decay=cfg.weight_decay,
        optim="adamw_torch",

        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        max_grad_norm=cfg.max_grad_norm,

        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,

        bf16=(
                cfg.torch_dtype == "bfloat16"
                or cfg.torch_dtype == torch.bfloat16
        ),
        fp16=(
                cfg.torch_dtype == "float16"
                or cfg.torch_dtype == torch.float16
        ),

        dataloader_num_workers=cfg.dataloader_num_workers,

        # wrapper 中的 prepare_model_for_kbit_training 已处理梯度检查点。
        gradient_checkpointing=False,

        # 必须为 False，否则 Trainer 可能删除 BioMedCLIP 输入。
        remove_unused_columns=False,

        # fixed 分支已在 fusion module 中通过 requires_grad=False 冻结。
        # 因此剩余可训练参数均应参与当前计算图，不需要额外搜索 unused parameters。
        ddp_find_unused_parameters=False,

        report_to="none",
    )

    trainer_kwargs = {}

    if cfg.use_discriminative_lr:
        optimizer = build_discriminative_adamw(
            model,
            lora_learning_rate=(
                cfg.lora_learning_rate
            ),
            visual_expert_learning_rate=(
                cfg.visual_expert_learning_rate
            ),
            router_gate_learning_rate=(
                cfg.router_gate_learning_rate
            ),
            visual_norm_learning_rate=(
                cfg.visual_norm_learning_rate
            ),
            weight_decay=cfg.weight_decay,
            adam_beta1=training_args.adam_beta1,
            adam_beta2=training_args.adam_beta2,
            adam_epsilon=training_args.adam_epsilon,
        )

        trainer_kwargs["optimizers"] = (
            optimizer,
            None,
        )

        ddp_print(
            format_optimizer_groups(optimizer),
            print_rank=cfg.print_rank,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        **trainer_kwargs,
    )

    # ==========================================
    # 7. 正式训练
    # ==========================================
    trainer.train()

    # ==========================================
    # 8. 保存最终权重
    # ==========================================
    if local_rank in [-1, 0]:
        final_save_path = os.path.join(output_dir, "final_weights")
        os.makedirs(final_save_path, exist_ok=True)

        trainer.save_model(final_save_path)
        processor.save_pretrained(final_save_path)

        ddp_print(
            f"\n🎉 训练完成！LoRA 权重与 Processor 已保存至: {final_save_path}",
            print_rank=cfg.print_rank,
        )

        if (
            cfg.enable_visual_adapter
            and hasattr(model.base_model.model.model.visual, "res_adapter")
        ):
            adapter_module = model.base_model.model.model.visual.res_adapter
            adapter_state_dict = adapter_module.state_dict()

            adapter_save_path = os.path.join(final_save_path, "visual_adapter.pt")
            torch.save(adapter_state_dict, adapter_save_path)

            ddp_print(
                f"✨ RoMA-Net V2-lite visual adapter 权重已保存至: {adapter_save_path}",
                print_rank=cfg.print_rank,
            )
        else:
            ddp_print(
                "ℹ️ 当前为 A0 / LoRA-only，未保存 visual_adapter.pt。",
                print_rank=cfg.print_rank,
            )

    # ==========================================
    # 9. 优雅释放 DDP 资源
    # ==========================================
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
