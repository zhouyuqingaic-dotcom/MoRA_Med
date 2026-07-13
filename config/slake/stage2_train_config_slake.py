import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Stage2TrainConfig:
    """
    Stage 2：SLAKE 微调配置。

    重要原则：
    - Stage 2 的模型结构必须与所加载的 Stage 1 消融结构一致；
    - Stage 1 和 Stage 2 使用相同的 LoRA 配置；
    - Stage 1 和 Stage 2 使用相同的 RoMA-Net V2-lite 配置；
    - BioMedCLIP 仍然冻结，只重新加载编码器。
    """

    # =========================================================
    # 1. 基础配置
    # =========================================================
    print_rank: int = 0

    # Stage 2 的随机种子
    seed: int = 2048

    # Stage 1 训练时使用的随机种子，用于推导权重目录
    stage1_seed: int = 2048

    # 必须与要加载的 Stage 1 消融保持一致
    ablation_id: str = "A5"

    output_root: str = "/home/yuqing/Models/MoRA_Med"

    # =========================================================
    # 2. SLAKE 数据
    # =========================================================
    slake_train_json_path: str = (
        "/home/yuqing/Datas/SLAKE/Slake1.0/train.json"
    )
    slake_val_json_path: str = (
        "/home/yuqing/Datas/SLAKE/Slake1.0/validate.json"
    )
    slake_test_json_path: str = (
        "/home/yuqing/Datas/SLAKE/Slake1.0/test.json"
    )
    slake_image_root: str = (
        "/home/yuqing/Datas/SLAKE/Slake1.0/imgs"
    )

    slake_max_size: int = 1024

    slake_instruction_suffix: str = (
        "Answer the question briefly and directly based on the image. "
        "Use a short medical term or phrase when possible. "
        "For yes/no questions, answer with yes or no. "
        "Do not add unnecessary explanation."
    )

    # Smoke test；None 表示使用完整训练集
    max_slake_train_samples: Optional[int] = None

    # =========================================================
    # 3. Qwen3-VL 与量化配置
    # =========================================================
    model_name_or_path: str = (
        "/home/yuqing/Models/Qwen3-VL-8B-Instruct"
    )

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    # =========================================================
    # 4. BioMedCLIP
    # =========================================================
    biomedclip_path: str = (
        "/home/yuqing/Models/"
        "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    use_cross_modal_prior: bool = True

    # =========================================================
    # 5. LoRA
    # 必须与 Stage 1 保持一致
    # =========================================================
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # =========================================================
    # 6. SLAKE 训练超参数
    # =========================================================
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_train_epochs: float = 3.0

    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2

    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 8

    # =========================================================
    # 7. RoMA-Net V2-lite
    # 必须与 Stage 1 保持一致
    # =========================================================
    enable_visual_adapter: bool = True

    visual_adapter_hidden_dim: int = 4096
    visual_adapter_r: int = 16
    router_hidden_dim: int = 128

    scale_mode: str = "learned"
    fixed_scale_weights: list[float] = field(
        default_factory=lambda: [
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ]
    )

    gate_mode: str = "learned"
    fixed_gate: float = 1.0
    gate_init: float = 0.5

    lambda_mode: str = "learnable"
    fixed_lambda: float = 0.1
    lambda_max: float = 1.0
    lambda_init: float = 0.1

    use_rms_norm: bool = True
    residual_norm_eps: float = 1e-6
    residual_norm_ratio_clip: Optional[float] = 10.0

    def __post_init__(self):
        if len(self.fixed_scale_weights) != 3:
            raise ValueError(
                "fixed_scale_weights 必须包含 3 个值，"
                "对应 F1/F3/F5。"
            )

        aid = self.ablation_id.upper()
        self.ablation_id = aid

        # -----------------------------------------------------
        # 必须与 Stage 1 TrainConfig 的消融定义完全相同
        # -----------------------------------------------------
        if aid == "A0":
            self.enable_visual_adapter = False
            self.scale_mode = "learned"
            self.gate_mode = "fixed"
            self.fixed_gate = 1.0
            self.lambda_mode = "fixed"
            self.fixed_lambda = 0.0
            self.use_rms_norm = False

        elif aid == "A1":
            self.enable_visual_adapter = True
            self.scale_mode = "learned"
            self.gate_mode = "fixed"
            self.fixed_gate = 1.0
            self.lambda_mode = "fixed"
            self.fixed_lambda = 0.1
            self.use_rms_norm = True

        elif aid == "A2":
            self.enable_visual_adapter = True
            self.scale_mode = "learned"
            self.gate_mode = "fixed"
            self.fixed_gate = 1.0
            self.lambda_mode = "learnable"
            self.lambda_init = 0.1
            self.lambda_max = 1.0
            self.use_rms_norm = True

        elif aid == "A3":
            self.enable_visual_adapter = True
            self.scale_mode = "learned"
            self.gate_mode = "learned"
            self.lambda_mode = "fixed"
            self.fixed_lambda = 0.1
            self.use_rms_norm = True

        elif aid == "A4":
            self.enable_visual_adapter = True
            self.scale_mode = "learned"
            self.gate_mode = "learned"
            self.lambda_mode = "learnable"
            self.lambda_init = 0.1
            self.lambda_max = 1.0
            self.use_rms_norm = False

        elif aid == "A5":
            self.enable_visual_adapter = True
            self.scale_mode = "learned"
            self.gate_mode = "learned"
            self.lambda_mode = "learnable"
            self.lambda_init = 0.1
            self.lambda_max = 1.0
            self.use_rms_norm = True

        else:
            raise ValueError(
                f"不支持的 ablation_id：{aid}。"
                "可选值为 A0、A1、A2、A3、A4、A5。"
            )

        # -----------------------------------------------------
        # Stage 1 权重目录
        # 命名规则与 Stage 1 TrainConfig 完全一致
        # -----------------------------------------------------
        stage1_experiment_dir = (
            f"Stage1_MIMIC_CXR_"
            f"{self.ablation_id}_"
            f"Experts-F1-F3-F5_"
            f"Scale-{self.scale_mode}_"
            f"Gate-{self.gate_mode}_"
            f"Lambda-{self.lambda_mode}_"
            f"RMS-{int(self.use_rms_norm)}_"
            f"Seed-{self.stage1_seed}"
        )

        self.stage1_weights_dir = os.path.join(
            self.output_root,
            stage1_experiment_dir,
            "final_weights",
        )

        # -----------------------------------------------------
        # Stage 2 输出目录
        # -----------------------------------------------------
        stage2_experiment_dir = (
            f"Stage2_SLAKE_"
            f"{self.ablation_id}_"
            f"Experts-F1-F3-F5_"
            f"Scale-{self.scale_mode}_"
            f"Gate-{self.gate_mode}_"
            f"Lambda-{self.lambda_mode}_"
            f"RMS-{int(self.use_rms_norm)}_"
            f"From-Stage1-Seed-{self.stage1_seed}_"
            f"Seed-{self.seed}"
        )

        self.output_dir = os.path.join(
            self.output_root,
            stage2_experiment_dir,
        )

        print(f"Stage 1 权重目录：{self.stage1_weights_dir}")
        print(f"Stage 2 输出目录：{self.output_dir}")
