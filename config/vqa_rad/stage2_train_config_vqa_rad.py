import os
from dataclasses import dataclass, field
from typing import Optional
from utils.ddp.ddp_utils import ddp_print

#获取命令行参数
import argparse
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ablation_id",
        type=str,
        default="A0",
        choices=["A0", "A1", "A2", "A3", "A4", "A5", "A6"],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--stage1_seed",
        type=int,
        default=2048,
    )

    # config 可能被 torchrun / 其他脚本 import，
    # 所以忽略当前 config 不认识的额外参数
    args, _ = parser.parse_known_args()

    return args
#执行参数获取
_cli_args=parse_args()

@dataclass
class Stage2TrainConfig:
    """
    Stage 2：VQA-RAD 微调配置。

    重要原则：
    - 模型结构必须与加载的 Stage 1 消融结构一致；
    - Stage 1 和 Stage 2 使用相同的 LoRA 与视觉专家配置；
    - A6-DLR 继续使用与 SLAKE 相同的分组学习率协议；
    - BioMedCLIP 始终冻结，仅用于生成路由特征。
    """

    # =========================================================
    # 1. 基础配置
    # =========================================================
    print_rank: int = 0

    # Stage 2 的随机种子
    # seed: int = 2048
    # 也可以获取命令行参数
    seed: int = getattr(_cli_args, "seed", 2048)

    # Stage 1 训练时使用的随机种子，用于推导权重目录
    # stage1_seed: int = 2048
    # 也可以获取命令行参数
    stage1_seed: int = getattr(_cli_args, "stage1_seed", 2048)

    # Stage 1 使用的 MIMIC-CXR 数据缓存前缀。
    # 必须与 Stage 1 TrainConfig 中的 mimic_cxr_cache_prefix 完全一致。
    # stage1_mimic_cxr_cache_prefix: str = (
    #     "mimic_cxr_train_clean_v2"
    # )
    # stage1_mimic_cxr_cache_prefix: str = (
    #     "mimic_cxr_train_clean_v2_screen80k_seed2048"
    # )
    # 当前使用 Stage 1 的 160k 中间规模实验
    stage1_mimic_cxr_cache_prefix: str = (
        "mimic_cxr_train_clean_v2_screen160k_seed2048"
    )

    # 必须与要加载的 Stage 1 消融保持一致
    # ablation_id: str = "A6" #"A0" #"A6" #"A6" #"A0" #"A5"
    # 也可以获取命令行参数
    ablation_id: str = getattr(_cli_args, "ablation_id", "A0")

    output_root: str = "/home/yuqing/Models/MoRA_Med"

    # =========================================================
    # 2. VQA-RAD 数据
    # =========================================================
    vqa_rad_train_jsonl_path: str = (
        "/home/yuqing/Datas/VQA-RAD/train.jsonl"
    )

    # 训练阶段不会使用，但 EvalConfig 可以继承该字段
    vqa_rad_test_jsonl_path: str = (
        "/home/yuqing/Datas/VQA-RAD/test_official.jsonl"
    )

    vqa_rad_image_root: str = (
        "/home/yuqing/Datas/VQA-RAD/images"
    )

    vqa_rad_max_size: int = 1024

    vqa_rad_instruction_suffix: str = (
        "Answer the question briefly and directly based on the image. "
        "Use a short medical term or phrase when possible. "
        "For yes/no questions, answer with yes or no. "
        "Do not add unnecessary explanation."
    )

    # Smoke test 时设为 32；正式训练设为 None
    max_vqa_rad_train_samples: Optional[int] = None

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
    # 6. VQA-RAD 训练超参数
    # =========================================================
    #重要，一定要根据卡数量调整，这回改变参数更新次数
    ## 双卡：
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 2
    #四卡：
    # per_device_train_batch_size = 1
    # gradient_accumulation_steps = 1

    #小数据集VQA-RAD改成10
    num_train_epochs: float = 10.0

    # =========================================================
    # A6-DLR 分组学习率
    # =========================================================
    #使用"A6-DLR"模式时候为true
    use_discriminative_lr: bool = True
    #使用"A0"模式时候设置为False
    # use_discriminative_lr: bool = False

    lr_recipe_name: str = "DLR"

    # Stage 2 加载的 Stage 1 是否不为 DLR 版本,这个设置为False。
    #A0时候设置为False
    # stage1_use_discriminative_lr: bool = False
    stage1_use_discriminative_lr: bool = True

    stage1_lr_recipe_name: str = "DLR"

    # 旧统一学习率，同时作为 Trainer 基础/显示学习率。
    learning_rate: float = 1e-5

    # Stage 2 parameter-group learning rates
    lora_learning_rate: float = 1e-5
    visual_expert_learning_rate: float = 3e-5
    router_gate_learning_rate: float = 2e-5
    visual_norm_learning_rate: float = 1e-5

    weight_decay: float = 0.01

    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 50
    max_grad_norm: float = 1.0

    logging_steps: int = 10
    #小数据集VQA-RAD改成200
    save_steps: int = 250
    save_total_limit: int = 40

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
    fixed_gate: float = 0.5
    gate_init: float = 0.5

    lambda_mode: str = "learnable"
    fixed_lambda: float = 0.9
    lambda_init: float = 0.9
    lambda_max: float = 1.0

    use_rms_norm: bool = True
    residual_norm_eps: float = 1e-6
    residual_norm_ratio_clip: Optional[float] = 10.0

    def __post_init__(self):
        if len(self.fixed_scale_weights) != 3:
            raise ValueError(
                "fixed_scale_weights 必须包含 3 个值，"
                "依次对应 Conv2D F3/F5/F7。"
            )

        aid = self.ablation_id.upper()
        self.ablation_id = aid

        # -----------------------------------------------------
        # 必须与 Stage 1 TrainConfig 的消融定义完全相同
        # -----------------------------------------------------
        if aid == "A0":
            # Pure Qwen3-VL + LoRA baseline
            self.enable_visual_adapter = False

            # A0 不使用 DLR
            self.use_discriminative_lr = False
            self.stage1_use_discriminative_lr = False

        elif aid == "A1":
            self.enable_visual_adapter = True
            self.scale_mode = "learned"
            self.gate_mode = "fixed"
            self.fixed_gate = 1.0
            self.lambda_mode = "fixed"
            self.fixed_lambda = 0.1
            self.use_rms_norm = True

            self.use_discriminative_lr = False
            self.stage1_use_discriminative_lr = False

        elif aid == "A2":
            self.enable_visual_adapter = True
            self.scale_mode = "learned"
            self.gate_mode = "fixed"
            self.fixed_gate = 1.0
            self.lambda_mode = "learnable"
            self.lambda_init = 0.1
            self.lambda_max = 1.0
            self.use_rms_norm = True

            self.use_discriminative_lr = False
            self.stage1_use_discriminative_lr = False

        elif aid == "A3":
            self.enable_visual_adapter = True
            self.scale_mode = "learned"
            self.gate_mode = "learned"
            self.lambda_mode = "fixed"
            self.fixed_lambda = 0.1
            self.use_rms_norm = True

            self.use_discriminative_lr = False
            self.stage1_use_discriminative_lr = False

        elif aid == "A4":
            self.enable_visual_adapter = True
            self.scale_mode = "learned"
            self.gate_mode = "learned"
            self.lambda_mode = "learnable"
            self.lambda_init = 0.1
            self.lambda_max = 1.0
            self.use_rms_norm = False

            self.use_discriminative_lr = False
            self.stage1_use_discriminative_lr = False

        elif aid == "A5":
            self.enable_visual_adapter = True
            self.scale_mode = "learned"
            self.gate_mode = "learned"
            self.lambda_mode = "learnable"
            self.lambda_init = 0.1
            self.lambda_max = 1.0
            self.use_rms_norm = True

            self.use_discriminative_lr = False
            self.stage1_use_discriminative_lr = False

        elif aid == "A6":
            # Full MoRA-Med
            self.enable_visual_adapter = True
            # Dynamic multi-scale routing
            self.scale_mode = "learned"
            # Adaptive gate
            self.gate_mode = "learned"
            # Learnable global residual scale
            self.lambda_mode = "learnable"
            # RMS Matching
            self.use_rms_norm = True
            # DLR
            self.use_discriminative_lr = True
            self.stage1_use_discriminative_lr = True

        else:
            raise ValueError(
                f"不支持的 ablation_id：{aid}。"
                "可选值为 A0、A1、A2、A3、A4、A5、A6。"
            )

        # -----------------------------------------------------
        # Stage 1 权重目录
        # 必须与 Stage 1 TrainConfig 的命名规则完全一致。
        # -----------------------------------------------------
        if self.ablation_id == "A0":
            stage1_a0_label = "A0-LoRAOnly-NoDLR"

            stage1_experiment_dir = (
                f"Stage1_MIMIC_CXR_"
                f"{self.stage1_mimic_cxr_cache_prefix}_"
                f"{stage1_a0_label}_"
                f"LoRA-r{self.lora_r}-Alpha-{self.lora_alpha}-"
                f"Dropout-{self.lora_dropout:g}_"
                f"Seed-{self.stage1_seed}"
            )


        elif self.ablation_id == "A6":
            stage1_a6_label = (
                f"A6-{self.stage1_lr_recipe_name}"
                if self.stage1_use_discriminative_lr
                else "A6"
            )

            if self.lambda_mode == "learnable":
                lambda_label = (
                    f"Lambda-learnable-Init-{self.lambda_init:g}"
                    f"-Max-{self.lambda_max:g}"
                )
            else:
                lambda_label = (
                    f"Lambda-fixed-{self.fixed_lambda:g}"
                )

            stage1_experiment_dir = (
                f"Stage1_MIMIC_CXR_"
                f"{self.stage1_mimic_cxr_cache_prefix}_"
                f"{stage1_a6_label}_"
                f"Experts-Conv2D-F3_F5_F7_"
                f"Scale-{self.scale_mode}_"
                f"Gate-{self.gate_mode}-Init-{self.gate_init:g}_"
                f"{lambda_label}_"
                f"RMS-{int(self.use_rms_norm)}_"
                f"Seed-{self.stage1_seed}"
            )

        else:
            stage1_experiment_dir = (
                f"Stage1_MIMIC_CXR_"
                f"{self.stage1_mimic_cxr_cache_prefix}_"
                f"{self.ablation_id}_"
                f"Experts-Conv2D-F3_F5_F7_"
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
        # 将 Stage 1 数据来源和 Conv2D 结构都写进目录名，
        # -----------------------------------------------------
        if self.ablation_id == "A0":
            stage2_a0_label = "A0-LoRAOnly-NoDLR"

            stage2_experiment_dir = (
                f"Stage2_VQA_RAD_"
                f"{self.stage1_mimic_cxr_cache_prefix}_"
                f"{stage2_a0_label}_"
                f"LoRA-r{self.lora_r}-Alpha-{self.lora_alpha}-"
                f"Dropout-{self.lora_dropout:g}_"
                f"From-Stage1-Seed-{self.stage1_seed}_"
                f"Seed-{self.seed}"
            )


        elif self.ablation_id == "A6":
            stage2_a6_label = (
                f"A6-{self.lr_recipe_name}"
                if self.use_discriminative_lr
                else "A6"
            )

            if self.lambda_mode == "learnable":
                lambda_label = (
                    f"Lambda-learnable-Init-{self.lambda_init:g}"
                    f"-Max-{self.lambda_max:g}"
                )
            else:
                lambda_label = (
                    f"Lambda-fixed-{self.fixed_lambda:g}"
                )

            stage2_experiment_dir = (
                f"Stage2_VQA_RAD_"
                f"{self.stage1_mimic_cxr_cache_prefix}_"
                f"{stage2_a6_label}_"
                f"Experts-Conv2D-F3_F5_F7_"
                f"Scale-{self.scale_mode}_"
                f"Gate-{self.gate_mode}-Init-{self.gate_init:g}_"
                f"{lambda_label}_"
                f"RMS-{int(self.use_rms_norm)}_"
                f"From-Stage1-Seed-{self.stage1_seed}_"
                f"Seed-{self.seed}"
            )

        else:
            stage2_experiment_dir = (
                f"Stage2_VQA_RAD_"
                f"{self.stage1_mimic_cxr_cache_prefix}_"
                f"{self.ablation_id}_"
                f"Experts-Conv2D-F3_F5_F7_"
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

        if self.enable_visual_adapter:
            if self.lambda_mode == "fixed":
                configured_lambda = self.fixed_lambda
            else:
                configured_lambda = self.lambda_init

            configured_gate = (
                self.gate_init
                if self.gate_mode == "learned"
                else self.fixed_gate
            )

            ddp_print(
                "视觉残差配置："
                f"lambda_mode={self.lambda_mode}, "
                f"lambda={configured_lambda}, "
                f"gate_mode={self.gate_mode}, "
                f"gate_init={configured_gate}, "
                f"初始lambda×gate={configured_lambda * configured_gate}"
            )

        if self.use_discriminative_lr:
            ddp_print(
                "Stage 2 分组学习率："
                f"LoRA={self.lora_learning_rate:g}, "
                f"Experts={self.visual_expert_learning_rate:g}, "
                f"Router/Gate={self.router_gate_learning_rate:g}, "
                f"Norm={self.visual_norm_learning_rate:g}"
            )

        ddp_print(f"Stage 1 权重目录：{self.stage1_weights_dir}")
        ddp_print(f"Stage 2 输出目录：{self.output_dir}")
        if self.enable_visual_adapter:
            ddp_print(
                "视觉专家结构：DWConv2D "
                "F3=3x3, F5=5x5, F7=7x7"
            )
        else:
            ddp_print(
                "视觉专家结构：A0 LoRA-only，未启用 visual adapter"
            )
