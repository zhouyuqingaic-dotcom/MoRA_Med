from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

from utils.ddp.ddp_utils import ddp_print


@dataclass
class Stage2TrainConfig:
    """
    VQA-Med 2021 Task 1 的 Stage 2 微调配置。

    设计原则
    --------
    1. Stage 2 的模型结构必须与所加载的 Stage 1 消融结构一致。
    2. Stage 1 与 Stage 2 使用相同的 LoRA 配置。
    3. A0 只加载 LoRA；A1～A6 加载 Visual Adapter。
    4. BioMedCLIP 始终冻结，只在启用 Visual Adapter 时作为路由条件。
    5. Train 与 Validation 是否合并，由
       stage2_trainer_vqa_med_2021.py 中的训练集构造决定。
    6. Test 只用于最终评测，不参与训练。
    """

    # =========================================================
    # 1. 基础配置
    # =========================================================
    print_rank: int = 0

    # Stage 2 随机种子。
    seed: int = 2048

    # Stage 1 来源权重的随机种子。
    stage1_seed: int = 2048

    # 必须与 Stage 1 TrainConfig 中的缓存前缀完全一致。
    stage1_mimic_cxr_cache_prefix: str = (
        "mimic_cxr_train_clean_v2_screen160k_seed2048"
    )

    # 必须与要加载的 Stage 1 消融一致。
    # 常用：
    #   A0：LoRA-only baseline
    #   A6：MoRA 强视觉残差版本
    ablation_id: str = "A6" #"A0"

    output_root: str = "/home/yuqing/Models/MoRA_Med"

    # 由 __post_init__ 自动生成。
    stage1_weights_dir: str = field(init=False)
    stage2_experiment_dir: str = field(init=False)
    output_dir: str = field(init=False)

    # =========================================================
    # 2. VQA-Med 2021 Task 1 数据
    # =========================================================
    vqa_med_2021_train_jsonl_path: str = (
        "/home/yuqing/Datas/VQA-Med-2021/jsonl/train.jsonl"
    )
    vqa_med_2021_validation_jsonl_path: str = (
        "/home/yuqing/Datas/VQA-Med-2021/jsonl/validation.jsonl"
    )
    vqa_med_2021_test_jsonl_path: str = (
        "/home/yuqing/Datas/VQA-Med-2021/jsonl/test.jsonl"
    )

    # 冻结官方划分规模，防止静默缺样本。
    vqa_med_2021_train_expected_count: int = 4500
    vqa_med_2021_validation_expected_count: int = 500
    vqa_med_2021_test_expected_count: int = 500

    # Dataset 初始化参数。
    #
    # JSONL 生成阶段已检查过图片，因此正式 DDP 训练默认不重复扫描。
    # 首次独立数据检查时可临时改为 True。
    vqa_med_2021_verify_images: bool = False
    vqa_med_2021_dataset_strict: bool = True

    # Qwen3-VL 图像最大边或 Collator 使用的图像尺寸上限。
    vqa_med_2021_max_size: int = 1024

    # Qwen3-VL 使用：原始问题 + 此回答格式指令。
    #
    # BioMedCLIP Router 仍应只读取原始 question，
    # 不应读取附加了 instruction_suffix 的文本。
    vqa_med_2021_instruction_suffix: str = (
        "Answer the question briefly and directly based on the image. "
        "State the primary abnormality, diagnosis, or imaging finding "
        "using a short medical term or phrase when possible. "
        "For yes/no questions, answer with yes or no. "
        "Do not add unnecessary explanation."
    )

    # Smoke test；None 表示使用完整划分。
    max_vqa_med_2021_train_samples: Optional[int] = None
    max_vqa_med_2021_validation_samples: Optional[int] = None
    max_vqa_med_2021_test_samples: Optional[int] = None

    # =========================================================
    # 3. Qwen3-VL 与量化
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
    # 6. Stage 2 训练超参数
    # =========================================================
    # 四卡默认：
    #   global batch = 4 GPUs × 4 samples × 1 accumulation = 16
    #
    # 双卡若希望保持相同 global batch，可改为：
    #   per_device_train_batch_size = 4
    #   gradient_accumulation_steps = 2
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1

    num_train_epochs: float = 3.0

    # A6-DLR 时设置为 True；A0 必须为 False。
    use_discriminative_lr: bool = True
    lr_recipe_name: str = "DLR"

    # 所加载的 Stage 1 是否为 DLR 版本。
    stage1_use_discriminative_lr: bool = True
    stage1_lr_recipe_name: str = "DLR"

    # 统一学习率，同时作为 Hugging Face Trainer 的基础显示学习率。
    learning_rate: float = 1e-5

    # DLR 参数组学习率。
    lora_learning_rate: float = 1e-5
    visual_expert_learning_rate: float = 3e-5
    router_gate_learning_rate: float = 2e-5
    visual_norm_learning_rate: float = 1e-5

    weight_decay: float = 0.01

    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    logging_steps: int = 10

    # Train-only 时 4500 条；Train+Validation 时 5000 条。
    # 四卡 global batch=16、训练 3 epochs 时，
    # 总 optimizer steps 分别约为 846 或 939。
    # 每 100 steps 保存一次，便于按现有 SLAKE/VQA-Med 2019
    # 流程比较 checkpoint-* 与 final_weights。
    save_steps: int = 100
    save_total_limit: int = 20

    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 8

    # =========================================================
    # 7. Visual Adapter / MoRA
    # 必须与 Stage 1 保持一致
    # =========================================================
    enable_visual_adapter: bool = True

    visual_adapter_hidden_dim: int = 4096
    visual_adapter_r: int = 16
    router_hidden_dim: int = 128

    # F3 / F5 / F7 路由。
    scale_mode: str = "learned"
    fixed_scale_weights: list[float] = field(
        default_factory=lambda: [
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ]
    )

    # 样本级 residual gate。
    gate_mode: str = "learned"
    fixed_gate: float = 1.0
    gate_init: float = 0.5

    # 全局残差系数 lambda。
    lambda_mode: str = "learnable"
    fixed_lambda: float = 0.1
    lambda_max: float = 1.0
    lambda_init: float = 0.1

    # 视觉残差 RMS 对齐。
    use_rms_norm: bool = True
    residual_norm_eps: float = 1e-6
    residual_norm_ratio_clip: Optional[float] = 10.0

    def __post_init__(self) -> None:
        self._validate_common_fields()
        self._apply_ablation_definition()
        self._validate_ablation_fields()
        self._build_stage1_weights_dir()
        self._build_stage2_output_dir()
        self._print_summary()

    # =========================================================
    # 8. 配置校验
    # =========================================================
    def _validate_common_fields(self) -> None:
        if len(self.fixed_scale_weights) != 3:
            raise ValueError(
                "fixed_scale_weights 必须包含 3 个值，"
                "依次对应 Conv2D F3/F5/F7。"
            )

        if any(
            (not isinstance(value, (int, float)))
            or (not math.isfinite(float(value)))
            or float(value) < 0.0
            for value in self.fixed_scale_weights
        ):
            raise ValueError(
                "fixed_scale_weights 必须全部是有限的非负数。"
            )

        if sum(float(value) for value in self.fixed_scale_weights) <= 0.0:
            raise ValueError(
                "fixed_scale_weights 的和必须大于 0。"
            )

        expected_counts = {
            "train": self.vqa_med_2021_train_expected_count,
            "validation": self.vqa_med_2021_validation_expected_count,
            "test": self.vqa_med_2021_test_expected_count,
        }
        for split, count in expected_counts.items():
            if not isinstance(count, int) or count <= 0:
                raise ValueError(
                    f"{split} expected_count 必须是正整数，当前为 {count!r}。"
                )

        sample_limits = {
            "train": (
                self.max_vqa_med_2021_train_samples,
                self.vqa_med_2021_train_expected_count,
            ),
            "validation": (
                self.max_vqa_med_2021_validation_samples,
                self.vqa_med_2021_validation_expected_count,
            ),
            "test": (
                self.max_vqa_med_2021_test_samples,
                self.vqa_med_2021_test_expected_count,
            ),
        }
        for split, (limit, full_count) in sample_limits.items():
            if limit is None:
                continue
            if not isinstance(limit, int) or limit <= 0:
                raise ValueError(
                    f"max_{split}_samples 必须是正整数或 None，"
                    f"当前为 {limit!r}。"
                )
            if limit > full_count:
                raise ValueError(
                    f"max_{split}_samples={limit} 超过官方规模 {full_count}。"
                )

        for name, path in {
            "train_jsonl": self.vqa_med_2021_train_jsonl_path,
            "validation_jsonl": self.vqa_med_2021_validation_jsonl_path,
            "test_jsonl": self.vqa_med_2021_test_jsonl_path,
            "model_name_or_path": self.model_name_or_path,
            "biomedclip_path": self.biomedclip_path,
            "output_root": self.output_root,
        }.items():
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"{name} 不能为空。")

        if not isinstance(
            self.vqa_med_2021_instruction_suffix,
            str,
        ) or not self.vqa_med_2021_instruction_suffix.strip():
            raise ValueError(
                "vqa_med_2021_instruction_suffix 不能为空。"
            )

        positive_int_fields = {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "dataloader_num_workers": self.dataloader_num_workers,
            "visual_adapter_hidden_dim": self.visual_adapter_hidden_dim,
            "visual_adapter_r": self.visual_adapter_r,
            "router_hidden_dim": self.router_hidden_dim,
            "vqa_med_2021_max_size": self.vqa_med_2021_max_size,
        }
        for name, value in positive_int_fields.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"{name} 必须是正整数，当前为 {value!r}。"
                )

        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs 必须大于 0。")

        if self.warmup_steps < 0:
            raise ValueError("warmup_steps 不能小于 0。")

        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm 必须大于 0。")

        if self.residual_norm_eps <= 0:
            raise ValueError("residual_norm_eps 必须大于 0。")

        if (
            self.residual_norm_ratio_clip is not None
            and self.residual_norm_ratio_clip <= 0
        ):
            raise ValueError(
                "residual_norm_ratio_clip 必须大于 0 或为 None。"
            )

    # =========================================================
    # 9. A0～A6 消融定义
    # 必须与 Stage 1 配置保持一致
    # =========================================================
    def _apply_ablation_definition(self) -> None:
        aid = str(self.ablation_id).strip().upper()
        self.ablation_id = aid

        if aid == "A0":
            self.enable_visual_adapter = False
            self.scale_mode = "learned"
            self.gate_mode = "fixed"
            self.fixed_gate = 1.0
            self.lambda_mode = "fixed"
            self.fixed_lambda = 0.0
            self.use_rms_norm = False

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
            # 强视觉残差注入实验：
            # effective residual scale = fixed_lambda × gate
            self.enable_visual_adapter = True
            self.scale_mode = "learned"

            self.gate_mode = "learned"
            self.gate_init = 0.5

            self.lambda_mode = "fixed"
            self.fixed_lambda = 1.0

            self.use_rms_norm = True

            # A6 是否使用 DLR，由用户配置的两个开关决定。
            # 不在这里强制覆盖：
            #   use_discriminative_lr
            #   stage1_use_discriminative_lr

        else:
            raise ValueError(
                f"不支持的 ablation_id：{aid!r}。"
                "可选值为 A0、A1、A2、A3、A4、A5、A6。"
            )

    def _validate_ablation_fields(self) -> None:
        if self.scale_mode not in {"learned", "fixed"}:
            raise ValueError(
                f"scale_mode 必须是 learned 或 fixed，当前为 {self.scale_mode!r}。"
            )

        if self.gate_mode not in {"learned", "fixed"}:
            raise ValueError(
                f"gate_mode 必须是 learned 或 fixed，当前为 {self.gate_mode!r}。"
            )

        if self.lambda_mode not in {"learnable", "fixed"}:
            raise ValueError(
                "lambda_mode 必须是 learnable 或 fixed，"
                f"当前为 {self.lambda_mode!r}。"
            )

        if not (0.0 <= self.fixed_gate <= 1.0):
            raise ValueError("fixed_gate 必须位于 [0, 1]。")

        if not (0.0 <= self.gate_init <= 1.0):
            raise ValueError("gate_init 必须位于 [0, 1]。")

        if self.fixed_lambda < 0.0:
            raise ValueError("fixed_lambda 不能小于 0。")

        if self.lambda_init < 0.0:
            raise ValueError("lambda_init 不能小于 0。")

        if self.lambda_max <= 0.0:
            raise ValueError("lambda_max 必须大于 0。")

        if self.lambda_init > self.lambda_max:
            raise ValueError("lambda_init 不能大于 lambda_max。")

        if (
            self.ablation_id != "A6"
            and (
                self.use_discriminative_lr
                or self.stage1_use_discriminative_lr
            )
        ):
            raise ValueError(
                "当前工程的 DLR 命名与参数组方案只用于 A6。"
            )

    # =========================================================
    # 10. Stage 1 权重目录
    # =========================================================
    def _build_stage1_weights_dir(self) -> None:
        if self.ablation_id == "A0":
            stage1_experiment_dir = (
                f"Stage1_MIMIC_CXR_"
                f"{self.stage1_mimic_cxr_cache_prefix}_"
                f"A0_LoRAOnly_"
                f"Seed-{self.stage1_seed}"
            )

        elif self.ablation_id == "A6":
            stage1_a6_label = (
                f"A6-{self.stage1_lr_recipe_name}"
                if self.stage1_use_discriminative_lr
                else "A6"
            )

            stage1_experiment_dir = (
                f"Stage1_MIMIC_CXR_"
                f"{self.stage1_mimic_cxr_cache_prefix}_"
                f"{stage1_a6_label}_"
                f"Experts-Conv2D-F3_F5_F7_"
                f"Scale-{self.scale_mode}_"
                f"Gate-{self.gate_mode}-Init-{self.gate_init:g}_"
                f"Lambda-fixed-{self.fixed_lambda:g}_"
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

    # =========================================================
    # 11. Stage 2 输出目录
    # =========================================================
    def _build_stage2_output_dir(self) -> None:
        if self.ablation_id == "A0":
            self.stage2_experiment_dir = (
                f"Stage2_VQA_MED_2021_"
                f"{self.stage1_mimic_cxr_cache_prefix}_"
                f"A0_LoRAOnly_"
                f"From-Stage1-Seed-{self.stage1_seed}_"
                f"Seed-{self.seed}"
            )

        elif self.ablation_id == "A6":
            stage2_a6_label = (
                f"A6-{self.lr_recipe_name}"
                if self.use_discriminative_lr
                else "A6"
            )

            self.stage2_experiment_dir = (
                f"Stage2_VQA_MED_2021_"
                f"{self.stage1_mimic_cxr_cache_prefix}_"
                f"{stage2_a6_label}_"
                f"Experts-Conv2D-F3_F5_F7_"
                f"Scale-{self.scale_mode}_"
                f"Gate-{self.gate_mode}-Init-{self.gate_init:g}_"
                f"Lambda-fixed-{self.fixed_lambda:g}_"
                f"RMS-{int(self.use_rms_norm)}_"
                f"From-Stage1-Seed-{self.stage1_seed}_"
                f"Seed-{self.seed}"
            )

        else:
            self.stage2_experiment_dir = (
                f"Stage2_VQA_MED_2021_"
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
            self.stage2_experiment_dir,
        )

    # =========================================================
    # 12. 配置摘要
    # =========================================================
    def _print_summary(self) -> None:
        if self.enable_visual_adapter:
            configured_lambda = (
                self.fixed_lambda
                if self.lambda_mode == "fixed"
                else self.lambda_init
            )
            configured_gate = (
                self.gate_init
                if self.gate_mode == "learned"
                else self.fixed_gate
            )

            ddp_print(
                "视觉残差配置："
                f"lambda_mode={self.lambda_mode}, "
                f"lambda={configured_lambda:g}, "
                f"gate_mode={self.gate_mode}, "
                f"gate={configured_gate:g}, "
                f"初始 lambda×gate="
                f"{configured_lambda * configured_gate:g}",
                print_rank=self.print_rank,
            )

        if self.use_discriminative_lr:
            ddp_print(
                "Stage 2 分组学习率："
                f"LoRA={self.lora_learning_rate:g}, "
                f"Experts={self.visual_expert_learning_rate:g}, "
                f"Router/Gate={self.router_gate_learning_rate:g}, "
                f"Norm={self.visual_norm_learning_rate:g}",
                print_rank=self.print_rank,
            )

        ddp_print(
            f"VQA-Med 2021 Train JSONL："
            f"{self.vqa_med_2021_train_jsonl_path}",
            print_rank=self.print_rank,
        )
        ddp_print(
            "VQA-Med 2021 数据配置："
            "Train 与 Validation 是否合并由 Trainer 决定；"
            "Test 不参与训练。",
            print_rank=self.print_rank,
        )
        ddp_print(
            f"Stage 1 权重目录：{self.stage1_weights_dir}",
            print_rank=self.print_rank,
        )
        ddp_print(
            f"Stage 2 输出目录：{self.output_dir}",
            print_rank=self.print_rank,
        )

        if self.enable_visual_adapter:
            ddp_print(
                "视觉专家结构：DWConv2D F3=3×3, F5=5×5, F7=7×7",
                print_rank=self.print_rank,
            )
        else:
            ddp_print(
                "视觉专家结构：A0 LoRA-only，未启用 Visual Adapter",
                print_rank=self.print_rank,
            )