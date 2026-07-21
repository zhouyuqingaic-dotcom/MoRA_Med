from dataclasses import dataclass, field
from typing import Optional
from utils.ddp.ddp_utils import ddp_print

@dataclass
class TrainConfig:
    """
    Stage 1: MIMIC-CXR Train Config

    对应 RoMA-Net V2-lite 消融体系：

    A0:
        Qwen3-VL + LoRA
        enable_visual_adapter=False

    A1:
        v2 fixed-alpha, w/o soft gate
        scale_mode="learned"
        gate_mode="fixed"
        fixed_gate=1.0
        lambda_mode="fixed"
        fixed_lambda=0.1
        use_rms_norm=True

    A2:
        v2 learnable-lambda, w/o soft gate
        scale_mode="learned"
        gate_mode="fixed"
        fixed_gate=1.0
        lambda_mode="learnable"
        lambda_init=0.1
        lambda_max=1.0
        use_rms_norm=True

    A3:
        v2 with soft gate, fixed-alpha
        scale_mode="learned"
        gate_mode="learned"
        lambda_mode="fixed"
        fixed_lambda=0.1
        use_rms_norm=True

    A4 optional:
        Full w/o RMS
        scale_mode="learned"
        gate_mode="learned"
        lambda_mode="learnable"
        use_rms_norm=False

    A5:
        Full RoMA-Net V2-lite
        scale_mode="learned"
        gate_mode="learned"
        lambda_mode="learnable"
        lambda_init=0.1
        lambda_max=1.0
        use_rms_norm=True

    A6:
        Strong visual residual injection
        scale_mode="learned"
        gate_mode="learned"
        gate_init=0.5
        lambda_mode="fixed"
        fixed_lambda=1.0
        use_rms_norm=True
    """

    # =========================================================
    # 1. 基础配置
    # =========================================================
    print_rank: int = 0
    seed: int = 2048

    # 当前消融实验 ID
    # 可选: "A0", "A1", "A2", "A3", "A4", "A5"
    ablation_id: str = "A6" #"A0" #"A5" #"A5" #"A5" #"A4" #"A3" #"A2" #"A1" "A0"

    # 统一输出根目录
    output_root: str = "/home/yuqing/Models/MoRA_Med"

    # =========================================================
    # 2. MIMIC-CXR 数据集配置
    # =========================================================
    mimic_cxr_root: str = "/home/yuqing/Datas/mimic-cxr-jpg-2.1.0"
    mimic_cxr_metadata_csv: str = "/home/yuqing/Datas/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-metadata.csv.gz"
    mimic_cxr_image_root: str = "/home/yuqing/Datas/mimic-cxr-jpg-2.1.0/files"
    mimic_cxr_report_root: str = "/home/yuqing/Datas/mimic-cxr-jpg-2.1.0/reports"

    # 可选: "impression", "findings", "full_report"
    mimic_cxr_target_section: str = "impression"

    mimic_cxr_load_report: bool = True
    mimic_cxr_view_positions: Optional[list[str]] = None
    mimic_cxr_drop_empty_target: bool = True

    # 缓存配置
    mimic_cxr_cache_dir: str = "/home/yuqing/Datas/mimic-cxr-jpg-2.1.0/cache"
    mimic_cxr_use_indices_cache: bool = True

    # 四卡正式训练时保持 False。
    mimic_cxr_rebuild_indices_cache: bool = False

    # # v2 表示使用 mimic_cxr_text_train_cleaning
    # # 过滤清洗后无效的监督文本。
    # mimic_cxr_cache_prefix: str = "mimic_cxr_train_clean_v2"
    #换成生成80k子集(用于快速验证)
    mimic_cxr_cache_prefix: str = "mimic_cxr_train_clean_v2_screen80k_seed2048"

    # MIMIC-CXR 专属指令
    mimic_cxr_instruction_suffix: str = (
        "Act as an expert radiologist. "
        "Carefully analyze this chest radiograph and provide a comprehensive clinical interpretation."
    )

    # 图像最大尺寸
    mimic_cxr_max_size: int = 1024

    # Smoke test 用；None 表示全量训练
    max_mimic_cxr_train_samples: Optional[int] = None

    # =========================================================
    # 3. Qwen3-VL 模型与量化配置
    # =========================================================
    model_name_or_path: str = "/home/yuqing/Models/Qwen3-VL-8B-Instruct"

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    # =========================================================
    # 4. BioMedCLIP 配置
    # =========================================================
    biomedclip_path: str = "/home/yuqing/Models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    biomedclip_model_name: str = "ViT-B-16"

    # 是否使用 V2-lite 的 cross-modal prior:
    # [c_I; c_Q; c_I*c_Q; |c_I-c_Q|; cos(c_I,c_Q)]
    # False 时退化为 [c_I; c_Q]
    use_cross_modal_prior: bool = True

    # =========================================================
    # 5. LoRA 配置
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
    # 6. 训练超参数
    # =========================================================
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_train_epochs: float = 1.0

    # =========================================================
    # A6-DLR 分组学习率
    # =========================================================

    # False 表示使用旧 A6 的统一学习率。
    # True 表示使用新的 A6-DLR。
    use_discriminative_lr: bool = True

    # 用于输出目录命名。
    lr_recipe_name: str = "DLR"

    # 旧统一学习率。
    # 同时作为 TrainingArguments 的基础/显示学习率。
    learning_rate: float = 2e-5

    # Stage 1 parameter-group learning rates
    lora_learning_rate: float = 2e-5
    visual_expert_learning_rate: float = 1e-4
    router_gate_learning_rate: float = 5e-5
    visual_norm_learning_rate: float = 2e-5

    weight_decay: float = 0.01

    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2

    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 12

    # =========================================================
    # 7. RoMA-Net V2-lite Visual Adapter 配置
    # =========================================================

    # A0 时为 False，其余 A1/A2/A3/A4/A5/A6 为 True
    enable_visual_adapter: bool = True

    # Qwen3-VL-8B 视觉 token hidden dim
    visual_adapter_hidden_dim: int = 4096

    # Adapter bottleneck ratio
    visual_adapter_r: int = 16

    # Router MLP hidden dim
    router_hidden_dim: int = 128

    # -----------------------------
    # Scale routing
    # -----------------------------
    # learned: BioMedCLIP-aware router 学习 π3/π5/π7
    # fixed: 使用 fixed_scale_weights
    scale_mode: str = "learned"

    fixed_scale_weights: list[float] = field(
        default_factory=lambda: [
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ]
    )

    # -----------------------------
    # Soft residual gate
    # -----------------------------
    # learned: g = sigmoid(MLP_g(h_route))
    # fixed: g = fixed_gate
    gate_mode: str = "learned"

    # w/o soft gate 时设为 1.0
    fixed_gate: float = 1.0

    # learned gate 的初始化值
    gate_init: float = 0.5

    # -----------------------------
    # Residual scale lambda
    # -----------------------------
    # learnable: lambda = lambda_max * sigmoid(lambda_a)
    # fixed: lambda = fixed_lambda，即 fixed-alpha 消融
    lambda_mode: str = "learnable"

    # fixed-alpha 消融时使用
    fixed_lambda: float = 0.1

    # learnable lambda 的上界和初始化值
    lambda_max: float = 1.0
    lambda_init: float = 0.1

    # -----------------------------
    # RMS residual normalization
    # -----------------------------
    use_rms_norm: bool = True
    residual_norm_eps: float = 1e-6
    residual_norm_ratio_clip: Optional[float] = 10.0


    def __post_init__(self):
        # =====================================================
        # 1. dtype / flash attention 检查
        # =====================================================
        if self.attn_implementation == "flash_attention_2" and self.torch_dtype != "bfloat16":
            print("⚠️ Warning: flash_attention_2 is best paired with bfloat16!")

        # =====================================================
        # 2. 根据 ablation_id 自动设置消融开关
        # =====================================================
        aid = self.ablation_id.upper()
        self.ablation_id = aid

        if aid == "A0":
            # -------------------------------------------------
            # Qwen3-VL + LoRA
            # 不启用 RoMA-Net visual adapter。
            # -------------------------------------------------
            self.enable_visual_adapter = False

            # 以下配置在 A0 中不会被真正使用。
            # 保留合法值是为了统一日志输出和配置检查。
            self.scale_mode = "learned"

            self.gate_mode = "fixed"
            self.fixed_gate = 1.0

            self.lambda_mode = "fixed"
            self.fixed_lambda = 0.0

            self.use_rms_norm = False

        elif aid == "A1":
            # -------------------------------------------------
            # 学习多尺度路由 π，但不使用 soft gate。
            # residual 强度使用固定 alpha/lambda。
            # -------------------------------------------------
            self.enable_visual_adapter = True

            self.scale_mode = "learned"

            self.gate_mode = "fixed"
            self.fixed_gate = 1.0

            self.lambda_mode = "fixed"
            self.fixed_lambda = 0.1

            self.use_rms_norm = True

        elif aid == "A2":
            # -------------------------------------------------
            # 不使用 soft gate，但学习全局 residual scale λ。
            # -------------------------------------------------
            self.enable_visual_adapter = True

            self.scale_mode = "learned"

            self.gate_mode = "fixed"
            self.fixed_gate = 1.0

            self.lambda_mode = "learnable"
            self.lambda_init = 0.1
            self.lambda_max = 1.0

            self.use_rms_norm = True

        elif aid == "A3":
            # -------------------------------------------------
            # 使用 soft gate，但 residual scale λ 固定。
            # -------------------------------------------------
            self.enable_visual_adapter = True

            self.scale_mode = "learned"

            self.gate_mode = "learned"

            self.lambda_mode = "fixed"
            self.fixed_lambda = 0.1

            self.use_rms_norm = True

        elif aid == "A4":
            # -------------------------------------------------
            # 完整 RoMA-Net V2-lite，但不使用 RMS normalization。
            # -------------------------------------------------
            self.enable_visual_adapter = True

            self.scale_mode = "learned"

            self.gate_mode = "learned"

            self.lambda_mode = "learnable"
            self.lambda_init = 0.1
            self.lambda_max = 1.0

            self.use_rms_norm = False

        elif aid == "A5":
            # -------------------------------------------------
            # 完整 RoMA-Net V2-lite。
            # -------------------------------------------------
            self.enable_visual_adapter = True

            self.scale_mode = "learned"

            self.gate_mode = "learned"

            self.lambda_mode = "learnable"
            self.lambda_init = 0.1
            self.lambda_max = 1.0

            self.use_rms_norm = True

        elif aid == "A6":
            # -------------------------------------------------
            # 强视觉残差注入实验。
            #
            # effective residual scale = fixed_lambda * gate
            # 初始状态约为：
            # 1.0 * 0.5 = 0.5
            #
            # lambda 固定，仅由样本级 Gate 控制残差强度。
            # -------------------------------------------------
            self.enable_visual_adapter = True

            # BioMedCLIP-conditioned F3/F5/F7 动态路由
            self.scale_mode = "learned"

            # 样本级 residual gate
            self.gate_mode = "learned"
            self.gate_init = 0.5

            # 强制固定 lambda=1.0
            self.lambda_mode = "fixed"
            self.fixed_lambda = 1.0

            # 保持 residual RMS normalization
            self.use_rms_norm = True

        else:
            raise ValueError(
                f"不支持的 ablation_id: {self.ablation_id}。"
                "可选值为 A0、A1、A2、A3、A4、A5、A6。"
            )


        # =====================================================
        # 3. 模式合法性检查
        # =====================================================
        if self.scale_mode not in {"learned", "fixed"}:
            raise ValueError(f"不支持的 scale_mode: {self.scale_mode}")

        if self.gate_mode not in {"learned", "fixed"}:
            raise ValueError(f"不支持的 gate_mode: {self.gate_mode}")

        if self.lambda_mode not in {"learnable", "fixed"}:
            raise ValueError(f"不支持的 lambda_mode: {self.lambda_mode}")

        if len(self.fixed_scale_weights) != 3:
            raise ValueError(
                "fixed_scale_weights 必须包含 3 个值，"
                "依次对应 Conv2D F3/F5/F7。"
            )

        if self.lambda_max <= 0:
            raise ValueError("lambda_max 必须大于 0。")

        if not (0.0 < self.lambda_init < self.lambda_max):
            # A0 时 lambda_init 不实际使用，但仍保持默认合法
            raise ValueError(
                f"lambda_init 必须满足 0 < lambda_init < lambda_max，"
                f"当前 lambda_init={self.lambda_init}, lambda_max={self.lambda_max}"
            )

        if not (0.0 <= self.fixed_gate <= 1.0):
            raise ValueError("fixed_gate 必须位于 [0, 1]。")

        if not (0.0 < self.gate_init < 1.0):
            raise ValueError("gate_init 必须位于 (0, 1)。")

        # =====================================================
        # 4. 输出目录
        # =====================================================
        if self.ablation_id == "A0":
            # 纯 LoRA baseline，不包含视觉适配器、Router、Gate 和 Lambda。
            experiment_name = (
                f"Stage1_MIMIC_CXR_"
                f"{self.mimic_cxr_cache_prefix}_"
                f"A0_LoRAOnly_"
                f"Seed-{self.seed}"
            )

        elif self.ablation_id == "A6":
            a6_label = (
                f"A6-{self.lr_recipe_name}"
                if self.use_discriminative_lr
                else "A6"
            )

            experiment_name = (
                f"Stage1_MIMIC_CXR_"
                f"{self.mimic_cxr_cache_prefix}_"
                f"{a6_label}_"
                f"Experts-Conv2D-F3_F5_F7_"
                f"Scale-{self.scale_mode}_"
                f"Gate-{self.gate_mode}-Init-{self.gate_init:g}_"
                f"Lambda-fixed-{self.fixed_lambda:g}_"
                f"RMS-{int(self.use_rms_norm)}_"
                f"Seed-{self.seed}"
            )

        else:
            experiment_name = (
                f"Stage1_MIMIC_CXR_"
                f"{self.mimic_cxr_cache_prefix}_"
                f"{self.ablation_id}_"
                f"Experts-Conv2D-F3_F5_F7_"
                f"Scale-{self.scale_mode}_"
                f"Gate-{self.gate_mode}_"
                f"Lambda-{self.lambda_mode}_"
                f"RMS-{int(self.use_rms_norm)}_"
                f"Seed-{self.seed}"
            )

        self.output_dir = (
            f"{self.output_root}/"
            f"{experiment_name}"
        )

        if self.enable_visual_adapter:
            if self.lambda_mode == "fixed":
                initial_lambda = self.fixed_lambda
            else:
                initial_lambda = self.lambda_init

            initial_effective_scale = initial_lambda * (
                self.gate_init
                if self.gate_mode == "learned"
                else self.fixed_gate
            )

            ddp_print(
                "视觉残差初始配置："
                f"lambda={initial_lambda}, "
                f"gate={self.gate_init if self.gate_mode == 'learned' else self.fixed_gate}, "
                f"lambda×gate={initial_effective_scale}"
            )

        if self.use_discriminative_lr:
            ddp_print(
                "Stage 1 分组学习率："
                f"LoRA={self.lora_learning_rate:g}, "
                f"Experts={self.visual_expert_learning_rate:g}, "
                f"Router/Gate={self.router_gate_learning_rate:g}, "
                f"Norm={self.visual_norm_learning_rate:g}"
            )

        ddp_print(f"当前输出目录为: {self.output_dir}")
