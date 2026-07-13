from dataclasses import dataclass, field
from typing import Optional


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

    A5:
        Full RoMA-Net V2-lite
        scale_mode="learned"
        gate_mode="learned"
        lambda_mode="learnable"
        lambda_init=0.1
        lambda_max=1.0
        use_rms_norm=True

    A4 optional:
        Full w/o RMS
        scale_mode="learned"
        gate_mode="learned"
        lambda_mode="learnable"
        use_rms_norm=False
    """

    # =========================================================
    # 1. 基础配置
    # =========================================================
    print_rank: int = 0
    seed: int = 2048

    # 当前消融实验 ID
    # 可选: "A0", "A1", "A2", "A3", "A4", "A5"
    ablation_id: str = "A5" #"A5" #"A4" #"A3" #"A2" #"A1" "A0"

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
    mimic_cxr_rebuild_indices_cache: bool = False
    mimic_cxr_cache_prefix: str = "mimic_cxr"

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

    learning_rate: float = 2e-5
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

    # A0 时为 False，其余 A1/A2/A3/A4/A5 为 True
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
    # learned: BioMedCLIP-aware router 学习 π1/π3/π5
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

        else:
            raise ValueError(
                f"不支持的 ablation_id: {self.ablation_id}。"
                "可选值为 A0、A1、A2、A3、A4、A5。"
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
                "对应 F1/F3/F5。"
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
        self.output_dir = (
            f"{self.output_root}/"
            f"Stage1_MIMIC_CXR_"
            f"{self.ablation_id}_"
            f"Experts-F1-F3-F5_"
            f"Scale-{self.scale_mode}_"
            f"Gate-{self.gate_mode}_"
            f"Lambda-{self.lambda_mode}_"
            f"RMS-{int(self.use_rms_norm)}_"
            f"Seed-{self.seed}"
        )

        print(f"当前输出目录为: {self.output_dir}")
