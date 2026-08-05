import os
from dataclasses import dataclass

from config.vqa_med_2019.stage2_train_config_vqa_med_2019 import (
    Stage2TrainConfig,
)


@dataclass
class Stage2EvalConfig(Stage2TrainConfig):
    """
    VQA-Med 2019 Stage 2 checkpoint 评测配置。

    直接继承 Stage2TrainConfig，确保训练与评测使用完全一致的：

    - A0～A6 消融定义；
    - LoRA 参数；
    - Conv2D F3/F5/F7 视觉专家；
    - Router、Gate、Lambda 和 RMS 配置；
    - DLR 配置；
    - Stage 2 实验目录命名。
    """

    # =========================================================
    # 1. 要评测的 Stage 2 实验
    # =========================================================

    # A6 + 两个 DLR 开关为 True，对应 A6-DLR。
    ablation_id: str = "A0" #"A6"

    # Stage 2 训练随机种子。
    seed: int = 2048

    # Stage 1 来源权重的随机种子。
    stage1_seed: int = 2048

    # 当前评测 A6-DLR。
    # 若评测普通 A6，将这两个字段改为 False。
    use_discriminative_lr: bool = False
    stage1_use_discriminative_lr: bool = False

    # =========================================================
    # 2. 评测输出
    # =========================================================

    eval_output_root: str = (
        "/home/yuqing/Models/MoRA_Med/"
        "Eval_VQA_MED_2019"
    )

    # True：
    #   只评测 final_weights。
    #
    # False：
    #   评测全部 checkpoint-*，最后再评测 final_weights。
    eval_final_weights_only: bool = False

    # =========================================================
    # 3. 生成参数
    # =========================================================

    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0

    # =========================================================
    # 4. DataLoader
    # =========================================================

    per_device_eval_batch_size: int = 4
    dataloader_num_workers: int = 4

    def __post_init__(self):
        """
        先通过 Stage2TrainConfig 生成对应的 Stage 2 训练目录，
        再将 output_dir 切换为评测结果目录。
        """

        # 根据 A0～A6、DLR、Stage 1 seed 等配置，
        # 生成与训练时完全一致的 Stage 2 实验目录。
        super().__post_init__()

        # super().__post_init__ 生成的 output_dir，
        # 此时是 Stage 2 训练目录。
        self.stage2_run_dir = self.output_dir

        run_name = os.path.basename(
            os.path.normpath(
                self.stage2_run_dir
            )
        )

        # 后续评测结果统一写入：
        #
        # Eval_VQA_MED_2019/
        # └── Stage2_VQA_MED_2019_.../
        self.output_dir = os.path.join(
            self.eval_output_root,
            run_name,
        )

        print("\n" + "=" * 60)
        print(
            "VQA-Med 2019 Stage 2 评测配置"
            f" | 消融={self.ablation_id}"
        )
        print(
            f"Stage 2 训练目录："
            f"{self.stage2_run_dir}"
        )
        print(
            f"评测结果目录："
            f"{self.output_dir}"
        )
        print(
            "评测范围："
            + (
                "仅 final_weights"
                if self.eval_final_weights_only
                else "全部 checkpoint + final_weights"
            )
        )
        print("=" * 60 + "\n")
