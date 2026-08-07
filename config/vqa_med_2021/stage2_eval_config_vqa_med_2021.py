import os
from dataclasses import dataclass

from config.vqa_med_2021.stage2_train_config_vqa_med_2021 import (
    Stage2TrainConfig,
)


@dataclass
class Stage2EvalConfig(Stage2TrainConfig):
    """
    VQA-Med 2021 Task 1 Stage 2 checkpoint 评测配置。

    结构与 SLAKE、VQA-Med 2019 的 EvalConfig 保持一致：

    - 直接继承 Stage2TrainConfig；
    - 使用同一套 A0～A6 消融定义；
    - 使用同一套 LoRA、Visual Adapter、Router、Gate、Lambda、RMS；
    - 根据 Stage2TrainConfig 自动定位 Stage 2 训练目录；
    - eval_final_weights_only=False 时评测 checkpoint-* + final_weights；
    - 所有节点直接在 test.jsonl 上评测并生成排行榜。

    注意：
    Train 与 Validation 如何组合，只在
    training/stage2_trainer_vqa_med_2021.py 的
    ConcatDataset 列表中设置，与本评测配置无关。
    """

    # =========================================================
    # 1. 要评测的 Stage 2 实验
    # =========================================================

    # 常用：
    #   "A0"：LoRA-only baseline
    #   "A6"：MoRA 强视觉残差
    ablation_id: str = "A6" #"A0"  # "A6"

    # Stage 2 训练随机种子。
    seed: int = 2048

    # Stage 1 来源权重随机种子。
    stage1_seed: int = 2048

    # 评测 A6-DLR 时，将以下两个字段改为 True。
    # 普通 A0 / A6 保持 False。
    use_discriminative_lr: bool = True
    stage1_use_discriminative_lr: bool = True

    # =========================================================
    # 2. 评测输出
    # =========================================================

    eval_output_root: str = (
        "/home/yuqing/Models/MoRA_Med/"
        "Eval_VQA_MED_2021"
    )

    # True：
    #   只评测 final_weights。
    #
    # False：
    #   按 step 顺序评测全部 checkpoint-*，
    #   最后再评测 final_weights。
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
        先由 Stage2TrainConfig 生成对应的 Stage 2 训练目录，
        再将 output_dir 切换为统一评测结果目录。
        """
        super().__post_init__()

        # super().__post_init__ 生成的 output_dir 是 Stage 2 训练目录。
        self.stage2_run_dir = self.output_dir

        run_name = os.path.basename(
            os.path.normpath(
                self.stage2_run_dir
            )
        )

        # 评测结果写入：
        #
        # Eval_VQA_MED_2021/
        # └── Stage2_VQA_MED_2021_.../
        self.output_dir = os.path.join(
            self.eval_output_root,
            run_name,
        )

        print("\n" + "=" * 60)
        print(
            "VQA-Med 2021 Stage 2 评测配置"
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
                else "全部 checkpoint-* + final_weights"
            )
        )
        print(
            "评测数据：VQA-Med 2021 Task 1 test.jsonl"
        )
        print("=" * 60 + "\n")