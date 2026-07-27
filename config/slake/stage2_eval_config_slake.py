import os
from dataclasses import dataclass

from config.slake.stage2_train_config_slake import Stage2TrainConfig


@dataclass
class Stage2EvalConfig(Stage2TrainConfig):
    """
    SLAKE Stage 2 checkpoint 评测配置。

    直接继承 Stage2TrainConfig，确保：
    - A0～A5 消融定义一致；
    - LoRA 结构一致；
    - RoMA-Net V2-lite 结构一致；
    - Stage 2 权重目录命名一致。
    """

    # 必须对应要评测的 Stage 2 实验
    ablation_id: str = "A0" #"A0" #"A5"

    # Stage 2 训练种子
    seed: int = 2048

    # Stage 1 来源权重的种子
    stage1_seed: int = 2048

    # True: 只评测 final_weights
    # False: 评测 checkpoint-* 和 final_weights
    eval_final_weights_only: bool = False

    # 评测结果根目录
    eval_output_root: str = (
        "/home/yuqing/Models/MoRA_Med/Eval_SLAKE"
    )

    # 生成参数
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0

    # DataLoader
    per_device_eval_batch_size: int = 4
    dataloader_num_workers: int = 4

    def __post_init__(self):
        super().__post_init__()

        self.stage2_run_dir = self.output_dir
        run_name = os.path.basename(self.stage2_run_dir)
        self.output_dir = os.path.join(self.eval_output_root, run_name)

        print("\n" + "=" * 60)
        print(f"SLAKE Stage 2 评测 | 消融={self.ablation_id}")
        print(f"训练目录：{self.stage2_run_dir}")
        print(f"输出目录：{self.output_dir}")
        print(
            "评测节点："
            + (
                "final_weights"
                if self.eval_final_weights_only
                else "checkpoint-* + final_weights"
            )
        )
        print("=" * 60 + "\n")
