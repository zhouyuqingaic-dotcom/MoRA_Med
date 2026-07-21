from collections import defaultdict

import torch
import torch.nn as nn


EXPERT_NAME_PARTS = (
    "adapter_f3.",
    "adapter_f5.",
    "adapter_f7.",
)

ROUTER_GATE_NAME_PARTS = (
    "router_backbone.",
    "scale_head.",
    "gate_head.",
)


def build_discriminative_adamw(
    model: nn.Module,
    *,
    lora_learning_rate: float,
    visual_expert_learning_rate: float,
    router_gate_learning_rate: float,
    visual_norm_learning_rate: float,
    weight_decay: float,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
) -> torch.optim.AdamW:
    """
    为 LoRA、视觉专家、Router/Gate 和 LayerNorm
    建立不同学习率的 AdamW 参数组。

    A6 中 lambda_a 已被冻结，不会进入优化器。
    """

    # 精确识别所有 LayerNorm 参数。
    norm_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.LayerNorm)
        for parameter in module.parameters(recurse=False)
    }

    buckets = defaultdict(list)
    unmatched_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        # 先判断 Norm，确保 router_backbone 中的 LayerNorm
        # 被分到 Norm，而不是 Router/Gate。
        if id(parameter) in norm_parameter_ids:
            component = "visual_norm"

        elif "lora_" in name:
            component = "lora"

        elif any(part in name for part in EXPERT_NAME_PARTS):
            component = "visual_experts"

        elif any(part in name for part in ROUTER_GATE_NAME_PARTS):
            component = "router_gate"

        # 兼容未来非 A6 的 learnable lambda。
        elif name.endswith("lambda_a"):
            component = "router_gate"

        else:
            unmatched_parameters.append(name)
            continue

        # Norm、bias 和 lambda 不做 weight decay。
        no_decay = (
            id(parameter) in norm_parameter_ids
            or name.endswith(".bias")
            or name.endswith("lambda_a")
        )

        buckets[(component, no_decay)].append(parameter)

    # 防止新增可训练模块后被静默遗漏。
    if unmatched_parameters:
        preview = "\n".join(
            f"  - {name}"
            for name in unmatched_parameters[:50]
        )

        raise RuntimeError(
            "发现未分类的可训练参数：\n"
            f"{preview}"
        )

    learning_rates = {
        "lora": float(lora_learning_rate),
        "visual_experts": float(
            visual_expert_learning_rate
        ),
        "router_gate": float(
            router_gate_learning_rate
        ),
        "visual_norm": float(
            visual_norm_learning_rate
        ),
    }

    parameter_groups = []

    # 把 LoRA 放在第一个组。
    # Hugging Face 日志通常显示第一个 group 的 LR，
    # 因此日志中的 learning_rate 会对应 LoRA LR。
    component_order = (
        "lora",
        "visual_experts",
        "router_gate",
        "visual_norm",
    )

    for component in component_order:
        for no_decay in (False, True):
            parameters = buckets.get(
                (component, no_decay),
                [],
            )

            if not parameters:
                continue

            parameter_groups.append(
                {
                    "params": parameters,
                    "lr": learning_rates[component],
                    "weight_decay": (
                        0.0
                        if no_decay
                        else float(weight_decay)
                    ),
                    "group_name": (
                        f"{component}/no_decay"
                        if no_decay
                        else f"{component}/decay"
                    ),
                }
            )

    if not parameter_groups:
        raise RuntimeError(
            "优化器没有找到任何可训练参数。"
        )

    return torch.optim.AdamW(
        parameter_groups,
        betas=(
            float(adam_beta1),
            float(adam_beta2),
        ),
        eps=float(adam_epsilon),
    )


def format_optimizer_groups(
    optimizer: torch.optim.Optimizer,
) -> str:
    """
    打印每个参数组的学习率、weight decay 和参数量。
    """

    lines = [
        "Discriminative AdamW parameter groups:"
    ]

    for group in optimizer.param_groups:
        parameter_count = sum(
            parameter.numel()
            for parameter in group["params"]
        )

        lines.append(
            "  - "
            f"{group.get('group_name', 'unnamed')}: "
            f"lr={group['lr']:.3e}, "
            f"weight_decay={group['weight_decay']:.3g}, "
            f"params={parameter_count:,}"
        )

    return "\n".join(lines)
