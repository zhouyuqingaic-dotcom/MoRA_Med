import math
import torch
import torch.nn.functional as F


def visual_token_split(
    x: torch.Tensor,
    grid_thw: torch.Tensor,
) -> list[tuple[torch.Tensor, tuple[int, int, int]]]:
    """
    将 Qwen-VL 展平后的视觉 token 按 grid_thw 切回每张图片，
    同时返回每张图片经过 spatial merge 后的二维网格形状。

    Args:
        x:
            展平后的视觉 token。
            shape: [1, N_total, D]

        grid_thw:
            每张图片/视频对应的原始视觉网格。
            shape: [num_images, 3]
            每行内容为 [t, h, w]。

            Qwen-VL 经过 spatial merge 后：
                H = h // 2
                W = w // 2

            因此每张图片的 token 数为：
                N_i = t * H * W

    Returns:
        一个列表，每个元素为：

            (
                sub_x,              # [1, N_i, D]
                (t, H, W),          # 合并后的二维网格形状
            )

        例如：

            [
                (sub_x_0, (t0, H0, W0)),
                (sub_x_1, (t1, H1, W1)),
                ...
            ]
    """
    if x.ndim != 3:
        raise ValueError(
            "visual_token_split 要求 x 为三维张量 "
            f"[1, N_total, D]，实际 shape={tuple(x.shape)}"
        )

    if x.size(0) != 1:
        raise ValueError(
            "当前视觉 token 切分逻辑要求 x 的 batch 维为 1，"
            f"实际 batch size={x.size(0)}。"
        )

    if grid_thw.ndim != 2 or grid_thw.size(-1) != 3:
        raise ValueError(
            "grid_thw 必须为 [num_images, 3]，"
            f"实际 shape={tuple(grid_thw.shape)}"
        )

    # grid_thw 可能位于 GPU。
    # 转到 CPU 后统一解析，避免循环中反复调用 .item()。
    grid_rows = grid_thw.detach().cpu().tolist()

    split_results: list[
        tuple[torch.Tensor, tuple[int, int, int]]
    ] = []

    start = 0
    total_tokens = x.size(1)
    spatial_merge_size = 2

    for image_index, row in enumerate(grid_rows):
        t, h, w = (int(value) for value in row)

        if t <= 0 or h <= 0 or w <= 0:
            raise ValueError(
                f"第 {image_index} 张图片的 grid_thw 非法："
                f"t={t}, h={h}, w={w}"
            )

        # 当前项目按照 Qwen-VL spatial merge size=2
        # 计算 merger 后的二维 token 网格。
        if (
            h % spatial_merge_size != 0
            or w % spatial_merge_size != 0
        ):
            raise ValueError(
                f"第 {image_index} 张图片的视觉网格不能被 "
                f"spatial_merge_size={spatial_merge_size} 整除："
                f"t={t}, h={h}, w={w}"
            )

        grid_h = h // spatial_merge_size
        grid_w = w // spatial_merge_size

        token_count = t * grid_h * grid_w
        end = start + token_count

        if end > total_tokens:
            raise ValueError(
                f"第 {image_index} 张图片切分越界："
                f"start={start}, end={end}, "
                f"输入 token 总数={total_tokens}, "
                f"grid_shape=({t}, {grid_h}, {grid_w})"
            )

        sub_x = x[:, start:end, :]

        if sub_x.size(1) != token_count:
            raise ValueError(
                f"第 {image_index} 张图片的视觉 token 数量错误："
                f"期望={token_count}, 实际={sub_x.size(1)}, "
                f"grid_shape=({t}, {grid_h}, {grid_w})"
            )

        split_results.append(
            (
                sub_x,
                (t, grid_h, grid_w),
            )
        )

        start = end

    # 防止 grid_thw 少描述了一部分 token。
    if start != total_tokens:
        raise ValueError(
            "视觉 Token 数量对齐失败："
            f"grid_thw 算出的总数={start}, "
            f"输入序列长度={total_tokens}"
        )

    return split_results


def safe_logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def normalize_fixed_weights(weights: list[float], num_experts: int = 4) -> list[float]:
    if weights is None:
        weights = [1.0 / num_experts] * num_experts

    if len(weights) != num_experts:
        raise ValueError(f"fixed weights 必须包含 {num_experts} 个数。")

    s = float(sum(weights))
    if s <= 0:
        raise ValueError("fixed weights 的和必须大于 0。")

    return [float(w) / s for w in weights]


def build_biomedclip_route_features(
    biomed_img_feat: torch.Tensor,
    biomed_txt_feat: torch.Tensor,
    use_cross_modal_prior: bool = True,
) -> torch.Tensor:
    """
    Router 输入特征。

    use_cross_modal_prior=False:
        [c_I; c_Q]

    use_cross_modal_prior=True:
        [c_I; c_Q; c_I*c_Q; |c_I-c_Q|; cos(c_I,c_Q)]
    """
    if not use_cross_modal_prior:
        return torch.cat([biomed_img_feat, biomed_txt_feat], dim=-1)

    img = F.normalize(biomed_img_feat, dim=-1)
    txt = F.normalize(biomed_txt_feat, dim=-1)

    prod = img * txt
    diff = torch.abs(img - txt)
    cos = torch.sum(img * txt, dim=-1, keepdim=True)

    return torch.cat([img, txt, prod, diff, cos], dim=-1)


def compute_bounded_lambda(
    lambda_a: torch.Tensor,
    lambda_max: float,
) -> torch.Tensor:
    """
    λ = λ_max * sigmoid(a)
    """
    return float(lambda_max) * torch.sigmoid(lambda_a.float())


def _compute_rms_match_tensors(
    x: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-6,
    ratio_clip: float | None = 10.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    统一计算 RMS Matching 使用的中间量。

    该函数是 rms_normalize_residual() 与诊断逻辑的共同数学来源，
    避免模型计算和日志统计分别实现一套公式后产生偏差。

    Args:
        x:
            原始视觉特征，shape 通常为 [1, N, D]。

        residual:
            RMS Matching 前的 routed residual，
            shape 与 x 一致。

        eps:
            与正式 RMS Matching 完全一致的数值稳定项。

        ratio_clip:
            RMS 缩放系数的上限。
            None 表示不截断。

    Returns:
        (
            rms_x_eps,
            rms_r_eps,
            ratio_unclipped,
            ratio_applied,
            ratio_clipped,
        )

        其中：
            rms_x_eps:
                sqrt(mean(x^2) + eps)

            rms_r_eps:
                sqrt(mean(residual^2) + eps)

            ratio_unclipped:
                rms_x_eps / rms_r_eps

            ratio_applied:
                实际用于 residual 缩放的系数。

            ratio_clipped:
                bool tensor，表示是否触发 ratio_clip。
    """
    x_fp32 = x.float()
    r_fp32 = residual.float()

    rms_x_eps = torch.sqrt(
        torch.mean(
            x_fp32 ** 2,
            dim=(-2, -1),
            keepdim=True,
        )
        + eps
    )

    rms_r_eps = torch.sqrt(
        torch.mean(
            r_fp32 ** 2,
            dim=(-2, -1),
            keepdim=True,
        )
        + eps
    )

    ratio_unclipped = (
        rms_x_eps
        / rms_r_eps
    )

    if ratio_clip is None:
        ratio_applied = ratio_unclipped
        ratio_clipped = torch.zeros_like(
            ratio_unclipped,
            dtype=torch.bool,
        )
    else:
        ratio_applied = torch.clamp(
            ratio_unclipped,
            max=ratio_clip,
        )

        ratio_clipped = (
            ratio_unclipped
            > float(ratio_clip)
        )

    return (
        rms_x_eps,
        rms_r_eps,
        ratio_unclipped,
        ratio_applied,
        ratio_clipped,
    )


def compute_rms_match_statistics(
    x: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-6,
    ratio_clip: float | None = 10.0,
) -> dict[str, torch.Tensor]:
    """
    返回与正式 RMS Matching 完全一致的诊断张量。

    该函数只负责暴露统计信息，不修改输入张量，
    也不参与 residual injection。

    建议仅在 diagnostics_enabled=True 时调用。

    Returns:
        {
            "rms_x_eps": Tensor,
            "rms_r_eps": Tensor,
            "ratio_unclipped": Tensor,
            "ratio_applied": Tensor,
            "ratio_clipped": BoolTensor,
        }
    """
    (
        rms_x_eps,
        rms_r_eps,
        ratio_unclipped,
        ratio_applied,
        ratio_clipped,
    ) = _compute_rms_match_tensors(
        x=x,
        residual=residual,
        eps=eps,
        ratio_clip=ratio_clip,
    )

    return {
        "rms_x_eps": rms_x_eps,
        "rms_r_eps": rms_r_eps,
        "ratio_unclipped": ratio_unclipped,
        "ratio_applied": ratio_applied,
        "ratio_clipped": ratio_clipped,
    }


def rms_normalize_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-6,
    ratio_clip: float | None = 10.0,
) -> torch.Tensor:
    """
    对单张图的 token + hidden 维度整体算 RMS。

    x/residual: [1, N, D]

    注意：
        模型实际使用的缩放系数与
        compute_rms_match_statistics() 共用同一套内部计算，
        因此诊断结果与正式前向保持一致。
    """
    (
        _,
        _,
        _,
        ratio_applied,
        _,
    ) = _compute_rms_match_tensors(
        x=x,
        residual=residual,
        eps=eps,
        ratio_clip=ratio_clip,
    )

    return residual * ratio_applied.to(
        device=residual.device,
        dtype=residual.dtype,
    )