import math
import torch
import torch.nn.functional as F


def visual_token_split(x: torch.Tensor, grid_thw: torch.Tensor) -> list[torch.Tensor]:
    """
    将 Qwen-VL 展平视觉 token 按 grid_thw 切回每张图片，避免 1D conv 跨图片污染。

    x: [1, N_total, D]
    grid_thw: [B, 3], 每行 [t, h, w]
    return: List[[1, N_i, D]]
    """
    t = grid_thw[:, 0]
    h = grid_thw[:, 1]
    w = grid_thw[:, 2]

    tokens_per_image = [int(n) for n in (t * (h // 2) * (w // 2)).tolist()]

    if sum(tokens_per_image) != x.size(1):
        raise ValueError(
            f"视觉 Token 数量对齐失败："
            f"grid_thw 算出总数={sum(tokens_per_image)}, 输入序列长度={x.size(1)}"
        )

    return list(torch.split(x, tokens_per_image, dim=1))


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


def rms_normalize_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-6,
    ratio_clip: float | None = 10.0,
) -> torch.Tensor:
    """
    对单张图的 token + hidden 维度整体算 RMS。

    x/residual: [1, N, D]
    """
    x_fp32 = x.float()
    r_fp32 = residual.float()

    rms_x = torch.sqrt(torch.mean(x_fp32 ** 2, dim=(-2, -1), keepdim=True) + eps)
    rms_r = torch.sqrt(torch.mean(r_fp32 ** 2, dim=(-2, -1), keepdim=True) + eps)

    ratio = rms_x / rms_r

    if ratio_clip is not None:
        ratio = torch.clamp(ratio, max=ratio_clip)

    return residual * ratio.to(device=residual.device, dtype=residual.dtype)
