import math
from typing import Tuple

import torch
import torch.nn as nn


class BasicVisualAdapter(nn.Module):
    """
    基本视觉适配器：Single-Scale Conv2D Residual Expert。

    输入视觉 token:
        x: [1, N, hidden_dim]

    根据 grid_shape=(t, h, w) 恢复二维网格:
        [1, N, C]
        -> [t, h, w, C]
        -> [t, C, h, w]

    其中 t 被视为 Conv2D 的 batch 维度，不在时间维进行卷积。

    三个专家使用真实的密集二维卷积：
        F3: kernel_size=3, receptive field=3x3
        F5: kernel_size=5, receptive field=5x5
        F7: kernel_size=7, receptive field=7x7
    """

    def __init__(
        self,
        hidden_dim: int,
        r: int = 16,
        kernel_size: int = 3,
    ):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim 必须为正整数，当前为 {hidden_dim}"
            )

        if r <= 0:
            raise ValueError(
                f"r 必须为正整数，当前为 {r}"
            )

        if hidden_dim % r != 0:
            raise ValueError(
                "hidden_dim 必须能够被 r 整除："
                f"hidden_dim={hidden_dim}, r={r}"
            )

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size 必须为正奇数，"
                f"当前为 {kernel_size}"
            )

        self.hidden_dim = hidden_dim
        self.r = r
        self.kernel_size = kernel_size

        # Conv2D reshape 时需要使用，因此保存为类属性
        self.bottleneck_dim = hidden_dim // r

        self.down = nn.Linear(
            hidden_dim,
            self.bottleneck_dim,
            bias=False,
        )

        # 保持输入输出空间尺寸不变
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=self.bottleneck_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=1,
            groups=self.bottleneck_dim,
            bias=False,
        )

        self.act = nn.GELU()

        self.up = nn.Linear(
            self.bottleneck_dim,
            hidden_dim,
            bias=False,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Depthwise Conv2D 使用标准 Kaiming 初始化
        nn.init.kaiming_uniform_(
            self.conv.weight,
            a=math.sqrt(5),
        )

        # 保持原有策略：
        # up 零初始化，使训练开始时 residual branch 近似为 0
        nn.init.zeros_(self.up.weight)

    def forward(
        self,
        x: torch.Tensor,
        grid_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Args:
            x:
                单张图片对应的视觉 token，
                shape 为 [1, N, hidden_dim]。

            grid_shape:
                Qwen 空间合并后的网格形状 (t, h, w)。

                必须满足：
                    N == t * h * w

        Returns:
            residual:
                shape 与 x 相同，为 [1, N, hidden_dim]。
        """

        if x.ndim != 3:
            raise ValueError(
                "Conv2D VisualAdapter 要求 x 为三维张量 "
                "[1, N, D]，"
                f"当前 shape={tuple(x.shape)}"
            )

        batch_size, token_count, hidden_dim = x.shape

        # 当前 Fusion 是按图片切分后逐张调用专家，
        # 因此这里明确要求 batch_size == 1。
        if batch_size != 1:
            raise ValueError(
                "Conv2D VisualAdapter 要求每次输入一张图片的 token，"
                f"即 batch_size=1，当前为 {batch_size}"
            )

        if hidden_dim != self.hidden_dim:
            raise ValueError(
                "输入 hidden_dim 与 Adapter 配置不一致："
                f"input={hidden_dim}, "
                f"expected={self.hidden_dim}"
            )

        if len(grid_shape) != 3:
            raise ValueError(
                "grid_shape 必须是 (t, h, w)，"
                f"当前为 {grid_shape}"
            )

        t, grid_h, grid_w = (
            int(grid_shape[0]),
            int(grid_shape[1]),
            int(grid_shape[2]),
        )

        if t <= 0 or grid_h <= 0 or grid_w <= 0:
            raise ValueError(
                "grid_shape 中的 t、h、w 必须均为正整数，"
                f"当前为 {(t, grid_h, grid_w)}"
            )

        expected_tokens = t * grid_h * grid_w

        if token_count != expected_tokens:
            raise ValueError(
                "视觉 token 数量与二维网格不匹配："
                f"tokens={token_count}, "
                f"grid_shape={(t, grid_h, grid_w)}, "
                f"expected_tokens={expected_tokens}"
            )

        # [1, N, hidden_dim]
        # -> [1, N, bottleneck_dim]
        h = self.down(x)

        # [1, N, C]
        # -> [t, H, W, C]
        h = h.reshape(
            t,
            grid_h,
            grid_w,
            self.bottleneck_dim,
        )

        # [t, H, W, C]
        # -> [t, C, H, W]
        #
        # t 作为 Conv2D 的 batch 维度，
        # 不在时间维度上做卷积。
        h = h.permute(0, 3, 1, 2).contiguous()

        # Depthwise Conv2D
        h = self.conv(h)

        # [t, C, H, W]
        # -> [t, H, W, C]
        h = h.permute(0, 2, 3, 1).contiguous()

        # [t, H, W, C]
        # -> [1, N, C]
        h = h.reshape(
            1,
            expected_tokens,
            self.bottleneck_dim,
        )

        h = self.act(h)

        # [1, N, bottleneck_dim]
        # -> [1, N, hidden_dim]
        return self.up(h)


class VisualAdapter_F3(BasicVisualAdapter):
    """
    小尺度二维残差专家。

    使用完整的 depthwise 3x3 Conv2D。
    """

    def __init__(
        self,
        hidden_dim: int,
        r: int = 16,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            r=r,
            kernel_size=3,
        )


class VisualAdapter_F5(BasicVisualAdapter):
    """
    中尺度二维残差专家。

    使用完整的 depthwise 5x5 Conv2D，
    不再使用 3x3 dilation=2 模拟感受野。
    """

    def __init__(
        self,
        hidden_dim: int,
        r: int = 16,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            r=r,
            kernel_size=5,
        )


class VisualAdapter_F7(BasicVisualAdapter):
    """
    大尺度二维残差专家。

    使用完整的 depthwise 7x7 Conv2D。
    """

    def __init__(
        self,
        hidden_dim: int,
        r: int = 16,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            r=r,
            kernel_size=7,
        )
