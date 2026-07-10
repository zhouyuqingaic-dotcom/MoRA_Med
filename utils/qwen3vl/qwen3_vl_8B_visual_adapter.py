import math
import torch
import torch.nn as nn


class BasicVisualAdapter(nn.Module):
    """
    基本视觉适配器 Single-Scale Residual Expert。

    通过 kernel_size 和 dilation 控制 1D token 序列上的感受野：
    RF = (kernel_size - 1) * dilation + 1

    F1: kernel=1, dilation=1, RF=1
    F3: kernel=3, dilation=1, RF=3
    F5: kernel=3, dilation=2, RF=5
    F7: kernel=3, dilation=3, RF=7
    """

    def __init__(
        self,
        hidden_dim: int,
        r: int = 16,
        kernel_size: int = 1,
        dilation: int = 1,
    ):
        super().__init__()

        bottleneck_dim = hidden_dim // r

        self.down = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim, bias=False)

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_conv = kernel_size > 1

        if self.use_conv:
            padding = (kernel_size - 1) * dilation // 2
            self.conv = nn.Conv1d(
                in_channels=bottleneck_dim,
                out_channels=bottleneck_dim,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=bottleneck_dim,
                bias=False,
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.use_conv:
            nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))

        # 关键：up 零初始化，保证训练初期 residual branch 近似 0
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        h = self.down(x)

        if self.use_conv:
            h = h.transpose(-1, -2)  # [B, C, N]
            h = self.conv(h)
            h = h.transpose(-1, -2)  # [B, N, C]

        h = self.act(h)
        return self.up(h)


class VisualAdapter_F1(BasicVisualAdapter):
    """
    Point-wise residual expert, RF=1。
    """
    def __init__(self, hidden_dim: int, r: int = 16):
        super().__init__(
            hidden_dim=hidden_dim,
            r=r,
            kernel_size=1,
            dilation=1,
        )


class VisualAdapter_F3(BasicVisualAdapter):
    """
    Short-range residual expert, RF=3。
    """
    def __init__(self, hidden_dim: int, r: int = 16):
        super().__init__(
            hidden_dim=hidden_dim,
            r=r,
            kernel_size=3,
            dilation=1,
        )


class VisualAdapter_F5(BasicVisualAdapter):
    """
    Mid-range residual expert, RF=5。
    通过 kernel=3, dilation=2 实现，而不是直接 kernel=5。
    """
    def __init__(self, hidden_dim: int, r: int = 16):
        super().__init__(
            hidden_dim=hidden_dim,
            r=r,
            kernel_size=3,
            dilation=2,
        )


class VisualAdapter_F7(BasicVisualAdapter):
    """
    Long-range local residual expert, RF=7。
    通过 kernel=3, dilation=3 实现。
    """
    def __init__(self, hidden_dim: int, r: int = 16):
        super().__init__(
            hidden_dim=hidden_dim,
            r=r,
            kernel_size=3,
            dilation=3,
        )
