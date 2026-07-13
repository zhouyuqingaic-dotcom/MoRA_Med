import torch
import torch.nn as nn

from utils.qwen3vl.qwen3_vl_8B_visual_adapters_fusion_utils import (
    visual_token_split,
    safe_logit,
    normalize_fixed_weights,
    build_biomedclip_route_features,
    compute_bounded_lambda,
    rms_normalize_residual,
)


class Qwen3VLMoEVisualAdapterFusion(nn.Module):
    """
    消融友好的 V2-lite fusion module。

    支持：
    - 三专家 F1/F3/F5
    - learned / fixed scale routing
    - learned / fixed soft gate
    - learnable / fixed lambda
    - RMS residual norm on/off
    """


    def __init__(
        self,
        hidden_dim: int,
        adapter_f1: nn.Module,
        adapter_f3: nn.Module,
        adapter_f5: nn.Module,
        biomedclip_image_encoder_dim: int = 512,
        biomedclip_text_encoder_dim: int = 512,
        router_hidden_dim: int = 128,

        scale_mode: str = "learned",
        fixed_scale_weights: list[float] | None = None,

        gate_mode: str = "learned",
        fixed_gate: float = 1.0,
        gate_init: float = 0.5,

        lambda_mode: str = "learnable",
        fixed_lambda: float = 0.1,
        lambda_max: float = 1.0,
        lambda_init: float = 0.1,

        use_rms_norm: bool = True,
        residual_norm_eps: float = 1e-6,
        residual_norm_ratio_clip: float | None = 10.0,

        use_cross_modal_prior: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.scale_mode = scale_mode
        self.gate_mode = gate_mode
        self.lambda_mode = lambda_mode

        self.fixed_gate = float(fixed_gate)
        self.fixed_lambda = float(fixed_lambda)

        self.lambda_max = float(lambda_max)
        self.use_rms_norm = bool(use_rms_norm)
        self.residual_norm_eps = residual_norm_eps
        self.residual_norm_ratio_clip = residual_norm_ratio_clip
        self.use_cross_modal_prior = use_cross_modal_prior

        self.norm = nn.LayerNorm(hidden_dim)

        self.adapter_f1 = adapter_f1
        self.adapter_f3 = adapter_f3
        self.adapter_f5 = adapter_f5

        if use_cross_modal_prior:
            if biomedclip_image_encoder_dim != biomedclip_text_encoder_dim:
                raise ValueError(
                    "use_cross_modal_prior=True 时，"
                    "biomedclip_image_encoder_dim 必须等于 biomedclip_text_encoder_dim。"
                )
            router_input_dim = biomedclip_image_encoder_dim * 4 + 1
        else:
            router_input_dim = biomedclip_image_encoder_dim + biomedclip_text_encoder_dim

        self.router_backbone = nn.Sequential(
            nn.Linear(router_input_dim, router_hidden_dim),
            nn.LayerNorm(router_hidden_dim),
            nn.GELU(),
        )

        self.scale_head = nn.Linear(
            router_hidden_dim,
            3,
        )
        self.gate_head = nn.Linear(router_hidden_dim, 1)

        fixed_scale_weights = normalize_fixed_weights(
            fixed_scale_weights,
            num_experts=3,
        )
        self.register_buffer(
            "fixed_scale_weights",
            torch.tensor(fixed_scale_weights, dtype=torch.float32),
            persistent=False,
        )

        lambda_ratio = float(lambda_init) / float(lambda_max)
        self.lambda_a = nn.Parameter(
            torch.tensor(safe_logit(lambda_ratio), dtype=torch.float32)
        )

        # 首先检查消融模式是否合法。
        self._validate_modes()

        # 初始化路由 head。
        self._reset_router_parameters(gate_init=gate_init)

        # 冻结当前消融模式下不会参与训练的分支。
        self._freeze_disabled_branches()

    def _validate_modes(self):
        if self.scale_mode not in {"learned", "fixed"}:
            raise ValueError(f"未知 scale_mode: {self.scale_mode}")

        if self.gate_mode not in {"learned", "fixed"}:
            raise ValueError(f"未知 gate_mode: {self.gate_mode}")

        if self.lambda_mode not in {"learnable", "fixed"}:
            raise ValueError(f"未知 lambda_mode: {self.lambda_mode}")

    def _reset_router_parameters(self, gate_init: float):
        # 初始 scale routing 为均匀分布
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

        # 初始 gate 为 gate_init
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, safe_logit(gate_init))

    def _freeze_disabled_branches(self):
        """
        根据当前消融模式冻结不参与前向计算的参数。

        这样做有三个作用：
        1. trainable parameter 统计更加准确；
        2. 优化器不会收集无效参数；
        3. 避免 DDP 将固定分支识别为 unused parameters。
        """

        # -------------------------------------------------
        # 固定尺度路由时，不训练 scale head。
        # -------------------------------------------------
        if self.scale_mode == "fixed":
            self.scale_head.requires_grad_(False)

        # -------------------------------------------------
        # 固定 residual gate 时，不训练 gate head。
        # -------------------------------------------------
        if self.gate_mode == "fixed":
            self.gate_head.requires_grad_(False)

        # -------------------------------------------------
        # 固定 lambda 时，不训练 lambda_a。
        #
        # 保留 lambda_a 这个参数对象，可以维持不同消融实验之间
        # state_dict 的字段结构一致，只关闭它的梯度即可。
        # -------------------------------------------------
        if self.lambda_mode == "fixed":
            self.lambda_a.requires_grad_(False)

        # -------------------------------------------------
        # router_backbone 同时为 scale head 和 gate head 服务。
        #
        # 只有当 scale 和 gate 都是 fixed 时，
        # router_backbone 才完全不参与任何可学习路由分支，
        # 此时应一并冻结。
        # -------------------------------------------------
        if self.scale_mode == "fixed" and self.gate_mode == "fixed":
            self.router_backbone.requires_grad_(False)

    def _get_scale_weights(self, route_hidden: torch.Tensor) -> torch.Tensor:
        if self.scale_mode == "learned":
            scale_logits = self.scale_head(route_hidden)
            return torch.softmax(scale_logits, dim=-1)

        bsz = route_hidden.size(0)
        return self.fixed_scale_weights.to(
            device=route_hidden.device,
            dtype=route_hidden.dtype,
        ).unsqueeze(0).expand(bsz, -1)

    def _get_soft_gate(self, route_hidden: torch.Tensor) -> torch.Tensor:
        if self.gate_mode == "learned":
            return torch.sigmoid(self.gate_head(route_hidden))

        return torch.full(
            (route_hidden.size(0), 1),
            fill_value=self.fixed_gate,
            device=route_hidden.device,
            dtype=route_hidden.dtype,
        )

    def _get_lambda(self, x: torch.Tensor) -> torch.Tensor:
        if self.lambda_mode == "learnable":
            return compute_bounded_lambda(
                lambda_a=self.lambda_a,
                lambda_max=self.lambda_max,
            ).to(device=x.device, dtype=x.dtype)

        return torch.tensor(
            self.fixed_lambda,
            device=x.device,
            dtype=x.dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        biomed_img_feat: torch.Tensor,
        biomed_txt_feat: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        if biomed_img_feat is None or biomed_txt_feat is None:
            raise ValueError("动态融合模式下，必须传入 biomed_img_feat 和 biomed_txt_feat。")

        if grid_thw is None:
            raise ValueError("动态融合模式下，必须传入 grid_thw。")

        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(0)

        route_features = build_biomedclip_route_features(
            biomed_img_feat=biomed_img_feat,
            biomed_txt_feat=biomed_txt_feat,
            use_cross_modal_prior=self.use_cross_modal_prior,
        )

        route_hidden = self.router_backbone(route_features)

        scale_weights = self._get_scale_weights(route_hidden)
        soft_gate = self._get_soft_gate(route_hidden)
        lambda_value = self._get_lambda(x)

        self.latest_routing_weights = scale_weights.detach().clone()
        self.latest_soft_gate = soft_gate.detach().clone()
        self.latest_lambda = lambda_value.detach().clone()

        x_splits = visual_token_split(x, grid_thw)

        if scale_weights.size(0) != len(x_splits):
            raise ValueError(
                f"Router batch size 与视觉图片数不一致："
                f"router B={scale_weights.size(0)}, visual splits={len(x_splits)}"
            )

        out_list = []

        for i, sub_x in enumerate(x_splits):
            w1 = scale_weights[i, 0]
            w3 = scale_weights[i, 1]
            w5 = scale_weights[i, 2]

            g = soft_gate[i].view(
                1,
                1,
                1,
            )

            norm_sub_x = self.norm(
                sub_x
            )

            res1 = self.adapter_f1(
                norm_sub_x
            )
            res3 = self.adapter_f3(
                norm_sub_x
            )
            res5 = self.adapter_f5(
                norm_sub_x
            )

            residual = (
                    w1 * res1
                    + w3 * res3
                    + w5 * res5
            )

            if self.use_rms_norm:
                residual = rms_normalize_residual(
                    x=sub_x,
                    residual=residual,
                    eps=self.residual_norm_eps,
                    ratio_clip=self.residual_norm_ratio_clip,
                )

            sub_out = sub_x + lambda_value * g * residual
            out_list.append(sub_out)

        out = torch.cat(out_list, dim=1)

        if is_2d:
            out = out.squeeze(0)

        return out
