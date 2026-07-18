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
    - 三专家 Conv2D F3/F5/F7
    - learned / fixed scale routing
    - learned / fixed soft gate
    - learnable / fixed lambda
    - RMS residual norm on/off

    Router 的三个输出依次对应：
        index 0 -> F3
        index 1 -> F5
        index 2 -> F7
    """

    def __init__(
        self,
        hidden_dim: int,
        adapter_f3: nn.Module,
        adapter_f5: nn.Module,
        adapter_f7: nn.Module,
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

        # 三个完整密集 DWConv2D 专家：
        # F3 -> 3x3
        # F5 -> 5x5
        # F7 -> 7x7
        self.adapter_f3 = adapter_f3
        self.adapter_f5 = adapter_f5
        self.adapter_f7 = adapter_f7

        if use_cross_modal_prior:
            if biomedclip_image_encoder_dim != biomedclip_text_encoder_dim:
                raise ValueError(
                    "use_cross_modal_prior=True 时，"
                    "biomedclip_image_encoder_dim 必须等于 "
                    "biomedclip_text_encoder_dim。"
                )

            router_input_dim = (
                biomedclip_image_encoder_dim * 4 + 1
            )
        else:
            router_input_dim = (
                biomedclip_image_encoder_dim
                + biomedclip_text_encoder_dim
            )

        self.router_backbone = nn.Sequential(
            nn.Linear(
                router_input_dim,
                router_hidden_dim,
            ),
            nn.LayerNorm(router_hidden_dim),
            nn.GELU(),
        )

        # 三个输出依次对应 F3、F5、F7。
        self.scale_head = nn.Linear(
            router_hidden_dim,
            3,
        )

        self.gate_head = nn.Linear(
            router_hidden_dim,
            1,
        )

        fixed_scale_weights = normalize_fixed_weights(
            fixed_scale_weights,
            num_experts=3,
        )

        self.register_buffer(
            "fixed_scale_weights",
            torch.tensor(
                fixed_scale_weights,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        lambda_ratio = (
            float(lambda_init)
            / float(lambda_max)
        )

        self.lambda_a = nn.Parameter(
            torch.tensor(
                safe_logit(lambda_ratio),
                dtype=torch.float32,
            )
        )

        # 检查消融模式是否合法。
        self._validate_modes()

        # 初始化 Router head。
        self._reset_router_parameters(
            gate_init=gate_init,
        )

        # 冻结当前消融模式中不会参与训练的分支。
        self._freeze_disabled_branches()

    def _validate_modes(self):
        if self.scale_mode not in {
            "learned",
            "fixed",
        }:
            raise ValueError(
                f"未知 scale_mode: {self.scale_mode}"
            )

        if self.gate_mode not in {
            "learned",
            "fixed",
        }:
            raise ValueError(
                f"未知 gate_mode: {self.gate_mode}"
            )

        if self.lambda_mode not in {
            "learnable",
            "fixed",
        }:
            raise ValueError(
                f"未知 lambda_mode: {self.lambda_mode}"
            )

    def _reset_router_parameters(
        self,
        gate_init: float,
    ):
        # 初始 scale routing 为均匀分布：
        # F3 = F5 = F7 = 1/3。
        nn.init.zeros_(
            self.scale_head.weight
        )
        nn.init.zeros_(
            self.scale_head.bias
        )

        # 初始 gate 为 gate_init。
        nn.init.zeros_(
            self.gate_head.weight
        )
        nn.init.constant_(
            self.gate_head.bias,
            safe_logit(gate_init),
        )

    def _freeze_disabled_branches(self):
        """
        根据当前消融模式冻结不参与前向计算的参数。

        作用：
        1. trainable parameter 统计更准确；
        2. 优化器不会收集无效参数；
        3. 避免 DDP 将固定分支视为 unused parameters。
        """

        # 固定尺度路由时，不训练 scale head。
        if self.scale_mode == "fixed":
            self.scale_head.requires_grad_(
                False
            )

        # 固定 residual gate 时，不训练 gate head。
        if self.gate_mode == "fixed":
            self.gate_head.requires_grad_(
                False
            )

        # 固定 lambda 时，不训练 lambda_a。
        #
        # 保留 lambda_a 参数对象，使不同消融实验之间
        # state_dict 的字段结构尽量一致。
        if self.lambda_mode == "fixed":
            self.lambda_a.requires_grad_(
                False
            )

        # Router backbone 同时服务于 scale head 和 gate head。
        # 只有两者都固定时，backbone 才完全不参与训练。
        if (
            self.scale_mode == "fixed"
            and self.gate_mode == "fixed"
        ):
            self.router_backbone.requires_grad_(
                False
            )

    def _get_scale_weights(
        self,
        route_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            scale_weights: [num_images, 3]

            三个维度依次表示：
                [:, 0] -> F3
                [:, 1] -> F5
                [:, 2] -> F7
        """
        if self.scale_mode == "learned":
            scale_logits = self.scale_head(
                route_hidden
            )

            return torch.softmax(
                scale_logits,
                dim=-1,
            )

        batch_size = route_hidden.size(0)

        return self.fixed_scale_weights.to(
            device=route_hidden.device,
            dtype=route_hidden.dtype,
        ).unsqueeze(0).expand(
            batch_size,
            -1,
        )

    def _get_soft_gate(
        self,
        route_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if self.gate_mode == "learned":
            return torch.sigmoid(
                self.gate_head(route_hidden)
            )

        return torch.full(
            (
                route_hidden.size(0),
                1,
            ),
            fill_value=self.fixed_gate,
            device=route_hidden.device,
            dtype=route_hidden.dtype,
        )

    def _get_lambda(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.lambda_mode == "learnable":
            return compute_bounded_lambda(
                lambda_a=self.lambda_a,
                lambda_max=self.lambda_max,
            ).to(
                device=x.device,
                dtype=x.dtype,
            )

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
        """
        Args:
            x:
                Qwen 视觉塔输出。

                支持：
                    [N_total, D]
                    [1, N_total, D]

            biomed_img_feat:
                BioMedCLIP 图像特征。
                第一维必须对应图片数量。

            biomed_txt_feat:
                BioMedCLIP 问题文本特征。
                第一维必须对应图片数量。

            grid_thw:
                每张图片对应的原始 Qwen 视觉网格。
                shape: [num_images, 3]

        Returns:
            与输入 x 形状一致的视觉特征。
        """
        if (
            biomed_img_feat is None
            or biomed_txt_feat is None
        ):
            raise ValueError(
                "动态融合模式下，必须传入 "
                "biomed_img_feat 和 biomed_txt_feat。"
            )

        if grid_thw is None:
            raise ValueError(
                "动态融合模式下，必须传入 grid_thw。"
            )

        # Qwen 的视觉输出可能是 [N, D]。
        # visual_token_split 统一要求 [1, N, D]。
        is_2d = x.dim() == 2

        if is_2d:
            x = x.unsqueeze(0)

        if x.dim() != 3:
            raise ValueError(
                "Fusion 要求 x 为 [N, D] 或 [1, N, D]，"
                f"当前 shape={tuple(x.shape)}"
            )

        route_features = (
            build_biomedclip_route_features(
                biomed_img_feat=biomed_img_feat,
                biomed_txt_feat=biomed_txt_feat,
                use_cross_modal_prior=(
                    self.use_cross_modal_prior
                ),
            )
        )

        route_hidden = self.router_backbone(
            route_features
        )

        # scale_weights:
        #   [:, 0] -> F3
        #   [:, 1] -> F5
        #   [:, 2] -> F7
        scale_weights = self._get_scale_weights(
            route_hidden
        )

        soft_gate = self._get_soft_gate(
            route_hidden
        )

        lambda_value = self._get_lambda(
            x
        )

        # 保留最近一次路由结果，供训练日志和评测诊断使用。
        self.latest_routing_weights = (
            scale_weights.detach().clone()
        )

        self.latest_soft_gate = (
            soft_gate.detach().clone()
        )

        self.latest_lambda = (
            lambda_value.detach().clone()
        )

        # 每个元素为：
        # (
        #     sub_x: [1, N_i, D],
        #     grid_shape: (t, H, W),
        # )
        x_splits = visual_token_split(
            x,
            grid_thw,
        )

        num_visual_items = len(x_splits)

        if scale_weights.size(0) != num_visual_items:
            raise ValueError(
                "Router batch size 与视觉图片数不一致："
                f"router B={scale_weights.size(0)}, "
                f"visual splits={num_visual_items}"
            )

        if soft_gate.size(0) != num_visual_items:
            raise ValueError(
                "Gate batch size 与视觉图片数不一致："
                f"gate B={soft_gate.size(0)}, "
                f"visual splits={num_visual_items}"
            )

        out_list = []

        for i, (
            sub_x,
            grid_shape,
        ) in enumerate(x_splits):
            # Router 的三个输出依次对应 F3、F5、F7。
            w3 = scale_weights[i, 0]
            w5 = scale_weights[i, 1]
            w7 = scale_weights[i, 2]

            # [1] -> [1, 1, 1]
            # 与 [1, N_i, D] residual 广播相乘。
            gate = soft_gate[i].view(
                1,
                1,
                1,
            )

            # 每张图片独立进行 LayerNorm 和 Conv2D，
            # 避免不同图片之间发生卷积污染。
            norm_sub_x = self.norm(
                sub_x
            )

            # 每个 Adapter 内部根据 grid_shape 将：
            # [1, N_i, D]
            # -> [t, C, H, W]
            # -> DWConv2D
            # -> [1, N_i, D]
            res3 = self.adapter_f3(
                norm_sub_x,
                grid_shape,
            )

            res5 = self.adapter_f5(
                norm_sub_x,
                grid_shape,
            )

            res7 = self.adapter_f7(
                norm_sub_x,
                grid_shape,
            )

            # Question-conditioned 三尺度动态融合。
            residual = (
                w3 * res3
                + w5 * res5
                + w7 * res7
            )

            if self.use_rms_norm:
                residual = rms_normalize_residual(
                    x=sub_x,
                    residual=residual,
                    eps=self.residual_norm_eps,
                    ratio_clip=(
                        self.residual_norm_ratio_clip
                    ),
                )

            sub_out = (
                sub_x
                + lambda_value
                * gate
                * residual
            )

            out_list.append(
                sub_out
            )

        # 恢复为原始展平视觉 token 顺序。
        out = torch.cat(
            out_list,
            dim=1,
        )

        if out.size(1) != x.size(1):
            raise RuntimeError(
                "Fusion 输出 token 数发生变化："
                f"input={x.size(1)}, "
                f"output={out.size(1)}"
            )

        if is_2d:
            out = out.squeeze(0)

        return out
