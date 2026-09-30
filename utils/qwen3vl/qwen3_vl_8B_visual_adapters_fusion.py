import torch
import torch.nn as nn

from utils.qwen3vl.qwen3_vl_8B_visual_adapters_fusion_utils import (
    visual_token_split,
    safe_logit,
    normalize_fixed_weights,
    build_biomedclip_route_features,
    compute_bounded_lambda,
    rms_normalize_residual,
    compute_rms_match_statistics,
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

        # ---------------------------------------------------------
        # Evaluation-only diagnostics.
        #
        # 这些字段不注册为 parameter / buffer，
        # 不进入 state_dict，不影响旧 checkpoint 兼容性。
        # 默认关闭，因此正常训练和正常评测不会增加额外统计开销。
        # ---------------------------------------------------------
        self.diagnostics_enabled = False
        self.diagnostic_records = []

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

    def enable_diagnostics(
        self,
        enabled: bool = True,
    ):
        """
        开启 / 关闭推理期诊断。

        默认关闭。
        关闭时同时清空旧记录，避免不同评测条件之间串数据。
        """
        self.diagnostics_enabled = bool(
            enabled
        )

        if not self.diagnostics_enabled:
            self.reset_diagnostics()

    def reset_diagnostics(self):
        """
        清空当前累计的诊断记录。
        建议每个 evaluation batch 开始前调用一次。
        """
        self.diagnostic_records = []

    def pop_diagnostics(self) -> list[dict]:
        """
        取出并清空当前累计的诊断记录。
        """
        records = self.diagnostic_records
        self.diagnostic_records = []
        return records

    @staticmethod
    def _compute_plain_rms(
        tensor: torch.Tensor,
    ) -> float:
        """
        用于描述实际 tensor 幅值的 RMS。

        注意：
        这里只用于日志，不参与模型计算；
        不加入 residual_norm_eps，避免改变其物理解释。
        """
        value = tensor.detach().float()

        rms = torch.sqrt(
            torch.mean(
                value.square()
            )
        )

        return float(rms.item())

    def forward(
        self,
        x: torch.Tensor,
        biomed_img_feat: torch.Tensor,
        biomed_txt_feat: torch.Tensor,
        grid_thw: torch.Tensor,
        router_txt_feat: torch.Tensor | None = None,
        stream_name: str | None = None,
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
                BioMedCLIP 原始正确问题的文本特征。
                第一维必须对应图片数量。

            router_txt_feat:
                可选的 Router-only 文本特征。

                None:
                    Router 和 Gate 都使用 biomed_txt_feat，
                    与原始实现保持一致。

                非 None:
                    仅 Router 使用 router_txt_feat；
                    Gate 仍然使用 biomed_txt_feat。

                该参数仅用于评测阶段的问题置换诊断。

            stream_name:
                当前视觉特征流名称，仅用于诊断记录。

                例如：
                    "pooled"
                    "deepstack_0"
                    "deepstack_1"

                不参与模型计算。

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

        # Router-only question intervention:
        # 仅允许在 learned routing 的评测阶段使用。
        if router_txt_feat is not None:
            if self.training:
                raise RuntimeError(
                    "router_txt_feat 仅用于评测阶段的 "
                    "Router-only intervention。"
                    "请先调用 model.eval()。"
                )

            if self.scale_mode != "learned":
                raise ValueError(
                    "router_txt_feat 只适用于 "
                    "scale_mode='learned'。"
                )

            if (
                router_txt_feat.shape
                != biomed_txt_feat.shape
            ):
                raise ValueError(
                    "router_txt_feat 与 biomed_txt_feat "
                    "shape 必须一致："
                    f"router={tuple(router_txt_feat.shape)}, "
                    f"original={tuple(biomed_txt_feat.shape)}"
                )

            if (
                router_txt_feat.device
                != biomed_txt_feat.device
            ):
                raise ValueError(
                    "router_txt_feat 与 biomed_txt_feat "
                    "必须位于同一 device。"
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

        # ---------------------------------------------------------
        # Original image-question prior.
        #
        # Gate 永远由正确的问题条件控制。
        # 正常模式下 Router 也直接复用这个 hidden，
        # 从而保持与旧代码相同的计算路径。
        # ---------------------------------------------------------
        original_route_features = (
            build_biomedclip_route_features(
                biomed_img_feat=biomed_img_feat,
                biomed_txt_feat=biomed_txt_feat,
                use_cross_modal_prior=(
                    self.use_cross_modal_prior
                ),
            )
        )

        original_route_hidden = (
            self.router_backbone(
                original_route_features
            )
        )

        # Gate 始终使用原始正确问题。
        soft_gate = self._get_soft_gate(
            original_route_hidden
        )

        # ---------------------------------------------------------
        # Router input.
        #
        # Normal:
        #     Router(I, Q)
        #
        # Router-only shuffle:
        #     Router(I, Q_shuffle)
        #
        # 图像始终保持为当前样本的原图。
        # ---------------------------------------------------------
        if router_txt_feat is None:
            scale_route_hidden = (
                original_route_hidden
            )
        else:
            shuffled_route_features = (
                build_biomedclip_route_features(
                    biomed_img_feat=biomed_img_feat,
                    biomed_txt_feat=router_txt_feat,
                    use_cross_modal_prior=(
                        self.use_cross_modal_prior
                    ),
                )
            )

            scale_route_hidden = (
                self.router_backbone(
                    shuffled_route_features
                )
            )

        # scale_weights:
        #   [:, 0] -> F3
        #   [:, 1] -> F5
        #   [:, 2] -> F7
        scale_weights = (
            self._get_scale_weights(
                scale_route_hidden
            )
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

            # -----------------------------------------------------
            # Question-conditioned 三尺度残差融合。
            #
            # raw_residual:
            #     RMS Matching 之前的原始 routed residual。
            # -----------------------------------------------------
            raw_residual = (
                w3 * res3
                + w5 * res5
                + w7 * res7
            )

            # used_residual:
            #     真正进入 residual injection 的 residual。
            #
            # A2 / no-RMS:
            #     used_residual == raw_residual
            #
            # A6 / RMS:
            #     used_residual == RMS-matched residual
            used_residual = raw_residual

            if self.use_rms_norm:
                used_residual = (
                    rms_normalize_residual(
                        x=sub_x,
                        residual=raw_residual,
                        eps=self.residual_norm_eps,
                        ratio_clip=(
                            self.residual_norm_ratio_clip
                        ),
                    )
                )

            # 真正注入原视觉特征的 residual update。
            delta = (
                lambda_value
                * gate
                * used_residual
            )

            sub_out = (
                sub_x
                + delta
            )

            # -----------------------------------------------------
            # Evaluation-only diagnostics.
            #
            # 这里只记录 detached scalar，
            # 不改变模型计算，也不保存完整 tensor。
            # -----------------------------------------------------
            if self.diagnostics_enabled:
                rms_x = (
                    self._compute_plain_rms(
                        sub_x
                    )
                )

                rms_raw = (
                    self._compute_plain_rms(
                        raw_residual
                    )
                )

                rms_used = (
                    self._compute_plain_rms(
                        used_residual
                    )
                )

                rms_delta = (
                    self._compute_plain_rms(
                        delta
                    )
                )

                # -------------------------------------------------
                # RMS Matching internal statistics.
                #
                # 只在模型实际启用 RMS Matching 时记录。
                # 这里与 rms_normalize_residual() 共用同一套
                # utils 数学实现，避免“模型算一套、日志算一套”。
                #
                # A2 / no-RMS:
                #     rho_* = None
                #
                # A6 / RMS:
                #     记录 unclipped / applied / clipped。
                # -------------------------------------------------
                if self.use_rms_norm:
                    rms_match_stats = (
                        compute_rms_match_statistics(
                            x=sub_x,
                            residual=raw_residual,
                            eps=self.residual_norm_eps,
                            ratio_clip=(
                                self.residual_norm_ratio_clip
                            ),
                        )
                    )

                    rho_unclipped = float(
                        rms_match_stats[
                            "ratio_unclipped"
                        ]
                        .detach()
                        .float()
                        .reshape(-1)[0]
                        .item()
                    )

                    rho_applied = float(
                        rms_match_stats[
                            "ratio_applied"
                        ]
                        .detach()
                        .float()
                        .reshape(-1)[0]
                        .item()
                    )

                    rho_clipped = bool(
                        rms_match_stats[
                            "ratio_clipped"
                        ]
                        .detach()
                        .reshape(-1)[0]
                        .item()
                    )
                else:
                    rho_unclipped = None
                    rho_applied = None
                    rho_clipped = None

                # 防止极端情况下除以 0。
                if rms_x > 0.0:
                    ratio_raw = (
                        rms_raw / rms_x
                    )

                    ratio_used = (
                        rms_used / rms_x
                    )

                    ratio_inject = (
                        rms_delta / rms_x
                    )
                else:
                    ratio_raw = None
                    ratio_used = None
                    ratio_inject = None

                self.diagnostic_records.append(
                    {
                        "visual_item_index": i,
                        "stream_name": (
                            stream_name
                            if stream_name is not None
                            else "unknown"
                        ),
                        "use_rms_norm": (
                            self.use_rms_norm
                        ),
                        "rms_x": rms_x,
                        "rms_raw": rms_raw,
                        "rms_used": rms_used,
                        "rms_delta": rms_delta,
                        "ratio_raw": ratio_raw,
                        "ratio_used": ratio_used,
                        "ratio_inject": (
                            ratio_inject
                        ),

                        # RMS Matching scaling diagnostics.
                        # 对 no-RMS 消融（例如 A2）保持为 None，
                        # 避免记录并不存在于实际模型路径中的“假想 rho”。
                        "rho_unclipped": (
                            rho_unclipped
                        ),
                        "rho_applied": (
                            rho_applied
                        ),
                        "rho_clipped": (
                            rho_clipped
                        ),

                        "routing_f3": float(
                            w3.detach()
                            .float()
                            .item()
                        ),
                        "routing_f5": float(
                            w5.detach()
                            .float()
                            .item()
                        ),
                        "routing_f7": float(
                            w7.detach()
                            .float()
                            .item()
                        ),
                        "gate": float(
                            gate.detach()
                            .float()
                            .reshape(-1)[0]
                            .item()
                        ),
                        "lambda": float(
                            lambda_value.detach()
                            .float()
                            .reshape(-1)[0]
                            .item()
                        ),
                    }
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