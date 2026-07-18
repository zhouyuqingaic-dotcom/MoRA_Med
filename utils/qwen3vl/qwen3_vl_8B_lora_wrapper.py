import types
from typing import Optional, List

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import PreTrainedModel

from utils.ddp.ddp_utils import ddp_print
from utils.qwen3vl.qwen3_vl_8B_visual_adapter import (
    VisualAdapter_F3,
    VisualAdapter_F5,
    VisualAdapter_F7,
)

from utils.qwen3vl.qwen3_vl_8B_visual_adapters_fusion import (
    Qwen3VLMoEVisualAdapterFusion,
)


# =====================================================================
# 冻结的 BioMedCLIP 跨模态特征提取器
# =====================================================================
class FrozenBioMedCLIPFeatureExtractor(nn.Module):
    """
    Frozen BioMedCLIP feature extractor.

    只负责提取 BioMedCLIP image/text features。
    不参与梯度更新。
    """

    def __init__(self, raw_model: nn.Module):
        super().__init__()
        self.model = raw_model
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, biomed_img_tensors, biomed_txt_tokens):
        with torch.no_grad():
            img_feat, txt_feat, _ = self.model(
                biomed_img_tensors,
                biomed_txt_tokens,
            )

            img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
            txt_feat = torch.nn.functional.normalize(txt_feat, dim=-1)

        return img_feat, txt_feat


# =====================================================================
# Qwen3-VL + LoRA + V2-lite Visual Residual Adapter Wrapper
# =====================================================================
class Qwen3VLLoraAndVisualAdapterWrapper:
    """
    Qwen3-VL LoRA + RoMA-Net V2-lite visual residual adapter wrapper.

    支持消融：
    A0:
        Qwen3-VL + LoRA
        enable_visual_adapter=False

    A1:
        v2 fixed-alpha, w/o soft gate
        scale_mode="learned"
        gate_mode="fixed"
        fixed_gate=1.0
        lambda_mode="fixed"
        fixed_lambda=alpha
        use_rms_norm=True

    A2:
        v2 learnable-lambda, w/o soft gate
        scale_mode="learned"
        gate_mode="fixed"
        fixed_gate=1.0
        lambda_mode="learnable"
        use_rms_norm=True

    A3:
        v2 with soft gate, fixed-alpha
        scale_mode="learned"
        gate_mode="learned"
        lambda_mode="fixed"
        fixed_lambda=alpha
        use_rms_norm=True

    A5:
        Full RoMA-Net V2-lite
        scale_mode="learned"
        gate_mode="learned"
        lambda_mode="learnable"
        use_rms_norm=True

    A4 optional:
        Full w/o RMS
        scale_mode="learned"
        gate_mode="learned"
        lambda_mode="learnable"
        use_rms_norm=False
    """

    def __init__(
        self,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: List[str],
        gradient_checkpointing: bool,

        visual_adapter_hidden_dim: int = 4096,
        visual_adapter_r: int = 16,

        # A0 控制：是否挂载 visual residual adapter
        enable_visual_adapter: bool = True,

        # BioMedCLIP
        biomed_extractor: Optional[nn.Module] = None,
        use_cross_modal_prior: bool = True,

        # Router hidden dim
        router_hidden_dim: int = 128,

        # Scale routing
        scale_mode: str = "learned",                 # learned / fixed
        fixed_scale_weights: Optional[List[float]] = None,

        # Soft gate
        gate_mode: str = "learned",                  # learned / fixed
        fixed_gate: float = 1.0,
        gate_init: float = 0.5,

        # Residual scale
        lambda_mode: str = "learnable",              # learnable / fixed
        fixed_lambda: float = 0.1,
        lambda_max: float = 1.0,
        lambda_init: float = 0.1,

        # RMS residual normalization
        use_rms_norm: bool = True,
        residual_norm_eps: float = 1e-6,
        residual_norm_ratio_clip: Optional[float] = 10.0,
    ):
        # LoRA
        self.r = lora_r
        self.alpha = lora_alpha
        self.dropout = lora_dropout
        self.target_modules = lora_target_modules
        self.gradient_checkpointing = gradient_checkpointing

        # Visual adapter
        self.visual_adapter_hidden_dim = visual_adapter_hidden_dim
        self.visual_adapter_r = visual_adapter_r
        self.enable_visual_adapter = enable_visual_adapter

        # BioMedCLIP
        self.biomed_extractor = biomed_extractor
        self.use_cross_modal_prior = use_cross_modal_prior

        # Router
        self.router_hidden_dim = router_hidden_dim

        # Scale routing
        self.scale_mode = scale_mode
        self.fixed_scale_weights = fixed_scale_weights

        # Soft gate
        self.gate_mode = gate_mode
        self.fixed_gate = fixed_gate
        self.gate_init = gate_init

        # Lambda
        self.lambda_mode = lambda_mode
        self.fixed_lambda = fixed_lambda
        self.lambda_max = lambda_max
        self.lambda_init = lambda_init

        # RMS
        self.use_rms_norm = use_rms_norm
        self.residual_norm_eps = residual_norm_eps
        self.residual_norm_ratio_clip = residual_norm_ratio_clip

        self._validate_modes()

    def _validate_modes(self):
        if self.scale_mode not in {"learned", "fixed"}:
            raise ValueError(f"未知 scale_mode: {self.scale_mode}")

        if self.gate_mode not in {"learned", "fixed"}:
            raise ValueError(f"未知 gate_mode: {self.gate_mode}")

        if self.lambda_mode not in {"learnable", "fixed"}:
            raise ValueError(f"未知 lambda_mode: {self.lambda_mode}")

    def _patch_forward_to_drop_biomed_kwargs(self, peft_model: PreTrainedModel):
        """
        A0 LoRA-only 时也可能 trainer/collator 传入 biomed_image_tensors / biomed_text_tokens。
        原始 Qwen forward 不认识这些参数，所以这里要安全 pop 掉。
        """
        peft_model.original_forward = peft_model.forward

        def patched_model_forward(self, *args, **kwargs):
            kwargs.pop("biomed_image_tensors", None)
            kwargs.pop("biomed_text_tokens", None)
            return self.original_forward(*args, **kwargs)

        peft_model.forward = types.MethodType(patched_model_forward, peft_model)

    def wrap(self, model: PreTrainedModel) -> PreTrainedModel:
        # ===============================================================
        # 1. 准备 QLoRA / LoRA 训练环境
        # ===============================================================
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=self.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        if hasattr(model, "config"):
            model.config.use_cache = False

        peft_model = get_peft_model(
            model,
            LoraConfig(
                r=self.r,
                lora_alpha=self.alpha,
                target_modules=self.target_modules,
                lora_dropout=self.dropout,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )

        if hasattr(peft_model, "enable_input_require_grads"):
            peft_model.enable_input_require_grads()

        # ===============================================================
        # 2. A0: Qwen3-VL + LoRA only，不挂 visual adapter
        # ===============================================================
        if not self.enable_visual_adapter:
            self._patch_forward_to_drop_biomed_kwargs(peft_model)

            ddp_print("\n✅ [A0 / LoRA Only] 未挂载 visual residual adapter。")
            peft_model.print_trainable_parameters()
            return peft_model

        # ===============================================================
        # 3. V2-lite 需要 BioMedCLIP extractor
        # ===============================================================
        if self.biomed_extractor is None:
            raise ValueError(
                "V2-lite visual adapter 已启用，但 biomed_extractor=None。"
                "请在 Trainer 中传入 FrozenBioMedCLIPFeatureExtractor 实例。"
            )

        # ===============================================================
        # 4. 定位 Qwen3-VL 视觉塔
        # ===============================================================
        ddp_print(
            "\n✨ [RoMA-Net V2-lite-3S Conv2D] "
            "正在挂载三尺度 DWConv2D F3/F5/F7 "
            "(3x3 / 5x5 / 7x7) "
            "soft-gated visual residual adapter..."
        )
        ddp_print(
            f"    expert_type=DWConv2D, "
            f"expert_scales=3x3/5x5/7x7, "
            f"scale_mode={self.scale_mode}, "
            f"gate_mode={self.gate_mode}, "
            f"lambda_mode={self.lambda_mode}, "
            f"use_rms_norm={self.use_rms_norm}"
        )

        vision_tower = peft_model.base_model.model.model.visual

        hidden_dim = self.visual_adapter_hidden_dim
        r = self.visual_adapter_r

        ref_param = next(vision_tower.parameters())
        adapter_device = ref_param.device
        adapter_dtype = ref_param.dtype if ref_param.is_floating_point() else torch.bfloat16

        # ===============================================================
        # 5. 实例化三个 DWConv2D residual experts: F3/F5/F7
        # ===============================================================
        adapter_f3 = VisualAdapter_F3(
            hidden_dim=hidden_dim,
            r=r,
        )
        adapter_f5 = VisualAdapter_F5(
            hidden_dim=hidden_dim,
            r=r,
        )
        adapter_f7 = VisualAdapter_F7(
            hidden_dim=hidden_dim,
            r=r,
        )

        # ===============================================================
        # 6. 实例化统一的消融友好 fusion module
        # ===============================================================
        fusion_layer = Qwen3VLMoEVisualAdapterFusion(
            hidden_dim=hidden_dim,
            adapter_f3=adapter_f3,
            adapter_f5=adapter_f5,
            adapter_f7=adapter_f7,

            router_hidden_dim=self.router_hidden_dim,

            scale_mode=self.scale_mode,
            fixed_scale_weights=self.fixed_scale_weights,

            gate_mode=self.gate_mode,
            fixed_gate=self.fixed_gate,
            gate_init=self.gate_init,

            lambda_mode=self.lambda_mode,
            fixed_lambda=self.fixed_lambda,
            lambda_max=self.lambda_max,
            lambda_init=self.lambda_init,

            use_rms_norm=self.use_rms_norm,
            residual_norm_eps=self.residual_norm_eps,
            residual_norm_ratio_clip=(
                self.residual_norm_ratio_clip
            ),

            use_cross_modal_prior=(
                self.use_cross_modal_prior
            ),
        )

        # 挂到视觉塔上，使其被 PyTorch/PEFT 正确追踪
        vision_tower.res_adapter = fusion_layer.to(
            device=adapter_device,
            dtype=adapter_dtype,
        )

        # 挂载冻结 BioMedCLIP extractor
        peft_model.biomed_extractor = self.biomed_extractor.to(
            device=adapter_device,
            dtype=adapter_dtype,
        )
        peft_model.biomed_extractor.eval()

        for param in peft_model.biomed_extractor.parameters():
            param.requires_grad = False

        # ===============================================================
        # 7. 第一重 patch：外层 model.forward
        #    负责截获 biomed_image_tensors / biomed_text_tokens
        # ===============================================================
        peft_model.original_forward = peft_model.forward

        def patched_model_forward(self, *args, **kwargs):
            biomed_img = kwargs.pop("biomed_image_tensors", None)
            biomed_txt = kwargs.pop("biomed_text_tokens", None)

            visual = self.base_model.model.model.visual

            # 每次 forward 前先清空，避免上一个 batch 的 BioMedCLIP 特征残留
            visual.current_biomed_img_feat = None
            visual.current_biomed_txt_feat = None

            if biomed_img is not None and biomed_txt is not None:
                if not hasattr(self, "biomed_extractor"):
                    raise ValueError(
                        "当前 batch 传入了 BioMedCLIP 输入，但 peft_model 上没有 biomed_extractor。"
                    )

                img_f, txt_f = self.biomed_extractor(
                    biomed_img,
                    biomed_txt,
                )

                visual.current_biomed_img_feat = img_f
                visual.current_biomed_txt_feat = txt_f

            return self.original_forward(*args, **kwargs)

        peft_model.forward = types.MethodType(
            patched_model_forward,
            peft_model,
        )

        # ===============================================================
        # 8. 第二重 patch：视觉塔 forward
        #    负责把 visual tokens 送入 V2-lite residual adapter
        # ===============================================================
        vision_tower.original_forward = vision_tower.forward

        def patched_vision_forward(self, *args, **kwargs):
            outputs = self.original_forward(*args, **kwargs)

            img_f = getattr(self, "current_biomed_img_feat", None)
            txt_f = getattr(self, "current_biomed_txt_feat", None)

            grid_thw = kwargs.get("grid_thw", None)
            if grid_thw is None and len(args) > 1:
                grid_thw = args[1]

            if img_f is None or txt_f is None:
                raise ValueError(
                    "V2-lite visual adapter 已启用，但当前 forward 没有 BioMedCLIP 特征。"
                    "请确认 collator/trainer 已传入 biomed_image_tensors 和 biomed_text_tokens。"
                )

            if grid_thw is None:
                raise ValueError(
                    "V2-lite visual adapter 已启用，但 vision forward 中没有拿到 grid_thw。"
                )

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                outputs.pooler_output = self.res_adapter(
                    outputs.pooler_output,
                    biomed_img_feat=img_f,
                    biomed_txt_feat=txt_f,
                    grid_thw=grid_thw,
                )

            if hasattr(outputs, "deepstack_features") and outputs.deepstack_features is not None:
                outputs.deepstack_features = [
                    self.res_adapter(
                        x,
                        biomed_img_feat=img_f,
                        biomed_txt_feat=txt_f,
                        grid_thw=grid_thw,
                    )
                    for x in outputs.deepstack_features
                ]

            return outputs

        vision_tower.forward = types.MethodType(
            patched_vision_forward,
            vision_tower,
        )

        ddp_print(
            "✅ [RoMA-Net V2-lite Conv2D F3/F5/F7] "
            "visual residual adapter 挂载成功！"
        )
        peft_model.print_trainable_parameters()

        return peft_model
