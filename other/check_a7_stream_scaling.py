"""Tiny CPU correctness check; does not load Qwen3-VL, BioMedCLIP or data."""
import torch
import torch.nn as nn

from utils.qwen3vl.qwen3_vl_8B_visual_adapters_fusion import Qwen3VLMoEVisualAdapterFusion


class OnesExpert(nn.Module):
    def forward(self, x, grid_shape):
        return torch.ones_like(x)


def make_fusion(*, scales=None, use_rms=False):
    return Qwen3VLMoEVisualAdapterFusion(
        hidden_dim=4,
        adapter_f3=OnesExpert(),
        adapter_f5=OnesExpert(),
        adapter_f7=OnesExpert(),
        biomedclip_image_encoder_dim=2,
        biomedclip_text_encoder_dim=2,
        lambda_init=0.9,
        lambda_max=1.0,
        gate_init=0.5,
        use_rms_norm=use_rms,
        fixed_stream_scales=scales,
    ).eval()


def main():
    x = torch.full((4, 4), 2.0)
    f = torch.zeros(1, 2)
    grid = torch.tensor([[1, 4, 4]])
    scales = [1.0, 2.0, 3.0, 4.0]  # toy values ONLY, not experimental coefficients
    fixed = make_fusion(scales=scales)
    a6 = make_fusion(use_rms=True)
    with torch.no_grad():
        for i, c in enumerate(scales):
            actual = fixed(x, f, f, grid, stream_id=i)
            expected = x + 0.9 * 0.5 * c * torch.ones_like(x)
            assert torch.allclose(actual, expected, atol=1e-6), (i, actual, expected)
        rms_out = a6(x, f, f, grid, stream_id=3)
        # RMS_x/RMS_r ~= 2; residual = 1; injection ~= 0.9.
        assert torch.allclose(rms_out, x + 0.9 * torch.ones_like(x), atol=1e-5)
        assert not any('fixed_stream_scales' in key for key in fixed.state_dict())
        try:
            fixed(x, f, f, grid, stream_id=4)
        except ValueError:
            pass
        else:
            raise AssertionError('A7 should reject invalid stream ids')
    print('PASS: A7 stream ids, fixed multipliers, unchanged RMS path, state_dict compatibility')


if __name__ == '__main__':
    main()
