"""A7: static per-visual-stream residual multipliers.

Fill the four coefficients ONCE using training-only calibration.
A7 must not be trained with guessed values or test-set statistics.
Both training stages and both SLAKE evaluators read this same file.
"""
import math

# Order: pooled, deepstack_0, deepstack_1, deepstack_2.
# EXAMPLE of format ONLY: (1.0, 10.0, 10.0, 5.0)
# These example numbers are NOT measured or recommended.
# Before formal A7 runs, replace None with your four calibrated values.
A7_FIXED_STREAM_SCALES: tuple[float, float, float, float] | None = None


def get_a7_fixed_stream_scales() -> list[float]:
    values = A7_FIXED_STREAM_SCALES
    if values is None:
        raise ValueError(
            "A7 has no calibrated coefficients. Edit "
            "config/a7_settings.py: A7_FIXED_STREAM_SCALES = "
            "(c_pooled, c_deepstack_0, c_deepstack_1, c_deepstack_2). "
            "Use TRAIN data only; never choose these from the test set."
        )
    if len(values) != 4:
        raise ValueError(f"A7 needs exactly four coefficients, got {len(values)}")
    scales = [float(v) for v in values]
    if not all(math.isfinite(v) and 0.0 < v <= 10.0 for v in scales):
        raise ValueError(
            "A7 scales must be finite values in (0, 10], "
            "consistent with the original capped-RMS amplification bound."
        )
    return scales


def a7_experiment_label(scales: list[float]) -> str:
    """Stable, legible folder identifier; stages share one exact rule."""
    if len(scales) != 4:
        raise ValueError("A7 label needs four scales")
    tokens = [f"{float(v):.10g}".replace(".", "p") for v in scales]
    return "A7-DLR-StreamFixed-" + "-".join(tokens)
