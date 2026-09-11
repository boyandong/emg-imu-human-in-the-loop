from __future__ import annotations

import numpy as np

from emgforce.processing.training_preprocessing import (
    TARGET_MEDIAN_ABS, filter_training_emg,
)


def estimate_realtime_scale(raw: np.ndarray, sample_rate: float) -> float:
    """Estimate the fixed per-donning scale from a guided calibration block."""
    filtered, _ = filter_training_emg(raw, sample_rate)
    reference = float(np.median(np.abs(filtered)))
    if not np.isfinite(reference) or reference <= np.finfo(np.float32).eps:
        raise ValueError("校准信号幅值过低，无法计算推理缩放系数")
    return reference / TARGET_MEDIAN_ABS


def fixed_lag_model_window(
    raw: np.ndarray,
    sample_rate: float,
    scale_counts_per_unit: float,
    model_window_samples: int,
    lag_samples: int,
    expected_channels: int | None = None,
) -> np.ndarray:
    """Filter a rolling buffer and return a stable window before its right edge.

    The training chain is zero phase, so live inference deliberately reports a
    fixed-lag estimate instead of pretending the newest sample can be filtered
    without future context.
    """
    signal = np.asarray(raw)
    required = int(model_window_samples) + int(lag_samples)
    if signal.ndim != 2 or signal.shape[1] < 1:
        raise ValueError(f"实时推理输入必须是二维数组 [samples, channels]，实际为 {signal.shape}")
    channels = signal.shape[1]
    if expected_channels is not None and channels != expected_channels:
        raise ValueError(f"实时推理输入通道数不匹配：期望 {expected_channels}，实际为 {channels}")
    if len(signal) < required:
        raise ValueError(f"实时缓冲不足：至少需要 {required} 点，实际 {len(signal)} 点")
    if not np.isfinite(scale_counts_per_unit) or scale_counts_per_unit <= 0:
        raise ValueError("实时推理缩放系数无效")
    filtered, _ = filter_training_emg(signal, sample_rate)
    end = len(filtered) - int(lag_samples)
    start = end - int(model_window_samples)
    output = (filtered[start:end] / float(scale_counts_per_unit)).astype(np.float32)
    if output.shape != (model_window_samples, channels) or not np.isfinite(output).all():
        raise RuntimeError(f"实时预处理输出异常：{output.shape}")
    return output

