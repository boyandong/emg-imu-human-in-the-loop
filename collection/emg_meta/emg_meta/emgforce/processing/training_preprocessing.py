from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


PREPROCESSING_VERSION = "meta_8ch_v1"
HIGH_PASS_HZ = 40.0
HIGH_PASS_ORDER = 4
NOTCH_Q = 30.0
TARGET_MEDIAN_ABS = 16.0


@dataclass(frozen=True, slots=True)
class TrainingPreprocessing:
    signal: np.ndarray
    scale_counts_per_unit: float
    notch_frequencies_hz: tuple[float, ...]


def notch_frequencies(sample_rate: float) -> tuple[float, ...]:
    """Return 50 Hz mains harmonics below Nyquist, capped at 450 Hz."""
    maximum = min(450.0, sample_rate / 2.0 - 1.0)
    return tuple(float(value) for value in np.arange(50.0, maximum + 0.1, 50.0))


def filter_training_emg(
    raw: np.ndarray, sample_rate: float
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Apply the unscaled, zero-phase Meta training filter chain."""
    signal = np.asarray(raw, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] < 1:
        raise ValueError(f"EMG 训练输入必须是二维数组，实际为 {signal.shape}")
    if len(signal) < 32:
        raise ValueError("EMG 数据过短，无法执行零相位训练滤波")
    if sample_rate <= 2.0 * HIGH_PASS_HZ:
        raise ValueError("EMG 采样率过低，无法执行 40 Hz 高通滤波")

    signal = signal - np.median(signal, axis=0, keepdims=True)
    high_pass = butter(
        HIGH_PASS_ORDER, HIGH_PASS_HZ, btype="highpass",
        fs=sample_rate, output="sos",
    )
    signal = sosfiltfilt(high_pass, signal, axis=0)

    notches = notch_frequencies(sample_rate)
    for frequency in notches:
        b, a = iirnotch(frequency, NOTCH_Q, fs=sample_rate)
        signal = filtfilt(b, a, signal, axis=0)
    if not np.isfinite(signal).all():
        raise ValueError("EMG 训练滤波产生了非有限数值")
    return signal, notches


def preprocess_training_emg(raw: np.ndarray, sample_rate: float) -> TrainingPreprocessing:
    """Create deterministic Meta-training input while leaving raw data untouched.

    Device packets contain signed 24-bit ADC counts, not calibrated microvolts.
    After zero-phase filtering, a single session-global robust scale maps the
    median absolute carrier amplitude to ``TARGET_MEDIAN_ABS``.  The scale is
    persisted in the export so the operation is explicit and reproducible.
    """
    signal, notches = filter_training_emg(raw, sample_rate)

    reference = float(np.median(np.abs(signal)))
    if not np.isfinite(reference) or reference <= np.finfo(np.float32).eps:
        raise ValueError("EMG 训练信号没有可用幅值，无法计算稳定缩放")
    scale = reference / TARGET_MEDIAN_ABS
    output = (signal / scale).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("EMG 训练预处理产生了非有限数值")
    return TrainingPreprocessing(output, scale, notches)
