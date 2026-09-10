"""EMG 实时显示滤波器。

本模块中的滤波器均采用“有状态、逐样本”方式运行，适合直接处理串口连续
到达的数据。滤波只用于界面显示，原始数据的保存逻辑不在本模块中。
"""

from __future__ import annotations

import math
from collections.abc import Sequence


class RunningMeanRemoval:
    """使用指数滑动均值估计并去除单通道直流分量。"""

    def __init__(self, sample_rate: float, time_constant_s: float = 1.0) -> None:
        # 将时间常数转换成逐样本更新系数。系数越小，均值变化越平缓。
        self.gain = 1.0 - math.exp(-1.0 / (sample_rate * time_constant_s))
        self.reset()

    def reset(self) -> None:
        # 每个通道拥有独立的均值状态，首次输入时用该样本初始化。
        self.mean = 0.0
        self.initialized = False

    def process(self, value: float) -> float:
        x = float(value)
        if not self.initialized:
            self.mean = x
            self.initialized = True
        else:
            # mean[n] = mean[n-1] + gain * (x[n] - mean[n-1])
            self.mean += self.gain * (x - self.mean)
        return x - self.mean


class Biquad:
    """归一化二阶 IIR（Biquad）滤波器的直接形式实现。"""

    def __init__(
        self, b0: float, b1: float, b2: float, a1: float, a2: float
    ) -> None:
        # 系数已除以 a0，因此运行时只需保存 b0~b2、a1~a2。
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.a1 = a1
        self.a2 = a2
        self.reset()

    def reset(self) -> None:
        # x1/x2 为前两个输入；y1/y2 为前两个输出。
        self.x1 = self.x2 = 0.0
        self.y1 = self.y2 = 0.0

    def process(self, value: float) -> float:
        x = float(value)
        # y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2]
        #        - a1*y[n-1] - a2*y[n-2]
        y = (
            self.b0 * x
            + self.b1 * self.x1
            + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2
        )
        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return y


class FirstOrderLowPass:
    """一阶双线性变换低通，用于产生较柔和的高频滚降。"""

    def __init__(self, sample_rate: float, cutoff_hz: float) -> None:
        _validate_frequency(sample_rate, cutoff_hz)
        # 对模拟一阶低通进行预畸变和双线性变换。
        k = math.tan(math.pi * cutoff_hz / sample_rate)
        norm = 1.0 / (1.0 + k)
        self.b0 = k * norm
        self.b1 = self.b0
        self.a1 = (k - 1.0) * norm
        self.reset()

    def reset(self) -> None:
        self.x1 = 0.0
        self.y1 = 0.0

    def process(self, value: float) -> float:
        x = float(value)
        y = self.b0 * x + self.b1 * self.x1 - self.a1 * self.y1
        self.x1 = x
        self.y1 = y
        return y


def _validate_frequency(sample_rate: float, frequency: float) -> None:
    """确保截止频率位于直流与奈奎斯特频率之间。"""
    if not 0 < frequency < sample_rate / 2:
        raise ValueError("filter frequency must be between 0 and Nyquist")


def butterworth_highpass(sample_rate: float, cutoff_hz: float) -> Biquad:
    """设计二阶 Butterworth 高通滤波器。"""
    _validate_frequency(sample_rate, cutoff_hz)
    omega = 2.0 * math.pi * cutoff_hz / sample_rate
    cosine = math.cos(omega)
    alpha = math.sin(omega) / math.sqrt(2.0)
    a0 = 1.0 + alpha
    return Biquad(
        ((1.0 + cosine) / 2.0) / a0,
        (-(1.0 + cosine)) / a0,
        ((1.0 + cosine) / 2.0) / a0,
        (-2.0 * cosine) / a0,
        (1.0 - alpha) / a0,
    )


def butterworth_lowpass(sample_rate: float, cutoff_hz: float) -> Biquad:
    """设计二阶 Butterworth 低通滤波器。"""
    _validate_frequency(sample_rate, cutoff_hz)
    omega = 2.0 * math.pi * cutoff_hz / sample_rate
    cosine = math.cos(omega)
    alpha = math.sin(omega) / math.sqrt(2.0)
    a0 = 1.0 + alpha
    return Biquad(
        ((1.0 - cosine) / 2.0) / a0,
        (1.0 - cosine) / a0,
        ((1.0 - cosine) / 2.0) / a0,
        (-2.0 * cosine) / a0,
        (1.0 - alpha) / a0,
    )


def notch(sample_rate: float, frequency_hz: float, q: float = 35.0) -> Biquad:
    """设计窄带陷波器；Q 越高，被抑制的频带越窄。"""
    _validate_frequency(sample_rate, frequency_hz)
    omega = 2.0 * math.pi * frequency_hz / sample_rate
    cosine = math.cos(omega)
    alpha = math.sin(omega) / (2.0 * q)
    a0 = 1.0 + alpha
    return Biquad(
        1.0 / a0,
        (-2.0 * cosine) / a0,
        1.0 / a0,
        (-2.0 * cosine) / a0,
        (1.0 - alpha) / a0,
    )


class EmgDisplayFilterBank:
    """多通道 EMG 实时显示预处理滤波器组。

    每个通道都有互不共享的滤波状态，处理顺序为：
    动态均值消除 -> 20 Hz 高通 -> 90 Hz 低通。

    原始数据始终不滤波保存。50 Hz 陷波不默认启用，只有质量检查确认
    存在明显市电污染时才应开启，避免无条件破坏有效频谱。
    """

    def __init__(self, channels: int = 8, sample_rate: float = 250.0) -> None:
        self.channels = channels
        self.sample_rate = sample_rate
        # 每个通道分别创建一套滤波器，避免通道间的历史状态互相影响。
        self._pipelines = [self._new_pipeline() for _ in range(channels)]

    def _new_pipeline(self):
        """创建单个通道的完整串联滤波链。"""
        return (
            # 1. 去除缓慢变化的基线和直流偏置。
            RunningMeanRemoval(self.sample_rate, time_constant_s=1.0),
            # 2. 构成适合 250 Hz 采样（Nyquist 125 Hz）的 20~90 Hz 带通。
            butterworth_highpass(self.sample_rate, 20.0),
            butterworth_lowpass(self.sample_rate, 90.0),
        )

    def reset(self) -> None:
        """清除所有通道、所有滤波级的历史状态。"""
        for pipeline in self._pipelines:
            for stage in pipeline:
                stage.reset()

    def process(self, values: Sequence[float]) -> tuple[float, ...]:
        """处理一个采样时刻的全部通道，并返回对应的滤波结果。"""
        if len(values) != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {len(values)}")
        filtered: list[float] = []
        for value, pipeline in zip(values, self._pipelines):
            output = float(value)
            # 当前通道样本依次通过该通道的所有滤波级。
            for stage in pipeline:
                output = stage.process(output)
            filtered.append(output)
        return tuple(filtered)
