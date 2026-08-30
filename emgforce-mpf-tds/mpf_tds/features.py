from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MPFConfig:
    sample_rate_hz: int = 200
    channels: int = 8
    window_samples: int = 20
    output_stride_samples: int = 4
    fft_samples: int = 16
    fft_stride_samples: int = 1
    frequency_bins: tuple[tuple[float, float], ...] = (
        (0.0, 12.5), (12.5, 25.0), (25.0, 37.5),
        (37.5, 50.0), (50.0, 75.0), (75.0, 100.0),
    )
    circular_diagonals: int = 4
    matrix_log_epsilon: float = 1e-6
    matrix_log_mode: str = "clamp"

    @property
    def left_context_samples(self) -> int:
        return self.window_samples - self.fft_stride_samples + self.fft_samples - 1

    @property
    def output_dim(self) -> int:
        return len(self.frequency_bins) * self.circular_diagonals * self.channels

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def for_sample_rate(cls, sample_rate_hz: int) -> "MPFConfig":
        if sample_rate_hz == 200:
            return cls()
        if sample_rate_hz == 2000:
            return cls(
                sample_rate_hz=2000,
                window_samples=200,
                output_stride_samples=40,
                fft_samples=64,
                fft_stride_samples=10,
                frequency_bins=(
                    (0.0, 62.5), (62.5, 125.0), (125.0, 250.0),
                    (250.0, 375.0), (375.0, 687.5), (687.5, 1000.0),
                ),
            )
        raise ValueError("只支持 200 Hz 或 2000 Hz MPF 预设")


class MultiBandMatrixPowerFeatures(nn.Module):
    """Sample-rate-aware 192-D multiband matrix-power features.

    Input is ``[batch, channels, samples]`` and output is
    ``[batch, frames, frequency_bands * diagonals * channels]``.
    The 2000 Hz preset is paper/Meta-equivalent; the default 200 Hz preset
    preserves a 50 Hz output rate while restricting bands below Nyquist.
    """

    def __init__(self, config: MPFConfig | None = None) -> None:
        super().__init__()
        self.config = config or MPFConfig()
        if self.config.matrix_log_mode not in {"clamp", "legacy_nan_to_zero"}:
            raise ValueError(
                "matrix_log_mode 必须是 clamp 或 legacy_nan_to_zero")
        if self.config.matrix_log_epsilon <= 0:
            raise ValueError("matrix_log_epsilon 必须为正数")
        hann = torch.hann_window(self.config.fft_samples, periodic=False)
        self.register_buffer("hann", hann, persistent=False)
        self.register_buffer("hann_norm", torch.linalg.vector_norm(hann), persistent=False)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1] != self.config.channels:
            raise ValueError(f"MPF 输入必须为 [batch,{self.config.channels},samples]")
        if signal.shape[-1] <= self.config.left_context_samples:
            raise ValueError(f"MPF 输入至少需要 {self.config.left_context_samples + 1} 个采样点")
        spectrum = torch.stft(
            signal.reshape(signal.shape[0] * signal.shape[1], -1),
            n_fft=self.config.fft_samples,
            hop_length=self.config.fft_stride_samples,
            window=self.hann.to(signal), center=False, normalized=False,
            onesided=True, return_complex=True,
        ) / self.hann_norm.to(signal)
        stft_frames = self.config.window_samples // self.config.fft_stride_samples
        stft_stride = self.config.output_stride_samples // self.config.fft_stride_samples
        spectrum = spectrum.unfold(-1, stft_frames, stft_stride)
        _, frequency_count, output_frames, window_frames = spectrum.shape
        spectrum = spectrum.reshape(
            signal.shape[0], signal.shape[1], frequency_count, output_frames, window_frames,
        ).permute(0, 3, 2, 1, 4)  # [B,T,F,C,W]
        cross_spectral_density = (
            spectrum @ spectrum.transpose(-2, -1).conj()
        ) / window_frames
        cross_spectral_density = cross_spectral_density.abs().square()
        frequencies = torch.fft.fftfreq(
            self.config.fft_samples, d=1.0 / self.config.sample_rate_hz,
            device=signal.device,
        )[:frequency_count].abs()
        features: list[torch.Tensor] = []
        for lower, upper in self.config.frequency_bins:
            mask = (frequencies > lower) & (frequencies <= upper)
            if not bool(mask.any()):
                raise ValueError(f"频带 [{lower},{upper}] 没有 FFT bin")
            matrix = cross_spectral_density[:, :, mask].mean(dim=2)
            eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
            if self.config.matrix_log_mode == "legacy_nan_to_zero":
                # Revision 2 reproduced Meta's public implementation exactly.
                # Keep it available only so existing trained weights retain the
                # feature distribution on which they were fitted.
                eigenvalues = eigenvalues.log().nan_to_num(nan=0.0, neginf=0.0)
            else:
                eigenvalues = eigenvalues.clamp_min(
                    self.config.matrix_log_epsilon).log()
            matrix_log = (eigenvectors * eigenvalues.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)
            diagonals = []
            channel_index = torch.arange(self.config.channels, device=signal.device)
            for offset in range(self.config.circular_diagonals):
                diagonals.append(matrix_log[..., channel_index, (channel_index + offset) % self.config.channels])
            features.append(torch.cat(diagonals, dim=-1))
        output = torch.cat(features, dim=-1)
        if not torch.isfinite(output).all():
            raise FloatingPointError("MPF 特征出现 NaN 或 Inf")
        return output
