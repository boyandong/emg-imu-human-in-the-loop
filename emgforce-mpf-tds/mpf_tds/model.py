from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class MPFTDSConfig:
    input_dim: int = 192
    feature_dim: int = 256
    output_channels: int = 4
    num_scales: int = 6
    blocks_per_scale: int = 2
    kernel_size: int = 3
    dropout: float = 0.1
    causal: bool = True
    implementation_revision: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


class TDSBlock(nn.Module):
    def __init__(self, dim: int, dilation: int, kernel_size: int = 3,
                 dropout: float = 0.1, causal: bool = True) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1) if causal else dilation * (kernel_size - 1) // 2
        self.right_padding = 0 if causal else dilation * (kernel_size - 1) - self.left_padding
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, dilation=dilation, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.depthwise(F.pad(x.transpose(1, 2), (self.left_padding, self.right_padding))).transpose(1, 2)
        x = self.norm1(x + F.leaky_relu(temporal, 0.1))
        return self.norm2(x + self.ffn(x))


class MPFTDSNetwork(nn.Module):
    """Six-scale TDS network reconstructed from the architecture in the paper."""

    def __init__(self, config: MPFTDSConfig | None = None) -> None:
        super().__init__()
        self.config = config or MPFTDSConfig()
        cfg = self.config
        self.input_projection = nn.Sequential(nn.Linear(cfg.input_dim, cfg.feature_dim), nn.LeakyReLU(0.1))
        if cfg.blocks_per_scale != 2:
            raise ValueError("论文结构要求每个尺度恰好两个 TDS block")
        self.scale_first = nn.ModuleList([
            TDSBlock(cfg.feature_dim, 2 ** scale, cfg.kernel_size, cfg.dropout, cfg.causal)
            for scale in range(cfg.num_scales)
        ])
        self.scale_second = nn.ModuleList([
            TDSBlock(cfg.feature_dim, 2 ** scale, cfg.kernel_size, cfg.dropout, cfg.causal)
            for scale in range(cfg.num_scales)
        ])
        self.output = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.feature_dim), nn.LeakyReLU(0.1),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.feature_dim, cfg.feature_dim),
            nn.LeakyReLU(0.1), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.feature_dim, cfg.output_channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError(f"TDS 输入必须为 [batch,time,{self.config.input_dim}]")
        base = self.input_projection(features)
        previous: torch.Tensor | None = None
        for scale, (first_block, second_block) in enumerate(zip(self.scale_first, self.scale_second)):
            factor = 2 ** scale
            if factor > 1:
                # A left-padded moving average implements the paper's 2^s
                # pooling scale without leaking future samples.
                pooled = F.avg_pool1d(
                    F.pad(base.transpose(1, 2), (factor - 1, 0)), factor, stride=1,
                ).transpose(1, 2)
            else:
                pooled = base
            current = first_block(pooled)
            if previous is not None:
                current = current + previous
            previous = second_block(current)
        assert previous is not None
        return self.output(previous)
