from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None
    nn = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("install the 'neural' extra to use HLA neural representations")


@dataclass(frozen=True, slots=True)
class HLAEncoderConfig:
    representation: str = "R2"
    stream_rates_hz: tuple[int, ...] = (200, 500, 1000)
    metadata_dim: int = 11
    feature_dim: int | None = None
    temporal_hidden_dim: int = 24
    stream_embedding_dim: int = 24
    token_dim: int = 64
    representation_dim: int = 96
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.representation not in {"R0", "R1-core", "R2", "R3", "R2-wide"}:
            raise ValueError("representation must be R0, R1-core, R2, R3, or R2-wide")
        rates = tuple(map(int, self.stream_rates_hz))
        if rates != tuple(sorted(set(rates))) or any(rate <= 0 for rate in rates):
            raise ValueError("stream rates must be unique, sorted, and positive")
        object.__setattr__(self, "stream_rates_hz", rates)
        expected_feature_dim = 6 if self.representation == "R0" else 11
        feature_dim = expected_feature_dim if self.feature_dim is None else int(self.feature_dim)
        if self.representation in {"R0", "R1-core", "R3"} and feature_dim != expected_feature_dim:
            raise ValueError(
                f"{self.representation} requires {expected_feature_dim} feature values per channel"
            )
        object.__setattr__(self, "feature_dim", feature_dim)
        dimensions = (
            self.metadata_dim, feature_dim, self.temporal_hidden_dim,
            self.stream_embedding_dim, self.token_dim, self.representation_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all HLA encoder dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0,1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


if nn is not None:
    class _ChannelTemporalEncoder(nn.Module):
        """One temporal encoder shared by every channel within a rate stream."""

        def __init__(self, hidden_dim: int, output_dim: int, dropout: float) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv1d(1, hidden_dim, kernel_size=9, stride=2, padding=4),
                nn.GroupNorm(1, hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, stride=2, padding=3),
                nn.GroupNorm(1, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.output = nn.Linear(hidden_dim * 2, output_dim)

        def forward(self, signal: "torch.Tensor") -> "torch.Tensor":
            if signal.ndim != 3:
                raise ValueError("each signal stream must have shape [batch,channels,time]")
            batch, channels, samples = signal.shape
            if samples < 3:
                raise ValueError("signal streams require at least three samples")
            encoded = self.network(signal.reshape(batch * channels, 1, samples))
            pooled = torch.cat((encoded.mean(dim=-1), encoded.amax(dim=-1)), dim=-1)
            return self.output(pooled).reshape(batch, channels, -1)


    class _MaskedSetPool(nn.Module):
        def __init__(self, input_dim: int, token_dim: int, output_dim: int, dropout: float) -> None:
            super().__init__()
            self.token = nn.Sequential(
                nn.Linear(input_dim, token_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(token_dim, token_dim), nn.GELU(),
            )
            self.attention = nn.Linear(token_dim, 1)
            self.output = nn.Sequential(
                nn.Linear(token_dim * 2, output_dim), nn.GELU(), nn.Dropout(dropout),
            )

        def forward(self, values: "torch.Tensor", channel_mask: "torch.Tensor") -> "torch.Tensor":
            if values.ndim != 3 or channel_mask.shape != values.shape[:2]:
                raise ValueError("values and channel_mask must be [batch,channels,...] and [batch,channels]")
            mask = channel_mask.to(dtype=torch.bool)
            if not torch.all(mask.any(dim=1)):
                raise ValueError("every item needs at least one usable channel")
            tokens = self.token(values)
            logits = self.attention(tokens).squeeze(-1).masked_fill(~mask, float("-inf"))
            weights = torch.softmax(logits, dim=1)
            attentive = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
            masked = tokens.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            maximum = masked.amax(dim=1)
            return self.output(torch.cat((attentive, maximum), dim=-1))


    class HLAHardwareFlexibleEncoder(nn.Module):
        """Permutation-invariant variable-channel encoder for HLA R0/R1/R2/R3."""

        def __init__(self, config: HLAEncoderConfig | None = None) -> None:
            super().__init__()
            self.config = config or HLAEncoderConfig()
            cfg = self.config
            self.uses_raw = cfg.representation in {"R2", "R3", "R2-wide"}
            self.uses_features = cfg.representation in {"R0", "R1-core", "R3"}
            self.stream_encoders = nn.ModuleDict({
                str(rate): _ChannelTemporalEncoder(
                    cfg.temporal_hidden_dim, cfg.stream_embedding_dim, cfg.dropout,
                )
                for rate in cfg.stream_rates_hz
            }) if self.uses_raw else nn.ModuleDict()
            raw_input = len(cfg.stream_rates_hz) * (cfg.stream_embedding_dim + 1)
            raw_token_dim = cfg.token_dim * (2 if cfg.representation == "R2-wide" else 1)
            self.raw_pool = (
                _MaskedSetPool(
                    raw_input + cfg.metadata_dim + 1,
                    raw_token_dim,
                    cfg.representation_dim,
                    cfg.dropout,
                ) if self.uses_raw else None
            )
            self.feature_pool = (
                _MaskedSetPool(
                    cfg.feature_dim + cfg.metadata_dim + 1,
                    cfg.token_dim,
                    cfg.representation_dim,
                    cfg.dropout,
                ) if self.uses_features else None
            )
            fusion_input = cfg.representation_dim * (int(self.uses_raw) + int(self.uses_features))
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input, cfg.representation_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )

        def _validate_common(
            self,
            metadata: "torch.Tensor",
            quality: "torch.Tensor",
            channel_mask: "torch.Tensor",
        ) -> tuple[int, int]:
            if metadata.ndim != 3 or metadata.shape[-1] != self.config.metadata_dim:
                raise ValueError("metadata has the wrong shape or feature dimension")
            batch, channels = metadata.shape[:2]
            if quality.shape == (batch, channels):
                quality = quality.unsqueeze(-1)
            if quality.shape != (batch, channels, 1):
                raise ValueError("quality must be [batch,channels] or [batch,channels,1]")
            if channel_mask.shape != (batch, channels):
                raise ValueError("channel_mask must be [batch,channels]")
            return batch, channels

        def forward(
            self,
            *,
            streams: Mapping[int, "torch.Tensor"] | None,
            stream_availability: "torch.Tensor | None",
            metadata: "torch.Tensor",
            quality: "torch.Tensor",
            channel_mask: "torch.Tensor",
            feature_tokens: "torch.Tensor | None" = None,
        ) -> "torch.Tensor":
            batch, channels = self._validate_common(metadata, quality, channel_mask)
            q = quality.unsqueeze(-1) if quality.ndim == 2 else quality
            branches: list[torch.Tensor] = []
            if self.uses_raw:
                if streams is None or stream_availability is None:
                    raise ValueError("raw representations require streams and availability")
                if stream_availability.shape != (batch, len(self.config.stream_rates_hz)):
                    raise ValueError("stream_availability has the wrong shape")
                stream_tokens: list[torch.Tensor] = []
                for index, rate in enumerate(self.config.stream_rates_hz):
                    available = stream_availability[:, index].to(metadata.dtype)
                    if rate in streams:
                        signal = streams[rate]
                        if signal.shape[:2] != (batch, channels):
                            raise ValueError(f"{rate} Hz stream has the wrong batch/channel shape")
                        encoded = self.stream_encoders[str(rate)](signal)
                    else:
                        if torch.any(available > 0):
                            raise ValueError(f"{rate} Hz is marked available but the stream is absent")
                        encoded = metadata.new_zeros((batch, channels, self.config.stream_embedding_dim))
                    encoded = encoded * available[:, None, None]
                    stream_tokens.extend((encoded, available[:, None, None].expand(-1, channels, 1)))
                raw_values = torch.cat((*stream_tokens, metadata, q), dim=-1)
                branches.append(self.raw_pool(raw_values, channel_mask))
            if self.uses_features:
                if feature_tokens is None or feature_tokens.shape != (
                    batch, channels, self.config.feature_dim,
                ):
                    raise ValueError("feature representation requires correctly shaped feature_tokens")
                feature_values = torch.cat((feature_tokens, metadata, q), dim=-1)
                branches.append(self.feature_pool(feature_values, channel_mask))
            result = self.fusion(torch.cat(branches, dim=-1))
            if not torch.isfinite(result).all():
                raise ValueError("HLA encoder produced NaN or Inf")
            return result


    class HLAMultiDatasetModel(nn.Module):
        """Shared HLA encoder with explicit dataset-specific classification heads."""

        def __init__(
            self,
            dataset_classes: Mapping[str, int],
            config: HLAEncoderConfig | None = None,
        ) -> None:
            super().__init__()
            if not dataset_classes or any(count < 2 for count in dataset_classes.values()):
                raise ValueError("each dataset head requires at least two classes")
            self.encoder = HLAHardwareFlexibleEncoder(config)
            self.heads = nn.ModuleDict({
                str(dataset): nn.Linear(self.encoder.config.representation_dim, int(count))
                for dataset, count in dataset_classes.items()
            })

        def forward(self, dataset_id: str, **inputs: "torch.Tensor") -> dict[str, "torch.Tensor"]:
            if dataset_id not in self.heads:
                raise ValueError(f"unknown dataset head {dataset_id!r}")
            representation = self.encoder(**inputs)
            return {"representation": representation, "logits": self.heads[dataset_id](representation)}


    def hla_parameter_count(model: nn.Module) -> int:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

else:  # pragma: no cover - keep non-neural installs usable
    class HLAHardwareFlexibleEncoder:
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


    class HLAMultiDatasetModel:
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


    def hla_parameter_count(*_: Any, **__: Any) -> int:
        _require_torch()
