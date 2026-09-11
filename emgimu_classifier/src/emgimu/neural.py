from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .baseline import HeadPrediction
from .state import Direction, Gesture

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
        raise RuntimeError("install the 'neural' extra to use the causal network")


@dataclass(frozen=True, slots=True)
class DualBranchConfig:
    emg_channels: int = 8
    imu_channels: int = 6
    hidden_dim: int = 48
    embedding_dim: int = 64
    direction_classes: int = 7
    gesture_classes: int = 4
    dropout: float = 0.10
    modality_dropout: float = 0.20
    tcn_dilations: tuple[int, ...] = (1, 2, 4, 8)
    dual_emg_representation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


if nn is not None:
    class CausalConv1d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
            super().__init__()
            self.left_padding = dilation * (kernel_size - 1)
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.conv(F.pad(x, (self.left_padding, 0)))


    class ResidualTCNBlock(nn.Module):
        def __init__(self, channels: int, dilation: int, dropout: float) -> None:
            super().__init__()
            self.depthwise = CausalConv1d(channels, channels, 3, dilation)
            self.pointwise = nn.Conv1d(channels, channels, 1)
            self.norm = nn.GroupNorm(1, channels)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            update = self.pointwise(F.gelu(self.depthwise(x)))
            return self.norm(x + self.dropout(update))


    class CircularEMGEncoder(nn.Module):
        def __init__(
            self, hidden_dim: int, embedding_dim: int, dropout: float,
            dilations: tuple[int, ...],
            dual_representation: bool,
        ) -> None:
            super().__init__()
            self.spatial = nn.Conv2d(1, hidden_dim, kernel_size=(3, 5))
            self.shape_spatial = (
                nn.Conv2d(1, hidden_dim, kernel_size=(3, 5))
                if dual_representation else None
            )
            self.shape_gate_logit = (
                nn.Parameter(torch.tensor(-1.0)) if dual_representation else None
            )
            self.tcn = nn.Sequential(*[
                ResidualTCNBlock(hidden_dim, dilation, dropout) for dilation in dilations
            ])
            self.output = nn.Linear(hidden_dim, embedding_dim)

        def forward(self, emg: "torch.Tensor") -> "torch.Tensor":
            # [B,T,8] -> [B,1,8,T]. Time is left-padded; channels wrap as a ring.
            x = emg.transpose(1, 2).unsqueeze(1)
            x = F.pad(x, (4, 0, 0, 0))
            x = F.pad(x, (0, 0, 1, 1), mode="circular")
            x = F.gelu(self.spatial(x)).mean(dim=2)
            if self.shape_spatial is not None:
                scale = torch.sqrt(torch.mean(emg.square(), dim=(1, 2), keepdim=True) + 1e-6)
                shape = (emg / scale).transpose(1, 2).unsqueeze(1)
                shape = F.pad(shape, (4, 0, 0, 0))
                shape = F.pad(shape, (0, 0, 1, 1), mode="circular")
                shape = F.gelu(self.shape_spatial(shape)).mean(dim=2)
                x = x + torch.sigmoid(self.shape_gate_logit) * shape
            x = self.tcn(x)
            return self.output(x[:, :, -1])


    class IMUEncoder(nn.Module):
        def __init__(
            self, hidden_dim: int, embedding_dim: int, dropout: float,
            dilations: tuple[int, ...],
        ) -> None:
            super().__init__()
            self.input = nn.Conv1d(6, hidden_dim, 1)
            self.tcn = nn.Sequential(*[
                ResidualTCNBlock(hidden_dim, dilation, dropout) for dilation in dilations
            ])
            self.output = nn.Linear(hidden_dim, embedding_dim)

        def forward(self, imu: "torch.Tensor") -> "torch.Tensor":
            x = self.tcn(F.gelu(self.input(imu.transpose(1, 2))))
            return self.output(x[:, :, -1])


    class DualBranchCausalNet(nn.Module):
        """Physiology-informed late fusion with penalized cross-modal gates."""

        def __init__(self, config: DualBranchConfig | None = None) -> None:
            super().__init__()
            self.config = config or DualBranchConfig()
            cfg = self.config
            self.emg_encoder = CircularEMGEncoder(
                cfg.hidden_dim, cfg.embedding_dim, cfg.dropout, cfg.tcn_dilations,
                cfg.dual_emg_representation,
            )
            self.imu_encoder = IMUEncoder(
                cfg.hidden_dim, cfg.embedding_dim, cfg.dropout, cfg.tcn_dilations,
            )
            self.emg_to_direction = nn.Linear(cfg.embedding_dim, cfg.embedding_dim)
            self.imu_to_gesture = nn.Linear(cfg.embedding_dim, cfg.embedding_dim)
            # Sigmoid(-2) starts near 0.12: the non-primary modality begins as weak evidence.
            self.direction_gate_logit = nn.Parameter(torch.tensor(-2.0))
            self.gesture_gate_logit = nn.Parameter(torch.tensor(-2.0))
            self.direction_head = nn.Linear(cfg.embedding_dim, cfg.direction_classes)
            self.gesture_head = nn.Linear(cfg.embedding_dim, cfg.gesture_classes)
            self.direction_aux = nn.Linear(cfg.embedding_dim, cfg.direction_classes)
            self.gesture_aux = nn.Linear(cfg.embedding_dim, cfg.gesture_classes)

        def _drop_modality(self, value: "torch.Tensor") -> "torch.Tensor":
            if not self.training or self.config.modality_dropout <= 0:
                return value
            keep = torch.rand((len(value), 1), device=value.device) >= self.config.modality_dropout
            return value * keep / (1.0 - self.config.modality_dropout)

        def forward(self, emg: "torch.Tensor", imu: "torch.Tensor") -> dict[str, "torch.Tensor"]:
            if emg.ndim != 3 or emg.shape[-1] != 8:
                raise ValueError("emg input must be [batch,time,8]")
            if imu.ndim != 3 or imu.shape[-1] != 6:
                raise ValueError("imu input must be [batch,time,6]")
            emg_embedding = self.emg_encoder(emg)
            imu_embedding = self.imu_encoder(imu)
            cross_emg = self._drop_modality(emg_embedding)
            cross_imu = self._drop_modality(imu_embedding)
            direction_gate = torch.sigmoid(self.direction_gate_logit)
            gesture_gate = torch.sigmoid(self.gesture_gate_logit)
            direction_embedding = imu_embedding + direction_gate * self.emg_to_direction(cross_emg)
            gesture_embedding = emg_embedding + gesture_gate * self.imu_to_gesture(cross_imu)
            return {
                "direction": self.direction_head(direction_embedding),
                "gesture": self.gesture_head(gesture_embedding),
                "direction_aux": self.direction_aux(imu_embedding),
                "gesture_aux": self.gesture_aux(emg_embedding),
                "gate_penalty": direction_gate.square() + gesture_gate.square(),
            }


    def dual_branch_loss(
        output: dict[str, "torch.Tensor"],
        direction: "torch.Tensor",
        gesture: "torch.Tensor",
        *,
        direction_weight: "torch.Tensor | None" = None,
        gesture_weight: "torch.Tensor | None" = None,
        sample_weight: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        per_sample = F.cross_entropy(
            output["direction"], direction, weight=direction_weight, reduction="none",
        )
        per_sample = per_sample + F.cross_entropy(
            output["gesture"], gesture, weight=gesture_weight, reduction="none",
        )
        auxiliary = F.cross_entropy(
            output["direction_aux"], direction, weight=direction_weight, reduction="none",
        )
        auxiliary = auxiliary + F.cross_entropy(
            output["gesture_aux"], gesture, weight=gesture_weight, reduction="none",
        )
        combined = per_sample + 0.2 * auxiliary
        primary = combined.mean() if sample_weight is None else (combined * sample_weight).mean()
        return primary + 0.01 * output["gate_penalty"]


    class NeuralPredictor:
        def __init__(
            self,
            model: "DualBranchCausalNet",
            *,
            direction_temperature: float = 1.0,
            gesture_temperature: float = 1.0,
            direction_threshold: float = 0.0,
            gesture_threshold: float = 0.0,
            auxiliary_disagreement_threshold: float = 0.75,
            metadata: dict[str, Any] | None = None,
            device: str = "cpu",
        ) -> None:
            self.model = model.to(device).eval()
            self.device = device
            self.direction_temperature = float(direction_temperature)
            self.gesture_temperature = float(gesture_temperature)
            self.direction_threshold = float(direction_threshold)
            self.gesture_threshold = float(gesture_threshold)
            self.auxiliary_disagreement_threshold = float(auxiliary_disagreement_threshold)
            self.metadata = dict(metadata or {})

        @torch.inference_mode()
        def predict(self, normalized_emg: np.ndarray, normalized_imu: np.ndarray) -> HeadPrediction:
            emg = torch.as_tensor(normalized_emg, dtype=torch.float32, device=self.device)[None]
            imu = torch.as_tensor(normalized_imu, dtype=torch.float32, device=self.device)[None]
            output = self.model(emg, imu)
            d_prob = torch.softmax(output["direction"] / self.direction_temperature, dim=-1)[0]
            h_prob = torch.softmax(output["gesture"] / self.gesture_temperature, dim=-1)[0]
            d_index = int(d_prob.argmax())
            h_index = int(h_prob.argmax())
            q_d = float(d_prob[d_index])
            q_h = float(h_prob[h_index])
            d_margin = q_d - float(torch.topk(d_prob, min(2, len(d_prob))).values[-1])
            h_margin = q_h - float(torch.topk(h_prob, min(2, len(h_prob))).values[-1])
            d_aux = torch.softmax(output["direction_aux"], dim=-1)[0]
            h_aux = torch.softmax(output["gesture_aux"], dim=-1)[0]
            d_disagrees = int(d_aux.argmax()) != d_index and float(d_aux.max()) >= self.auxiliary_disagreement_threshold
            h_disagrees = int(h_aux.argmax()) != h_index and float(h_aux.max()) >= self.auxiliary_disagreement_threshold
            direction = (
                Direction(d_index)
                if q_d >= self.direction_threshold and not d_disagrees else Direction.UNKNOWN
            )
            gesture = (
                Gesture(h_index)
                if q_h >= self.gesture_threshold and not h_disagrees else Gesture.UNKNOWN
            )
            return HeadPrediction(
                direction=direction, gesture=gesture, q_direction=q_d, q_gesture=q_h,
                direction_margin=d_margin, gesture_margin=h_margin,
                direction_confidence_rejected=q_d < self.direction_threshold,
                gesture_confidence_rejected=q_h < self.gesture_threshold,
                direction_disagreement_rejected=d_disagrees,
                gesture_disagreement_rejected=h_disagrees,
            )

else:  # pragma: no cover - lightweight placeholders keep non-neural CLI usable
    class DualBranchCausalNet:
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()

    class NeuralPredictor:
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()

    def dual_branch_loss(*_: Any, **__: Any) -> Any:
        _require_torch()


def augment_training_batch(
    emg: "torch.Tensor",
    imu: "torch.Tensor",
    *,
    generator: "torch.Generator | None" = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    _require_torch()
    if emg.ndim != 3 or imu.ndim != 3:
        raise ValueError("training batches must be rank-3 tensors")
    result_emg = emg.clone()
    result_imu = imu.clone()
    batch = len(emg)
    shifts = torch.randint(0, 8, (batch,), generator=generator, device=emg.device)
    for index, shift in enumerate(shifts.tolist()):
        result_emg[index] = torch.roll(result_emg[index], shift, dims=-1)
    amplitude = 0.8 + 0.4 * torch.rand((batch, 1, 1), generator=generator, device=emg.device)
    result_emg *= amplitude
    drop = torch.rand((batch,), generator=generator, device=emg.device) < 0.25
    channels = torch.randint(0, 8, (batch,), generator=generator, device=emg.device)
    for index in torch.nonzero(drop, as_tuple=False).flatten().tolist():
        result_emg[index, :, channels[index]] = 0.0
    result_emg += torch.randn(result_emg.shape, generator=generator, device=emg.device) * 0.01
    imu_bias = (torch.rand((batch, 1, 6), generator=generator, device=imu.device) - 0.5) * 0.04
    result_imu += imu_bias
    return result_emg, result_imu
