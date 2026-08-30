from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .features import MPFConfig, MultiBandMatrixPowerFeatures
from .model import MPFTDSConfig, MPFTDSNetwork


class MPFTDSRuntime:
    def __init__(self, artifact: str | Path, bundle: object) -> None:
        self.bundle = bundle
        payload = torch.load(str(artifact), map_location="cpu", weights_only=False)
        revision = int(payload.get("implementation_revision", 0))
        feature_payload = dict(payload.get("feature_config", {}))
        if revision == 2:
            # Revision 2 artifacts predate the explicit mode field and were
            # trained with Meta's nan/negative-infinity-to-zero behavior.
            feature_payload["matrix_log_mode"] = "legacy_nan_to_zero"
        elif revision == 3:
            feature_payload.setdefault("matrix_log_mode", "clamp")
        else:
            raise ValueError(
                f"不支持的 MPF+TDS 实现版本 {revision}；请重新训练或升级运行时")
        self.feature_config = MPFConfig(**feature_payload)
        self.network_config = MPFTDSConfig(**payload.get("network_config", {}))
        if self.network_config.input_dim != self.feature_config.output_dim:
            raise ValueError(
                "模型特征维度不一致："
                f"MPF 输出 {self.feature_config.output_dim}，"
                f"TDS 输入 {self.network_config.input_dim}")
        if int(bundle.input_channels) != self.feature_config.channels:
            raise ValueError(
                "模型包通道数与 MPF 特征配置不一致："
                f"清单 {bundle.input_channels}，特征 {self.feature_config.channels}")
        self.features = MultiBandMatrixPowerFeatures(self.feature_config).eval()
        self.model = MPFTDSNetwork(self.network_config).eval()
        self.model.load_state_dict(payload["state_dict"], strict=True)
        torch.set_num_threads(max(1, min(8, __import__("os").cpu_count() or 1)))

    def predict(self, emg: np.ndarray) -> np.ndarray:
        signal = np.asarray(emg, dtype=np.float32)
        if signal.ndim != 2 or signal.shape[1] != self.feature_config.channels:
            raise ValueError(f"模型输入应为 [samples,{self.feature_config.channels}]")
        tensor = torch.from_numpy(np.ascontiguousarray(signal.T[None]))
        with torch.inference_mode():
            features = self.features(tensor)
            probabilities = torch.sigmoid(self.model(features))[0].transpose(0, 1).cpu().numpy()
        return probabilities.astype(np.float32, copy=False)
