from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from emgforce.algorithms import META_CONV_LSTM, get_algorithm


@dataclass(frozen=True, slots=True)
class ModelBundle:
    root: Path
    model_id: str
    artifact: Path
    sha256: str
    labels: tuple[str, ...]
    display_names: dict[str, str]
    sample_rate: int
    input_channels: int
    output_channels: int
    metadata: dict[str, Any]
    preprocessing: dict[str, Any]
    algorithm_id: str = META_CONV_LSTM
    runtime_backend: str = "lightning_conv_lstm"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def load_model_bundle(root: Path, verify_hash: bool = True) -> ModelBundle:
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少模型清单：{manifest_path}")
    manifest = _read_json(manifest_path)
    artifact_meta = manifest.get("artifact", {})
    network = manifest.get("network", {})
    signal_meta = manifest.get("signal", {})
    artifact = root / str(artifact_meta.get("filename", "model.ts"))
    if not artifact.is_file():
        raise FileNotFoundError(f"缺少推理模型：{artifact}")
    expected_hash = str(artifact_meta.get("sha256", "")).lower()
    if verify_hash and expected_hash:
        actual_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"模型 SHA-256 不匹配：期望 {expected_hash}，实际 {actual_hash}")

    labels_meta = _read_json(root / str(manifest.get("labels_file", "labels.json")))
    preprocessing = _read_json(
        root / str(manifest.get("preprocessing_file", "preprocessing.json")))
    labels = tuple(str(item) for item in labels_meta.get("model_outputs", []))
    display_names = {
        str(key): str(value)
        for key, value in labels_meta.get("display_names", {}).items()
    }
    input_channels = int(signal_meta.get("input_channels", network.get("input_channels", 0)))
    output_channels = int(network.get("output_channels", 0))
    sample_rate = int(signal_meta.get("sample_rate_hz", network.get("sample_rate_hz", 0)))
    algorithm_id = str(manifest.get("algorithm_id", META_CONV_LSTM))
    spec = get_algorithm(algorithm_id)
    runtime_backend = str(manifest.get("runtime_backend", spec.runtime_backend))
    if input_channels <= 0 or output_channels != len(labels) or sample_rate <= 0:
        raise ValueError(
            f"模型清单与上位机不兼容：通道数({input_channels})和采样率({sample_rate})必须为正数，且输出数({output_channels})与标签数({len(labels)})一致")
    if input_channels != spec.input_channels or sample_rate != spec.sample_rate_hz:
        raise ValueError(
            f"{spec.display_name} 要求 {spec.input_channels} 通道/{spec.sample_rate_hz} Hz，"
            f"模型包为 {input_channels} 通道/{sample_rate} Hz")
    return ModelBundle(
        root=root,
        model_id=str(manifest.get("model_id", root.name)),
        artifact=artifact,
        sha256=expected_hash,
        labels=labels,
        display_names=display_names,
        sample_rate=sample_rate,
        input_channels=input_channels,
        output_channels=output_channels,
        metadata=manifest,
        preprocessing=preprocessing,
        algorithm_id=algorithm_id,
        runtime_backend=runtime_backend,
    )


def discover_model_bundles(models_root: Path) -> list[ModelBundle]:
    models_root = Path(models_root)
    if not models_root.exists():
        return []
    bundles: list[ModelBundle] = []
    for manifest in sorted(models_root.rglob("manifest.json")):
        try:
            bundle = load_model_bundle(manifest.parent, verify_hash=False)
            artifact_format = str(
                bundle.metadata.get("artifact", {}).get("format", "")).lower()
            if artifact_format in {"pytorch_lightning", "mpf_tds_state_dict"}:
                bundles.append(bundle)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return bundles


class TorchScriptGestureModel:
    """NumPy-facing wrapper around a TorchScript artifact."""

    def __init__(self, bundle: ModelBundle, verify_hash: bool = True) -> None:
        self.bundle = load_model_bundle(bundle.root, verify_hash=verify_hash)
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "未安装 PyTorch，无法加载实时识别模型；请更新环境") from exc
        self._torch = torch
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        self._model = torch.jit.load(str(self.bundle.artifact), map_location="cpu")
        self._model.eval()
        warmup = torch.zeros(1, self.bundle.input_channels, 4000, dtype=torch.float32)
        with torch.inference_mode():
            self._model(warmup)

    def predict(self, emg: np.ndarray) -> np.ndarray:
        """Return sigmoid probabilities as ``[outputs, timesteps]``."""
        signal = np.asarray(emg, dtype=np.float32)
        if signal.ndim != 2 or signal.shape[1] != self.bundle.input_channels:
            raise ValueError(
                f"模型输入应为 [samples,{self.bundle.input_channels}]，实际为 {signal.shape}")
        if len(signal) < 21:
            raise ValueError("模型输入至少需要 21 个采样点")
        tensor = self._torch.from_numpy(np.ascontiguousarray(signal.T[None, ...]))
        with self._torch.inference_mode():
            logits = self._model(tensor)
            probabilities = self._torch.sigmoid(logits)[0].cpu().numpy()
        if probabilities.shape[0] != self.bundle.output_channels:
            raise RuntimeError(f"模型输出形状异常：{probabilities.shape}")
        return probabilities.astype(np.float32, copy=False)


class LightningGestureModel:
    """NumPy-facing wrapper around a PyTorch Lightning Checkpoint (.ckpt)."""

    def __init__(self, bundle: ModelBundle, ckpt_path: Path, config_path: Path | None = None) -> None:
        self.bundle = bundle
        try:
            import torch
            from hydra.utils import get_class
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise RuntimeError("未安装 PyTorch/Hydra/OmegaConf，无法加载 .ckpt 权重文件") from exc

        self._torch = torch
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

        if config_path is None:
            config_path = bundle.root / str(bundle.metadata.get("training_config_file", "training_config.yaml"))
        if not config_path.is_file():
            raise FileNotFoundError(f"缺少模型配置文件：{config_path}")

        config = OmegaConf.load(str(config_path))
        if hasattr(config, "lightning_module"):
            module_config = config.lightning_module
        else:
            module_config = config

        target = str(module_config.get("_target_", ""))
        if not target:
            raise ValueError("训练配置 lightning_module 缺少 Hydra _target_")
        module_class = get_class(target)
        self._model = module_class.load_from_checkpoint(
            str(ckpt_path), map_location="cpu")
        self._model.eval()

        warmup = torch.zeros(1, self.bundle.input_channels, 4000, dtype=torch.float32)
        with torch.inference_mode():
            self._model(warmup)

    def predict(self, emg: np.ndarray) -> np.ndarray:
        """Return sigmoid probabilities as ``[outputs, timesteps]``."""
        signal = np.asarray(emg, dtype=np.float32)
        if signal.ndim != 2 or signal.shape[1] != self.bundle.input_channels:
            raise ValueError(
                f"模型输入应为 [samples,{self.bundle.input_channels}]，实际为 {signal.shape}")
        if len(signal) < 21:
            raise ValueError("模型输入至少需要 21 个采样点")
        tensor = self._torch.from_numpy(np.ascontiguousarray(signal.T[None, ...]))
        with self._torch.inference_mode():
            logits = self._model(tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
            probabilities = self._torch.sigmoid(logits)[0].cpu().numpy()
        if probabilities.shape[0] != self.bundle.output_channels:
            raise RuntimeError(f"模型输出形状异常：{probabilities.shape}")
        return probabilities.astype(np.float32, copy=False)


def create_gesture_model(bundle: ModelBundle, verify_hash: bool = True):
    """Create the runtime selected by the model bundle's algorithm/backend."""
    if verify_hash:
        bundle = load_model_bundle(bundle.root, verify_hash=True)
    artifact_meta = bundle.metadata.get("artifact", {})
    fmt = str(artifact_meta.get("format", "")).lower()
    if bundle.runtime_backend == "lightning_conv_lstm" and (
            fmt == "pytorch_lightning" or bundle.artifact.suffix == ".ckpt"):
        ckpt_path = bundle.artifact
        config_path = bundle.root / str(bundle.metadata.get("training_config_file", "training_config.yaml"))
        return LightningGestureModel(bundle, ckpt_path, config_path)
    if bundle.runtime_backend == "mpf_tds" and fmt == "mpf_tds_state_dict":
        from mpf_tds.runtime import MPFTDSRuntime
        return MPFTDSRuntime(bundle.artifact, bundle)
    raise ValueError(f"模型包后端与权重格式不匹配：{bundle.runtime_backend}/{fmt}")
