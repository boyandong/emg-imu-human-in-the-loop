from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from .features import MPFConfig, MultiBandMatrixPowerFeatures


LABELS = ("index_press", "index_release", "middle_press", "middle_release",
          "thumb_click", "thumb_down", "thumb_in", "thumb_out", "thumb_up")
LABEL_SHIFT_SEC = 0.100
LABEL_HALF_WIDTH_SEC = 0.020
EXPECTED_PREPROCESSING_VERSION = "meta_8ch_v1"


def _validate_training_dataset(
    dataset: h5py.Dataset, path: Path, config: MPFConfig,
) -> None:
    fields = dataset.dtype.fields or {}
    if "emg" not in fields or "time" not in fields:
        raise ValueError(f"训练文件 /data 缺少 emg 或 time 字段：{path}")
    emg_shape = fields["emg"][0].shape
    channels = int(emg_shape[0]) if emg_shape else 0
    sample_rate = float(dataset.attrs.get("sample_rate", 0.0))
    version = str(dataset.attrs.get("preprocessing_version", ""))
    if version != EXPECTED_PREPROCESSING_VERSION:
        raise ValueError(
            f"训练文件预处理版本必须为 {EXPECTED_PREPROCESSING_VERSION}，"
            f"实际为 {version or '未预处理'}：{path}")
    if not np.isclose(sample_rate, config.sample_rate_hz):
        raise ValueError(
            f"训练文件采样率必须为 {config.sample_rate_hz} Hz，"
            f"实际为 {sample_rate:g} Hz：{path}")
    if channels != config.channels:
        raise ValueError(
            f"训练文件通道数必须为 {config.channels}，实际为 {channels}：{path}")


def load_split(
    data_root: str | Path,
    split: str,
    sequence_frames: int = 400,
    feature_config: MPFConfig | None = None,
) -> TensorDataset:
    root = Path(data_root)
    rows = list(csv.DictReader((root / "discrete_gestures_corpus.csv").open(encoding="utf-8")))
    config = feature_config or MPFConfig()
    extractor = MultiBandMatrixPowerFeatures(config).eval()
    feature_parts: list[torch.Tensor] = []; target_parts: list[torch.Tensor] = []
    required_samples = extractor.config.left_context_samples + 1 + (sequence_frames - 1) * 40
    sequence_stride_samples = sequence_frames * extractor.config.output_stride_samples
    for row in rows:
        if row["split"] != split:
            continue
        path = root / row["dataset"]
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                raise ValueError(f"训练文件缺少 /data：{path}")
            dataset = handle["data"]
            _validate_training_dataset(dataset, path, config)
            signal = np.asarray(dataset["emg"], dtype=np.float32)
            sample_times = np.asarray(dataset["time"], dtype=np.float64)
        prompts = pd.read_hdf(path, "prompts")
        # Adjacent training sequences overlap only by MPF left context so their
        # 50 Hz output frames are continuous and no events disappear at boundaries.
        for start in range(0, len(signal) - required_samples + 1, sequence_stride_samples):
            raw = torch.from_numpy(signal[start:start + required_samples].T[None])
            with torch.inference_mode():
                features = extractor(raw)[0]
            output_times = sample_times[
                start + extractor.config.left_context_samples:start + required_samples:40
            ][:len(features)]
            targets = torch.zeros((len(features), len(LABELS)), dtype=torch.float32)
            for prompt in prompts.itertuples():
                name = str(prompt.name)
                if name not in LABELS:
                    continue
                center = float(prompt.time) + LABEL_SHIFT_SEC
                active = np.flatnonzero(
                    (output_times >= center - LABEL_HALF_WIDTH_SEC)
                    & (output_times <= center + LABEL_HALF_WIDTH_SEC))
                if len(active):
                    targets[torch.from_numpy(active), LABELS.index(name)] = 1.0
            feature_parts.append(features); target_parts.append(targets)
    if not feature_parts:
        raise ValueError(f"数据集中没有 {split} split")
    return TensorDataset(torch.stack(feature_parts), torch.stack(target_parts))
