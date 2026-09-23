"""Data-free, hash-checked runtime for the experimental Song 8-channel F0 model."""
from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos
from scipy.special import softmax

from emgforce.algorithms import SONG_REAL8_LOCAL
from .model_bundle import ModelBundle


LABELS = ("fist", "index_pinch", "neutral", "open_hand")
DISPLAY = {"fist": "握拳", "index_pinch": "食指捏合", "neutral": "自然静息", "open_hand": "张开手"}
MODEL_NAME = "song_f0_model.json"
MANIFEST_NAME = "song_manifest.json"


@dataclass(frozen=True, slots=True)
class SongModelInfo:
    directory: Path
    model_id: str
    validation_accuracy: float
    validation_macro_f1: float

    @property
    def combo_key(self) -> str:
        return f"song::{self.directory}"


def song_key_path(value: object) -> Path | None:
    text = str(value)
    return Path(text[6:]) if text.startswith("song::") else None


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Song 模型 JSON 顶层必须是对象：{path}")
    return value


def discover_song_models(models_root: Path) -> list[SongModelInfo]:
    found = []
    for path in sorted(Path(models_root).glob("*/" + MANIFEST_NAME)):
        try:
            manifest = _read_json(path)
            if manifest["algorithm_id"] != SONG_REAL8_LOCAL:
                continue
            if not (path.parent / MODEL_NAME).is_file():
                continue
            found.append(SongModelInfo(path.parent, str(manifest["model_id"]),
                                       float(manifest["validation_trial_accuracy"]),
                                       float(manifest["validation_trial_macro_f1"])))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return found


class SongLocalRuntime:
    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        manifest = _read_json(self.directory / MANIFEST_NAME)
        artifact = self.directory / MODEL_NAME
        if (manifest.get("format_version") != 1 or
                manifest.get("algorithm_id") != SONG_REAL8_LOCAL or
                manifest.get("artifact") != MODEL_NAME or
                manifest.get("model_status") != "exploratory_one_person_one_day_not_formal_frozen"):
            raise ValueError("Song 模型清单与实验运行时不兼容")
        actual_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual_hash != manifest.get("sha256"):
            raise ValueError("Song 模型 SHA-256 不匹配")
        model = _read_json(artifact)
        if (model.get("format_version") != 1 or
                model.get("model_kind") != "song_real8_causal_f0_logistic" or
                model.get("sample_rate_hz") != 250 or model.get("channels") != 8 or
                model.get("window_samples") != 50 or model.get("hop_samples") != 25 or
                tuple(model.get("classes", ())) != LABELS):
            raise ValueError("Song 模型通道、采样率、窗口或类别不兼容")
        if model.get("filter") != {
            "highpass_hz": 40.0, "highpass_order": 4,
            "notches_hz": [50.0, 100.0], "notch_q": 30.0,
            "implementation": "causal_sosfilt_zero_initial_state_continuous",
        }:
            raise ValueError("Song 模型因果滤波契约不匹配")
        self.thresholds = self._array(model, "f0_thresholds", (8,), positive=True)
        self.scaler_mean = self._array(model, "standard_scaler_mean", (48,))
        self.scaler_scale = self._array(model, "standard_scaler_scale", (48,), positive=True)
        self.coef = self._array(model, "logistic_coef", (4, 48))
        self.intercept = self._array(model, "logistic_intercept", (4,))
        self.manifest = manifest
        self.artifact = artifact
        self.sha256 = actual_hash
        self._filters = [butter(4, 40.0, btype="highpass", fs=250.0, output="sos")]
        for frequency in (50.0, 100.0):
            b, a = iirnotch(frequency, Q=30.0, fs=250.0)
            self._filters.append(tf2sos(b, a))
        self.reset()

    @staticmethod
    def _array(model: dict, name: str, shape: tuple[int, ...], *, positive=False):
        value = np.asarray(model[name], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all() or (positive and np.any(value <= 0)):
            raise ValueError(f"Song 模型字段无效：{name}")
        return value

    def make_bundle(self) -> ModelBundle:
        return ModelBundle(root=self.directory, model_id=str(self.manifest["model_id"]),
                           artifact=self.artifact, sha256=self.sha256,
                           labels=LABELS, display_names=DISPLAY,
                           sample_rate=250, input_channels=8, output_channels=4,
                           algorithm_id=SONG_REAL8_LOCAL, runtime_backend="song_real8_f0",
                           preprocessing={"online_event_threshold": 0.5, "window_samples": 50,
                                          "hop_samples": 25},
                           metadata={"experimental_adapter": True,
                                     "song_real8_local": True,
                                     "training": {"val_accuracy": self.manifest["validation_trial_accuracy"],
                                                  "val_macro_f1": self.manifest["validation_trial_macro_f1"]},
                                     "source_checkpoint": {"path": str(self.artifact), "sha256": self.sha256},
                                     "model_status": self.manifest["model_status"]})

    def reset(self):
        self._states = [np.zeros((len(sos), 2, 8), dtype=np.float64) for sos in self._filters]
        self._window = deque(maxlen=50)
        self._since_reset = 0
        self._last_index: int | None = None

    def _filter_block(self, raw: np.ndarray) -> np.ndarray:
        values = np.asarray(raw, dtype=np.float64)
        for index, sos in enumerate(self._filters):
            values, self._states[index] = sosfilt(sos, values, axis=0, zi=self._states[index])
        return values.astype(np.float32)

    def predict_filtered_window(self, filtered: np.ndarray) -> np.ndarray:
        x = np.asarray(filtered, dtype=np.float64)
        if x.shape != (50, 8) or not np.isfinite(x).all():
            raise ValueError("Song 模型窗口必须是有限的 50×8 因果滤波 EMG")
        differences = np.diff(x, axis=0)
        rms = np.sqrt(np.mean(x * x, axis=0))
        mav = np.mean(np.abs(x), axis=0)
        wl = np.sum(np.abs(differences), axis=0)
        threshold = self.thresholds[None, :]
        zc = np.sum((x[:-1] * x[1:] < 0) & (np.abs(differences) > threshold), axis=0)
        left = x[1:-1] - x[:-2]
        right = x[1:-1] - x[2:]
        ssc = np.sum((left * right > 0) & (np.maximum(np.abs(left), np.abs(right)) > threshold), axis=0)
        wamp = np.sum(np.abs(differences) > threshold, axis=0)
        features = np.concatenate((rms, mav, wl, zc, ssc, wamp)).astype(np.float32)
        standardized = ((features - self.scaler_mean) / self.scaler_scale)
        probabilities = softmax(self.coef @ standardized + self.intercept)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Song 模型输出异常")
        return probabilities.astype(np.float32)

    def ingest(self, raw: np.ndarray, indices: np.ndarray) -> tuple[bool, list[tuple[int, np.ndarray]]]:
        raw = np.asarray(raw)
        indices = np.asarray(indices, dtype=np.int64)
        if raw.ndim != 2 or raw.shape[1] != 8 or indices.shape != (len(raw),):
            raise ValueError("Song 实时输入必须是 [samples,8] 且索引数量一致")
        frames: list[tuple[int, np.ndarray]] = []
        discontinuity = False
        if not len(raw):
            return False, frames
        boundaries = [0, *(np.flatnonzero(np.diff(indices) != 1) + 1).tolist(), len(raw)]
        for part in range(len(boundaries) - 1):
            left, right = boundaries[part], boundaries[part + 1]
            if part > 0 or (self._last_index is not None and int(indices[left]) != self._last_index + 1):
                self.reset()
                discontinuity = True
            filtered = self._filter_block(raw[left:right])
            for offset, sample in enumerate(filtered):
                sample_id = int(indices[left + offset])
                self._window.append(sample)
                self._since_reset += 1
                self._last_index = sample_id
                if self._since_reset >= 50 and (self._since_reset - 50) % 25 == 0:
                    frames.append((sample_id, self.predict_filtered_window(np.stack(self._window))))
        return discontinuity, frames
