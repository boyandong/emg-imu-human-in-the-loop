"""Hash-checked, window-level runtime for the exploratory Song 28-state model.

This module deliberately has no stream alignment or UI hook. The caller must
provide a causal 50×8 EMG window and the matched 22×6 native IMU window.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax


MODEL_NAME = "song_joint28_model.json"
MANIFEST_NAME = "song_joint28_manifest.json"


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"model JSON must be an object: {path}")
    return value


class SongJoint28WindowRuntime:
    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        self.manifest = _read_object(self.directory / MANIFEST_NAME)
        self.artifact = self.directory / MODEL_NAME
        if (self.manifest.get("format_version") != 1
                or self.manifest.get("artifact") != MODEL_NAME
                or self.manifest.get("model_status") != "exploratory_one_person_one_day_offline_only"):
            raise ValueError("incompatible experimental Song 28-state manifest")
        self.sha256 = hashlib.sha256(self.artifact.read_bytes()).hexdigest()
        if self.sha256 != self.manifest.get("sha256"):
            raise ValueError("Song 28-state model artifact hash mismatch")
        model = _read_object(self.artifact)
        if (model.get("format_version") != 1
                or model.get("model_kind") != "song_28_source_factorized_f0v2_f2a_f3c_f6"
                or model.get("emg_rate_hz") != 250 or model.get("imu_rate_hz") != 112
                or model.get("emg_channels") != 8 or model.get("imu_channels") != 6
                or model.get("emg_window_samples") != 50 or model.get("imu_window_samples") != 22
                or model.get("covariance_shrinkage") != 0.05):
            raise ValueError("incompatible Song 28-state signal or feature contract")
        self.hand_classes = tuple(model["hand_classes"])
        self.arm_classes = tuple(model["arm_classes"])
        self.joint_classes = tuple(model["joint_classes"])
        self.joint_indices = np.asarray(model["joint_indices"], dtype=np.int64)
        if (len(set(self.hand_classes)) != 4 or len(set(self.arm_classes)) != 7
                or len(set(self.joint_classes)) != 28 or self.joint_indices.shape != (28, 2)
                or set(map(tuple, self.joint_indices)) != {(a, h) for a in range(7) for h in range(4)}
                or any(self.joint_classes[i] != f"{self.arm_classes[a]}_{self.hand_classes[h]}"
                       for i, (a, h) in enumerate(self.joint_indices))):
            raise ValueError("28-state label mapping is incomplete or misordered")
        self.thresholds = self._array(model["hand"], "rest_thresholds", (8,), positive=True)
        self.hand_mean = self._array(model["hand"], "scaler_mean", (104,))
        self.hand_scale = self._array(model["hand"], "scaler_scale", (104,), positive=True)
        self.hand_coef = self._array(model["hand"], "coef", (4, 104))
        self.hand_intercept = self._array(model["hand"], "intercept", (4,))
        self.arm_mean = self._array(model["arm"], "scaler_mean", (13,))
        self.arm_scale = self._array(model["arm"], "scaler_scale", (13,), positive=True)
        self.arm_coef = self._array(model["arm"], "coef", (7, 13))
        self.arm_intercept = self._array(model["arm"], "intercept", (7,))

    @staticmethod
    def _array(group: dict, field: str, shape: tuple[int, ...], *, positive: bool = False) -> np.ndarray:
        value = np.asarray(group[field], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all() or (positive and np.any(value <= 0)):
            raise ValueError(f"invalid Song 28-state field: {field}")
        return value

    @staticmethod
    def _normalized_covariance(x: np.ndarray) -> np.ndarray:
        centered = x - x.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / (len(x) - 1)
        trace = np.trace(covariance)
        covariance = 0.95 * covariance + 0.05 * trace / 8 * np.eye(8)
        return covariance / max(float(np.trace(covariance)), 1e-12)

    def hand_features(self, filtered_emg: np.ndarray) -> np.ndarray:
        x = np.asarray(filtered_emg, dtype=np.float64)
        if x.shape != (50, 8) or not np.isfinite(x).all():
            raise ValueError("Song 28-state EMG requires finite 50×8 causal-filtered samples")
        difference = np.diff(x, axis=0)
        rms = np.sqrt(np.mean(x * x, axis=0))
        mav = np.mean(np.abs(x), axis=0)
        wl = np.sum(np.abs(difference), axis=0)
        zc = np.sum((x[:-1] * x[1:] < 0) & (np.abs(difference) > self.thresholds), axis=0)
        left, right = x[1:-1] - x[:-2], x[1:-1] - x[2:]
        ssc = np.sum((left * right > 0) &
                     (np.maximum(np.abs(left), np.abs(right)) > self.thresholds), axis=0)
        wamp = np.sum(np.abs(difference) > self.thresholds, axis=0)
        f0 = np.concatenate((rms, mav, wl, zc, ssc, wamp)).astype(np.float32)
        covariance = self._normalized_covariance(x)
        row, col = np.triu_indices(8)
        f2a = covariance[row, col].copy()
        f2a[row != col] *= np.sqrt(2.0)
        columns = []
        for lag in range(1, 5):
            values = covariance[np.arange(8), (np.arange(8) + lag) % 8]
            columns.extend((values.mean(), np.median(values), values.std(),
                            np.quantile(values, 0.25), np.quantile(values, 0.75)))
        f3c = np.asarray(columns, dtype=np.float32)
        return np.concatenate((f0, f2a.astype(np.float32), f3c)).astype(np.float32)

    @staticmethod
    def arm_features(imu_window: np.ndarray) -> np.ndarray:
        imu = np.asarray(imu_window, dtype=np.float64)
        if imu.shape != (22, 6) or not np.isfinite(imu).all():
            raise ValueError("Song 28-state IMU requires finite 22×6 matched samples")
        features = []
        for sensor in (imu[:, :3], imu[:, 3:]):
            norm = np.linalg.norm(sensor, axis=1)
            features.extend((norm.mean(), norm.std(), np.sqrt(np.mean(norm * norm)),
                             norm.max(), np.ptp(norm)))
        gravity = imu[:, :3].mean(axis=0)
        features.extend(gravity / max(float(np.linalg.norm(gravity)), 1e-10))
        return np.nan_to_num(np.asarray(features, dtype=np.float32))

    def predict_filtered_window(self, filtered_emg: np.ndarray,
                                imu_window: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hand = self.hand_features(filtered_emg)
        arm = self.arm_features(imu_window)
        p_hand = softmax(self.hand_coef @ ((hand - self.hand_mean) / self.hand_scale)
                         + self.hand_intercept)
        p_arm = softmax(self.arm_coef @ ((arm - self.arm_mean) / self.arm_scale)
                        + self.arm_intercept)
        joint = p_arm[self.joint_indices[:, 0]] * p_hand[self.joint_indices[:, 1]]
        joint /= joint.sum()
        if not np.isfinite(joint).all():
            raise RuntimeError("Song 28-state model produced nonfinite probabilities")
        return p_hand, p_arm, joint
