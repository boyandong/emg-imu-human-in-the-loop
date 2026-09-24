"""Hash-checked, window-level runtime for the exploratory Song 28-state model.

This module deliberately has no stream alignment or UI hook. The caller must
provide a causal 50×8 EMG window and the matched 22×6 native IMU window.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos
from scipy.special import softmax


MODEL_NAME = "song_joint28_model.json"
MANIFEST_NAME = "song_joint28_manifest.json"
COMBO_PREFIX = "song28::"
ARM_DISPLAY = {"still": "手臂静止", "up": "向上", "down": "向下", "left": "向左",
               "right": "向右", "forward": "向前", "backward": "向后"}
HAND_DISPLAY = {"neutral": "静息", "index_pinch": "食指捏合", "fist": "握拳", "open_hand": "张开"}


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"model JSON must be an object: {path}")
    return value


def song_joint28_key_path(value: object) -> Path | None:
    name = str(value)
    return Path(name[len(COMBO_PREFIX):]) if name.startswith(COMBO_PREFIX) else None


def discover_song_joint28_bundles(models_root: Path) -> list[Path]:
    built_in = Path(__file__).resolve().parents[2] / "model_assets/song_joint28_window"
    candidates = [built_in, *sorted(Path(models_root).glob("*/" + MANIFEST_NAME))]
    found = []
    for path in candidates:
        try:
            SongJoint28WindowRuntime(path)
            if path.resolve() not in found:
                found.append(path.resolve())
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return found


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

    def make_bundle(self):
        # Keep offline export independent of the collection application's imports.
        from emgforce.inference.model_bundle import ModelBundle

        display = {}
        for name in self.joint_classes:
            for arm in self.arm_classes:
                prefix = arm + "_"
                if name.startswith(prefix):
                    display[name] = f"{ARM_DISPLAY[arm]} · {HAND_DISPLAY[name[len(prefix):]]}"
                    break
        return ModelBundle(
            root=self.directory, model_id="Song 手势×手臂 28 类（S01/S02）",
            artifact=self.artifact, sha256=self.sha256, labels=self.joint_classes,
            display_names=display, sample_rate=250, input_channels=8, output_channels=28,
            algorithm_id="song_joint28_local_v1", runtime_backend="song_joint28_source_factorized",
            preprocessing={"online_event_threshold": 0.15, "window_samples": 50,
                           "hop_samples": 25, "imu_window_samples": 22},
            metadata={"experimental_adapter": True, "song_joint28_local": True,
                      "model_status": self.manifest["model_status"],
                      "source_checkpoint": {"path": str(self.artifact), "sha256": self.sha256},
                      "training": {"source_sessions": ["S01", "S02"]}},
        )

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


class SongJoint28Stream:
    """Causal 25-sample-hop stream with indexed native IMU and a watermark.

    An EMG frame is emitted only after an IMU sample mapped *after* its ending
    EMG index arrives. This makes the last 22 IMU samples with index <= end
    final, including when packets from the two sensors arrive in separate calls.
    """

    def __init__(self, model: SongJoint28WindowRuntime):
        self.model = model
        self._filters = [butter(4, 40.0, btype="highpass", fs=250.0, output="sos")]
        for frequency in (50.0, 100.0):
            b, a = iirnotch(frequency, Q=30.0, fs=250.0)
            self._filters.append(tf2sos(b, a))
        self.reset()

    def reset(self) -> None:
        self._states = [np.zeros((len(sos), 2, 8), dtype=np.float64) for sos in self._filters]
        self._emg = deque(maxlen=50)
        self._imu: deque[tuple[int, np.ndarray]] = deque(maxlen=128)
        self._pending: deque[tuple[int, np.ndarray]] = deque(maxlen=40)
        self._last_emg_index: int | None = None
        self._last_imu_index: int | None = None
        self._since_reset = 0
        self.dropped_frames = 0

    def ingest_emg(self, raw: np.ndarray, indices: np.ndarray) -> tuple[bool, list[tuple[int, np.ndarray]]]:
        values = np.asarray(raw)
        indices = np.asarray(indices, dtype=np.int64)
        if (values.ndim != 2 or values.shape[1] != 8 or indices.shape != (len(values),)
                or not np.isfinite(values).all()):
            raise ValueError("Song 28 stream EMG requires finite [samples,8] and matching indices")
        if not len(values):
            return False, []
        discontinuity = False
        emitted = []
        boundaries = [0, *(np.flatnonzero(np.diff(indices) != 1) + 1).tolist(), len(values)]
        for part in range(len(boundaries) - 1):
            left, right = boundaries[part], boundaries[part + 1]
            if (part > 0 or (self._last_emg_index is not None
                             and int(indices[left]) != self._last_emg_index + 1)):
                self.reset()
                discontinuity = True
                emitted = []
            filtered = np.asarray(values[left:right], dtype=np.float64)
            for i, sos in enumerate(self._filters):
                filtered, self._states[i] = sosfilt(sos, filtered, axis=0, zi=self._states[i])
            for offset, sample in enumerate(filtered.astype(np.float32)):
                index = int(indices[left + offset])
                self._emg.append(sample)
                self._last_emg_index = index
                self._since_reset += 1
                if self._since_reset >= 50 and (self._since_reset - 50) % 25 == 0:
                    if len(self._pending) == self._pending.maxlen:
                        self.dropped_frames += 1
                    self._pending.append((index, np.stack(self._emg)))
            emitted.extend(self._flush())
        return discontinuity, emitted

    def ingest_imu(self, accel: np.ndarray, gyro: np.ndarray,
                   emg_indices: np.ndarray) -> list[tuple[int, np.ndarray]]:
        accel = np.asarray(accel, dtype=np.float32)
        gyro = np.asarray(gyro, dtype=np.float32)
        indices = np.asarray(emg_indices, dtype=np.int64)
        if (accel.ndim != 2 or accel.shape[1] != 3 or gyro.shape != accel.shape
                or indices.shape != (len(accel),) or not np.isfinite(accel).all()
                or not np.isfinite(gyro).all() or np.any(np.diff(indices) < 0)
                or (len(indices) and self._last_imu_index is not None
                    and int(indices[0]) < self._last_imu_index)):
            raise ValueError("Song 28 stream IMU requires ordered finite accel/gyro and EMG indices")
        for index, a, g in zip(indices, accel, gyro):
            self._imu.append((int(index), np.concatenate((a, g))))
            self._last_imu_index = int(index)
        return self._flush()

    def _flush(self) -> list[tuple[int, np.ndarray]]:
        frames = []
        while self._pending and self._last_imu_index is not None and self._last_imu_index > self._pending[0][0]:
            end, emg = self._pending.popleft()
            eligible = [(index, value) for index, value in self._imu if index <= end]
            if len(eligible) < 22 or end - eligible[-1][0] > 8 or end - eligible[-22][0] > 60:
                self.dropped_frames += 1
                continue
            imu = np.stack([value for _, value in eligible[-22:]])
            _, _, joint = self.model.predict_filtered_window(emg, imu)
            frames.append((end, joint))
        return frames
