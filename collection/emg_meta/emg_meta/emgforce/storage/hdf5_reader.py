from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from emgforce.config import EMG_CHANNELS, SAMPLING_RATE


class Hdf5SessionReader:
    REQUIRED_META = {"participant_id", "session_id", "sampling_rate",
                     "num_emg_channels", "protocol_name"}

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def summary(self) -> dict[str, Any]:
        with h5py.File(self.path, "r") as h5:
            raw = h5["streams/emg/raw"]
            imu = h5["streams/imu/gyro"]
            trials = h5["trials"][:]
            events = h5["events"][:]
            valid = int(np.count_nonzero(trials["valid"])) if len(trials) else 0
            loss = sum(1 for value in events["event_type"] if self._text(value) == "PACKET_LOSS")
            rate = float(h5["meta"].attrs.get("sampling_rate", SAMPLING_RATE))
            return {
                "participant": self._text(h5["meta"].attrs.get("participant_id", "")),
                "session": self._text(h5["meta"].attrs.get("session_id", "")),
                "duration": len(raw) / rate,
                "emg_samples": len(raw), "imu_samples": len(imu),
                "trial_count": len(trials), "valid_trials": valid,
                "bad_trials": len(trials) - valid, "packet_loss_events": loss,
            }

    def validate(self) -> list[str]:
        problems: list[str] = []
        with h5py.File(self.path, "r") as h5:
            missing = self.REQUIRED_META - set(h5["meta"].attrs)
            if missing:
                problems.append("缺少 metadata: " + ", ".join(sorted(missing)))
            raw = h5["streams/emg/raw"]
            index = h5["streams/emg/sample_index"][:]
            if raw.ndim != 2 or raw.shape[1] != EMG_CHANNELS:
                problems.append(f"EMG shape 无效: {raw.shape}")
            if len(index) != len(raw) or (len(index) > 1 and np.any(np.diff(index) != 1)):
                problems.append("EMG sample_index 不连续或长度不匹配")
            n = len(raw)
            for row in h5["trials"]:
                for field in ("trial_start_sample", "prompt_start_sample", "trial_end_sample"):
                    value = int(row[field])
                    if value < -1 or value > n:
                        problems.append(f"试次 {int(row['trial_id'])} 的 {field} 越界")
            for row in h5["events"]:
                value = int(row["sample_index"])
                if value < 0 or value > n:
                    problems.append(f"Event {int(row['event_id'])} sample_index 越界")
            cue_events = h5.get("cue_events")
            if cue_events is not None:
                by_trial: dict[int, list[str]] = {}
                for row in cue_events:
                    value = int(row["sample_index"])
                    if value < 0 or value > n:
                        problems.append(
                            f"Cue trial {int(row['trial_id'])} sample_index 越界")
                    by_trial.setdefault(int(row["trial_id"]), []).append(
                        self._text(row["name"]))
                for row in h5["trials"]:
                    if not bool(row["valid"]):
                        continue
                    label = self._text(row["label"])
                    expected = {
                        "index_hold": ["index_press", "index_release"],
                        "middle_hold": ["middle_press", "middle_release"],
                    }.get(label)
                    if expected is not None and by_trial.get(int(row["trial_id"]), []) != expected:
                        problems.append(
                            f"有效保持试次 {int(row['trial_id'])} 缺少或错序的 press/release 事件")
        return problems

    def trials(self) -> np.ndarray:
        with h5py.File(self.path, "r") as h5:
            return h5["trials"][:]

    def trial_emg(self, trial_id: int, padding_sec: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        with h5py.File(self.path, "r") as h5:
            rows = h5["trials"][:]
            matches = rows[rows["trial_id"] == trial_id]
            if not len(matches):
                raise KeyError(f"试次 {trial_id} 不存在")
            row = matches[0]
            rate = float(h5["meta"].attrs.get("sampling_rate", SAMPLING_RATE))
            pad = round(padding_sec * rate)
            start = max(0, int(row["trial_start_sample"]) - pad)
            end = min(len(h5["streams/emg/raw"]), int(row["trial_end_sample"]) + pad)
            return h5["streams/emg/raw"][start:end], np.arange(start, end, dtype=np.int64)

    @staticmethod
    def _text(value: Any) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)
