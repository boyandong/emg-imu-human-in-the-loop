"""Explicit, local-only capture of live input and displayed model output."""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class LiveDiagnosticRecorder:
    """One operator-started diagnostic run; no labels or accuracy are inferred."""

    def __init__(self, root: Path, *, model_id: str, model_sha256: str,
                 labels: tuple[str, ...], sample_rate_hz: int,
                 threshold: float, hand: str) -> None:
        self.directory = Path(root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                                       + "-" + uuid4().hex[:8])
        self.directory.mkdir(parents=True, exist_ok=False)
        self.labels = tuple(labels)
        self.manifest = {
            "schema": "emgforce_live_diagnostic_v1", "status": "recording",
            "started_utc": _utc(), "model_id": str(model_id),
            "model_sha256": str(model_sha256), "labels": list(self.labels),
            "sample_rate_hz": int(sample_rate_hz), "threshold_at_start": float(threshold),
            "hand": str(hand), "ground_truth_available": False,
            "raw_samples": 0, "imu_samples": 0, "predictions": 0,
            "reported_lost_packets": 0,
        }
        self._h5 = h5py.File(self.directory / "signals.h5", "w")
        self._raw = self._h5.create_dataset("emg/raw", (0, 8), maxshape=(None, 8),
                                           chunks=(1024, 8), dtype="i4")
        self._indices = self._h5.create_dataset("emg/sample_index", (0,), maxshape=(None,),
                                               chunks=(1024,), dtype="i8")
        self._emg_received = self._h5.create_dataset("emg/received_ns", (0,), maxshape=(None,),
                                                    chunks=(1024,), dtype="i8")
        self._gyro = self._h5.create_dataset("imu/gyro_rad_s", (0, 3), maxshape=(None, 3),
                                            chunks=(512, 3), dtype="f4")
        self._accel = self._h5.create_dataset("imu/accel_m_s2", (0, 3), maxshape=(None, 3),
                                             chunks=(512, 3), dtype="f4")
        self._imu_received = self._h5.create_dataset("imu/received_ns", (0,), maxshape=(None,),
                                                    chunks=(512,), dtype="i8")
        self._csv_handle = (self.directory / "predictions.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_handle)
        self._writer.writerow(("received_utc", "output_sample_index", "peak_label",
                               "active_label", "events", "inference_ms", "threshold",
                               *(f"p_{label}" for label in self.labels)))
        self._annotation_handle = (self.directory / "manual_annotations.csv").open(
            "w", newline="", encoding="utf-8")
        self._annotation_writer = csv.writer(self._annotation_handle)
        self._annotation_writer.writerow(("marked_utc", "marked_monotonic_ns",
                                          "latest_emg_sample_index", "action", "event"))
        self.manifest["manual_annotations"] = 0
        self._write_manifest()
        self._closed = False

    def _write_manifest(self) -> None:
        path = self.directory / "manifest.json"
        temporary = self.directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        temporary.replace(path)

    def record_emg(self, raw: np.ndarray, indices: np.ndarray,
                   received_ns: np.ndarray | None = None) -> None:
        values = np.asarray(raw, dtype=np.int32)
        sample_indices = np.asarray(indices, dtype=np.int64)
        received = (np.full(len(values), -1, dtype=np.int64) if received_ns is None
                    else np.asarray(received_ns, dtype=np.int64))
        if values.ndim != 2 or values.shape[1] != 8 or sample_indices.shape != (len(values),):
            raise ValueError("diagnostic EMG must be aligned [samples, 8] plus indices")
        if received.shape != (len(values),):
            raise ValueError("diagnostic EMG receive timestamps must align with samples")
        if len(sample_indices) and (np.any(np.diff(sample_indices) <= 0) or
                                   (len(self._indices) and sample_indices[0] <= self._indices[-1])):
            raise ValueError("diagnostic EMG indices must increase")
        count = len(self._raw)
        self._raw.resize((count + len(values), 8))
        self._indices.resize((count + len(values),))
        self._emg_received.resize((count + len(values),))
        self._raw[count:] = values
        self._indices[count:] = sample_indices
        self._emg_received[count:] = received
        self.manifest["raw_samples"] += len(values)
        self._h5.flush()

    def record_imu(self, gyro: np.ndarray, accel: np.ndarray, received_ns: np.ndarray) -> None:
        gyro = np.asarray(gyro, dtype=np.float32)
        accel = np.asarray(accel, dtype=np.float32)
        received_ns = np.asarray(received_ns, dtype=np.int64)
        if gyro.ndim != 2 or gyro.shape[1] != 3 or accel.shape != gyro.shape or received_ns.shape != (len(gyro),):
            raise ValueError("diagnostic IMU streams must have aligned three-axis samples")
        count = len(self._gyro)
        for dataset, values in ((self._gyro, gyro), (self._accel, accel)):
            dataset.resize((count + len(values), 3))
            dataset[count:] = values
        self._imu_received.resize((count + len(received_ns),))
        self._imu_received[count:] = received_ns
        self.manifest["imu_samples"] += len(gyro)
        self._h5.flush()

    def record_prediction(self, frame, threshold: float) -> None:
        probabilities = np.asarray(frame.probabilities, dtype=np.float64)
        if tuple(frame.labels) != self.labels or probabilities.shape != (len(self.labels),) or not np.isfinite(probabilities).all():
            raise ValueError("diagnostic prediction labels or probabilities differ from loaded model")
        peak = self.labels[int(np.argmax(probabilities))]
        self._writer.writerow((_utc(), int(frame.output_sample_index), peak,
                               frame.active_label or "", "|".join(event.name for event in frame.events),
                               float(frame.inference_ms), float(threshold),
                               *(float(value) for value in probabilities)))
        self._csv_handle.flush()
        self.manifest["predictions"] += 1

    def record_packet_loss(self, lost: int) -> None:
        self.manifest["reported_lost_packets"] += max(0, int(lost))

    def record_annotation(self, action: str, event: str,
                          latest_sample_index: int | None) -> None:
        if action not in self.labels or event not in ("start", "end"):
            raise ValueError("manual annotation must name a model label and start/end")
        self._annotation_writer.writerow((_utc(), time.monotonic_ns(),
                                          "" if latest_sample_index is None else int(latest_sample_index),
                                          action, event))
        self._annotation_handle.flush()
        self.manifest["manual_annotations"] += 1

    def close(self) -> Path:
        if not self._closed:
            self._csv_handle.close()
            self._annotation_handle.close()
            self._h5.close()
            self.manifest["stopped_utc"] = _utc()
            self.manifest["status"] = "closed"
            self._write_manifest()
            self._closed = True
        return self.directory
