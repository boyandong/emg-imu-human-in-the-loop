from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from emgforce.config import EMG_CHANNELS, RECORDER_BATCH_SAMPLES
from emgforce.experiment.models import CueEvent, ExperimentEvent, TrialInfo


LOGGER = logging.getLogger(__name__)


class Hdf5Recorder:
    """Single-writer, unbounded-queue HDF5 recorder.

    Producers only copy/enqueue decoded values. Every HDF5 operation happens on
    the writer thread; display back-pressure therefore cannot lose recording data.
    """

    def __init__(self, batch_samples: int = RECORDER_BATCH_SAMPLES,
                 on_error: Callable[[str], None] | None = None) -> None:
        self.batch_samples = batch_samples
        self.on_error = on_error
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._startup_error: BaseException | None = None
        self.path: Path | None = None
        self.active = False
        self.emg_samples_written = 0
        self.imu_samples_written = 0

    def start(self, path: Path, metadata: dict[str, Any]) -> None:
        if self.active:
            raise RuntimeError("HDF5 记录器已启动")
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"实验文件已存在：{path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._ready.clear()
        self._closed.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._run, args=(path, dict(metadata)),
            name="hdf5-writer", daemon=False,
        )
        self._thread.start()
        if not self._ready.wait(10):
            raise TimeoutError("HDF5 Writer 启动超时")
        if self._startup_error is not None:
            raise RuntimeError(f"无法创建 HDF5: {self._startup_error}")
        self.active = True

    def enqueue_emg(self, raw: np.ndarray, sample_index: np.ndarray,
                    packet_seq: np.ndarray) -> None:
        self._put("emg", (
            np.asarray(raw, dtype=np.int32).copy(),
            np.asarray(sample_index, dtype=np.int64).copy(),
            np.asarray(packet_seq, dtype=np.uint8).copy(),
        ))

    def enqueue_imu(self, gyro: np.ndarray, accel: np.ndarray,
                    pc_monotonic_ns: np.ndarray, packet_seq: np.ndarray,
                    emg_sample_index: np.ndarray) -> None:
        self._put("imu", tuple(np.asarray(value).copy() for value in (
            gyro, accel, pc_monotonic_ns, packet_seq, emg_sample_index
        )))

    def enqueue_event(self, event: ExperimentEvent) -> None:
        self._put("event", event)

    def enqueue_trial(self, trial: TrialInfo) -> None:
        self._put("trial", trial)

    def enqueue_cue_event(self, event: CueEvent) -> None:
        self._put("cue_event", event)

    def stop(self, timeout: float = 15.0) -> None:
        if self._thread is None:
            return
        self._queue.put(("stop", None))
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("HDF5 Writer 未能安全停止")
        self._thread = None
        self.active = False
        if self._startup_error is not None:
            raise RuntimeError(f"HDF5 Writer 错误: {self._startup_error}")

    def _put(self, kind: str, payload: Any) -> None:
        if not self.active:
            raise RuntimeError("HDF5 记录器尚未启动")
        self._queue.put((kind, payload))

    def _run(self, path: Path, metadata: dict[str, Any]) -> None:
        handle: h5py.File | None = None
        try:
            handle = h5py.File(path, "x")
            datasets = self._create_schema(handle, metadata)
            self._ready.set()
            emg_buffer: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            buffered_samples = 0
            while True:
                kind, payload = self._queue.get()
                if kind == "emg":
                    emg_buffer.append(payload)
                    buffered_samples += len(payload[0])
                    if buffered_samples >= self.batch_samples:
                        self._flush_emg(datasets, emg_buffer)
                        emg_buffer.clear()
                        buffered_samples = 0
                elif kind == "imu":
                    self._append_imu(datasets, payload)
                elif kind == "event":
                    self._append_event(datasets["events"], payload)
                elif kind == "trial":
                    self._append_trial(datasets["trials"], payload)
                elif kind == "cue_event":
                    self._append_cue_event(datasets["cue_events"], payload)
                elif kind == "stop":
                    if emg_buffer:
                        self._flush_emg(datasets, emg_buffer)
                    handle.flush()
                    break
        except BaseException as exc:
            LOGGER.exception("HDF5 writer failure")
            self._startup_error = exc
            if self.on_error:
                self.on_error(str(exc))
        finally:
            self._ready.set()
            if handle is not None:
                handle.close()
            self.active = False
            self._closed.set()

    @staticmethod
    def _create_schema(handle: h5py.File, metadata: dict[str, Any]) -> dict[str, Any]:
        meta = handle.create_group("meta")
        for key, value in metadata.items():
            meta.attrs[key] = "" if value is None else value
        meta.attrs["schema_version"] = "2.1"
        meta.attrs["label_alignment_status"] = "raw_cues_only"
        streams = handle.create_group("streams")
        emg = streams.create_group("emg")
        imu = streams.create_group("imu")
        result: dict[str, Any] = {
            "emg_raw": emg.create_dataset("raw", (0, EMG_CHANNELS),
                maxshape=(None, EMG_CHANNELS), dtype="i4", chunks=(2000, EMG_CHANNELS)),
            "emg_index": emg.create_dataset("sample_index", (0,), maxshape=(None,),
                dtype="i8", chunks=(4000,)),
            "emg_seq": emg.create_dataset("packet_seq", (0,), maxshape=(None,),
                dtype="u1", chunks=(4000,)),
            "gyro": imu.create_dataset("gyro", (0, 3), maxshape=(None, 3),
                dtype="f4", chunks=(1000, 3)),
            "accel": imu.create_dataset("accel", (0, 3), maxshape=(None, 3),
                dtype="f4", chunks=(1000, 3)),
            "imu_ns": imu.create_dataset("pc_monotonic_ns", (0,), maxshape=(None,),
                dtype="i8", chunks=(2000,)),
            "imu_seq": imu.create_dataset("packet_seq", (0,), maxshape=(None,),
                dtype="u1", chunks=(2000,)),
            "imu_emg_index": imu.create_dataset("emg_sample_index", (0,), maxshape=(None,),
                dtype="i8", chunks=(2000,)),
        }
        string = h5py.string_dtype("utf-8")
        event_dtype = np.dtype([
            ("event_id", "i8"), ("sample_index", "i8"), ("time_sec", "f8"),
            ("pc_monotonic_ns", "i8"), ("event_type", string), ("label", string),
            ("trial_id", "i8"), ("stage_id", "i8"), ("donning_id", "i8"),
            ("note", string),
        ])
        trial_dtype = np.dtype([
            ("trial_id", "i8"), ("label", string), ("stage_id", "i8"),
            ("donning_id", "i8"), ("trial_start_sample", "i8"),
            ("rest_start_sample", "i8"), ("prompt_start_sample", "i8"),
            ("prompt_end_sample", "i8"), ("trial_end_sample", "i8"),
            ("valid", "?"), ("reject_reason", string), ("note", string),
        ])
        result["events"] = handle.create_dataset("events", (0,), maxshape=(None,),
                                                  dtype=event_dtype, chunks=(256,))
        result["trials"] = handle.create_dataset("trials", (0,), maxshape=(None,),
                                                  dtype=trial_dtype, chunks=(128,))
        gesture_dtype = np.dtype([
            ("name", string), ("sample_index", "i8"), ("time_sec", "f8"),
            ("trial_id", "i8"), ("stage_id", "i8"),
            ("scheduled_monotonic_ns", "i8"), ("emitted_monotonic_ns", "i8"),
        ])
        result["cue_events"] = handle.create_dataset(
            "cue_events", (0,), maxshape=(None,), dtype=gesture_dtype, chunks=(128,))
        result["cue_events"].attrs["time_reference"] = "session_relative_seconds"
        result["cue_events"].attrs["semantics"] = "ui_action_cue_not_movement_onset"
        return result

    def _flush_emg(self, d: dict[str, Any], blocks: list[tuple[np.ndarray, ...]]) -> None:
        values = [np.concatenate([block[i] for block in blocks], axis=0) for i in range(3)]
        for key, value in zip(("emg_raw", "emg_index", "emg_seq"), values):
            self._append(d[key], value)
        self.emg_samples_written += len(values[0])

    def _append_imu(self, d: dict[str, Any], payload: tuple[np.ndarray, ...]) -> None:
        for key, value in zip(("gyro", "accel", "imu_ns", "imu_seq", "imu_emg_index"), payload):
            self._append(d[key], value)
        self.imu_samples_written += len(payload[0])

    @staticmethod
    def _append(dataset: h5py.Dataset, values: np.ndarray) -> None:
        start = len(dataset)
        dataset.resize(start + len(values), axis=0)
        dataset[start:] = values

    @classmethod
    def _append_event(cls, dataset: h5py.Dataset, event: ExperimentEvent) -> None:
        row = np.array([(
            event.event_id, event.sample_index, event.time_sec, event.pc_monotonic_ns,
            event.event_type.value, event.label, event.trial_id, event.stage_id,
            event.donning_id, event.note,
        )], dtype=dataset.dtype)
        cls._append(dataset, row)

    @classmethod
    def _append_trial(cls, dataset: h5py.Dataset, trial: TrialInfo) -> None:
        row = np.array([(
            trial.trial_id, trial.label, trial.stage_id, trial.donning_id,
            trial.trial_start_sample, trial.rest_start_sample, trial.prompt_start_sample,
            trial.prompt_end_sample, trial.trial_end_sample, trial.valid,
            trial.reject_reason, trial.note,
        )], dtype=dataset.dtype)
        cls._append(dataset, row)

    @classmethod
    def _append_cue_event(cls, dataset: h5py.Dataset,
                          event: CueEvent) -> None:
        row = np.array([(
            event.name, event.sample_index, event.time_sec,
            event.trial_id, event.stage_id,
            event.scheduled_monotonic_ns, event.emitted_monotonic_ns,
        )], dtype=dataset.dtype)
        cls._append(dataset, row)
