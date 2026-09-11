from __future__ import annotations

import json
import logging
import time
import hashlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from emgforce import __version__
from emgforce.config import BAUDRATE, EMG_CHANNELS, IMU_SAMPLING_RATE, SAMPLING_RATE
from emgforce.collection_protocol import (
    FORMAL_PROTOCOL_NAME, QUALITY_THRESHOLD_VERSION, TIMESTAMP_SOURCE,
)
from emgforce.controller import AcquisitionController
from emgforce.processing.meta_corpus import resolve_dataset_split
from emgforce.processing.meta_alignment import export_meta_aligned
from emgforce.storage.hdf5_recorder import Hdf5Recorder
from emgforce.storage.session_paths import SessionPaths, build_session_paths
from emgforce.quality.session_readiness import generate_session_readiness

from .events import EventType
from .models import (CueEvent, ExperimentEvent, ParticipantInfo,
                     ProtocolConfig, SessionInfo)
from .prompt_engine import PromptEngine
from .trial_manager import TrialManager


LOGGER = logging.getLogger(__name__)


class ExperimentSession(QObject):
    session_started = Signal(object)
    session_stopped = Signal(object)
    event_created = Signal(object)
    recorder_error = Signal(str)
    status_changed = Signal(str)

    def __init__(self, acquisition: AcquisitionController, data_root: Path,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.acquisition = acquisition
        self.data_root = Path(data_root)
        self.prompt = PromptEngine(self)
        self.trials = TrialManager()
        self.recorder: Hdf5Recorder | None = None
        self.paths: SessionPaths | None = None
        self.participant: ParticipantInfo | None = None
        self.info: SessionInfo | None = None
        self.protocol: ProtocolConfig | None = None
        self.donning_id = 1
        self.stage_id = 1
        self._event_id = 0
        self._active = False
        self._log_handler: logging.Handler | None = None
        self.last_aligned_path: Path | None = None
        self.last_readiness: dict | None = None
        self._automatic_stage_name = ""
        self._wire_prompt()

    @property
    def active(self) -> bool:
        return self._active

    def start(self, participant: ParticipantInfo, info: SessionInfo,
              protocol: ProtocolConfig) -> SessionPaths:
        if self._active:
            raise RuntimeError("实验场次已在运行")
        participant.validate(); info.validate(); protocol.validate()
        if protocol.formal_collection and protocol.name != FORMAL_PROTOCOL_NAME:
            raise ValueError(f"正式采集协议必须为 {FORMAL_PROTOCOL_NAME}")
        if protocol.formal_collection and info.session_id.upper() == "S04" \
                and not info.model_frozen_confirmed:
            raise ValueError("S04 是正式测试；必须先确认模型、阈值和校准算法已经冻结")
        info.dataset_split = resolve_dataset_split(info.session_id, info.dataset_split)
        paths = build_session_paths(self.data_root, participant.participant_id, info.session_id)
        if paths.directory.exists():
            raise FileExistsError(f"实验场次目录已存在：{paths.directory}")
        paths.directory.mkdir(parents=True)
        self._install_session_log(paths.log)
        started = datetime.now().astimezone().isoformat()
        start_unix_time = time.time()
        protocol_payload = protocol.to_dict()
        protocol_json = json.dumps(protocol_payload, ensure_ascii=False, sort_keys=True)
        protocol_hash = protocol.source_sha256 or hashlib.sha256(
            protocol_json.encode("utf-8")).hexdigest()
        metadata = {
            **asdict(participant), "session_id": info.session_id,
            "experiment_name": info.experiment_name,
            "sampling_rate": SAMPLING_RATE, "num_emg_channels": EMG_CHANNELS,
            "emg_nominal_rate_hz": SAMPLING_RATE,
            "imu_nominal_rate_hz": IMU_SAMPLING_RATE,
            "baudrate": BAUDRATE, "serial_port": info.serial_port,
            "software_version": __version__, "protocol_name": protocol.name,
            "protocol_version": protocol.protocol_version or protocol.name,
            "protocol_file": protocol.source_filename,
            "protocol_file_sha256": protocol_hash,
            "dataset_split": info.dataset_split,
            "session_role": ("validation" if info.dataset_split == "val"
                             else ("final_test" if info.session_id.upper() == "S04"
                                   else info.dataset_split)),
            "protocol_json": protocol_json,
            "start_datetime": started,
            "start_unix_time": start_unix_time,
            "date": started[:10],
            "task": "discrete_gestures",
            "posture_name": protocol.posture_name or "unspecified",
            "posture_instruction": protocol.posture_instruction,
            "quality_report_json": info.quality_report_json,
            "donning_notes": info.donning_notes,
            "tested_arm": info.tested_arm,
            "channel1_orientation": info.channel1_orientation,
            "channel_1_orientation": info.channel1_orientation,
            "anatomical_marker": info.anatomical_marker,
            "strap_setting": info.strap_setting,
            "stabilization_sec": info.stabilization_sec,
            "physical_condition": info.physical_condition,
            "recorded_arm": info.recorded_arm or info.tested_arm,
            "donning_code": info.donning_code,
            "donning_id": info.donning_code,
            "anatomical_distance_mm": info.anatomical_distance_mm,
            "strap_scale": info.strap_scale,
            "strap_tightness": info.strap_tightness,
            "skin_condition": info.skin_condition,
            "fatigue_before": info.fatigue_before,
            "fatigue_after": info.fatigue_after,
            "reference_photo_name": info.reference_photo_name,
            "reference_photo_sha256": info.reference_photo_sha256,
            "operator_id": info.operator_id,
            "quality_override_reason": info.quality_override_reason,
            "quality_threshold_version": protocol.quality_threshold_version or QUALITY_THRESHOLD_VERSION,
            "model_frozen_confirmed": info.model_frozen_confirmed,
            "timestamp_source": TIMESTAMP_SOURCE,
            "device_timestamp_available": False,
        }
        config_payload = {
            "participant": asdict(participant), "session": asdict(info),
            "protocol": protocol.to_dict(), "automatic": {
                "start_datetime": started, "software_version": __version__,
                "sampling_rate": SAMPLING_RATE, "num_emg_channels": EMG_CHANNELS,
                "imu_sampling_rate": IMU_SAMPLING_RATE,
                "baudrate": BAUDRATE,
            },
        }
        paths.config.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        recorder = Hdf5Recorder(on_error=self.recorder_error.emit)
        try:
            recorder.start(paths.hdf5, metadata)
        except Exception:
            self._remove_session_log()
            raise
        self.participant, self.info, self.protocol = participant, info, protocol
        self.paths, self.recorder = paths, recorder
        self.last_aligned_path = None
        self.last_readiness = None
        self.donning_id, self.stage_id, self._event_id = 1, info.stage_id, 0
        self._automatic_stage_name = info.stage_name
        self.trials = TrialManager()
        self.acquisition.begin_recording(recorder)
        self._active = True
        LOGGER.info("session start: %s / %s", participant.participant_id, info.session_id)
        self._event(EventType.SESSION_START)
        self._event(EventType.DONNING_START)
        self._event(EventType.STAGE_START, note=info.stage_name)
        self.prompt.prepare(protocol)
        self.prompt.start()
        self.session_started.emit(paths)
        return paths

    def stop(self) -> Path | None:
        if not self._active:
            return self.paths.hdf5 if self.paths else None
        self.prompt.stop()
        if self.trials.current is not None:
            self.trials.mark_bad("operator_reject", "session stopped before trial completed")
            self._event(EventType.TRIAL_END, label=self.trials.current.label,
                        trial_id=self.trials.current.trial_id, note="session stopped")
            self.recorder.enqueue_trial(self.trials.end(self.acquisition.current_sample_index))
        self._event(EventType.STAGE_END)
        self._event(EventType.DONNING_END)
        self._event(EventType.SESSION_END)
        assert self.recorder is not None
        summary = self.acquisition.recording_summary()
        if self.info is not None:
            summary["fatigue_after"] = self.info.fatigue_after
        self.recorder.update_metadata(summary)
        self.acquisition.end_recording()
        self.recorder.stop()
        path = self.paths.hdf5 if self.paths else None
        if path is not None and self.paths is not None and self._uses_meta_discrete_labels():
            try:
                self.last_aligned_path = export_meta_aligned(
                    path, self.paths.aligned_hdf5)
            except Exception:
                # The immutable acquisition is already safely closed. Alignment
                # failure must never make the raw recording appear lost.
                LOGGER.exception("offline gesture alignment/export failed")
                self.status_changed.emit("ALIGNMENT_EXPORT_FAILED")
        if path is not None and self.paths is not None:
            try:
                self.last_readiness = generate_session_readiness(
                    path, self.paths.readiness_manifest)
                self.status_changed.emit(
                    "COLLECTION_READY" if self.last_readiness["status"] == "passed"
                    else "COLLECTION_GATE_FAILED")
            except Exception:
                LOGGER.exception("session collection readiness generation failed")
                self.status_changed.emit("COLLECTION_GATE_ERROR")
        self._active = False
        LOGGER.info("session stop: %s", path)
        self._remove_session_log()
        self.session_stopped.emit(path)
        return path

    def _uses_meta_discrete_labels(self) -> bool:
        return SAMPLING_RATE == 2000 and bool(self.protocol) and set(self.protocol.labels) == {
            "thumb_tap", "thumb_swipe_left", "thumb_swipe_right",
            "thumb_swipe_up", "thumb_swipe_down", "index_hold", "middle_hold",
        }

    def new_donning(self) -> None:
        self._require_active()
        self._event(EventType.DONNING_END)
        self.donning_id += 1
        self._event(EventType.DONNING_START)

    def new_stage(self, name: str = "") -> None:
        self._require_active()
        self._event(EventType.STAGE_END)
        self.stage_id += 1
        self._event(EventType.STAGE_START, note=name)
        self._automatic_stage_name = name

    def mark_bad(self, reason: str, note: str = "") -> None:
        trial = self.trials.mark_bad(reason, note)
        self._event(EventType.BAD_TRIAL, label=trial.label, trial_id=trial.trial_id,
                    note=json.dumps({"reject_reason": reason, "note": note}, ensure_ascii=False))

    def manual_mark(self, note: str = "") -> None:
        self._require_active()
        trial = self.trials.current
        self._event(EventType.MANUAL_MARK, label=trial.label if trial else "",
                    trial_id=trial.trial_id if trial else -1, note=note)

    def device_disconnected(self) -> None:
        if self._active:
            self._event(EventType.DEVICE_DISCONNECTED)
            self._invalidate_current_once("device_problem", "device disconnected during trial")

    def device_reconnected(self) -> None:
        if self._active:
            self._event(EventType.DEVICE_RECONNECTED)

    def packet_loss(self, lost: int, previous: int, current: int) -> None:
        if self._active:
            note = json.dumps({"lost_count": lost, "previous_seq": previous,
                               "current_seq": current})
            self._event(EventType.PACKET_LOSS, note=note)
            if lost >= 2:
                self._invalidate_current_once(
                    "packet_loss", f"lost {lost} frames before sequence {current}")

    def quality_alert(self, reason: str) -> None:
        if not self._active:
            return
        self._event(EventType.QUALITY_ALERT, note=reason)
        self._invalidate_current_once("signal_quality", reason)
        if self.trials.current is not None:
            self.prompt.repeat()

    def record_fatigue(self, block_index: int, score: int, note: str = "") -> None:
        self._require_active()
        if score not in range(0, 11):
            raise ValueError("疲劳/不适评分必须为 0–10")
        self._event(EventType.FATIGUE_REPORT, note=json.dumps({
            "block_index": block_index, "score": score, "note": note,
        }, ensure_ascii=False))

    def update_fatigue_after(self, score: int) -> None:
        self._require_active()
        if score not in range(0, 11):
            raise ValueError("采集后疲劳评分必须为 0–10")
        assert self.info is not None and self.recorder is not None
        self.info.fatigue_after = score
        self.recorder.update_metadata({"fatigue_after": score})

    def _invalidate_current_once(self, reason: str, note: str) -> None:
        trial = self.trials.current
        if trial is None or not trial.valid:
            return
        self.trials.mark_bad(reason, note)
        self._event(EventType.BAD_TRIAL, label=trial.label, trial_id=trial.trial_id,
                    note=json.dumps({"reject_reason": reason, "note": note}, ensure_ascii=False))

    def _wire_prompt(self) -> None:
        self.prompt.trial_started.connect(self._trial_started)
        self.prompt.rest_started.connect(self._rest_started)
        self.prompt.rest_ended.connect(self._rest_ended)
        self.prompt.prompt_started.connect(self._prompt_started)
        self.prompt.prompt_ended.connect(self._prompt_ended)
        self.prompt.trial_ended.connect(self._trial_ended)
        self.prompt.trial_skipped.connect(self._trial_skipped)
        self.prompt.trial_repeat_requested.connect(self._trial_repeat_requested)
        self.prompt.gesture_cued.connect(self._gesture_cued)
        self.prompt.block_break_started.connect(self._block_break_started)
        self.prompt.block_break_ended.connect(self._block_break_ended)
        self.prompt.finished.connect(lambda: self.status_changed.emit("PROTOCOL_FINISHED"))

    def _trial_started(self, trial_id: int, label: str) -> None:
        assert self.protocol is not None
        null_kind = self.protocol.null_kind(label)
        stage_name = None
        if null_kind == "timed":
            stage_name = "null_timed_snap_flick"
        elif null_kind == "continuous":
            stage_name = label
        if stage_name is not None and stage_name != self._automatic_stage_name:
            self._event(EventType.STAGE_END)
            self.stage_id += 1
            self._event(EventType.STAGE_START, note=stage_name)
            self._automatic_stage_name = stage_name
        index = self.acquisition.current_sample_index
        self.acquisition.reset_trial_quality_window()
        event_uid = self._trial_event_uid(trial_id)
        self.trials.start(trial_id, label, self.stage_id, self.donning_id, index,
                          self.prompt.current_onset_offset_ms,
                          trial_kind=self.prompt.current_trial_kind,
                          block_index=self.prompt.current_block_index,
                          attempt=self.prompt.current_attempt,
                          rerecord_of_trial_id=self.prompt.current_rerecord_of_trial_id,
                          event_uid=event_uid)
        self._event(EventType.TRIAL_START, label=label, trial_id=trial_id)
        if self.prompt.current_trial_kind == "calibration":
            self._event(EventType.CALIBRATION_BLOCK_START, label=label, trial_id=trial_id)

    def _rest_started(self, trial_id: int, label: str) -> None:
        self.trials.set_index("rest_start_sample", self.acquisition.current_sample_index)
        self._event(EventType.REST_START, label=label, trial_id=trial_id)

    def _rest_ended(self, trial_id: int, label: str) -> None:
        self._event(EventType.REST_END, label=label, trial_id=trial_id)

    def _prompt_started(self, trial_id: int, label: str) -> None:
        self.trials.set_index("prompt_start_sample", self.acquisition.current_sample_index)
        self._event(EventType.PROMPT_START, label=label, trial_id=trial_id)
        LOGGER.info("prompt start: trial=%s label=%s", trial_id, label)

    def _prompt_ended(self, trial_id: int, label: str) -> None:
        index = self.acquisition.current_sample_index
        self.trials.set_index("prompt_end_sample", index)
        current = self.trials.current
        if current is not None and current.stable_end_sample < 0:
            guard = round((self.protocol.transition_guard_ms if self.protocol else 0)
                          * SAMPLING_RATE / 1000)
            self.trials.set_index("stable_end_sample", max(current.stable_start_sample, index - guard))
        self._event(EventType.PROMPT_END, label=label, trial_id=trial_id)

    def _trial_ended(self, trial_id: int, label: str) -> None:
        self._event(EventType.TRIAL_END, label=label, trial_id=trial_id)
        if self.trials.current is not None and self.trials.current.trial_kind == "calibration":
            self._event(EventType.CALIBRATION_BLOCK_END, label=label, trial_id=trial_id)
            if not self.trials.current.valid and not self.prompt._repeat_after_current:
                self.prompt.repeat()
        assert self.recorder is not None
        self.recorder.enqueue_trial(self.trials.end(self.acquisition.current_sample_index))

    def _trial_skipped(self, trial_id: int, label: str) -> None:
        trial = self.trials.mark_bad("operator_reject", "trial skipped")
        self._event(EventType.BAD_TRIAL, label=label, trial_id=trial_id,
                    note='{"reject_reason":"operator_reject","note":"trial skipped"}')

    def _trial_repeat_requested(self, trial_id: int, label: str) -> None:
        trial = self.trials.current
        if trial is not None and trial.valid:
            self.trials.mark_bad("rerecorded", "operator requested immediate rerecord")
        self._event(EventType.RERECORD_REQUESTED, label=label, trial_id=trial_id,
                    note="original retained as invalid; replacement appended")

    def _block_break_started(self, block: int, total: int) -> None:
        self._event(EventType.BLOCK_BREAK_START,
                    note=json.dumps({"block": block, "total": total}))

    def _block_break_ended(self, block: int, total: int) -> None:
        self._event(EventType.BLOCK_BREAK_END,
                    note=json.dumps({"block": block, "total": total}))

    def _gesture_cued(self, trial_id: int, name: str) -> None:
        """Record sparse Meta-style cues; cue time is not biological onset."""
        mapped = self._meta_gesture_name(name)
        index = self.acquisition.current_sample_index
        emitted_ns = time.monotonic_ns()
        assert self.recorder is not None
        current = self.trials.current
        event_uid = current.event_uid if current is not None else ""
        if current is not None:
            if name == "hand:release":
                self.trials.set_index("release_prompt_sample", index)
                self.trials.set_index("stable_end_sample", index)
            else:
                guard = round((self.protocol.transition_guard_ms if self.protocol else 0)
                              * SAMPLING_RATE / 1000)
                candidate = index + guard
                if current.stable_start_sample < candidate:
                    self.trials.set_index("stable_start_sample", candidate)
        self.recorder.enqueue_cue_event(CueEvent(
            mapped, index, index / SAMPLING_RATE, trial_id, self.stage_id,
            self.prompt.scheduled_cue_monotonic_ns(trial_id, name), emitted_ns, event_uid))

    def _trial_event_uid(self, trial_id: int) -> str:
        participant = self.participant.participant_id if self.participant else "unknown"
        session = self.info.session_id if self.info else "unknown"
        return f"{participant}:{session}:trial:{trial_id}:attempt:{self.prompt.current_attempt}"

    def _meta_gesture_name(self, name: str) -> str:
        direct = {
            "thumb_tap": "thumb_click", "thumb_swipe_up": "thumb_up",
            "thumb_swipe_down": "thumb_down", "index_press": "index_press",
            "index_release": "index_release", "middle_press": "middle_press",
            "middle_release": "middle_release",
        }
        if name in direct:
            return direct[name]
        # Meta's in/out labels are anatomical. For a right hand, screen-left is
        # toward the palm; the relationship is mirrored for a left hand.
        hand = self.participant.dominant_hand if self.participant else ""
        if name == "thumb_swipe_left":
            return "thumb_out" if hand == "left" else "thumb_in"
        if name == "thumb_swipe_right":
            return "thumb_in" if hand == "left" else "thumb_out"
        return name

    def _event(self, event_type: EventType, label: str = "", trial_id: int = -1,
               note: str = "") -> ExperimentEvent:
        index = self.acquisition.current_sample_index
        event = ExperimentEvent(self._event_id, index, index / SAMPLING_RATE,
            time.monotonic_ns(), event_type, label, trial_id,
            self.stage_id, self.donning_id, note)
        self._event_id += 1
        assert self.recorder is not None
        self.recorder.enqueue_event(event)
        self.event_created.emit(event)
        return event

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("实验场次尚未启动")

    def _install_session_log(self, path: Path) -> None:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logging.getLogger().addHandler(handler)
        self._log_handler = handler

    def _remove_session_log(self) -> None:
        if self._log_handler:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None
