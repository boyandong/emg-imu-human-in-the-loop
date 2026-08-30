from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from emgforce import __version__
from emgforce.config import BAUDRATE, EMG_CHANNELS, SAMPLING_RATE
from emgforce.controller import AcquisitionController
from emgforce.processing.meta_corpus import resolve_dataset_split
from emgforce.processing.meta_alignment import export_meta_aligned
from emgforce.storage.hdf5_recorder import Hdf5Recorder
from emgforce.storage.session_paths import SessionPaths, build_session_paths

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
        info.dataset_split = resolve_dataset_split(info.session_id, info.dataset_split)
        paths = build_session_paths(self.data_root, participant.participant_id, info.session_id)
        if paths.directory.exists():
            raise FileExistsError(f"实验场次目录已存在：{paths.directory}")
        paths.directory.mkdir(parents=True)
        self._install_session_log(paths.log)
        started = datetime.now().astimezone().isoformat()
        start_unix_time = time.time()
        metadata = {
            **asdict(participant), "session_id": info.session_id,
            "experiment_name": info.experiment_name,
            "sampling_rate": SAMPLING_RATE, "num_emg_channels": EMG_CHANNELS,
            "baudrate": BAUDRATE, "serial_port": info.serial_port,
            "software_version": __version__, "protocol_name": protocol.name,
            "dataset_split": info.dataset_split,
            "protocol_json": json.dumps(protocol.to_dict(), ensure_ascii=False),
            "start_datetime": started,
            "start_unix_time": start_unix_time,
            "task": "discrete_gestures",
            "posture_name": protocol.posture_name or "unspecified",
            "posture_instruction": protocol.posture_instruction,
        }
        config_payload = {
            "participant": asdict(participant), "session": asdict(info),
            "protocol": protocol.to_dict(), "automatic": {
                "start_datetime": started, "software_version": __version__,
                "sampling_rate": SAMPLING_RATE, "num_emg_channels": EMG_CHANNELS,
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
        self.acquisition.end_recording()
        assert self.recorder is not None
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
        self._active = False
        LOGGER.info("session stop: %s", path)
        self._remove_session_log()
        self.session_stopped.emit(path)
        return path

    def _uses_meta_discrete_labels(self) -> bool:
        return bool(self.protocol) and set(self.protocol.labels) == {
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

    def device_reconnected(self) -> None:
        if self._active:
            self._event(EventType.DEVICE_RECONNECTED)

    def packet_loss(self, lost: int, previous: int, current: int) -> None:
        if self._active:
            note = json.dumps({"lost_count": lost, "previous_seq": previous,
                               "current_seq": current})
            self._event(EventType.PACKET_LOSS, note=note)

    def _wire_prompt(self) -> None:
        self.prompt.trial_started.connect(self._trial_started)
        self.prompt.rest_started.connect(self._rest_started)
        self.prompt.rest_ended.connect(self._rest_ended)
        self.prompt.prompt_started.connect(self._prompt_started)
        self.prompt.prompt_ended.connect(self._prompt_ended)
        self.prompt.trial_ended.connect(self._trial_ended)
        self.prompt.trial_skipped.connect(self._trial_skipped)
        self.prompt.gesture_cued.connect(self._gesture_cued)
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
        self.trials.start(trial_id, label, self.stage_id, self.donning_id, index)
        self._event(EventType.TRIAL_START, label=label, trial_id=trial_id)

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
        self.trials.set_index("prompt_end_sample", self.acquisition.current_sample_index)
        self._event(EventType.PROMPT_END, label=label, trial_id=trial_id)

    def _trial_ended(self, trial_id: int, label: str) -> None:
        self._event(EventType.TRIAL_END, label=label, trial_id=trial_id)
        assert self.recorder is not None
        self.recorder.enqueue_trial(self.trials.end(self.acquisition.current_sample_index))

    def _trial_skipped(self, trial_id: int, label: str) -> None:
        trial = self.trials.mark_bad("operator_reject", "trial skipped")
        self._event(EventType.BAD_TRIAL, label=label, trial_id=trial_id,
                    note='{"reject_reason":"operator_reject","note":"trial skipped"}')

    def _gesture_cued(self, trial_id: int, name: str) -> None:
        """Record sparse Meta-style cues; cue time is not biological onset."""
        mapped = self._meta_gesture_name(name)
        index = self.acquisition.current_sample_index
        emitted_ns = time.monotonic_ns()
        assert self.recorder is not None
        self.recorder.enqueue_cue_event(CueEvent(
            mapped, index, index / SAMPLING_RATE, trial_id, self.stage_id,
            self.prompt.scheduled_deadline_monotonic_ns, emitted_ns))

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
