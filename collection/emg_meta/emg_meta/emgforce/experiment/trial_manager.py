from __future__ import annotations

from dataclasses import replace

from .models import TrialInfo


class TrialManager:
    """Owns trial indices and annotations; it never inspects EMG amplitude."""

    REJECT_REASONS = (
        "wrong_gesture", "no_gesture", "multiple_gestures",
        "participant_error", "device_problem", "operator_reject", "other",
        "rerecorded", "signal_quality", "packet_loss", "calibration_failed",
    )

    def __init__(self) -> None:
        self.trials: list[TrialInfo] = []
        self.current: TrialInfo | None = None

    def start(self, trial_id: int, label: str, stage_id: int,
              donning_id: int, sample_index: int,
              relative_onset_offset_ms: int = 0, *, trial_kind: str = "formal",
              block_index: int = 0, attempt: int = 1,
              rerecord_of_trial_id: int = -1, event_uid: str = "") -> TrialInfo:
        if self.current is not None:
            raise RuntimeError("已有尚未结束的试次")
        self.current = TrialInfo(trial_id, label, stage_id, donning_id,
                                 trial_start_sample=sample_index,
                                 relative_onset_offset_ms=relative_onset_offset_ms,
                                 trial_kind=trial_kind, block_index=block_index,
                                 attempt=attempt,
                                 rerecord_of_trial_id=rerecord_of_trial_id,
                                 event_uid=event_uid)
        return self.current

    def set_index(self, field: str, sample_index: int) -> None:
        if self.current is None or field not in {
            "rest_start_sample", "prompt_start_sample", "prompt_end_sample",
            "stable_start_sample", "stable_end_sample", "release_prompt_sample",
        }:
            raise RuntimeError("试次状态或索引字段无效")
        setattr(self.current, field, sample_index)

    def mark_bad(self, reason: str, note: str = "") -> TrialInfo:
        if self.current is None:
            raise RuntimeError("当前没有试次")
        if reason not in self.REJECT_REASONS:
            raise ValueError("无效 reject_reason")
        self.current.valid = False
        self.current.reject_reason = reason
        self.current.note = note
        self.current.discard_reason = reason
        self.current.completion_status = "invalid"
        return self.current

    def end(self, sample_index: int) -> TrialInfo:
        if self.current is None:
            raise RuntimeError("当前没有试次")
        self.current.trial_end_sample = sample_index
        if self.current.completion_status == "pending":
            self.current.completion_status = "completed" if self.current.valid else "invalid"
        completed = replace(self.current)
        self.trials.append(completed)
        self.current = None
        return completed
