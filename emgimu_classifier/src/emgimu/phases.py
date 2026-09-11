from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .state import Phase


@dataclass(frozen=True, slots=True)
class PhaseUpdate:
    stable_label: int
    phase: Phase
    confidence: float
    changed: bool = False


class PhaseTracker:
    """Two-frame hysteresis with explicit onset, release and transition states."""

    def __init__(
        self,
        idle_label: IntEnum | int,
        unknown_label: IntEnum | int = -1,
        *,
        confirm_frames: int = 2,
        hold_after_frames: int = 3,
    ) -> None:
        if confirm_frames < 1 or hold_after_frames < 1:
            raise ValueError("phase frame counts must be positive")
        self.idle_label = int(idle_label)
        self.unknown_label = int(unknown_label)
        self.confirm_frames = int(confirm_frames)
        self.hold_after_frames = int(hold_after_frames)
        self.reset()

    def reset(self) -> None:
        self.stable = self.idle_label
        self.candidate = self.idle_label
        self.candidate_count = 0
        self.stable_frames = 0

    def update(self, label: IntEnum | int, confidence: float, *, valid: bool = True) -> PhaseUpdate:
        label = int(label)
        confidence = max(0.0, min(1.0, float(confidence)))
        if not valid or label == self.unknown_label:
            self.candidate = self.unknown_label
            self.candidate_count = 0
            self.stable_frames = 0
            return PhaseUpdate(self.unknown_label, Phase.UNKNOWN, confidence)

        if label != self.stable:
            if label != self.candidate:
                self.candidate = label
                self.candidate_count = 1
            else:
                self.candidate_count += 1

            if self.stable == self.idle_label and label != self.idle_label:
                pending_phase = Phase.ONSET
            elif self.stable != self.idle_label and label == self.idle_label:
                pending_phase = Phase.RELEASE
            else:
                pending_phase = Phase.TRANSITION

            if self.candidate_count < self.confirm_frames:
                return PhaseUpdate(self.stable, pending_phase, confidence)

            self.stable = label
            self.candidate_count = 0
            self.stable_frames = 0
            phase = Phase.IDLE if label == self.idle_label else Phase.ACTIVE
            return PhaseUpdate(self.stable, phase, confidence, changed=True)

        self.candidate = label
        self.candidate_count = 0
        self.stable_frames += 1
        if label == self.idle_label:
            phase = Phase.IDLE
        elif self.stable_frames >= self.hold_after_frames:
            phase = Phase.HOLD
        else:
            phase = Phase.ACTIVE
        return PhaseUpdate(self.stable, phase, confidence)


class OpenAwareGestureTracker(PhaseTracker):
    """Gesture tracker that preserves an observed Open pose through EMG decay."""

    def __init__(
        self,
        idle_label: IntEnum | int,
        open_label: IntEnum | int,
        unknown_label: IntEnum | int = -1,
        *,
        confirm_frames: int = 2,
        hold_after_frames: int = 3,
        open_release_frames: int = 8,
    ) -> None:
        super().__init__(
            idle_label, unknown_label,
            confirm_frames=confirm_frames, hold_after_frames=hold_after_frames,
        )
        if open_release_frames < confirm_frames:
            raise ValueError("open_release_frames must be at least confirm_frames")
        self.open_label = int(open_label)
        self.open_release_frames = int(open_release_frames)
        self.open_release_count = 0

    def reset(self) -> None:
        super().reset()
        self.open_release_count = 0

    def update(self, label: IntEnum | int, confidence: float, *, valid: bool = True) -> PhaseUpdate:
        label = int(label)
        if self.stable == self.open_label and valid and label == self.idle_label:
            self.open_release_count += 1
            if self.open_release_count < self.open_release_frames:
                self.stable_frames += 1
                return PhaseUpdate(self.open_label, Phase.HOLD, confidence)
            self.stable = self.idle_label
            self.candidate = self.idle_label
            self.candidate_count = 0
            self.stable_frames = 0
            self.open_release_count = 0
            return PhaseUpdate(self.idle_label, Phase.RELEASE, confidence, changed=True)
        if label != self.idle_label:
            self.open_release_count = 0
        return super().update(label, confidence, valid=valid)
