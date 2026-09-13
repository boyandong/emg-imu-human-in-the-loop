from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class TrialRef:
    dataset_id: str
    subject_id: str
    session_id: str
    trial_id: str
    task_label: int
    duration_seconds: float
    sensor_view: str = "default"

    @property
    def trial_key(self) -> tuple[str, str, str, str]:
        return self.dataset_id, self.subject_id, self.session_id, self.trial_id


@dataclass(frozen=True, slots=True)
class PersonalizationSplit:
    dataset_id: str
    target_subject: str
    calibration_session: str
    gesture_budget_per_class: int
    rest_seconds: float
    seed: int
    source_training: tuple[TrialRef, ...]
    calibration: tuple[TrialRef, ...]
    same_session_evaluation: tuple[TrialRef, ...]
    cross_session_evaluation: tuple[TrialRef, ...]

    def assert_disjoint(self) -> None:
        calibration = {row.trial_key for row in self.calibration}
        same = {row.trial_key for row in self.same_session_evaluation}
        cross = {row.trial_key for row in self.cross_session_evaluation}
        source = {row.trial_key for row in self.source_training}
        intersections = {
            "calibration/same_session": calibration & same,
            "calibration/cross_session": calibration & cross,
            "calibration/source": calibration & source,
            "same_session/source": same & source,
            "cross_session/source": cross & source,
        }
        failures = {name: values for name, values in intersections.items() if values}
        if failures:
            raise ValueError(f"trial leakage between split roles: {failures}")


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def _unique_trials(records: Sequence[TrialRef]) -> dict[tuple[str, str, str, str], TrialRef]:
    unique: dict[tuple[str, str, str, str], TrialRef] = {}
    for row in records:
        if row.duration_seconds <= 0:
            raise ValueError(f"trial {row.trial_id!r} must have positive duration")
        previous = unique.get(row.trial_key)
        if previous is not None and (
            previous.task_label != row.task_label
            or not np.isclose(previous.duration_seconds, row.duration_seconds)
        ):
            raise ValueError(f"sensor views disagree about trial {row.trial_key}")
        unique.setdefault(row.trial_key, row)
    return unique


def _expand_views(records: Sequence[TrialRef], keys: set[tuple[str, str, str, str]]) -> tuple[TrialRef, ...]:
    return tuple(sorted(
        (row for row in records if row.trial_key in keys),
        key=lambda row: (*map(str, row.trial_key), row.sensor_view),
    ))


def build_personalization_split(
    records: Iterable[TrialRef],
    *,
    target_subject: str,
    calibration_session: str | None,
    active_task_labels: Sequence[int],
    neutral_task_label: int,
    gesture_budget_per_class: int,
    rest_seconds: float,
    seed: int,
) -> PersonalizationSplit:
    """Create whole-trial LOSO calibration and evaluation roles.

    Multiple sensor views of one physical trial always receive the same role.
    Gesture and rest budgets are selected only from the earliest requested target
    session.  All later target sessions remain untouched cross-session evaluation.
    """

    rows = tuple(records)
    if not rows:
        raise ValueError("records cannot be empty")
    datasets = {row.dataset_id for row in rows}
    if len(datasets) != 1:
        raise ValueError("one personalization split must contain exactly one dataset")
    if gesture_budget_per_class < 0 or rest_seconds < 0:
        raise ValueError("calibration budgets cannot be negative")
    labels = tuple(map(int, active_task_labels))
    if not labels or neutral_task_label in labels or len(labels) != len(set(labels)):
        raise ValueError("active_task_labels must be unique and exclude neutral_task_label")
    target_rows = tuple(row for row in rows if row.subject_id == target_subject)
    if not target_rows:
        raise ValueError(f"target subject {target_subject!r} is absent")
    sessions = sorted({row.session_id for row in target_rows}, key=_natural_key)
    selected_session = sessions[0] if calibration_session is None else calibration_session
    if selected_session not in sessions:
        raise ValueError(f"calibration session {selected_session!r} is absent for target subject")

    unique = _unique_trials(target_rows)
    candidates = [row for row in unique.values() if row.session_id == selected_session]
    rng = np.random.default_rng(seed)
    calibration_keys: set[tuple[str, str, str, str]] = set()
    for label in labels:
        available = sorted(
            (row for row in candidates if row.task_label == label),
            key=lambda row: _natural_key(row.trial_id),
        )
        if len(available) < gesture_budget_per_class:
            raise ValueError(
                f"subject {target_subject} session {selected_session} label {label} has "
                f"{len(available)} trials; needs {gesture_budget_per_class}"
            )
        if gesture_budget_per_class:
            indices = rng.choice(len(available), size=gesture_budget_per_class, replace=False)
            calibration_keys.update(available[int(index)].trial_key for index in indices)

    rest = sorted(
        (row for row in candidates if row.task_label == neutral_task_label),
        key=lambda row: _natural_key(row.trial_id),
    )
    if rest_seconds:
        order = rng.permutation(len(rest))
        accumulated = 0.0
        for index in order:
            row = rest[int(index)]
            calibration_keys.add(row.trial_key)
            accumulated += row.duration_seconds
            if accumulated >= rest_seconds:
                break
        if accumulated < rest_seconds:
            raise ValueError(
                f"only {accumulated:.3f}s neutral data available; needs {rest_seconds:.3f}s"
            )

    source_keys = {
        row.trial_key for row in rows if row.subject_id != target_subject
    }
    same_keys = {
        row.trial_key for row in target_rows
        if row.session_id == selected_session and row.trial_key not in calibration_keys
    }
    selected_index = sessions.index(selected_session)
    later_sessions = set(sessions[selected_index + 1:])
    cross_keys = {
        row.trial_key for row in target_rows if row.session_id in later_sessions
    }
    result = PersonalizationSplit(
        dataset_id=next(iter(datasets)),
        target_subject=target_subject,
        calibration_session=selected_session,
        gesture_budget_per_class=gesture_budget_per_class,
        rest_seconds=float(rest_seconds),
        seed=int(seed),
        source_training=_expand_views(rows, source_keys),
        calibration=_expand_views(rows, calibration_keys),
        same_session_evaluation=_expand_views(rows, same_keys),
        cross_session_evaluation=_expand_views(rows, cross_keys),
    )
    result.assert_disjoint()
    return result
