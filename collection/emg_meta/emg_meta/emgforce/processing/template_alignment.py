from __future__ import annotations

"""Session-level forced alignment for prompted discrete sEMG gestures.

This module follows the public method in Kaifosh & Reardon (2025): prompted
times initialize gesture-specific session templates, template estimation and
event-time inference alternate, and nearby events are constrained jointly.
Meta's private protocol priors and global templates were not released, so the
constants and session-bootstrap recentering below are explicit local choices.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from mpf_tds.features import MPFConfig, MultiBandMatrixPowerFeatures

from .training_preprocessing import filter_training_emg


FRAME_STRIDE_SEC = 0.020
SEARCH_START_SEC = 0.080
SEARCH_END_SEC = 1.200
INITIAL_DELAY_SEC = 0.450
TEMPLATE_BEFORE_SEC = 0.200
TEMPLATE_AFTER_SEC = 0.200
MIN_TEMPLATE_EVENTS = 3
MIN_TEMPLATE_CORRELATION = 0.25
MIN_CORRELATION_MARGIN = 0.05
MAX_ITERATIONS = 8
CONVERGENCE_SEC = 0.005
BEAM_WIDTH = 64
MIN_EVENT_GAP_SEC = 0.040
RERP_RIDGE = 1e-3
MAX_RECENTER_FRAMES = 5
METHOD_NAME = "session_rerp_template_beam_v3"
FEATURE_NAME = "matrix_log_mpf_8ch_192d_v2"
RECENTER_NAME = "optional_global_template_time_axis_correlation_v1"


@dataclass(frozen=True, slots=True)
class TemplateAlignedEvent:
    name: str
    trial_id: int
    stage_id: int
    cue_sample_index: int
    session_aligned_sample_index: int
    global_recentered_sample_index: int
    aligned_sample_index: int
    offset_ms: float
    template_correlation: float
    second_best_correlation: float
    confidence: float
    search_boundary_hit: bool
    exclusion_reason: str
    method: str
    included: bool = False


@dataclass(frozen=True, slots=True)
class TemplateAlignmentResult:
    events: list[TemplateAlignedEvent]
    frame_samples: np.ndarray
    feature_timeseries: np.ndarray
    gesture_templates: dict[str, np.ndarray]
    template_offsets_frames: np.ndarray
    match_offsets_ms: np.ndarray
    match_scores: np.ndarray
    iterations: int
    converged: bool
    global_recenter_offsets_frames: dict[str, int]
    global_template_applied: bool


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def template_quality_reasons(*, correlation: float, second_best: float,
                             confidence: float, boundary_hit: bool,
                             minimum_confidence: float) -> list[str]:
    reasons: list[str] = []
    if boundary_hit:
        reasons.append("search_boundary_hit")
    if confidence < minimum_confidence:
        reasons.append("low_confidence")
    if correlation < MIN_TEMPLATE_CORRELATION:
        reasons.append("low_template_correlation")
    if correlation - second_best < MIN_CORRELATION_MARGIN:
        reasons.append("ambiguous_template_peak")
    return reasons


def mpf_features(raw: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the official-equivalent 50 Hz matrix-log MPF representation.

    Alignment and MPF-TDS training now share the same feature implementation.
    The frame sample is the last input sample available to that causal MPF
    frame, matching ``mpf_tds.data.load_split``.
    """
    signal = np.asarray(raw)
    if signal.ndim != 2 or signal.shape[1] != 8:
        raise ValueError(f"MPF 对齐输入必须是 [samples,8]，实际为 {signal.shape}")
    rate = int(round(sample_rate))
    if not np.isclose(sample_rate, rate) or rate != 2000:
        raise ValueError("MPF 对齐目前要求 2000 Hz 采样率")
    filtered, _ = filter_training_emg(signal, sample_rate)
    config = MPFConfig(sample_rate_hz=rate, channels=signal.shape[1])
    extractor = MultiBandMatrixPowerFeatures(config).eval()
    tensor = torch.from_numpy(filtered.astype(np.float32).T[None])
    with torch.inference_mode():
        output = extractor(tensor)[0].cpu().numpy().astype(np.float32, copy=False)
    frame_samples = config.left_context_samples + np.arange(
        len(output), dtype=np.int64) * config.output_stride_samples
    return output, frame_samples


# Compatibility alias for callers that imported the old local approximation.
mpf_like_features = mpf_features


def _snippet(features: np.ndarray, center: int, offsets: np.ndarray) -> np.ndarray | None:
    indices = center + offsets
    if len(indices) == 0 or indices[0] < 0 or indices[-1] >= len(features):
        return None
    return features[indices]


def _estimate_templates(features: np.ndarray, names: list[str], centers: np.ndarray,
                         offsets: np.ndarray) -> dict[str, np.ndarray]:
    """Estimate overlapping gesture responses with an rERP design matrix."""
    if not len(features) or not len(offsets):
        return {}
    supported = []
    for name in sorted(set(names)):
        complete = sum(
            event_name == name and _snippet(features, int(center), offsets) is not None
            for event_name, center in zip(names, centers)
        )
        if complete >= MIN_TEMPLATE_EVENTS:
            supported.append(name)
    if not supported:
        return {}

    class_index = {name: index for index, name in enumerate(supported)}
    width = len(offsets)
    # The intercept absorbs the session background. Each event adds one copy of
    # its class response, so simultaneous/nearby gestures are disentangled by
    # the joint regression rather than contaminating event-triggered averages.
    design = np.zeros((len(features), 1 + len(supported) * width), dtype=np.float32)
    design[:, 0] = 1.0
    for name, center in zip(names, centers):
        if name not in class_index:
            continue
        columns = 1 + class_index[name] * width + np.arange(width)
        rows = int(center) + offsets
        valid = (rows >= 0) & (rows < len(features))
        design[rows[valid], columns[valid]] += 1.0

    gram = design.T @ design
    penalty = np.eye(gram.shape[0], dtype=np.float32) * RERP_RIDGE
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        gram.astype(np.float64) + penalty,
        (design.T @ features).astype(np.float64),
    )
    return {
        name: coefficients[
            1 + index * width:1 + (index + 1) * width
        ].astype(np.float32)
        for name, index in class_index.items()
    }


def _global_recenter_offsets(
    templates: dict[str, np.ndarray],
    global_templates: dict[str, np.ndarray] | None,
) -> dict[str, int]:
    """Find event-time shifts by correlating templates only along time."""
    if not global_templates:
        return {name: 0 for name in templates}
    shifts: dict[str, int] = {}
    for name, session in templates.items():
        reference = global_templates.get(name)
        if reference is None or reference.shape != session.shape:
            shifts[name] = 0
            continue
        best_score = -np.inf
        best_shift = 0
        for shift in range(-MAX_RECENTER_FRAMES, MAX_RECENTER_FRAMES + 1):
            # Shifting an event by +s changes its template to phi(t+s).
            if shift >= 0:
                left, right = session[shift:], reference[:len(reference) - shift]
            else:
                left, right = session[:len(session) + shift], reference[-shift:]
            if len(left) < 3:
                continue
            score = _cosine_score(left, right)
            if score > best_score:
                best_score, best_shift = score, shift
        shifts[name] = best_shift
    return shifts


def _cosine_score(template: np.ndarray, candidate: np.ndarray) -> float:
    # Remove a per-feature temporal baseline so stationary mains/background
    # power cannot dominate the template match.
    left = template - np.mean(template, axis=0, keepdims=True)
    right = candidate - np.mean(candidate, axis=0, keepdims=True)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.sum(left * right) / denominator) if denominator > 1e-12 else -1.0


def _score_candidates(features: np.ndarray, names: list[str], cue_frames: np.ndarray,
                      templates: dict[str, np.ndarray], template_offsets: np.ndarray,
                      candidate_offsets: np.ndarray) -> np.ndarray:
    scores = np.full((len(names), len(candidate_offsets)), np.nan, dtype=np.float32)
    for event_index, (name, cue) in enumerate(zip(names, cue_frames)):
        template = templates.get(name)
        if template is None:
            continue
        for candidate_index, delay in enumerate(candidate_offsets):
            candidate = _snippet(features, int(cue + delay), template_offsets)
            if candidate is not None:
                scores[event_index, candidate_index] = _cosine_score(template, candidate)
    return scores


def _overlap_groups(cue_frames: np.ndarray, candidate_offsets: np.ndarray) -> list[range]:
    if not len(cue_frames):
        return []
    groups: list[range] = []
    start = 0
    current_end = int(cue_frames[0] + candidate_offsets[-1])
    for index in range(1, len(cue_frames)):
        next_start = int(cue_frames[index] + candidate_offsets[0])
        if next_start > current_end:
            groups.append(range(start, index)); start = index
        current_end = max(current_end, int(cue_frames[index] + candidate_offsets[-1]))
    groups.append(range(start, len(cue_frames)))
    return groups


def _beam_align(cue_frames: np.ndarray, scores: np.ndarray,
                candidate_offsets: np.ndarray, minimum_gap_frames: int) -> np.ndarray:
    selected = np.zeros(len(cue_frames), dtype=np.int64)
    for group in _overlap_groups(cue_frames, candidate_offsets):
        states: list[tuple[float, int, tuple[int, ...]]] = [(0.0, -10**12, ())]
        for event_index in group:
            valid = np.flatnonzero(np.isfinite(scores[event_index]))
            if not len(valid):
                fallback = int(np.argmin(np.abs(candidate_offsets - np.median(candidate_offsets))))
                valid = np.asarray([fallback])
            candidates: list[tuple[float, int, tuple[int, ...]]] = []
            for total, last_time, path in states:
                for candidate_index in valid:
                    event_time = int(cue_frames[event_index] + candidate_offsets[candidate_index])
                    if event_time < last_time + minimum_gap_frames:
                        continue
                    score = float(scores[event_index, candidate_index]) \
                        if np.isfinite(scores[event_index, candidate_index]) else -1.0
                    candidates.append((total + score, event_time,
                                       path + (int(candidate_index),)))
            if not candidates:
                # Protocol ordering is more important than a failed local fit.
                candidates = [(total - 1.0, max(last_time + minimum_gap_frames,
                               int(cue_frames[event_index] + candidate_offsets[0])),
                               path + (0,)) for total, last_time, path in states]
            candidates.sort(key=lambda item: item[0], reverse=True)
            states = candidates[:BEAM_WIDTH]
        best_path = states[0][2]
        for event_index, candidate_index in zip(group, best_path):
            selected[event_index] = candidate_index
    return selected


def align_session_templates(raw: np.ndarray, cues: np.ndarray, trials: np.ndarray,
                            sample_rate: float,
                            minimum_confidence: float = 0.20,
                            global_templates: dict[str, np.ndarray] | None = None,
                            ) -> TemplateAlignmentResult:
    """Jointly align all Meta gesture cues in one recording session."""
    features, frame_samples = mpf_features(raw, sample_rate)
    names: list[str] = [_text(row["name"]) for row in cues]
    cue_samples = np.asarray([int(row["sample_index"]) for row in cues], dtype=np.int64)
    cue_frames = np.searchsorted(frame_samples, cue_samples).astype(np.int64)
    template_offsets = np.arange(
        -round(TEMPLATE_BEFORE_SEC / FRAME_STRIDE_SEC),
        round(TEMPLATE_AFTER_SEC / FRAME_STRIDE_SEC) + 1, dtype=np.int64)
    candidate_offsets = np.arange(
        round(SEARCH_START_SEC / FRAME_STRIDE_SEC),
        round(SEARCH_END_SEC / FRAME_STRIDE_SEC) + 1, dtype=np.int64)
    centers = cue_frames + round(INITIAL_DELAY_SEC / FRAME_STRIDE_SEC)
    converged = False
    scores = np.full((len(cues), len(candidate_offsets)), np.nan, dtype=np.float32)
    templates: dict[str, np.ndarray] = {}
    iterations = 0
    for iterations in range(1, MAX_ITERATIONS + 1):
        templates = _estimate_templates(features, names, centers, template_offsets)
        scores = _score_candidates(
            features, names, cue_frames, templates, template_offsets, candidate_offsets)
        chosen = _beam_align(
            cue_frames, scores, candidate_offsets,
            max(1, round(MIN_EVENT_GAP_SEC / FRAME_STRIDE_SEC)))
        updated = cue_frames + candidate_offsets[chosen]
        if np.max(np.abs(updated - centers), initial=0) * FRAME_STRIDE_SEC <= CONVERGENCE_SEC:
            centers = updated; converged = True; break
        centers = updated

    # Stabilize the final session coordinate before optional global recentering.
    for _ in range(2):
        templates = _estimate_templates(features, names, centers, template_offsets)
        scores = _score_candidates(
            features, names, cue_frames, templates, template_offsets, candidate_offsets)
        chosen = _beam_align(
            cue_frames, scores, candidate_offsets,
            max(1, round(MIN_EVENT_GAP_SEC / FRAME_STRIDE_SEC)))
        updated = cue_frames + candidate_offsets[chosen]
        if np.array_equal(updated, centers):
            converged = True
            break
        centers = updated
    # Recompute once more if the final search moved an event, so exported
    # scores are exactly the scores used to place the green review line.
    templates = _estimate_templates(features, names, centers, template_offsets)
    scores = _score_candidates(
        features, names, cue_frames, templates, template_offsets, candidate_offsets)
    chosen = _beam_align(
        cue_frames, scores, candidate_offsets,
        max(1, round(MIN_EVENT_GAP_SEC / FRAME_STRIDE_SEC)))
    final_centers = cue_frames + candidate_offsets[chosen]
    converged = converged or np.array_equal(final_centers, centers)
    centers = final_centers
    recenter_offsets = _global_recenter_offsets(templates, global_templates)
    global_template_applied = bool(global_templates) and any(
        name in global_templates and global_templates[name].shape == template.shape
        for name, template in templates.items())

    valid_trials = {int(row["trial_id"]) for row in trials if bool(row["valid"])}
    provisional: list[TemplateAlignedEvent] = []
    for index, row in enumerate(cues):
        row_scores = scores[index]
        finite = np.flatnonzero(np.isfinite(row_scores))
        selected_index = int(chosen[index])
        best = float(row_scores[selected_index]) if np.isfinite(row_scores[selected_index]) else -1.0
        separated = [candidate for candidate in finite
                     if abs(int(candidate) - selected_index) >= 3]
        second = max((float(row_scores[candidate]) for candidate in separated), default=-1.0)
        # Confidence combines absolute template agreement and uniqueness of the
        # match.  Boundary hits are never auto-accepted.
        agreement = np.clip((best - 0.05) / 0.45, 0.0, 1.0)
        uniqueness = np.clip((best - second) / 0.12, 0.0, 1.0)
        confidence = float(0.7 * agreement + 0.3 * uniqueness)
        boundary = selected_index in {0, len(candidate_offsets) - 1}
        aligned_sample = int(np.clip(
            frame_samples[int(np.clip(centers[index], 0, len(frame_samples) - 1))],
            0, max(0, len(raw) - 1)))
        recentered_frame = int(np.clip(
            centers[index] + recenter_offsets.get(names[index], 0),
            0, len(frame_samples) - 1))
        recentered_sample = int(np.clip(
            frame_samples[recentered_frame], 0, max(0, len(raw) - 1)))
        cue_sample = int(row["sample_index"])
        reasons = template_quality_reasons(
            correlation=best, second_best=second, confidence=confidence,
            boundary_hit=boundary, minimum_confidence=minimum_confidence)
        provisional.append(TemplateAlignedEvent(
            name=names[index], trial_id=int(row["trial_id"]),
            stage_id=int(row["stage_id"]), cue_sample_index=cue_sample,
            session_aligned_sample_index=aligned_sample,
            global_recentered_sample_index=recentered_sample,
            aligned_sample_index=recentered_sample,
            offset_ms=(recentered_sample - cue_sample) * 1000.0 / sample_rate,
            template_correlation=best, second_best_correlation=second,
            confidence=confidence, search_boundary_hit=boundary,
            exclusion_reason=",".join(reasons), method=METHOD_NAME,
        ))
    trial_ok: dict[int, bool] = {}
    for event in provisional:
        trial_ok[event.trial_id] = trial_ok.get(event.trial_id, True) and not bool(
            event.exclusion_reason)
    events = [TemplateAlignedEvent(
        **{field: getattr(event, field) for field in (
            "name", "trial_id", "stage_id", "cue_sample_index",
            "session_aligned_sample_index", "global_recentered_sample_index",
            "aligned_sample_index", "offset_ms", "template_correlation",
            "second_best_correlation", "confidence", "search_boundary_hit", "method")},
        included=event.trial_id in valid_trials and trial_ok.get(event.trial_id, False),
        exclusion_reason=(
            event.exclusion_reason
            if event.exclusion_reason else
            ("paired_event_failed" if not trial_ok.get(event.trial_id, False) else
             ("invalid_trial" if event.trial_id not in valid_trials else ""))
        ),
    ) for event in provisional]
    match_offsets_ms = np.empty((len(cues), len(candidate_offsets)), dtype=np.float64)
    for event_index, (cue_frame, cue_sample) in enumerate(zip(cue_frames, cue_samples)):
        candidate_frames = np.clip(
            cue_frame + candidate_offsets, 0, max(0, len(frame_samples) - 1))
        match_offsets_ms[event_index] = (
            frame_samples[candidate_frames] - cue_sample) * 1000.0 / sample_rate
    return TemplateAlignmentResult(
        events=events, frame_samples=frame_samples, feature_timeseries=features,
        gesture_templates=templates, template_offsets_frames=template_offsets,
        match_offsets_ms=match_offsets_ms,
        match_scores=scores, iterations=iterations, converged=converged,
        global_recenter_offsets_frames=recenter_offsets,
        global_template_applied=global_template_applied,
    )
