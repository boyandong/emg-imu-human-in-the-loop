from __future__ import annotations

import logging
import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .meta_corpus import refresh_corpus_entry, upsert_corpus_entry
from .global_alignment_template import load_global_template
from .template_alignment import (
    FEATURE_NAME, METHOD_NAME, MIN_CORRELATION_MARGIN,
    MIN_TEMPLATE_CORRELATION, RECENTER_NAME, TemplateAlignedEvent,
    align_session_templates, template_quality_reasons,
)
from .training_preprocessing import (
    HIGH_PASS_HZ, HIGH_PASS_ORDER, NOTCH_Q, PREPROCESSING_VERSION,
    TARGET_MEDIAN_ABS, preprocess_training_emg,
)


LOGGER = logging.getLogger(__name__)
META_GESTURE_NAMES = {
    "index_press", "index_release", "middle_press", "middle_release",
    "thumb_click", "thumb_down", "thumb_in", "thumb_out", "thumb_up",
}
RELEASE_EVENTS = {"index_release", "middle_release"}
MIN_ALIGNMENT_CONFIDENCE = 0.20
ACCEPTED_REVIEW_STATUSES = {"auto_accepted", "accepted", "manually_adjusted"}


@dataclass(frozen=True, slots=True)
class AlignedCue:
    name: str
    trial_id: int
    stage_id: int
    cue_sample_index: int
    aligned_sample_index: int
    offset_ms: float
    confidence: float
    method: str
    included: bool = False


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, min(int(width), len(values)))
    if width == 1:
        return values.copy()
    padded = np.pad(values, (width // 2, width - 1 - width // 2), mode="edge")
    cumulative = np.cumsum(np.r_[0.0, padded], dtype=np.float64)
    return (cumulative[width:] - cumulative[:-width]) / width


def _energy_envelope(emg: np.ndarray, sample_rate: float) -> np.ndarray:
    signal = np.asarray(emg, dtype=np.float64)
    # Remove slow electrode/ADC drift without changing the raw stored samples.
    baseline_width = max(3, round(0.100 * sample_rate))
    high_passed = np.empty_like(signal)
    for channel in range(signal.shape[1]):
        high_passed[:, channel] = signal[:, channel] - _smooth(
            signal[:, channel], baseline_width)
    power = np.sqrt(np.mean(high_passed * high_passed, axis=1))
    return _smooth(power, max(3, round(0.025 * sample_rate)))


def _first_sustained(mask: np.ndarray, minimum: int) -> int | None:
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run >= minimum:
            return index - run + 1
    return None


def align_cue(emg: np.ndarray, cue_offset: int, sample_rate: float,
              name: str) -> tuple[int, float, str]:
    """Estimate one event near its UI cue using a session-local energy change.

    This is a transparent first-pass aligner for well-separated prompts. It is
    deliberately labelled ``energy_*_v1`` rather than claiming to reproduce
    Meta's unpublished forced-alignment implementation.
    """
    envelope = _energy_envelope(emg, sample_rate)
    cue = int(np.clip(cue_offset, 0, len(envelope) - 1))
    noise_start = max(0, cue - round(0.45 * sample_rate))
    noise_end = max(noise_start + 1, cue - round(0.15 * sample_rate))
    search_end = min(len(envelope), cue + round(0.80 * sample_rate))
    if search_end <= cue + 1:
        return cue, 0.0, "insufficient_window"

    before = envelope[noise_start:noise_end]
    noise = max(float(np.median(np.abs(before - np.median(before)))), 1e-12)
    search = envelope[cue:search_end]
    minimum = max(1, round(0.020 * sample_rate))

    if name in RELEASE_EVENTS:
        active_start = max(0, cue - round(0.30 * sample_rate))
        active_end = max(active_start + 1, cue - round(0.03 * sample_rate))
        active = float(np.median(envelope[active_start:active_end]))
        late_start = min(len(envelope) - 1, cue + round(0.30 * sample_rate))
        late_end = min(len(envelope), cue + round(0.75 * sample_rate))
        rest = float(np.percentile(envelope[late_start:late_end], 30)) \
            if late_end > late_start else float(np.min(search))
        contrast = active - rest
        threshold = rest + 0.65 * max(contrast, 0.0)
        local = _first_sustained(search <= threshold, minimum)
        if local is None or contrast <= 0:
            derivative = _smooth(np.diff(search, prepend=search[0]), minimum)
            local = int(np.argmin(derivative))
            method = "energy_fall_fallback_v1"
        else:
            method = "energy_fall_v1"
        confidence = float(np.clip(contrast / (6.0 * noise), 0.0, 1.0))
    else:
        resting = float(np.median(before))
        high = float(np.percentile(search, 90))
        contrast = high - resting
        threshold = resting + max(4.0 * noise, 0.25 * max(contrast, 0.0))
        local = _first_sustained(search >= threshold, minimum)
        if local is None or contrast <= 0:
            derivative = _smooth(np.diff(search, prepend=search[0]), minimum)
            local = int(np.argmax(derivative))
            method = "energy_rise_fallback_v1"
        else:
            method = "energy_rise_v1"
        confidence = float(np.clip(contrast / (6.0 * noise), 0.0, 1.0))

    return cue + int(local), confidence, method


def _align_all(handle: h5py.File) -> list[AlignedCue]:
    raw = handle["streams/emg/raw"]
    cues = handle["cue_events"][:]
    trials = handle["trials"][:]
    rate = float(handle["meta"].attrs["sampling_rate"])
    valid_trials = {int(row["trial_id"]) for row in trials if bool(row["valid"])}
    provisional: list[AlignedCue] = []
    for cue in cues:
        name = _text(cue["name"])
        if name not in META_GESTURE_NAMES:
            continue
        cue_index = int(cue["sample_index"])
        start = max(0, cue_index - round(0.65 * rate))
        end = min(len(raw), cue_index + round(1.00 * rate))
        if end - start < round(0.30 * rate):
            aligned, confidence, method = cue_index, 0.0, "insufficient_window"
        else:
            local, confidence, method = align_cue(
                raw[start:end], cue_index - start, rate, name)
            aligned = start + local
        provisional.append(AlignedCue(
            name=name, trial_id=int(cue["trial_id"]), stage_id=int(cue["stage_id"]),
            cue_sample_index=cue_index, aligned_sample_index=aligned,
            offset_ms=(aligned - cue_index) * 1000.0 / rate,
            confidence=confidence, method=method,
        ))

    # A trial is exported only when every event belonging to it is trustworthy.
    trial_ok: dict[int, bool] = {}
    for event in provisional:
        trial_ok[event.trial_id] = (
            trial_ok.get(event.trial_id, True)
            and event.confidence >= MIN_ALIGNMENT_CONFIDENCE)
    return [AlignedCue(
        **{field: getattr(event, field) for field in (
            "name", "trial_id", "stage_id", "cue_sample_index",
            "aligned_sample_index", "offset_ms", "confidence", "method")},
        included=event.trial_id in valid_trials and trial_ok.get(event.trial_id, False),
    ) for event in provisional]


def _stage_frame(handle: h5py.File, start_unix: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    active: tuple[float, str] | None = None
    for event in handle["events"]:
        kind = _text(event["event_type"])
        when = start_unix + float(event["time_sec"])
        if kind == "STAGE_START":
            active = (when, _text(event["note"]) or "default")
        elif kind == "STAGE_END" and active is not None:
            rows.append({"start": active[0], "end": when, "name": active[1]})
            active = None
    return pd.DataFrame(rows, columns=["start", "end", "name"])


def export_meta_aligned(
    source: Path,
    target: Path | None = None,
    *,
    global_template_path: Path | None = None,
    register_corpus: bool = True,
) -> Path:
    """Create a Meta-readable v3 aligned file without modifying source."""
    source = Path(source)
    target = Path(target) if target else source.with_name("session_meta_aligned.hdf5")
    if target.exists():
        raise FileExistsError(f"对齐导出文件已存在：{target}")
    global_bundle = load_global_template(global_template_path) \
        if global_template_path is not None else None

    with h5py.File(source, "r") as src:
        if "cue_events" not in src:
            raise ValueError("原始文件缺少 cue_events，无法执行时间对齐")
        rate = float(src["meta"].attrs["sampling_rate"])
        start_unix = float(src["meta"].attrs["start_unix_time"])
        raw = src["streams/emg/raw"][:]
        sample_index = src["streams/emg/sample_index"][:]
        all_cues = src["cue_events"][:]
        target_cues = all_cues[np.asarray([
            _text(row["name"]) in META_GESTURE_NAMES for row in all_cues])]
        alignment = align_session_templates(raw, target_cues,
                                            src["trials"][:], rate,
                                            MIN_ALIGNMENT_CONFIDENCE,
                                            global_templates=(global_bundle.templates
                                                              if global_bundle else None))
        aligned = list(alignment.events)
        # Very small calibration/test recordings cannot form a session template.
        # Keep a transparent per-event fallback rather than inventing a template.
        for index, event in enumerate(aligned):
            if event.name in alignment.gesture_templates:
                continue
            cue_index = event.cue_sample_index
            start = max(0, cue_index - round(0.65 * rate))
            end = min(len(raw), cue_index + round(1.00 * rate))
            local, confidence, method = align_cue(
                raw[start:end], cue_index - start, rate, event.name)
            sample = start + local
            aligned[index] = TemplateAlignedEvent(
                name=event.name, trial_id=event.trial_id, stage_id=event.stage_id,
                cue_sample_index=cue_index, session_aligned_sample_index=sample,
                global_recentered_sample_index=sample, aligned_sample_index=sample,
                offset_ms=(sample - cue_index) * 1000.0 / rate,
                template_correlation=float("nan"),
                second_best_correlation=float("nan"), confidence=confidence,
                search_boundary_hit=False,
                exclusion_reason=("" if confidence >= MIN_ALIGNMENT_CONFIDENCE
                                  else "low_confidence"),
                method=f"{method}_small_session_fallback",
            )
        trial_ok: dict[int, bool] = {}
        for event in aligned:
            if event.method == METHOD_NAME:
                failed = template_quality_reasons(
                    correlation=event.template_correlation,
                    second_best=event.second_best_correlation,
                    confidence=event.confidence,
                    boundary_hit=event.search_boundary_hit,
                    minimum_confidence=MIN_ALIGNMENT_CONFIDENCE)
            else:
                failed = [] if event.confidence >= MIN_ALIGNMENT_CONFIDENCE \
                    else ["low_confidence"]
            trial_ok[event.trial_id] = trial_ok.get(event.trial_id, True) and not failed
        valid_trials = {int(row["trial_id"]) for row in src["trials"] if bool(row["valid"])}
        aligned = [replace(
            event,
            included=(event.trial_id in valid_trials
                      and trial_ok.get(event.trial_id, False)),
            exclusion_reason=(
                event.exclusion_reason if event.exclusion_reason else
                ("paired_event_failed" if not trial_ok.get(event.trial_id, False)
                 else ("invalid_trial" if event.trial_id not in valid_trials else ""))))
            for event in aligned]
        stages = _stage_frame(src, start_unix)
        metadata = {key: value for key, value in src["meta"].attrs.items()}

    included = [event for event in aligned if event.included]
    prompts = pd.DataFrame({
        "name": [event.name for event in included],
        "time": [start_unix + event.aligned_sample_index / rate for event in included],
    })
    training = preprocess_training_emg(raw, rate)
    data_dtype = np.dtype([("emg", "f4", (raw.shape[1],)), ("time", "f8")])
    data = np.empty(len(raw), dtype=data_dtype)
    data["emg"] = training.signal
    data["time"] = start_unix + sample_index / rate
    string = h5py.string_dtype("utf-8")
    alignment_dtype = np.dtype([
        ("name", string), ("trial_id", "i8"), ("stage_id", "i8"),
        ("cue_sample_index", "i8"), ("aligned_sample_index", "i8"),
        ("session_aligned_sample_index", "i8"),
        ("global_recentered_sample_index", "i8"),
        ("offset_ms", "f8"), ("template_correlation", "f8"),
        ("second_best_correlation", "f8"), ("confidence", "f8"),
        ("search_boundary_hit", "?"), ("exclusion_reason", string),
        ("method", string), ("included", "?"),
    ])
    alignment_rows = np.array([(
        event.name, event.trial_id, event.stage_id, event.cue_sample_index,
        event.aligned_sample_index, event.session_aligned_sample_index,
        event.global_recentered_sample_index, event.offset_ms,
        event.template_correlation, event.second_best_correlation,
        event.confidence, event.search_boundary_hit, event.exclusion_reason,
        event.method, event.included,
    ) for event in aligned], dtype=alignment_dtype)
    review_dtype = np.dtype([
        ("event_index", "i8"), ("reviewed_sample_index", "i8"),
        ("status", "S24"), ("note", "S256"), ("reviewed_at", "S40"),
    ])
    review_rows = np.array([(
        index, event.aligned_sample_index,
        b"auto_accepted" if event.included else b"auto_rejected", b"", b"",
    ) for index, event in enumerate(aligned)], dtype=review_dtype)

    with h5py.File(target, "x") as dst:
        dataset = dst.create_dataset("data", data=data, chunks=(min(2000, max(1, len(data))),))
        dataset.attrs["task"] = "discrete_gestures"
        dataset.attrs["sample_rate"] = rate
        dataset.attrs["num_emg_channels"] = raw.shape[1]
        dataset.attrs["emg_units"] = "normalized_device_counts"
        dataset.attrs["source_emg_units"] = "device_raw_counts"
        dataset.attrs["preprocessing_version"] = PREPROCESSING_VERSION
        dataset.attrs["highpass_hz"] = HIGH_PASS_HZ
        dataset.attrs["highpass_order"] = HIGH_PASS_ORDER
        dataset.attrs["zero_phase_filtering"] = True
        dataset.attrs["notch_frequencies_hz"] = np.asarray(
            training.notch_frequencies_hz, dtype=np.float64)
        dataset.attrs["notch_q"] = NOTCH_Q
        dataset.attrs["amplitude_normalization"] = "session_global_median_abs"
        dataset.attrs["normalization_target_median_abs"] = TARGET_MEDIAN_ABS
        dataset.attrs["device_counts_per_output_unit"] = training.scale_counts_per_unit
        qc = dst.create_dataset("alignment_events", data=alignment_rows,
                                maxshape=(None,), chunks=True)
        qc.attrs["algorithm"] = METHOD_NAME
        qc.attrs["feature_extractor"] = FEATURE_NAME
        qc.attrs["recentering"] = RECENTER_NAME
        qc.attrs["global_template_applied"] = alignment.global_template_applied
        qc.attrs["global_template_version"] = (
            global_bundle.version if global_bundle else "")
        qc.attrs["global_template_sha256"] = (
            global_bundle.sha256 if global_bundle else "")
        qc.attrs["global_template_participant_count"] = (
            global_bundle.participant_count if global_bundle else 0)
        qc.attrs["global_template_session_count"] = (
            global_bundle.session_count if global_bundle else 0)
        qc.attrs["iterations"] = alignment.iterations
        qc.attrs["converged"] = alignment.converged
        qc.attrs["minimum_confidence"] = MIN_ALIGNMENT_CONFIDENCE
        qc.attrs["minimum_template_correlation"] = MIN_TEMPLATE_CORRELATION
        qc.attrs["minimum_correlation_margin"] = MIN_CORRELATION_MARGIN
        qc.attrs["source_file"] = source.name
        score_set = dst.create_dataset(
            "alignment_match_scores", data=alignment.match_scores,
            chunks=(1, alignment.match_scores.shape[1]))
        score_set.attrs["offset_axis_dataset"] = "alignment_match_offsets_ms"
        dst.create_dataset(
            "alignment_match_offsets_ms", data=alignment.match_offsets_ms,
            chunks=(1, alignment.match_offsets_ms.shape[1]))
        template_group = dst.create_group("alignment_templates")
        template_group.attrs["offset_frames"] = alignment.template_offsets_frames
        template_group.attrs["frame_stride_ms"] = 20.0
        for name, template in alignment.gesture_templates.items():
            template_group.create_dataset(name, data=template, compression="gzip")
        recenter_group = dst.create_group("alignment_global_recenter_offsets_frames")
        for name, offset in alignment.global_recenter_offsets_frames.items():
            recenter_group.attrs[name] = int(offset)
        reviews = dst.create_dataset("alignment_reviews", data=review_rows,
                                     maxshape=(None,), chunks=True)
        reviews.attrs["accepted_statuses"] = ",".join(sorted(ACCEPTED_REVIEW_STATUSES))
        meta = dst.create_group("meta")
        for key, value in metadata.items():
            meta.attrs[key] = value
        meta.attrs["label_alignment_status"] = "session_rerp_aligned_v3"
        meta.attrs["training_label_shift_applied"] = False
        meta.attrs["training_target_pulse_window_sec"] = np.asarray([0.08, 0.12])
        meta.attrs["excluded_low_confidence_events"] = len(aligned) - len(included)
        meta.attrs["training_preprocessing_version"] = PREPROCESSING_VERSION
        meta.attrs["corpus_registration_enabled"] = register_corpus

    prompts.to_hdf(target, key="prompts", mode="a", format="fixed")
    stages.to_hdf(target, key="stages", mode="a", format="fixed")
    corpus_path = upsert_corpus_entry(target) if register_corpus else None
    LOGGER.info("aligned Meta export: %s (%s/%s cues included)",
                target, len(included), len(aligned))
    if corpus_path is not None:
        LOGGER.info("Meta corpus updated: %s", corpus_path)
    return target


def _decode_fixed(value: Any) -> str:
    return bytes(value).rstrip(b"\x00").decode("utf-8") \
        if isinstance(value, (bytes, np.bytes_)) else str(value)


def ensure_alignment_reviews(path: Path) -> None:
    """Add review rows to an older aligned export without changing its events."""
    path = Path(path)
    with h5py.File(path, "r+") as handle:
        if "alignment_reviews" in handle:
            return
        events = handle["alignment_events"][:]
        dtype = np.dtype([
            ("event_index", "i8"), ("reviewed_sample_index", "i8"),
            ("status", "S24"), ("note", "S256"), ("reviewed_at", "S40"),
        ])
        rows = np.array([(
            index, int(event["aligned_sample_index"]),
            b"auto_accepted" if bool(event["included"]) else b"auto_rejected",
            b"", b"",
        ) for index, event in enumerate(events)], dtype=dtype)
        dataset = handle.create_dataset("alignment_reviews", data=rows,
                                        maxshape=(None,), chunks=True)
        dataset.attrs["accepted_statuses"] = ",".join(sorted(ACCEPTED_REVIEW_STATUSES))


def rebuild_reviewed_prompts(path: Path) -> int:
    """Rebuild prompts from reviewed sample indices; alignment events stay immutable."""
    path = Path(path); ensure_alignment_reviews(path)
    with h5py.File(path, "r") as handle:
        events = handle["alignment_events"][:]
        reviews = handle["alignment_reviews"][:]
        rate = float(handle["data"].attrs["sample_rate"])
        start_unix = float(handle["meta"].attrs["start_unix_time"])
    trial_is_accepted: dict[int, bool] = {}
    for event, review in zip(events, reviews):
        trial_id = int(event["trial_id"])
        accepted_status = _decode_fixed(review["status"]) in ACCEPTED_REVIEW_STATUSES
        trial_is_accepted[trial_id] = trial_is_accepted.get(trial_id, True) and accepted_status
    accepted: list[tuple[str, float]] = []
    for event, review in zip(events, reviews):
        if trial_is_accepted.get(int(event["trial_id"]), False):
            accepted.append((
                _text(event["name"]),
                start_unix + int(review["reviewed_sample_index"]) / rate,
            ))
    prompts = pd.DataFrame(accepted, columns=["name", "time"])
    with h5py.File(path, "r+") as handle:
        if "prompts" in handle:
            del handle["prompts"]
        handle["meta"].attrs["reviewed_prompt_count"] = len(prompts)
    prompts.to_hdf(path, key="prompts", mode="a", format="fixed")
    refresh_corpus_entry(path)
    return len(prompts)


def update_alignment_review(path: Path, event_index: int, *, status: str,
                            reviewed_sample_index: int | None = None,
                            note: str = "") -> int:
    if status not in {"accepted", "manually_adjusted", "rejected"}:
        raise ValueError(f"无效审核状态：{status}")
    path = Path(path); ensure_alignment_reviews(path)
    with h5py.File(path, "r+") as handle:
        reviews = handle["alignment_reviews"]
        if not 0 <= event_index < len(reviews):
            raise IndexError("对齐事件索引越界")
        row = reviews[event_index]
        if reviewed_sample_index is not None:
            row["reviewed_sample_index"] = int(np.clip(
                reviewed_sample_index, 0, max(0, len(handle["data"]) - 1)))
        row["status"] = status.encode("utf-8")
        row["note"] = note[:80].encode("utf-8")
        row["reviewed_at"] = datetime.now(timezone.utc).isoformat().encode("utf-8")
        reviews[event_index] = row
    rebuild_reviewed_prompts(path)
    return int(row["reviewed_sample_index"])


def reject_alignment_trial(path: Path, trial_id: int, note: str = "") -> int:
    path = Path(path); ensure_alignment_reviews(path)
    changed = 0
    with h5py.File(path, "r+") as handle:
        events = handle["alignment_events"][:]
        reviews = handle["alignment_reviews"]
        stamp = datetime.now(timezone.utc).isoformat().encode("utf-8")
        for index, event in enumerate(events):
            if int(event["trial_id"]) != int(trial_id):
                continue
            row = reviews[index]
            row["status"] = b"rejected"
            row["note"] = note[:80].encode("utf-8")
            row["reviewed_at"] = stamp
            reviews[index] = row; changed += 1
    rebuild_reviewed_prompts(path)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align raw UI cues to EMG and export a separate Meta-format HDF5 file.")
    parser.add_argument("source", type=Path, help="raw session.h5")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--global-template", type=Path, default=None)
    parser.add_argument(
        "--no-register-corpus", action="store_true",
        help="Create a comparison export without adding it to the training corpus.")
    args = parser.parse_args()
    print(export_meta_aligned(
        args.source, args.output, global_template_path=args.global_template,
        register_corpus=not args.no_register_corpus))


if __name__ == "__main__":
    main()
