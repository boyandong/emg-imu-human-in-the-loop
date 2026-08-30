from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .meta_alignment import (
    _smooth, ensure_alignment_reviews, reject_alignment_trial, update_alignment_review,
)


def _text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).rstrip(b"\x00").decode("utf-8")
    return str(value)


def review_activity_envelopes(
        raw: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Create human-readable EMG activity traces for alignment review.

    The stored signal remains untouched.  For display only, an FFT mask removes
    DC/drift, frequencies outside the useful EMG band, and narrow 50 Hz mains
    harmonics.  A 25 ms RMS envelope then turns the dense carrier waveform into
    the slowly varying muscle activity that an operator can actually inspect.
    """
    signal = np.asarray(raw, dtype=np.float64)
    if signal.ndim != 2 or not len(signal):
        channels = signal.shape[1] if signal.ndim == 2 else 0
        return np.empty((0, channels)), np.empty(0)
    centered = signal - np.median(signal, axis=0, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=0)
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate)
    keep = (frequencies >= 20.0) & (frequencies <= min(450.0, sample_rate / 2 - 1.0))
    # Remove the mains fundamental and its harmonics.  The narrow display-only
    # notches preserve the broadband EMG burst while eliminating the sinusoid.
    for harmonic in np.arange(50.0, min(450.0, sample_rate / 2) + 0.1, 50.0):
        keep &= np.abs(frequencies - harmonic) > 3.0
    spectrum[~keep, :] = 0.0
    filtered = np.fft.irfft(spectrum, n=len(centered), axis=0)
    width = max(3, round(0.025 * sample_rate))
    activity = np.empty_like(filtered)
    for channel in range(filtered.shape[1]):
        activity[:, channel] = np.sqrt(np.maximum(
            _smooth(filtered[:, channel] ** 2, width), 0.0))
    combined = np.sqrt(np.maximum(_smooth(
        np.mean(filtered * filtered, axis=1), width), 0.0))
    return activity, combined


class AlignmentReviewStore:
    """Read and persist event reviews without ever opening the raw file writable."""

    def __init__(self, raw_path: Path, aligned_path: Path) -> None:
        self.raw_path = Path(raw_path)
        self.aligned_path = Path(aligned_path)
        ensure_alignment_reviews(self.aligned_path)
        self.events = np.empty(0)
        self.reviews = np.empty(0)
        self.sample_rate = 2000.0
        self.num_samples = 0
        self.refresh()

    def refresh(self) -> None:
        with h5py.File(self.aligned_path, "r") as handle:
            self.events = handle["alignment_events"][:]
            self.reviews = handle["alignment_reviews"][:]
            self.sample_rate = float(handle["data"].attrs["sample_rate"])
            self.num_samples = len(handle["data"])

    def indices(self, low_confidence_only: bool = False) -> list[int]:
        if not low_confidence_only:
            return list(range(len(self.events)))
        threshold = 0.0
        with h5py.File(self.aligned_path, "r") as handle:
            threshold = float(handle["alignment_events"].attrs.get(
                "minimum_confidence", 0.20))
        return [index for index, (event, review) in enumerate(zip(self.events, self.reviews))
                if (float(event["confidence"]) < threshold
                    or not bool(event["included"])
                    or _text(review["status"]) in {"auto_rejected", "rejected"})]

    def automatic_summary(self) -> dict[str, Any]:
        """Summarize immutable automatic QC decisions at event and Trial level."""
        fields = self.events.dtype.names or ()
        has_included = "included" in fields
        included = [bool(event["included"]) if has_included else True
                    for event in self.events]
        events_by_trial: dict[int, list[bool]] = defaultdict(list)
        reasons: Counter[str] = Counter()
        for event, event_included in zip(self.events, included):
            trial_id = int(event["trial_id"])
            events_by_trial[trial_id].append(event_included)
            if not event_included and "exclusion_reason" in fields:
                for reason in _text(event["exclusion_reason"]).split(","):
                    if reason:
                        reasons[reason] += 1
        included_trials = sorted(
            trial_id for trial_id, decisions in events_by_trial.items() if all(decisions))
        excluded_trials = sorted(set(events_by_trial) - set(included_trials))
        with h5py.File(self.aligned_path, "r") as handle:
            attrs = handle["alignment_events"].attrs
            thresholds = {
                "confidence": float(attrs.get("minimum_confidence", np.nan)),
                "correlation": float(attrs.get("minimum_template_correlation", np.nan)),
                "margin": float(attrs.get("minimum_correlation_margin", np.nan)),
            }
            algorithm = _text(attrs.get("algorithm", "unknown"))
        return {
            "algorithm": algorithm,
            "total_events": len(self.events),
            "included_events": sum(included),
            "excluded_events": len(included) - sum(included),
            "total_trials": len(events_by_trial),
            "included_trials": included_trials,
            "excluded_trials": excluded_trials,
            "reason_counts": dict(sorted(reasons.items())),
            "thresholds": thresholds,
        }

    def record(self, event_index: int) -> dict[str, Any]:
        event = self.events[event_index]; review = self.reviews[event_index]
        record = {
            "event_index": event_index,
            "name": _text(event["name"]),
            "trial_id": int(event["trial_id"]),
            "stage_id": int(event["stage_id"]),
            "cue_sample_index": int(event["cue_sample_index"]),
            "auto_aligned_sample_index": int(event["aligned_sample_index"]),
            "reviewed_sample_index": int(review["reviewed_sample_index"]),
            "offset_ms": (
                int(review["reviewed_sample_index"]) - int(event["cue_sample_index"]))
                * 1000.0 / self.sample_rate,
            "confidence": float(event["confidence"]),
            "method": _text(event["method"]),
            "status": _text(review["status"]),
            "note": _text(review["note"]),
            "reviewed_at": _text(review["reviewed_at"]),
        }
        fields = event.dtype.names or ()
        record.update({
            "template_alignment": "template_correlation" in fields,
            "template_correlation": float(event["template_correlation"])
                if "template_correlation" in fields else float("nan"),
            "second_best_correlation": float(event["second_best_correlation"])
                if "second_best_correlation" in fields else float("nan"),
            "search_boundary_hit": bool(event["search_boundary_hit"])
                if "search_boundary_hit" in fields else False,
            "exclusion_reason": _text(event["exclusion_reason"])
                if "exclusion_reason" in fields else "",
        })
        return record

    def match_curve(self, event_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return V2 template-match offsets (seconds) and scores, if present."""
        with h5py.File(self.aligned_path, "r") as handle:
            if "alignment_match_scores" not in handle:
                return np.empty(0), np.empty(0)
            dataset = handle["alignment_match_scores"]
            if "alignment_match_offsets_ms" in handle:
                offsets = np.asarray(
                    handle["alignment_match_offsets_ms"][event_index],
                    dtype=np.float64) / 1000.0
            else:
                offsets = np.asarray(dataset.attrs["offset_ms"], dtype=np.float64) / 1000.0
            return offsets, np.asarray(dataset[event_index], dtype=np.float64)

    def segment(self, event_index: int, before_sec: float = 0.50,
                after_sec: float = 1.50) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        record = self.record(event_index)
        cue = record["cue_sample_index"]
        start = max(0, cue - round(before_sec * self.sample_rate))
        end = min(self.num_samples, cue + round(after_sec * self.sample_rate))
        with h5py.File(self.raw_path, "r") as handle:
            raw = handle["streams/emg/raw"][start:end].astype(np.float64)
        x = (np.arange(start, end) - cue) / self.sample_rate
        activity, envelope = review_activity_envelopes(raw, self.sample_rate)
        aligned_sec = (record["reviewed_sample_index"] - cue) / self.sample_rate
        return x, activity, envelope, aligned_sec

    def accept(self, event_index: int, note: str = "") -> None:
        current = self.record(event_index)["reviewed_sample_index"]
        update_alignment_review(
            self.aligned_path, event_index, status="accepted",
            reviewed_sample_index=current, note=note)
        self.refresh()

    def adjust_ms(self, event_index: int, delta_ms: float, note: str = "") -> None:
        current = self.record(event_index)["reviewed_sample_index"]
        adjusted = current + round(delta_ms * self.sample_rate / 1000.0)
        update_alignment_review(
            self.aligned_path, event_index, status="manually_adjusted",
            reviewed_sample_index=adjusted, note=note)
        self.refresh()

    def reject_trial(self, event_index: int, note: str = "") -> int:
        changed = reject_alignment_trial(
            self.aligned_path, self.record(event_index)["trial_id"], note)
        self.refresh()
        return changed
