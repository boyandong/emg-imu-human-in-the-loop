"""Audit continuous Song S04 predictions without treating cue time as EMG onset."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from benchmarks.song_real8_study import _hash, _text, parse_label


def run(source: Path, bundle: Path, collection_root: Path, output: Path) -> dict:
    sys.path.insert(0, str(collection_root.resolve()))
    from emgforce.inference.song_local import LABELS, SongLocalRuntime, SongOnlineDecision

    session_dir = source / "2026-09-18_S04"
    session_file = session_dir / "session.h5"
    readiness = json.loads((session_dir / "SESSION_COLLECTION_READINESS.json").read_text(encoding="utf-8"))
    source_sha = _hash(session_file)
    if source_sha != readiness["hdf5_sha256"]:
        raise ValueError("S04 source hash differs from collection readiness audit")
    with h5py.File(session_file) as handle:
        meta = handle["meta"].attrs
        if (_text(meta["session_id"]) != "S04" or _text(meta["dataset_split"]) != "test" or
                int(meta["emg_nominal_rate_hz"]) != 250 or int(meta["num_emg_channels"]) != 8):
            raise ValueError("S04 metadata incompatible with this audit")
        raw = handle["streams/emg/raw"][:]
        indices = handle["streams/emg/sample_index"][:]
        trials = handle["trials"][:]
    if not np.array_equal(indices, np.arange(len(raw))):
        raise ValueError("S04 sample indices are not contiguous")

    runtime = SongLocalRuntime(bundle)
    decision = SongOnlineDecision()
    frame_indices, frame_probabilities = [], []
    decoded_labels = []
    for start in range(0, len(raw), 37):
        gap, frames = runtime.ingest(raw[start:start + 37], indices[start:start + 37])
        if gap:
            raise ValueError("S04 continuous replay unexpectedly reset")
        for index, probabilities in frames:
            frame_indices.append(index)
            frame_probabilities.append(probabilities)
            active, _ = decision.step(probabilities, threshold=0.5)
            decoded_labels.append(active if active is not None else "unknown")
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    probabilities = np.stack(frame_probabilities)
    predicted = np.asarray(LABELS)[np.argmax(probabilities, axis=1)]
    decoded = np.asarray(decoded_labels)
    whole_session_support = dict(Counter(predicted.tolist()))

    trial_true, trial_pred, trial_has_correct, stable_frame_true, stable_frame_pred = [], [], [], [], []
    stable_decoded = []
    trial_decoded_has_correct = []
    rest_frame_pred = []
    stable_support = Counter()
    rest_intervals = 0
    for row in trials:
        if _text(row["trial_kind"]) != "formal" or not bool(row["valid"]) or _text(row["completion_status"]) != "completed":
            continue
        _, hand = parse_label(_text(row["label"]))
        stable_start, stable_end = int(row["stable_start_sample"]), int(row["stable_end_sample"])
        # Only 200 ms windows fully inside the recorded stable cue interval.
        selected = np.flatnonzero((frame_indices - 49 >= stable_start) & (frame_indices < stable_end))
        if len(selected):
            trial_true.append(hand)
            trial_pred.append(LABELS[int(np.argmax(probabilities[selected].mean(axis=0)))])
            trial_has_correct.append(bool(np.any(predicted[selected] == hand)))
            stable_frame_true.extend([hand] * len(selected))
            stable_frame_pred.extend(predicted[selected].tolist())
            stable_decoded.extend(decoded[selected].tolist())
            trial_decoded_has_correct.append(bool(np.any(decoded[selected] == hand)))
            stable_support[hand] += len(selected)
        rest_start, prompt_start = int(row["rest_start_sample"]), int(row["prompt_start_sample"])
        selected_rest = np.flatnonzero((frame_indices - 49 >= rest_start) & (frame_indices < prompt_start))
        if len(selected_rest):
            rest_frame_pred.extend(predicted[selected_rest].tolist())
            rest_intervals += 1

    labels = list(LABELS)
    result = {
        "status": "continuous_cue_timeline_audited_not_physiological_onset_validated",
        "source_s04_sha256": source_sha,
        "model_sha256": runtime.sha256,
        "sample_count": len(raw), "duration_seconds_nominal": len(raw) / 250,
        "frame_count": len(frame_indices), "all_frame_predicted_support": whole_session_support,
        "formal_trials_with_full_stable_windows": len(trial_true),
        "stable_frames": len(stable_frame_true), "stable_frame_true_support": dict(stable_support),
        "stable_frame_predicted_support": dict(Counter(stable_frame_pred)),
        "stable_frame_accuracy": float(accuracy_score(stable_frame_true, stable_frame_pred)),
        "stable_frame_macro_f1": float(f1_score(stable_frame_true, stable_frame_pred, labels=labels,
                                                  average="macro", zero_division=0)),
        "online_decision_rule": {"probability_threshold": 0.5, "consecutive_frames": 3,
                                 "hop_ms": 100, "unresolved_label": "unknown"},
        "stable_decoded_frame_support": dict(Counter(stable_decoded)),
        "stable_decoded_frame_accuracy": float(accuracy_score(stable_frame_true, stable_decoded)),
        "stable_decoded_frame_macro_f1": float(f1_score(stable_frame_true, stable_decoded,
                                                          labels=labels, average="macro", zero_division=0)),
        "stable_decoded_frame_recall": dict(zip(labels, map(float, recall_score(
            stable_frame_true, stable_decoded, labels=labels, average=None, zero_division=0)))),
        "stable_trial_at_least_one_correct_decoded_frame_fraction": float(np.mean(trial_decoded_has_correct)),
        "stable_trial_mean_probability_accuracy": float(accuracy_score(trial_true, trial_pred)),
        "stable_trial_mean_probability_macro_f1": float(f1_score(trial_true, trial_pred, labels=labels,
                                                                   average="macro", zero_division=0)),
        "stable_trial_recall": dict(zip(labels, map(float, recall_score(trial_true, trial_pred,
                                                 labels=labels, average=None, zero_division=0)))),
        "stable_trial_confusion_matrix": confusion_matrix(trial_true, trial_pred, labels=labels).tolist(),
        "stable_trial_at_least_one_correct_frame_fraction": float(np.mean(trial_has_correct)),
        "rest_cue_intervals_with_full_windows": rest_intervals,
        "rest_cue_frames": len(rest_frame_pred),
        "rest_cue_predicted_support": dict(Counter(rest_frame_pred)),
        "rest_cue_neutral_fraction": float(np.mean(np.asarray(rest_frame_pred) == "neutral"))
        if rest_frame_pred else None,
        "labels_order": labels,
        "boundary": "All-frame support includes calibration, rest, transitions and uncued time. Decoder metrics replay the fixed live threshold and three-frame rule on every frame; Qt displays the latest state per processed input batch, so these are decoder-state rather than measured screen metrics. Trial/stable metrics use recorded cue intervals, not physiological onset. Rest intervals may contain residual prior movement. No physical USB, event-onset or UI-latency validation."
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "frame_count", "all_frame_predicted_support", "stable_frame_accuracy",
        "stable_decoded_frame_accuracy", "stable_trial_mean_probability_accuracy",
        "stable_trial_recall", "rest_cue_neutral_fraction")},
        ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.bundle, args.collection_root, args.output)
