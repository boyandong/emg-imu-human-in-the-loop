"""Cue-relative S04 event diagnostics from the existing continuous Song decoder."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.song_continuous_replay import replay_session
from benchmarks.song_real8_study import _text, parse_label


ROOT = Path(__file__).resolve().parent / "song_real8"
PROTOCOL_PATH = ROOT / "CUE_EVENT_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def summarize(rows: list[dict]) -> dict:
    labels = ("fist", "index_pinch", "neutral", "open_hand")
    by_class = {}
    for label in labels:
        part = [row for row in rows if row["hand"] == label]
        delays = np.asarray([row["first_correct_latency_seconds"] for row in part
                             if row["first_correct_latency_seconds"] is not None], dtype=float)
        by_class[label] = {
            "events": len(part),
            "ever_correct_decoded": sum(row["event_success"] for row in part),
            "endpoint_correct_decoded": sum(row["endpoint_correct"] for row in part),
            "late_rest_with_frames": sum(row["late_rest_frame_count"] > 0 for row in part),
            "late_rest_pre_prompt_active": sum(row["pre_prompt_activation"] is True for row in part),
            "first_correct_latency_median_seconds": float(np.median(delays)) if len(delays) else None,
            "first_correct_latency_p90_seconds": float(np.quantile(delays, 0.9)) if len(delays) else None,
        }
    active = [row for row in rows if row["hand"] != "neutral"]
    delays = np.asarray([row["first_correct_latency_seconds"] for row in active
                         if row["first_correct_latency_seconds"] is not None], dtype=float)
    return {
        "eligible_events": len(rows),
        "active_events": len(active),
        "active_ever_correct": sum(row["event_success"] for row in active),
        "active_endpoint_correct": sum(row["endpoint_correct"] for row in active),
        "active_missed": sum(not row["event_success"] for row in active),
        "late_rest_intervals_with_frames": sum(row["late_rest_frame_count"] > 0 for row in rows),
        "late_rest_intervals_with_active_display": sum(row["pre_prompt_activation"] is True for row in rows),
        "successful_active_first_correct_median_seconds": float(np.median(delays)) if len(delays) else None,
        "successful_active_first_correct_p90_seconds": float(np.quantile(delays, 0.9)) if len(delays) else None,
        "by_class": by_class,
    }


def run(source: Path, bundle: Path, collection_root: Path) -> dict:
    source_sha, model_sha, sample_count, trials, frame_indices, probabilities = replay_session(
        source, bundle, collection_root, "S04")
    if source_sha != PROTOCOL["source_sha256"] or model_sha != PROTOCOL["model_sha256"]:
        raise ValueError("cue event source or model differs from frozen protocol")
    sys.path.insert(0, str(collection_root.resolve()))
    from emgforce.inference.song_local import SongOnlineDecision

    decoder = SongOnlineDecision(consecutive_frames=3)
    decoded = []
    for probability in probabilities:
        state, _ = decoder.step(probability, threshold=0.5)
        decoded.append(state if state is not None else "unknown")
    decoded = np.asarray(decoded)
    rows = []
    for trial in trials:
        if (_text(trial["trial_kind"]) != "formal" or not bool(trial["valid"])
                or _text(trial["completion_status"]) != "completed"):
            continue
        _, hand = parse_label(_text(trial["label"]))
        stable_start, stable_end = int(trial["stable_start_sample"]), int(trial["stable_end_sample"])
        stable = np.flatnonzero((frame_indices - 49 >= stable_start) & (frame_indices < stable_end))
        if not len(stable):
            continue
        rest_start, prompt_start = int(trial["rest_start_sample"]), int(trial["prompt_start_sample"])
        late_start = max(rest_start, prompt_start - 100)
        late_rest = np.flatnonzero((frame_indices - 49 >= late_start) & (frame_indices < prompt_start))
        correct = np.flatnonzero(decoded[stable] == hand)
        first_latency = ((int(frame_indices[stable[correct[0]]]) - stable_start) / 250.0
                         if len(correct) and hand != "neutral" else None)
        rest_states = decoded[late_rest]
        active_rest = int(np.count_nonzero(np.isin(rest_states, ("fist", "index_pinch", "open_hand"))))
        rows.append({
            "trial_id": _text(trial["trial_id"]), "label": _text(trial["label"]), "hand": hand,
            "stable_start_sample": stable_start, "stable_end_sample": stable_end,
            "stable_frame_count": len(stable), "decoded_correct_frames": len(correct),
            "decoded_unknown_frames": int(np.count_nonzero(decoded[stable] == "unknown")),
            "event_success": bool(len(correct)), "endpoint_correct": bool(decoded[stable[-1]] == hand),
            "first_correct_latency_seconds": first_latency,
            "late_rest_frame_count": len(late_rest), "late_rest_active_frame_count": active_rest,
            "pre_prompt_activation": bool(active_rest) if len(late_rest) else None,
        })
    if len(rows) != 144 or len({row["trial_id"] for row in rows}) != 144:
        raise ValueError("S04 eligible event coverage changed")
    summary = summarize(rows)
    artifact = {
        "status": "cue_relative_offline_replay_not_physiological_onset_or_ui_latency",
        "protocol": PROTOCOL,
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "source_sha256": source_sha, "model_sha256": model_sha,
        "sample_count": sample_count, "decoded_frames": len(decoded),
        **summary,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "CUE_EVENT_ROWS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "CUE_EVENT_AUDIT.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: artifact[key] for key in (
        "eligible_events", "active_events", "active_ever_correct", "active_endpoint_correct",
        "late_rest_intervals_with_active_display", "successful_active_first_correct_median_seconds")}), flush=True)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("E:/qxy/emg_meta/emg_meta/data/Song"))
    parser.add_argument("--bundle", type=Path, default=Path(
        "collection/emg_meta/emg_meta/models/song_real8_f0"))
    parser.add_argument("--collection-root", type=Path, default=Path("collection/emg_meta/emg_meta"))
    args = parser.parse_args()
    run(args.source, args.bundle, args.collection_root)
