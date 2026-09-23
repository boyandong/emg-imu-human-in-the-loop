"""Describe cue-relative output timing for already-exported Song models.

Recorded cue boundaries are not physiological movement onsets. No model fitting,
threshold selection, or physical-device timing is performed here.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.song_continuous_replay import replay_session
from benchmarks.song_real8_study import _text, parse_label


def _summarize(rows: list[dict]) -> dict:
    latency = [row["decoded_latency_ms"] for row in rows if row["decoded_latency_ms"] is not None]
    return {
        "trials": len(rows),
        "already_correct_before_cue": sum(row["already_correct_before_cue"] for row in rows),
        "raw_correct_within_cue": sum(row["raw_latency_ms"] is not None for row in rows),
        "decoded_correct_within_cue": len(latency),
        "decoded_correct_within_cue_fraction": len(latency) / len(rows) if rows else None,
        "decoded_latency_median_ms_among_hits": float(np.median(latency)) if latency else None,
        "decoded_latency_p90_ms_among_hits": float(np.percentile(latency, 90)) if latency else None,
        "stable_cue_evaluable": sum(row["stable_cue_evaluable"] for row in rows),
        "decoded_correct_in_stable_cue": sum(row["decoded_correct_in_stable_cue"] for row in rows),
    }


def score_cue_response(trials, frame_indices: np.ndarray, probabilities: np.ndarray) -> dict:
    """Return trial-level cue timing with the unchanged live decision rule."""
    from emgforce.inference.song_local import LABELS, SongOnlineDecision

    if len(frame_indices) != len(probabilities) or len(frame_indices) == 0:
        raise ValueError("frames and probabilities must align and be nonempty")
    if not np.all(np.diff(frame_indices) > 0):
        raise ValueError("frame indices must be strictly increasing")

    raw = np.asarray(LABELS)[np.argmax(probabilities, axis=1)]
    decision = SongOnlineDecision()
    decoded = []
    for probability in probabilities:
        active, _ = decision.step(probability, threshold=0.5)
        decoded.append(active if active is not None else "unknown")
    decoded = np.asarray(decoded)

    trial_rows = []
    for row in trials:
        if (_text(row["trial_kind"]) != "formal" or not bool(row["valid"]) or
                _text(row["completion_status"]) != "completed"):
            continue
        arm, hand = parse_label(_text(row["label"]))
        start, end = int(row["prompt_start_sample"]), int(row["prompt_end_sample"])
        stable_start, stable_end = int(row["stable_start_sample"]), int(row["stable_end_sample"])
        if not (0 <= start < stable_start < stable_end <= end):
            raise ValueError("formal cue and stable interval are inconsistent")
        before = int(np.searchsorted(frame_indices, start, side="left")) - 1
        in_cue = np.flatnonzero((frame_indices >= start) & (frame_indices < end))
        in_stable = np.flatnonzero((frame_indices - 49 >= stable_start) & (frame_indices < stable_end))
        if len(in_cue) == 0:
            raise ValueError("formal trial has no cue prediction frame")

        def first_latency(labels):
            hits = in_cue[labels[in_cue] == hand]
            return round((int(frame_indices[hits[0]]) - start) * 1000 / 250, 3) if len(hits) else None

        trial_rows.append({
            "arm": arm, "hand": hand,
            "already_correct_before_cue": bool(before >= 0 and decoded[before] == hand),
            "raw_latency_ms": first_latency(raw),
            "decoded_latency_ms": first_latency(decoded),
            "stable_cue_evaluable": bool(len(in_stable)),
            "decoded_correct_in_stable_cue": bool(len(in_stable) and np.any(decoded[in_stable] == hand)),
        })
    if not trial_rows:
        raise ValueError("no completed formal trials")

    by_hand = defaultdict(list)
    for row in trial_rows:
        by_hand[row["hand"]].append(row)
    return {
        "all": _summarize(trial_rows),
        "by_hand": {hand: _summarize(by_hand[hand]) for hand in LABELS},
        "transition_needed": _summarize([row for row in trial_rows if not row["already_correct_before_cue"]]),
        "trial_rows": trial_rows,
    }


def run(source: Path, collection_root: Path, bundles: dict[str, Path], session_id: str, output: Path) -> dict:
    sys.path.insert(0, str(collection_root.resolve()))
    result = {
        "status": "cue_relative_offline_replay_not_physiological_onset_or_device_latency",
        "session": session_id,
        "sampling_rate_hz": 250,
        "model_results": {},
        "boundary": "First matching frame end within a recorded 1.3–1.5 s cue is measured from cue start; this is not human movement onset or actual UI/device latency. Trials already showing the correct state before cue are counted separately. Median/p90 condition on hits, so misses must be read with them. The fixed 0.5/three-frame live decoder runs continuously without trial resets. Same-person/day S04 was previously examined and these descriptive results cannot tune a product threshold.",
    }
    expected_sha = None
    comparison_rows = {}
    for name, bundle in bundles.items():
        source_sha, model_sha, sample_count, trials, indices, probabilities = replay_session(
            source, bundle, collection_root, session_id)
        if expected_sha is not None and source_sha != expected_sha:
            raise ValueError("model replays used different source files")
        expected_sha = source_sha
        score = score_cue_response(trials, indices, probabilities)
        comparison_rows[name] = score["trial_rows"]
        result["model_results"][name] = {
            "model_sha256": model_sha,
            "sample_count": sample_count,
            "frame_count": len(indices),
            "all": score["all"],
            "by_hand": score["by_hand"],
            "transition_needed": score["transition_needed"],
        }
        print(json.dumps({"model": name, "transition_needed": score["transition_needed"]},
                         ensure_ascii=False), flush=True)
    if "F0" in comparison_rows and "F0+SPD" in comparison_rows:
        left, right = comparison_rows["F0"], comparison_rows["F0+SPD"]
        if len(left) != len(right) or any(
                a["arm"] != b["arm"] or a["hand"] != b["hand"] for a, b in zip(left, right)):
            raise ValueError("paired model replays have different formal trials")
        def paired(rows):
            both = [(a, b) for a, b in rows if a["decoded_latency_ms"] is not None
                    and b["decoded_latency_ms"] is not None]
            differences = [b["decoded_latency_ms"] - a["decoded_latency_ms"] for a, b in both]
            return {
                "trials": len(rows),
                "both_hit": len(both),
                "F0_only_hit": sum(a["decoded_latency_ms"] is not None and
                                   b["decoded_latency_ms"] is None for a, b in rows),
                "SPD_only_hit": sum(a["decoded_latency_ms"] is None and
                                    b["decoded_latency_ms"] is not None for a, b in rows),
                "neither_hit": sum(a["decoded_latency_ms"] is None and
                                   b["decoded_latency_ms"] is None for a, b in rows),
                "SPD_faster_among_both_hits": sum(value < 0 for value in differences),
                "SPD_equal_among_both_hits": sum(value == 0 for value in differences),
                "SPD_slower_among_both_hits": sum(value > 0 for value in differences),
                "SPD_minus_F0_median_latency_ms_among_both_hits": float(np.median(differences)) if both else None,
            }
        pairs = list(zip(left, right))
        result["paired_comparison"] = {
            "all_formal": paired(pairs),
            "transition_needed_in_both": paired([(a, b) for a, b in pairs if not
                a["already_correct_before_cue"] and not b["already_correct_before_cue"]]),
        }
    result["source_sha256"] = expected_sha
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--f0-bundle", required=True, type=Path)
    parser.add_argument("--spd-bundle", required=True, type=Path)
    parser.add_argument("--session", default="S04", choices=("S03", "S04"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.collection_root,
        {"F0": args.f0_bundle, "F0+SPD": args.spd_bundle}, args.session, args.output)
