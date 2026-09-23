"""Choose a Song display persistence rule on S03, then replay S04 unchanged."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

from benchmarks.song_continuous_replay import replay_session
from benchmarks.song_real8_study import _text, parse_label


def cue_masks(trials: np.ndarray, frame_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    stable_truth = np.full(len(frame_indices), "", dtype=object)
    rest = np.zeros(len(frame_indices), dtype=bool)
    valid_trials = 0
    for row in trials:
        if (_text(row["trial_kind"]) != "formal" or not bool(row["valid"]) or
                _text(row["completion_status"]) != "completed"):
            continue
        _, hand = parse_label(_text(row["label"]))
        stable = ((frame_indices - 49 >= int(row["stable_start_sample"])) &
                  (frame_indices < int(row["stable_end_sample"])))
        if not np.any(stable):
            continue
        if np.any(stable_truth[stable] != ""):
            raise ValueError("overlapping Song stable cue intervals")
        stable_truth[stable] = hand
        rest |= ((frame_indices - 49 >= int(row["rest_start_sample"])) &
                 (frame_indices < int(row["prompt_start_sample"])))
        valid_trials += 1
    if not valid_trials or np.any(rest & (stable_truth != "")):
        raise ValueError("missing trials or rest/stable cue overlap")
    return stable_truth, rest, valid_trials


def score_rule(probabilities: np.ndarray, stable_truth: np.ndarray, rest: np.ndarray,
               consecutive_frames: int, ema_alpha: float) -> dict:
    from emgforce.inference.song_local import LABELS, SongOnlineDecision

    decoder = SongOnlineDecision(consecutive_frames)
    decoded = []
    smoothed = None
    for probability in probabilities:
        smoothed = (probability.copy() if smoothed is None else
                    ema_alpha * probability + (1 - ema_alpha) * smoothed)
        label, _ = decoder.step(smoothed, threshold=0.5)
        decoded.append(label if label is not None else "unknown")
    decoded = np.asarray(decoded)
    stable = stable_truth != ""
    truth, predicted = stable_truth[stable].astype(str), decoded[stable]
    return {
        "consecutive_frames": consecutive_frames,
        "ema_alpha": ema_alpha,
        "stable_frames": int(np.sum(stable)), "rest_frames": int(np.sum(rest)),
        "whole_stream_state_changes": int(np.count_nonzero(decoded[1:] != decoded[:-1])),
        "whole_stream_changes_per_minute": float(np.count_nonzero(decoded[1:] != decoded[:-1]) /
                                                 (len(decoded) * 0.1 / 60)),
        "stable_accuracy": float(accuracy_score(truth, predicted)),
        "stable_macro_f1": float(f1_score(truth, predicted, labels=LABELS,
                                           average="macro", zero_division=0)),
        "stable_recall": dict(zip(LABELS, map(float, recall_score(
            truth, predicted, labels=LABELS, average=None, zero_division=0)))),
        "stable_display_support": dict(Counter(predicted.tolist())),
        "rest_neutral_fraction": float(np.mean(decoded[rest] == "neutral")),
        "rest_active_fraction": float(np.mean(np.isin(decoded[rest],
                                                   [label for label in LABELS if label != "neutral"]))),
    }


def run(source: Path, bundle: Path, collection_root: Path, output: Path) -> dict:
    def evaluate(session_id: str) -> dict:
        print(f"replaying {session_id} Song 8-channel stream", flush=True)
        sha, model_sha, sample_count, trials, indices, probabilities = replay_session(
            source, bundle, collection_root, session_id)
        truth, rest, trial_count = cue_masks(trials, indices)
        rows = [score_rule(probabilities, truth, rest, count, alpha)
                for alpha in (0.25, 0.5, 0.75, 1.0) for count in (1, 2, 3)]
        return {
            "source_sha256": sha, "model_sha256": model_sha,
            "sample_count": sample_count, "frame_count": len(indices),
            "valid_formal_trials_with_stable_frames": trial_count,
            "rules": rows,
        }

    sessions = {"S03": evaluate("S03")}
    validation = {(row["ema_alpha"], row["consecutive_frames"]): row
                  for row in sessions["S03"]["rules"]}
    baseline = validation[(1.0, 3)]
    eligible = [row for row in validation.values()
                if (row["rest_neutral_fraction"] >= baseline["rest_neutral_fraction"] - 0.02 and
                    row["whole_stream_changes_per_minute"] <= baseline["whole_stream_changes_per_minute"])]
    preferred = max(eligible, key=lambda row: (row["stable_macro_f1"],
                                                row["stable_accuracy"], row["consecutive_frames"],
                                                row["ema_alpha"]))
    selected = ((preferred["ema_alpha"], preferred["consecutive_frames"])
                if preferred["stable_macro_f1"] >= baseline["stable_macro_f1"] + 0.02 else (1.0, 3))
    # The selection is fixed before the final session is even loaded.
    sessions["S04"] = evaluate("S04")
    final = {(row["ema_alpha"], row["consecutive_frames"]): row
             for row in sessions["S04"]["rules"]}
    result = {
        "status": "exploratory_same_person_day_decoder_rule_selection",
        "fixed_candidates": "0.5 probability threshold; causal probability EMA alpha 0.25/0.5/0.75/1.0 and one/two/three consecutive 100 ms frames",
        "selection": "Exploratory second-stage S03 selection after unsmoothed 1-frame rule was found to chatter: require rest-cue neutral fraction no more than 2 percentage points below current unsmoothed 3-frame rule and whole-stream state-change rate no higher than that rule; choose largest stable macro-F1, then accuracy, then more frames and alpha. Switch only for at least 2 points macro-F1 gain. S04 never selects.",
        "selected_rule": {"ema_alpha": selected[0], "consecutive_frames": selected[1]},
        "validation_gain_macro_f1_vs_current": validation[selected]["stable_macro_f1"] - baseline["stable_macro_f1"],
        "final_gain_macro_f1_vs_current": final[selected]["stable_macro_f1"] - final[(1.0, 3)]["stable_macro_f1"],
        "sessions": sessions,
        "boundary": "S03 collection readiness failed; S04 is same participant/day and its current-rule results had already been inspected before this candidate study. Recorded cue intervals are not physiological movement onsets, and rest may contain residual movement. Decoder-state replay is not physical USB/UI timing or cross-person/day validation."
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"selected_rule": result["selected_rule"],
                      "S03_gain_f1": result["validation_gain_macro_f1_vs_current"],
                      "S04_gain_f1": result["final_gain_macro_f1_vs_current"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.bundle, args.collection_root, args.output)
