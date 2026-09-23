"""S03-only temperature selection with live-decoder rest constraints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.song_continuous_replay import replay_session
from benchmarks.song_decoder_rule_study import cue_masks
from benchmarks.song_neutral_bias_study import late_rest_mask, score as score_stream
from benchmarks.song_probability_calibration_audit import calibration_bins
from benchmarks.song_real8_study import load_session
from benchmarks.song_spd_increment_study import metrics, trial_probabilities


TEMPERATURES = (0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0)


def apply_temperature(probability, temperature: float):
    p = np.asarray(probability, dtype=np.float64)
    if (not np.isfinite(temperature) or temperature <= 0 or p.ndim != 2 or
            not len(p) or not np.isfinite(p).all() or np.any(p < 0) or
            not np.allclose(p.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Finite normalized probability rows and positive temperature required")
    if temperature == 1.0:
        return p.copy()
    logits = np.log(np.maximum(p, 1e-12)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def load_evaluation(source, collection_root, bundle, session):
    sys.path.insert(0, str(collection_root.resolve()))
    from emgforce.inference.song_local import LABELS, SongLocalRuntime

    data = load_session(source / f"2026-09-18_{session}", session, filter_mode="causal")
    runtime = SongLocalRuntime(bundle)
    window_probability = np.stack([
        runtime.predict_filtered_window(window) for window in data["batch"].emg
    ])
    continuous = replay_session(source, bundle, collection_root, session)
    if continuous[0] != data["audit"]["sha256"] or continuous[1] != runtime.sha256:
        raise ValueError("Stable-window and continuous replays differ in source/model")
    truth, rest, formal_count = cue_masks(continuous[3], continuous[4])
    late_rest = late_rest_mask(continuous[3], continuous[4])
    if np.any(late_rest & ~rest):
        raise ValueError("late rest mask extends outside recorded rest")
    return {
        "source_sha256": continuous[0], "model_sha256": runtime.sha256,
        "data": data, "window_probability": window_probability,
        "continuous_probability": continuous[5], "truth": truth,
        "late_rest": late_rest, "continuous_formal_trials": formal_count,
        "labels": np.asarray(LABELS),
    }


def evaluate(evaluation, temperature: float):
    labels = evaluation["labels"]
    window = apply_temperature(evaluation["window_probability"], temperature)
    ids, truth, trial_probability = trial_probabilities(
        evaluation["data"]["hand"], window, evaluation["data"]["trial"], labels)
    trial = metrics(truth, trial_probability, labels)
    _, ece = calibration_bins(truth, trial_probability, labels)
    stream = score_stream(apply_temperature(evaluation["continuous_probability"], temperature),
                          evaluation["truth"], evaluation["late_rest"], 0.0)
    return {
        "temperature": temperature, "stable_formal_trials": len(ids),
        "stable_trial_accuracy": trial["accuracy"],
        "stable_trial_macro_f1": trial["macro_f1"],
        "stable_trial_log_loss": trial["log_loss"],
        "stable_trial_brier": trial["brier"],
        "stable_trial_top_label_ece_10_bins": ece,
        "stable_decoded_macro_f1": stream["stable_decoded_macro_f1"],
        "late_rest_decoded_active_fraction": stream["late_rest_decoded_active_fraction"],
        "whole_stream_state_changes_per_minute": stream["whole_stream_state_changes_per_minute"],
    }


def run(source: Path, collection_root: Path, bundle: Path, output: Path) -> dict:
    validation = load_evaluation(source, collection_root, bundle, "S03")
    candidates = [evaluate(validation, temperature) for temperature in TEMPERATURES]
    baseline = next(row for row in candidates if row["temperature"] == 1.0)
    eligible = [row for row in candidates if (
        row["late_rest_decoded_active_fraction"] <= baseline["late_rest_decoded_active_fraction"] + 0.02 and
        row["stable_decoded_macro_f1"] >= baseline["stable_decoded_macro_f1"] - 0.02 and
        row["whole_stream_state_changes_per_minute"] <=
        baseline["whole_stream_state_changes_per_minute"] * 1.2)]
    preferred = min(eligible, key=lambda row: (
        row["stable_trial_log_loss"], row["late_rest_decoded_active_fraction"],
        abs(row["temperature"] - 1)))
    selected = (preferred["temperature"] if
                preferred["stable_trial_log_loss"] <= baseline["stable_trial_log_loss"] - 0.01
                else 1.0)

    # The final session is loaded only after the validation-only choice.
    final = load_evaluation(source, collection_root, bundle, "S04")
    final_baseline = evaluate(final, 1.0)
    final_selected = evaluate(final, selected)
    result = {
        "status": "exploratory_same_person_day_temperature_tradeoff_not_deployed",
        "fixed_candidates": list(TEMPERATURES),
        "selection_rule": "S03 only: candidate late-rest decoded active fraction <= unscaled+2pp, stable decoded macro-F1 >= unscaled-2pp, state changes/min <= unscaled*1.2; then minimum stable-trial LogLoss, tie-break lower rest active and temperature nearer 1; switch only if LogLoss improves >=0.01. S04 loads after selection.",
        "selected_temperature": selected,
        "validation": {"source_sha256": validation["source_sha256"],
                       "model_sha256": validation["model_sha256"],
                       "continuous_formal_trials": validation["continuous_formal_trials"],
                       "candidates": candidates},
        "final": {"source_sha256": final["source_sha256"],
                  "model_sha256": final["model_sha256"],
                  "continuous_formal_trials": final["continuous_formal_trials"],
                  "unscaled": final_baseline, "S03_selected": final_selected},
        "boundary": "Only existing source-trained bundle probabilities are post-processed; no classifier fit. S03 collection readiness failed, and S04 had already been inspected in earlier Song work, so this is exploratory. Stable-trial metrics use cue intervals and no actual movement onset; late-rest cues may include residual movement. No physical device, new user/day or live latency validation.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"selected_temperature": selected,
                      "S03_eligible": len(eligible),
                      "S04_log_loss_unscaled": final_baseline["stable_trial_log_loss"],
                      "S04_log_loss_selected": final_selected["stable_trial_log_loss"],
                      "S04_late_rest_active_unscaled": final_baseline["late_rest_decoded_active_fraction"],
                      "S04_late_rest_active_selected": final_selected["late_rest_decoded_active_fraction"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.collection_root, args.bundle, args.output)
