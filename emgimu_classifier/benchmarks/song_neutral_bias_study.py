"""Validation-only neutral-bias selection for Song F0+SPD stream decoding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

from benchmarks.song_continuous_replay import replay_session
from benchmarks.song_decoder_rule_study import cue_masks
from benchmarks.song_real8_study import _text


def late_rest_mask(trials, indices):
    mask = np.zeros(len(indices), dtype=bool)
    for row in trials:
        if (_text(row["trial_kind"]) != "formal" or not bool(row["valid"]) or
                _text(row["completion_status"]) != "completed"):
            continue
        stable = ((indices - 49 >= int(row["stable_start_sample"])) &
                  (indices < int(row["stable_end_sample"])))
        if not np.any(stable):
            continue
        start = max(int(row["rest_start_sample"]), int(row["prompt_start_sample"]) - 100)
        end = int(row["prompt_start_sample"])
        mask |= (indices - 49 >= start) & (indices < end)
    if not np.any(mask):
        raise ValueError("no full windows in late pre-prompt rest")
    return mask


def score(probabilities, stable_truth, late_rest, bias):
    from emgforce.inference.song_local import LABELS, SongOnlineDecision

    p = np.asarray(probabilities, dtype=np.float64).copy()
    p[:, LABELS.index("neutral")] *= np.exp(bias)
    p /= p.sum(axis=1, keepdims=True)
    decoder = SongOnlineDecision()
    output = []
    for row in p:
        label, _ = decoder.step(row, 0.5)
        output.append(label if label is not None else "unknown")
    output = np.asarray(output)
    stable = stable_truth != ""
    true, predicted = stable_truth[stable].astype(str), output[stable]
    active = tuple(label for label in LABELS if label != "neutral")
    changes = int(np.count_nonzero(output[1:] != output[:-1]))
    return {
        "neutral_logit_bias": bias,
        "stable_frames": int(np.sum(stable)), "late_rest_frames": int(np.sum(late_rest)),
        "stable_decoded_accuracy": float(accuracy_score(true, predicted)),
        "stable_decoded_macro_f1": float(f1_score(true, predicted, labels=LABELS,
                                                   average="macro", zero_division=0)),
        "stable_decoded_recall": dict(zip(LABELS, map(float, recall_score(
            true, predicted, labels=LABELS, average=None, zero_division=0)))),
        "late_rest_decoded_active_fraction": float(np.mean(np.isin(output[late_rest], active))),
        "late_rest_decoded_neutral_fraction": float(np.mean(output[late_rest] == "neutral")),
        "whole_stream_state_changes_per_minute": changes / (len(output) * 0.1 / 60),
    }


def paired_stream(source, collection_root, f0_bundle, spd_bundle, session_id):
    print(f"replay {session_id} F0 and F0+SPD", flush=True)
    f0 = replay_session(source, f0_bundle, collection_root, session_id)
    spd = replay_session(source, spd_bundle, collection_root, session_id)
    for index in (0, 2):
        if f0[index] != spd[index]:
            raise ValueError("paired streams have different source or sample count")
    if not np.array_equal(f0[4], spd[4]):
        raise ValueError("paired frame indices differ")
    truth, rest, count = cue_masks(f0[3], f0[4])
    late = late_rest_mask(f0[3], f0[4])
    if np.any(late & (truth != "")) or np.any(late & ~rest):
        raise ValueError("late-rest windows must be strictly inside rest cues")
    return {"source_sha256": f0[0], "f0_model_sha256": f0[1],
            "spd_model_sha256": spd[1], "formal_trials": count,
            "truth": truth, "late": late, "f0_probability": f0[5],
            "spd_probability": spd[5]}


def run(source, collection_root, f0_bundle, spd_bundle, output):
    validation = paired_stream(source, collection_root, f0_bundle, spd_bundle, "S03")
    baseline = score(validation["f0_probability"], validation["truth"], validation["late"], 0.0)
    candidates = [score(validation["spd_probability"], validation["truth"],
                        validation["late"], bias)
                  for bias in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)]
    eligible = [row for row in candidates
                if (row["late_rest_decoded_active_fraction"] <=
                    baseline["late_rest_decoded_active_fraction"] + 0.02 and
                    row["stable_decoded_macro_f1"] >= baseline["stable_decoded_macro_f1"] + 0.02 and
                    row["whole_stream_state_changes_per_minute"] <=
                    baseline["whole_stream_state_changes_per_minute"] * 1.2)]
    selected = (max(eligible, key=lambda row: (row["stable_decoded_macro_f1"],
                                            -row["late_rest_decoded_active_fraction"],
                                            -row["neutral_logit_bias"]))
                if eligible else None)
    # Only now load S04, after S03 has fixed the rule or rejected all candidates.
    final_stream = paired_stream(source, collection_root, f0_bundle, spd_bundle, "S04")
    final = {"F0": score(final_stream["f0_probability"], final_stream["truth"],
                         final_stream["late"], 0.0),
             "F0_plus_SPD_unbiased": score(final_stream["spd_probability"],
                                            final_stream["truth"], final_stream["late"], 0.0)}
    if selected is not None:
        final["F0_plus_SPD_S03_selected"] = score(final_stream["spd_probability"],
                                                   final_stream["truth"], final_stream["late"],
                                                   selected["neutral_logit_bias"])
    result = {
        "status": "exploratory_one_person_day_neutral_bias_tradeoff",
        "fixed_candidates": "Add neutral logit bias 0/0.25/0.5/0.75/1/1.25/1.5 to F0+SPD probabilities; fixed 0.5 threshold and 3-frame online decision",
        "selection": "Use S03 only: late-rest decoded active fraction <= F0 +2 percentage points, stable decoded macro-F1 >= F0 +2 points, and whole-stream state changes/minute <= F0 *1.2; maximize F1, then lower rest active, then lower bias. S04 loads after this decision.",
        "selected_neutral_logit_bias": None if selected is None else selected["neutral_logit_bias"],
        "validation": {"source_sha256": validation["source_sha256"],
                       "f0_model_sha256": validation["f0_model_sha256"],
                       "spd_model_sha256": validation["spd_model_sha256"],
                       "formal_trials": validation["formal_trials"],
                       "F0": baseline, "F0_plus_SPD_candidates": candidates},
        "final": {"source_sha256": final_stream["source_sha256"],
                  "formal_trials": final_stream["formal_trials"], "scores": final},
        "boundary": "S03 failed formal collection readiness; S04 same participant/day and previously inspected for other Song experiments. Cue rest can contain residual movement; late rest is only the final 400 ms before prompt and not physiological ground truth. This offline decoder-state replay does not measure physical USB/UI timing or cross-person/day transfer."
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"selected_bias": result["selected_neutral_logit_bias"],
                      "S03_eligible": len(eligible),
                      "S04_baseline_f1": final["F0"]["stable_decoded_macro_f1"],
                      "S04_spd_f1": final["F0_plus_SPD_unbiased"]["stable_decoded_macro_f1"]}),
          flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--f0-bundle", required=True, type=Path)
    parser.add_argument("--spd-bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.collection_root, args.f0_bundle, args.spd_bundle, args.output)
