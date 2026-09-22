"""Recompute frozen EPN Anchor predictions from native data without fitting."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from emgimu.datasets.epn612 import load_epn612_windows
from emgimu.feature_bank.epn_calibration import _anchor_probability
from emgimu.feature_bank.epn_study import aggregate_trials


def replay(archive, run, correction):
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    phase = manifest["phase"]
    if phase not in ("validation", "final") or manifest["calibration_method"] != "anchor":
        raise ValueError("Only corrected frozen Anchor phases are supported")
    expected = correction["corrected_runs"][phase]
    if run.name != expected["run_id"]:
        raise ValueError("Not the audited corrected run")
    state_path = run / "fitted_source_state.pkl"
    state_bytes = state_path.read_bytes()
    if hashlib.sha256(state_bytes).hexdigest() != expected["files_sha256"][state_path.name]:
        raise ValueError("Untrusted or changed source state")
    # Pickle is loaded only after matching the hash of this locally produced state.
    state = pickle.loads(state_bytes)
    if set(state) != {"families", "scaler", "classifier"} or list(state["families"]) != manifest["families"]:
        raise ValueError("Unexpected source-state contents")
    state_before = pickle.dumps(state)
    splits = json.loads((run / "split_trial_ids.json").read_text(encoding="utf-8"))
    target = load_epn612_windows(archive, users=manifest["target_users"])
    parts = []
    labels = users = trials = weights = None
    for name, family in state["families"].items():
        part, y, u, t, w = aggregate_trials(family.transform(target.batch), target)
        if labels is not None:
            for actual, previous in ((y, labels), (u, users), (t, trials), (w, weights)):
                np.testing.assert_array_equal(actual, previous)
        parts.append(part)
        labels, users, trials, weights = y, u, t, w
    standardized = state["scaler"].transform(np.concatenate(parts, axis=1))
    population = state["classifier"].predict_proba(standardized)
    max_error, checked, trial_count = 0., 0, 0
    with np.load(run / "heldout_predictions.npz", allow_pickle=False) as saved:
        for subject in manifest["target_users"]:
            personal = users == subject
            for shots in (0, 1, 2, 5):
                key = f"user{subject}_shots{shots}"
                partition = splits["target_cases"][key]
                calibration = personal & np.isin(trials, partition["calibration"])
                evaluation = personal & np.isin(trials, partition["evaluation"])
                if set(partition["calibration"]) & set(partition["evaluation"]):
                    raise ValueError("Calibration/evaluation trial overlap")
                if set(trials[personal]) != set(partition["calibration"]) | set(partition["evaluation"]):
                    raise ValueError("Incomplete personal trial partition")
                if shots:
                    personal_probability = _anchor_probability(standardized[calibration], labels[calibration], standardized[evaluation])
                    weight = shots / (2.0 + shots)
                    predicted = (1.0 - weight) * population[evaluation] + weight * personal_probability
                    predicted /= predicted.sum(axis=1, keepdims=True)
                else:
                    predicted = population[evaluation]
                expected_probability = saved[f"{key}_probability"]
                np.testing.assert_array_equal(trials[evaluation], saved[f"{key}_trials"])
                np.testing.assert_array_equal(labels[evaluation], saved[f"{key}_labels"])
                np.testing.assert_array_equal(weights[evaluation], saved[f"{key}_weights"])
                error = float(np.max(np.abs(predicted - expected_probability)))
                np.testing.assert_allclose(predicted, expected_probability, rtol=0, atol=1e-12)
                max_error = max(max_error, error)
                checked += 1
                trial_count += len(expected_probability)
    if state_path.read_bytes() != state_bytes or pickle.dumps(state) != state_before:
        raise AssertionError("Frozen state file changed")
    audit = {"status": "ok", "phase": phase, "native_target_users": manifest["target_users"],
             "native_archive_bytes": archive.stat().st_size, "fresh_whole_archive_digest_computed": False,
             "probability_arrays_recomputed": checked, "evaluation_trial_predictions_recomputed": trial_count,
             "maximum_absolute_probability_error": max_error, "source_state_file_unchanged": True,
             "source_state_in_memory_unchanged": True,
             "source_state_sha256": hashlib.sha256(state_bytes).hexdigest(),
             "scope": "Native target windows -> saved source features/scaler/classifier -> calibration-only Anchor; no model refit."
                      " Historical family identity and device performance remain unverified."}
    (run / "native_replay_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("status", "phase", "probability_arrays_recomputed", "maximum_absolute_probability_error")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("correction_audit", type=Path)
    args = parser.parse_args()
    replay(args.archive, args.run, json.loads(args.correction_audit.read_text(encoding="utf-8")))
