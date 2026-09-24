"""Read back source OOF selection and every Song 28-state calibrated trial probability."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2.song_28_source_calibration import (
    PROTOCOL_PATH, ROOT, select_temperature, temperature_probability,
)
from benchmarks.song_28_spd_increment_study import detailed_metrics
from benchmarks.song_real8_study import parse_label


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify() -> dict:
    result = json.loads((ROOT / "SONG_28_SOURCE_CAL_RESULTS.json").read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    prior_path = ROOT / "SONG_ARM_CAL_RESULTS.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    old28 = json.loads((ROOT / "SONG_28_RESULTS.json").read_text(encoding="utf-8"))
    if (result["protocol"] != protocol
            or result["protocol_sha256"] != hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
            or result["reference_arm_results_sha256"] != hashlib.sha256(prior_path.read_bytes()).hexdigest()
            or result["runtime_versions"] != prior["runtime_versions"]
            or result["source_hdf5_sha256"] != prior["source_hdf5_sha256"]):
        raise ValueError("frozen source-calibration provenance changed")
    joint_classes = np.asarray(old28["classes"])
    hands = np.asarray(sorted({parse_label(label)[1] for label in joint_classes}))
    arms = np.asarray(sorted({parse_label(label)[0] for label in joint_classes}))
    oof = rows(ROOT / "SONG_28_SOURCE_CAL_OOF.csv")
    if len(oof) != result["source_oof_trial_rows"] or len(oof) != 285:
        raise ValueError("source OOF trial coverage changed")
    seen = set()
    hand_truth = []; arm_truth = []; hand_probability = []; arm_probability = []
    fold_counts = {}
    for row in oof:
        fit, heldout, trial = row["fit_session"], row["heldout_session"], row["trial_id"]
        if ([fit, heldout] not in protocol["source_oof_folds"]
                or not trial.startswith(heldout + ":") or trial in seen):
            raise ValueError("OOF row fitted on its own held-out session or duplicated")
        seen.add(trial)
        fold_counts[heldout] = fold_counts.get(heldout, 0) + 1
        hand_truth.append(row["hand_truth"]); arm_truth.append(row["arm_truth"])
        hand_probability.append([float(row[f"hand_p_{name}"]) for name in hands])
        arm_probability.append([float(row[f"arm_p_{name}"]) for name in arms])
    if fold_counts != {"S02": 142, "S01": 143}:
        raise ValueError("source OOF fold sizes changed")
    chosen = {}
    for branch, truth, probabilities, classes in (
            ("hand", hand_truth, hand_probability, hands),
            ("arm", arm_truth, arm_probability, arms)):
        p = np.asarray(probabilities)
        if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-12):
            raise ValueError(f"invalid OOF {branch} probabilities")
        selected, scores = select_temperature(np.asarray(truth), p, classes,
                                               protocol["temperature_candidates"])
        recorded = result["source_oof"][branch]
        if selected != recorded["selected_temperature"] or len(scores) != len(recorded["scores"]):
            raise ValueError(f"OOF {branch} temperature selection changed")
        for actual, saved in zip(scores, recorded["scores"]):
            if (actual["temperature"] != saved["temperature"] or
                    not np.isclose(actual["source_oof_trial_log_loss"],
                                   saved["source_oof_trial_log_loss"], rtol=0, atol=1e-12)):
                raise ValueError(f"OOF {branch} score changed")
        chosen[branch] = selected
    prediction_rows = rows(ROOT / "SONG_28_SOURCE_CAL_TRIAL_PREDICTIONS.csv")
    baseline = {(row["phase"], row["trial_id"]): row for row in
                rows(ROOT / "SONG_ARM_CAL_TRIAL_PREDICTIONS.csv")
                if row["arm"] == "source_arm_x_same_hand"}
    if len(prediction_rows) != result["prediction_rows"] or len(prediction_rows) != 568:
        raise ValueError("held-out prediction rows changed")
    checked = 0; max_baseline_error = 0.0; max_temperature_replay_error = 0.0
    for phase, session in (("validation", "S03"), ("final", "S04")):
        ids = result["sessions"][phase]["trial_ids"]
        by_arm = {}
        for name in protocol["comparison"]:
            subset = [row for row in prediction_rows if row["phase"] == phase and row["arm"] == name]
            if ([row["trial_id"] for row in subset] != ids
                    or any(row["session"] != session for row in subset)):
                raise ValueError(f"native trial order changed: {phase}/{name}")
            truth = np.asarray([row["label"] for row in subset])
            p = np.asarray([[float(row[f"p_{klass}"]) for klass in joint_classes] for row in subset])
            if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-12):
                raise ValueError("invalid held-out probability vector")
            actual = detailed_metrics(truth, p, joint_classes)
            saved = result["sessions"][phase]["arms"][name]
            for key in ("trials", "accuracy", "macro_f1", "log_loss", "brier", "arm_accuracy", "hand_accuracy"):
                if not np.isclose(actual[key], saved[key], rtol=0, atol=1e-12):
                    raise ValueError(f"saved metric changed: {phase}/{name}/{key}")
            if actual["confusion_matrix"] != saved["confusion_matrix"]:
                raise ValueError("joint confusion matrix changed")
            by_arm[name] = (truth, p)
            checked += 1
        unc_truth, unc = by_arm[protocol["comparison"][0]]
        cal_truth, cal = by_arm[protocol["comparison"][1]]
        if not np.array_equal(unc_truth, cal_truth) or not np.array_equal(unc.argmax(axis=1), cal.argmax(axis=1)):
            raise ValueError("temperature altered labels or hard decisions")
        for trial, label, p in zip(ids, unc_truth, unc):
            reference = baseline.get((phase, trial))
            if reference is None or reference["label"] != label:
                raise ValueError("frozen factorized trial missing")
            max_baseline_error = max(max_baseline_error, max(
                abs(float(reference[f"p_{klass}"]) - value) for klass, value in zip(joint_classes, p)))
        hand = np.stack([unc[:, [i for i, label in enumerate(joint_classes)
                                 if parse_label(label)[1] == klass]].sum(axis=1) for klass in hands], axis=1)
        arm = np.stack([unc[:, [i for i, label in enumerate(joint_classes)
                                if parse_label(label)[0] == klass]].sum(axis=1) for klass in arms], axis=1)
        hand = temperature_probability(hand, chosen["hand"])
        arm = temperature_probability(arm, chosen["arm"])
        replay = np.stack([hand[:, np.flatnonzero(hands == parse_label(label)[1])[0]] *
                           arm[:, np.flatnonzero(arms == parse_label(label)[0])[0]]
                           for label in joint_classes], axis=1)
        max_temperature_replay_error = max(max_temperature_replay_error,
                                           float(np.max(np.abs(replay - cal))))
    if max_baseline_error > 1e-12 or max_temperature_replay_error > 1e-12:
        raise ValueError("baseline parity or temperature transformation changed")
    if not np.isclose(max_baseline_error, result["max_uncalibrated_baseline_probability_error"], atol=1e-12):
        raise ValueError("recorded baseline parity differs")
    verification = {"status": "verified", "source_oof_trials": len(oof),
                    "heldout_native_trials": len(prediction_rows) // len(protocol["comparison"]),
                    "metric_groups_read_back": checked,
                    "temperatures_selected_from_source_oof_only": chosen,
                    "max_uncalibrated_baseline_probability_error": max_baseline_error,
                    "max_temperature_probability_replay_error": max_temperature_replay_error,
                    "hard_decisions_unchanged": True}
    (ROOT / "SONG_28_SOURCE_CAL_VERIFICATION.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    return verification


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
