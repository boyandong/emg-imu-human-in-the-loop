"""Fixed-family Song screening on identical one-person/day held-out sessions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.song_real8_study import _join_batches, load_session
from benchmarks.song_spd_increment_study import metrics, trial_probabilities
from emgimu.feature_bank.families import (
    BodyContextFamily, CspSpatialFamily, LocalDetailFamily, ScalePatternFamily,
    SpectralStateFamily, SpdTangentFamily, TemporalFormFamily,
    TraceCovarianceFamily,
)


SESSIONS = ("S01", "S02", "S03", "S04")
CLASSES = ("fist", "index_pinch", "neutral", "open_hand")
FAMILIES = {
    "F0": LocalDetailFamily,
    "F1": ScalePatternFamily,
    "F2a": TraceCovarianceFamily,
    "F2b": CspSpatialFamily,
    "F2c": SpdTangentFamily,
    "F4": SpectralStateFamily,
    "F5": TemporalFormFamily,
    "F6": BodyContextFamily,
}
SCREEN = {"F0": ("F0",)} | {f"F0_{name}": ("F0", name)
                              for name in FAMILIES if name != "F0"}
CORE = "F0_F2c"
CONDITIONAL = {f"Core_{name}": ("F0", "F2c", name)
               for name in FAMILIES if name not in ("F0", "F2c")}
ARMS = SCREEN | CONDITIONAL


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _per_class_f1(confusion: list[list[int]]) -> dict[str, float]:
    matrix = np.asarray(confusion, dtype=np.int64)
    result = {}
    for index, name in enumerate(CLASSES):
        denominator = matrix[index].sum() + matrix[:, index].sum()
        result[name] = float(2 * matrix[index, index] / denominator) if denominator else 0.0
    return result


def run(source: Path, previous: Path, output: Path, predictions: Path) -> dict:
    data = {sid: load_session(source / f"2026-09-18_{sid}", sid, "causal")
            for sid in SESSIONS}
    baseline = json.loads(previous.read_text(encoding="utf-8"))
    if baseline["status"] != "exploratory_one_person_one_day_leave_one_session_out":
        raise ValueError("prior Song held-out experiment has different semantics")
    rows, folds = [], {}
    for held_out in SESSIONS:
        training = tuple(sid for sid in SESSIONS if sid != held_out)
        if (baseline["folds"][held_out]["train_sessions"] != list(training) or
                baseline["folds"][held_out]["test_hdf5_sha256"] != data[held_out]["audit"]["sha256"]):
            raise ValueError("held-out split or source recording changed")
        train_batch = _join_batches([data[sid] for sid in training])
        train_y = np.concatenate([data[sid]["hand"] for sid in training])
        print(f"{held_out}: fit eight source-only feature families", flush=True)
        fitted = {}
        for name, factory in FAMILIES.items():
            family = factory()
            family.fit(train_batch, train_y if name == "F2b" else None)
            fitted[name] = {sid: family.transform(data[sid]["batch"])
                            for sid in (*training, held_out)}
        fold_scores, fold_predictions = {}, {}
        expected_ids, expected_truth = None, None
        for arm, members in ARMS.items():
            x_train = np.concatenate([
                np.concatenate([fitted[name][sid] for name in members], axis=1)
                for sid in training])
            x_test = np.concatenate([fitted[name][held_out] for name in members], axis=1)
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000, random_state=0))
            model.fit(x_train, train_y)
            classes = model[-1].classes_
            if tuple(classes) != CLASSES:
                raise ValueError("Song class order changed")
            ids, truth, probability = trial_probabilities(
                data[held_out]["hand"], model.predict_proba(x_test),
                data[held_out]["trial"], classes)
            if expected_ids is None:
                expected_ids, expected_truth = ids, truth
            elif not np.array_equal(ids, expected_ids) or not np.array_equal(truth, expected_truth):
                raise ValueError("family arm changed held-out trial membership")
            score = metrics(truth, probability, classes)
            score["per_class_f1"] = _per_class_f1(score["confusion_matrix"])
            fold_scores[arm] = score
            fold_predictions[arm] = probability
            if arm == CORE:
                old = baseline["folds"][held_out]["scores"]
                if any(abs(score[key] - old[key]) > 1e-8
                       for key in ("accuracy", "macro_f1", "log_loss", "brier")):
                    raise ValueError(f"{held_out}: frozen F0+F2c baseline failed replay")
            for trial, label, values in zip(ids, truth, probability, strict=True):
                rows.append({"held_out_session": held_out, "arm": arm,
                             "trial_id": trial, "true_label": label,
                             **{f"p_{name}": format(float(value), ".17g")
                                for name, value in zip(CLASSES, values, strict=True)}})
        fold_deltas = {}
        for added in (name for name in FAMILIES if name not in ("F0", "F2c")):
            arm = f"Core_{added}"
            core, candidate = fold_scores[CORE], fold_scores[arm]
            core_correct = np.asarray(CLASSES)[np.argmax(fold_predictions[CORE], axis=1)] == expected_truth
            added_correct = np.asarray(CLASSES)[np.argmax(fold_predictions[arm], axis=1)] == expected_truth
            fold_deltas[added] = {
                "delta_log_loss": core["log_loss"] - candidate["log_loss"],
                "delta_brier": core["brier"] - candidate["brier"],
                "delta_macro_f1": candidate["macro_f1"] - core["macro_f1"],
                "corrected_trials": int(np.sum(~core_correct & added_correct)),
                "new_errors": int(np.sum(core_correct & ~added_correct)),
            }
        folds[held_out] = {"training_sessions": list(training),
                           "trial_count": int(len(expected_ids)),
                           "scores": fold_scores, "conditional_vs_core": fold_deltas}
        print(f"{held_out}: core F1={fold_scores[CORE]['macro_f1']:.4f}, "
              f"screened {len(ARMS)} fixed arms", flush=True)
    predictions.parent.mkdir(parents=True, exist_ok=True)
    fields = ("held_out_session", "arm", "trial_id", "true_label") + tuple(f"p_{c}" for c in CLASSES)
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with predictions.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    if len(saved) != sum(fold["trial_count"] for fold in folds.values()) * len(ARMS):
        raise ValueError("saved family prediction rows are incomplete")
    prior_csv = previous.parent / baseline["predictions_csv"]
    if _sha256(prior_csv) != baseline["predictions_sha256"]:
        raise ValueError("prior Song core prediction file changed")
    with prior_csv.open(encoding="utf-8", newline="") as handle:
        prior_rows = {row["trial_id"]: row for row in csv.DictReader(handle)}
    core_rows = [row for row in saved if row["arm"] == CORE]
    if len(prior_rows) != len(core_rows) or {row["trial_id"] for row in core_rows} != set(prior_rows):
        raise ValueError("core trial identities differ from prior Song experiment")
    max_core_error = max(abs(float(row[f"p_{name}"]) -
                             float(prior_rows[row["trial_id"]][f"p_{name}"]))
                         for row in core_rows for name in CLASSES)
    if max_core_error > 1e-12:
        raise ValueError(f"core probability replay error {max_core_error}")
    pooled = {}
    classes = np.asarray(CLASSES)
    for arm in ARMS:
        subset = [row for row in saved if row["arm"] == arm]
        if len({row["trial_id"] for row in subset}) != len(subset):
            raise ValueError(f"repeated trial in {arm}")
        truth = np.asarray([row["true_label"] for row in subset])
        probability = np.asarray([[float(row[f"p_{name}"]) for name in CLASSES]
                                  for row in subset])
        pooled[arm] = metrics(truth, probability, classes)
        pooled[arm]["per_class_f1"] = _per_class_f1(pooled[arm]["confusion_matrix"])
        for sid in SESSIONS:
            part = [row for row in subset if row["held_out_session"] == sid]
            values = np.asarray([[float(row[f"p_{name}"]) for name in CLASSES]
                                 for row in part])
            check = metrics(np.asarray([row["true_label"] for row in part]), values, classes)
            if any(not np.isclose(check[key], folds[sid]["scores"][arm][key], atol=1e-12, rtol=0)
                   for key in ("accuracy", "macro_f1", "log_loss", "brier")):
                raise ValueError(f"saved {sid}/{arm} scores failed read-back")
    result = {
        "status": "exploratory_song_fixed_family_held_out_screen",
        "date": "2026-09-18", "participants": 1,
        "source_hdf5_sha256": {sid: data[sid]["audit"]["sha256"] for sid in SESSIONS},
        "prior_core_study_sha256": _sha256(previous),
        "prior_core_predictions_sha256": _sha256(prior_csv),
        "max_core_probability_replay_error": max_core_error,
        "family_ids": {name: FAMILIES[name]().family_id for name in FAMILIES},
        "arms": {name: list(members) for name, members in ARMS.items()},
        "core_arm": CORE, "folds": folds, "pooled_scores": pooled,
        "predictions_csv": predictions.name, "predictions_sha256": _sha256(predictions),
        "boundary": "Fixed causal F0 and seven reference families on identical cued stable trial IDs, with all feature state and classifiers fitted on three other sessions. F2b uses training labels only. One person, one day, S01-S03 readiness failures, and already inspected S04 make this exploratory within-day evidence, not cross-person/day validation or a live model-selection study. Reference families are not claimed historically exact.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "arms": len(ARMS),
                      "prediction_rows": len(saved)}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--prior-core", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.prior_core, args.output, args.predictions)
