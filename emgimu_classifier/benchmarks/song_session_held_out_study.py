"""Exploratory Song F0+SPD leave-one-session-out study, one person/day only."""
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
from emgimu.feature_bank.families import LocalDetailFamily, SpdTangentFamily


SESSIONS = ("S01", "S02", "S03", "S04")
CLASSES = ("fist", "index_pinch", "neutral", "open_hand")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(source: Path, output: Path, predictions: Path) -> dict:
    data = {}
    for sid in SESSIONS:
        data[sid] = load_session(source / f"2026-09-18_{sid}", sid, "causal")
    folds, rows = {}, []
    for held_out in SESSIONS:
        training = tuple(sid for sid in SESSIONS if sid != held_out)
        if len(training) != 3 or held_out in training:
            raise ValueError("session split leaked the held-out recording")
        print(f"held-out {held_out}: fit F0+SPD on {','.join(training)}", flush=True)
        batch = _join_batches([data[sid] for sid in training])
        f0 = LocalDetailFamily().fit(batch)
        spd = SpdTangentFamily().fit(batch)

        def features(sid: str) -> np.ndarray:
            return np.concatenate((f0.transform(data[sid]["batch"]),
                                   spd.transform(data[sid]["batch"])), axis=1)

        train_x = np.concatenate([features(sid) for sid in training])
        train_y = np.concatenate([data[sid]["hand"] for sid in training])
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=0))
        model.fit(train_x, train_y)
        classes = model[-1].classes_
        if tuple(classes) != CLASSES:
            raise ValueError("unexpected Song class order")
        ids, truth, probability = trial_probabilities(
            data[held_out]["hand"], model.predict_proba(features(held_out)),
            data[held_out]["trial"], classes)
        if len(set(ids)) != len(ids) or any(not trial.startswith(held_out + ":") for trial in ids):
            raise ValueError("held-out trial identities are invalid")
        fold_scores = metrics(truth, probability, classes)
        folds[held_out] = {
            "train_sessions": list(training),
            "test_session": held_out,
            "train_windows": int(len(train_y)),
            "test_windows": int(len(data[held_out]["hand"])),
            "scores": fold_scores,
            "source_hdf5_sha256": {sid: data[sid]["audit"]["sha256"] for sid in training},
            "test_hdf5_sha256": data[held_out]["audit"]["sha256"],
        }
        for trial, label, probs in zip(ids, truth, probability, strict=True):
            rows.append({"held_out_session": held_out, "trial_id": trial,
                         "true_label": label,
                         **{f"p_{name}": format(float(value), ".17g")
                            for name, value in zip(CLASSES, probs, strict=True)}})
        print(f"held-out {held_out}: {fold_scores['trials']} trials, "
              f"macro-F1={fold_scores['macro_f1']:.4f}", flush=True)
    if len(rows) != sum(fold["scores"]["trials"] for fold in folds.values()):
        raise ValueError("trial prediction count changed")
    predictions.parent.mkdir(parents=True, exist_ok=True)
    fields = ("held_out_session", "trial_id", "true_label") + tuple(f"p_{c}" for c in CLASSES)
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    # Read back the deliverable, so reported scores bind to the serialized probabilities.
    with predictions.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    if len(saved) != len(rows) or len({row["trial_id"] for row in saved}) != len(rows):
        raise ValueError("saved trial predictions are missing or repeated")
    for sid in SESSIONS:
        subset = [row for row in saved if row["held_out_session"] == sid]
        truth = np.asarray([row["true_label"] for row in subset])
        probability = np.asarray([[float(row[f"p_{name}"]) for name in CLASSES]
                                  for row in subset])
        verified = metrics(truth, probability, np.asarray(CLASSES))
        for key in ("accuracy", "macro_f1", "log_loss", "brier"):
            if not np.isclose(verified[key], folds[sid]["scores"][key], rtol=0, atol=1e-12):
                raise ValueError(f"saved {sid} {key} differs from in-memory score")
    overall_truth = np.asarray([row["true_label"] for row in saved])
    overall_p = np.asarray([[float(row[f"p_{name}"]) for name in CLASSES] for row in saved])
    result = {
        "status": "exploratory_one_person_one_day_leave_one_session_out",
        "protocol": "fixed causal F0+SPD; three complete sessions train, fourth held out; no target-session fitting or calibration",
        "session_date": "2026-09-18", "participant_count": 1,
        "classes": list(CLASSES), "folds": folds,
        "pooled_trial_scores": metrics(overall_truth, overall_p, np.asarray(CLASSES)),
        "predictions_csv": predictions.name,
        "predictions_sha256": _digest(predictions),
        "boundary": "Exploratory within-day session transfer on valid cued stable trials. S01-S03 failed collection readiness; S04 passed. Folds reuse a single person's same-day data and several outcomes were inspected earlier. This does not measure cross-person/day generalization, continuous live accuracy, or an independent final test.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "trials": len(saved),
                      "pooled_macro_f1": result["pooled_trial_scores"]["macro_f1"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.output, args.predictions)
