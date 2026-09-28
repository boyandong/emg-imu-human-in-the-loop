"""Independently audit saved DS2 v9 personal-calibration assignments and scores."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, log_loss


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "public_ds2_force_v9"
JOIN = ROOT / "DS2_V9_FORCE_TRIAL_JOIN.csv"
SHOTS = (0, 1, 2, 5)
SCHEDULE = {"unseen_high": (1, 0, 1, 0, 1), "product_all": (1, 0, 2, 1, 0)}
SUBJECTS = {"validation": range(13, 17), "final_descriptive": range(17, 21)}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check_score(rows: list[dict[str, str]], saved: dict) -> None:
    y = np.array([int(r["gesture_code"]) for r in rows])
    p = np.array([[float(r[f"p{k}"]) for k in range(4)] for r in rows])
    predicted = np.argmax(p, axis=1)
    assert len(rows) == saved["trials"]
    assert np.isclose(np.mean(predicted == y), saved["accuracy"], atol=1e-12)
    assert np.isclose(f1_score(y, predicted, labels=range(4), average="macro", zero_division=0),
                      saved["macro_f1"], atol=1e-12)
    assert np.isclose(log_loss(y, p, labels=range(4)), saved["log_loss"], atol=1e-12)
    assert np.isclose(np.mean(np.sum((p - np.eye(4)[y]) ** 2, axis=1)), saved["brier"], atol=1e-12)
    assert confusion_matrix(y, predicted, labels=range(4)).tolist() == saved["confusion"]
    for k in range(4):
        denom = int(np.sum(y == k))
        recall = float(np.sum((y == k) & (predicted == k)) / denom) if denom else 0.0
        assert np.isclose(recall, saved["recall_by_code"][str(k)], atol=1e-12)


def main() -> None:
    result = json.loads((DATA / "CALIBRATION_RESULTS.json").read_text(encoding="utf-8"))
    protocol = json.loads((DATA / "CALIBRATION_PROTOCOL.json").read_text(encoding="utf-8"))
    assert protocol["budgets_per_gesture"] == list(SHOTS)
    assert protocol["calibration_schedule"] == {k: list(v) for k, v in SCHEDULE.items()}
    assert digest(DATA / "CALIBRATION_PROTOCOL.json") == result["protocol_sha256"]
    assert digest(DATA / "CALIBRATION_ASSIGNMENT.csv") == result["calibration_assignment_sha256"]
    assert digest(DATA / "CALIBRATION_TRIAL_PREDICTIONS.csv") == result["trial_predictions_sha256"]
    assert digest(JOIN) == result["source_sha256"]["trial_join"]
    assert digest(DATA / "TRIAL_PREDICTIONS.csv") == result["source_sha256"]["parent_predictions"]
    join = {int(r["raw_trial_index_zero_based"]): r for r in read(JOIN)}
    assignment = read(DATA / "CALIBRATION_ASSIGNMENT.csv")
    predictions = read(DATA / "CALIBRATION_TRIAL_PREDICTIONS.csv")
    parent = {(r["training_mode"], r["arm"], r["split"], int(r["raw_trial_index_zero_based"])): r
              for r in read(DATA / "TRIAL_PREDICTIONS.csv")}
    assert len(predictions) == result["prediction_rows"] == 12576
    grouped_assignment = defaultdict(list)
    for row in assignment:
        mode, split, subject, gesture, slot = (row["training_mode"], row["split"],
                                               int(row["subject_folder"]), int(row["gesture_code"]),
                                               int(row["shot_slot"]))
        trial = int(row["raw_trial_index_zero_based"])
        assert subject in SUBJECTS[split] and gesture in range(4) and slot in range(1, 6)
        source = join[trial]
        assert (int(source["subject_folder"]), int(source["gesture_code_if_uniform"]),
                int(source["force_code"])) == (subject, gesture, int(row["force_code"]))
        assert int(row["force_code"]) == SCHEDULE[mode][slot - 1]
        grouped_assignment[(mode, split, subject, gesture)].append((slot, trial))
    assert len(assignment) == 2 * 8 * 4 * 5 and len(grouped_assignment) == 2 * 8 * 4
    reserved = defaultdict(set)
    for key, items in grouped_assignment.items():
        mode, split, subject, gesture = key
        assert sorted(slot for slot, _ in items) == list(range(1, 6))
        selected = [trial for _, trial in sorted(items)]
        assert len(set(selected)) == 5
        reserved[(mode, split, subject)].update(selected)
        used = set()
        for slot, force in enumerate(SCHEDULE[mode]):
            candidates = [trial for trial, source in join.items()
                          if source["subject_folder"] != "N/A"
                          and source["gesture_code_if_uniform"] != "N/A"
                          and int(source["subject_folder"]) == subject
                          and int(source["gesture_code_if_uniform"]) == gesture
                          and int(source["force_code"]) == force and trial not in used]
            expected = min(candidates, key=lambda t: hashlib.sha256(
                f"20260928|{subject}|{gesture}|{t}".encode()).digest())
            assert selected[slot] == expected
            used.add(expected)

    groups = defaultdict(list)
    trial_sets = defaultdict(set)
    probability_max_error = 0.0
    for row in predictions:
        mode, arm, split, shot = (row["training_mode"], row["arm"], row["split"],
                                  int(row["shots_per_gesture"]))
        subject, trial, gesture, force = (int(row["subject_folder"]),
                                          int(row["raw_trial_index_zero_based"]),
                                          int(row["gesture_code"]), int(row["force_code"]))
        assert shot in SHOTS and subject in SUBJECTS[split]
        source = join[trial]
        assert (int(source["subject_folder"]), int(source["gesture_code_if_uniform"]),
                int(source["force_code"])) == (subject, gesture, force)
        assert force == 2 if mode == "unseen_high" else trial not in reserved[(mode, split, subject)]
        p = np.array([float(row[f"p{k}"]) for k in range(4)])
        assert np.all(np.isfinite(p)) and np.all(p >= 0)
        assert abs(float(p.sum()) - 1) < 1e-12
        if shot == 0:
            original = parent[(mode, arm, split, trial)]
            probability_max_error = max(
                probability_max_error,
                max(abs(float(row[f"p{k}"]) - float(original[f"p{k}"])) for k in range(4)))
        groups[(mode, arm, split, shot)].append(row)
        trial_sets[(mode, arm, split, shot)].add(trial)
    assert probability_max_error == 0.0
    assert len(groups) == 48
    checked = 0
    for (mode, arm, split, shot), rows in groups.items():
        assert len(trial_sets[(mode, arm, split, shot)]) == len(rows)
        assert trial_sets[(mode, arm, split, shot)] == trial_sets[(mode, arm, split, 0)]
        saved = result["scores"][mode][arm][split][str(shot)]
        check_score(rows, saved["all"])
        checked += 1
        for force, score in saved["by_force_code"].items():
            check_score([r for r in rows if r["force_code"] == force], score)
            checked += 1
        for subject, score in saved["by_subject"].items():
            check_score([r for r in rows if r["subject_folder"] == subject], score)
            checked += 1
    curve = []
    for mode in SCHEDULE:
        for arm in ("F0", "F1", "F0_plus_F1"):
            for split in SUBJECTS:
                for shot in SHOTS:
                    score = result["scores"][mode][arm][split][str(shot)]["all"]
                    curve.append({"training_mode": mode, "arm": arm, "split": split,
                                  "shots_per_gesture": shot, "trials": score["trials"],
                                  "macro_f1": score["macro_f1"],
                                  "log_loss": score["log_loss"], "brier": score["brier"],
                                  "accuracy": score["accuracy"]})
    with (DATA / "CALIBRATION_CURVE.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(curve)
    audit = {"status": "pass", "assignment_rows": len(assignment),
             "prediction_rows": len(predictions), "score_cells_recomputed": checked,
             "curve_rows": len(curve),
             "zero_shot_max_probability_error_vs_parent": probability_max_error,
             "budget_evaluation_trial_sets_identical": True,
             "force_safe_and_nested_hash_schedule": True,
             "input_sha256": {name: digest(DATA / name) for name in
                              ("CALIBRATION_PROTOCOL.json", "CALIBRATION_RESULTS.json",
                               "CALIBRATION_ASSIGNMENT.csv", "CALIBRATION_TRIAL_PREDICTIONS.csv")}}
    (DATA / "CALIBRATION_VERIFICATION.json").write_bytes(
        (json.dumps(audit, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({k: v for k, v in audit.items() if k != "input_sha256"}))


if __name__ == "__main__":
    main()
