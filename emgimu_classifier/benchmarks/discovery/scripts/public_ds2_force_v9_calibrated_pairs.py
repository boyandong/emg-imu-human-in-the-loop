"""Post-hoc paired family comparisons from frozen DS2 v9 calibration predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, log_loss


BASE = Path(__file__).resolve().parents[1] / "public_ds2_force_v9"
ARMS = ("F0", "F1", "F0_plus_F1")
SPLITS = {"validation": range(13, 17), "final_descriptive": range(17, 21)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(rows: dict[int, dict]) -> dict:
    truth = np.array([rows[trial]["truth"] for trial in sorted(rows)])
    probs = np.array([rows[trial]["probs"] for trial in sorted(rows)])
    prediction = probs.argmax(axis=1)
    return {"trials": len(truth),
            "macro_f1": float(f1_score(truth, prediction, labels=range(4),
                                        average="macro", zero_division=0)),
            "log_loss": float(log_loss(truth, probs, labels=range(4))),
            "brier": float(np.mean(np.sum((probs - np.eye(4)[truth]) ** 2, axis=1)))}


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path = BASE) -> dict:
    audit = json.loads((BASE / "CALIBRATION_VERIFICATION.json").read_text(encoding="utf-8"))
    source_scores = json.loads((BASE / "CALIBRATION_RESULTS.json").read_text(encoding="utf-8"))["scores"]
    if audit["status"] != "pass" or audit["prediction_rows"] != 12576:
        raise ValueError("Calibration predictions have not passed independent verification")
    for name, digest in audit["input_sha256"].items():
        if sha(BASE / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    indexed = defaultdict(dict)
    with (BASE / "CALIBRATION_TRIAL_PREDICTIONS.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            mode, arm, split, shots, subject = (row["training_mode"], row["arm"], row["split"],
                                                 int(row["shots_per_gesture"]),
                                                 int(row["subject_folder"]))
            trial = int(row["raw_trial_index_zero_based"])
            key = (mode, arm, split, shots)
            if trial in indexed[key]:
                raise ValueError("Duplicate trial probability")
            indexed[key][trial] = {"truth": int(row["gesture_code"]),
                                   "force": int(row["force_code"]), "subject": subject,
                                   "probs": [float(row[f"p{k}"]) for k in range(4)]}
    increments, complements = [], []
    headline = {}
    for mode in ("unseen_high", "product_all"):
        conditions = ("force2",) if mode == "unseen_high" else ("all", "force0", "force1", "force2")
        headline[mode] = {}
        for split, subjects in SPLITS.items():
            headline[mode][split] = {}
            for shots in (0, 1, 2, 5):
                arms = {arm: indexed[(mode, arm, split, shots)] for arm in ARMS}
                ids = set(arms["F0"])
                if any(set(arms[arm]) != ids for arm in ARMS):
                    raise ValueError("Arms do not have paired evaluation trials")
                for trial in ids:
                    if len({(arms[arm][trial]["truth"], arms[arm][trial]["force"],
                             arms[arm][trial]["subject"]) for arm in ARMS}) != 1:
                        raise ValueError("Paired trial labels or grouping differ")
                for condition in conditions:
                    for subject in ("ALL", *(str(person) for person in subjects)):
                        selected = sorted(trial for trial in ids
                                          if (condition == "all" or arms["F0"][trial]["force"] == int(condition[-1]))
                                          and (subject == "ALL" or arms["F0"][trial]["subject"] == int(subject)))
                        if not selected:
                            raise ValueError("Empty paired condition cell")
                        subset = {arm: {trial: arms[arm][trial] for trial in selected} for arm in ARMS}
                        scores = {arm: score(subset[arm]) for arm in ARMS}
                        for arm in ARMS:
                            saved = source_scores[mode][arm][split][str(shots)]
                            reference = (saved["all"] if subject == "ALL" and (condition == "all" or mode == "unseen_high")
                                         else saved["by_force_code"][condition[-1]]
                                         if subject == "ALL" and condition.startswith("force")
                                         else saved["by_subject"][subject]
                                         if condition == "all" or mode == "unseen_high" else None)
                            if reference is not None:
                                for metric in ("trials", "macro_f1", "log_loss", "brier"):
                                    if not np.isclose(scores[arm][metric], reference[metric], atol=1e-12):
                                        raise ValueError(f"Saved score mismatch: {mode}/{arm}/{split}/{shots}/{condition}/{subject}/{metric}")
                        base, joint = scores["F0"], scores["F0_plus_F1"]
                        common = {"training_mode": mode, "split": split,
                                  "shots_per_gesture": shots, "condition": condition,
                                  "subject_folder": subject, "evaluation_trials": len(selected)}
                        increments.append({**common, "core_bank": "F0", "added_family": "F1",
                                           "delta_macro_f1": joint["macro_f1"] - base["macro_f1"],
                                           "delta_log_loss": joint["log_loss"] - base["log_loss"],
                                           "delta_brier": joint["brier"] - base["brier"],
                                           "F0_macro_f1": base["macro_f1"],
                                           "F1_alone_macro_f1": scores["F1"]["macro_f1"],
                                           "F0_plus_F1_macro_f1": joint["macro_f1"]})
                        for first, second in (("F0", "F0_plus_F1"), ("F1", "F0_plus_F1")):
                            truth = np.array([subset[first][trial]["truth"] for trial in selected])
                            a = np.array([subset[first][trial]["probs"] for trial in selected]).argmax(axis=1)
                            b = np.array([subset[second][trial]["probs"] for trial in selected]).argmax(axis=1)
                            a_error = (a != truth).astype(float)
                            b_error = (b != truth).astype(float)
                            corr = (float(np.corrcoef(a_error, b_error)[0, 1])
                                    if np.std(a_error) > 0 and np.std(b_error) > 0 else "N/A")
                            complements.append({**common, "family_a": first, "family_b": second,
                                                "error_correlation": corr,
                                                "disagreement_rate": float(np.mean(a != b)),
                                                "a_correct_b_wrong": float(np.mean((a == truth) & (b != truth))),
                                                "a_wrong_b_correct": float(np.mean((a != truth) & (b == truth)))})
                        if subject == "ALL" and condition in ("all", "force2"):
                            headline[mode][split][f"{shots}_{condition}"] = {
                                arm: scores[arm] for arm in ARMS}
    if len(increments) != 200 or len(complements) != 400:
        raise ValueError("Unexpected paired-cell count")
    output.mkdir(parents=True, exist_ok=True)
    inc_path = output / "CALIBRATION_PAIRED_INCREMENT.csv"
    comp_path = output / "CALIBRATION_PAIRED_COMPLEMENTARITY.csv"
    write(inc_path, increments)
    write(comp_path, complements)
    result = {"status": "post_hoc_descriptive_paired_calibration_analysis",
              "input_sha256": {"calibration_verification": sha(BASE / "CALIBRATION_VERIFICATION.json"),
                               "calibration_predictions": sha(BASE / "CALIBRATION_TRIAL_PREDICTIONS.csv"),
                               "runner": sha(Path(__file__))},
              "increment_rows": len(increments), "complementarity_rows": len(complements),
              "increment_sha256": sha(inc_path), "complementarity_sha256": sha(comp_path),
              "headline": headline,
              "boundary": "Post-hoc analysis of the frozen descriptive calibration trial predictions. Identical trial IDs across arms and budgets are checked. No tuning or family promotion is based on final users; three-channel public subjective-force results are not own-device live validation."}
    (output / "CALIBRATION_PAIRED_RESULTS.json").write_bytes(
        (json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(f"paired increments={len(increments)} complementarity={len(complements)}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE)
    run(parser.parse_args().output)
