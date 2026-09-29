"""Read-back family, conditional and error cells for the force extension."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2.family_screen import _score
from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parent
PREFIX = "FORCE_V1_EXTENSION"
CLASSES = [str(c) for c in range(7)]
BASE = "F0v2"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze() -> dict:
    result_path = ROOT / f"{PREFIX}_RESULTS.json"
    prediction_path = ROOT / f"{PREFIX}_PREDICTIONS.csv"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    arms = result["protocol"]["arms"]
    if (arms != [BASE, "F0v2+scale_pattern", "F0v2+ring_lag",
                 "F0v2+correlation_spectrum", "F0v2+frequency_direction"]
            or result["prediction_sha256"] != sha256(prediction_path)):
        raise AssertionError("frozen force extension changed")
    with prediction_path.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    if len(raw) != 5880:
        raise AssertionError("five-arm force prediction coverage changed")
    by_arm = {(r["phase"], r["trial_id"], r["arm"]): r for r in raw}
    if len(by_arm) != len(raw):
        raise AssertionError("duplicate arm/trial prediction")
    family, conditional, errors = [], [], []
    for phase in ("validation", "final"):
        trial_ids = result["split_trial_ids"][phase]["target_trials"]
        if (len(trial_ids) != 588 or len(set(trial_ids)) != 588
                or {r["trial_id"] for r in raw if r["phase"] == phase} != set(trial_ids)):
            raise AssertionError("native target trial inventory changed")
        for trial_id in trial_ids:
            identities = {(by_arm[(phase, trial_id, arm)]["subject"],
                           by_arm[(phase, trial_id, arm)]["condition"],
                           by_arm[(phase, trial_id, arm)]["label"]) for arm in arms}
            if len(identities) != 1:
                raise AssertionError("paired class or condition changed")
        groups = [("pooled", "ALL", "ALL", trial_ids)]
        for subject in result["protocol"][f"{phase}_subjects"]:
            groups.append(("subject", str(subject), "ALL", [trial_id for trial_id in trial_ids
                if int(by_arm[(phase, trial_id, BASE)]["subject"]) == subject]))
        for condition in result["protocol"]["target_conditions"]:
            groups.append(("condition", "ALL", condition, [trial_id for trial_id in trial_ids
                if by_arm[(phase, trial_id, BASE)]["condition"] == condition]))
        for scope, subject, condition, ids in groups:
            if not ids:
                raise AssertionError("empty force analysis cell")
            context = {"phase": phase, "scope": scope, "subject": subject,
                       "condition": condition, "evaluation_trials": len(ids)}
            scores = {}
            decisions = {}
            truth = np.asarray([int(by_arm[(phase, trial_id, BASE)]["label"]) for trial_id in ids])
            for arm in arms:
                rows = [by_arm[(phase, trial_id, arm)] for trial_id in ids]
                score_rows = [{"label": row["label"],
                               **{f"p_{c}": row[f"p_{c}"] for c in CLASSES}} for row in rows]
                measured = _score(score_rows, CLASSES)
                scores[arm] = measured
                probability = np.asarray([[float(row[f"p_{c}"]) for c in CLASSES] for row in rows])
                decisions[arm] = probability.argmax(axis=1)
                if scope == "pooled":
                    saved = result["scores"][phase][arm]["pooled"]
                    for metric in ("macro_f1", "accuracy", "log_loss", "brier"):
                        if abs(measured[metric] - saved[metric]) > 1e-10:
                            raise AssertionError(f"saved pooled score changed: {phase}/{arm}/{metric}")
                family.append({**context, "feature_family": arm,
                    "feature_dimension": sum(result["feature_dimensions"][name]
                                             for name in arm.split("+")),
                    **{metric: measured[metric] for metric in
                       ("macro_f1", "accuracy", "log_loss", "brier", "ece", "per_class_f1_json")}})
            core = scores[BASE]
            for arm in arms[1:]:
                added = arm.removeprefix("F0v2+")
                current = scores[arm]
                conditional.append({**context, "core_bank": BASE, "added_family": added,
                                    "delta_logloss": core["log_loss"] - current["log_loss"],
                                    "delta_macro_f1": current["macro_f1"] - core["macro_f1"],
                                    "delta_brier": core["brier"] - current["brier"]})
            for a, b in itertools.combinations(arms, 2):
                wrong_a = decisions[a] != truth
                wrong_b = decisions[b] != truth
                corr = (float(np.corrcoef(wrong_a.astype(float), wrong_b.astype(float))[0, 1])
                        if np.std(wrong_a) and np.std(wrong_b) else "N/A")
                errors.append({**context, "family_a": a, "family_b": b,
                               "error_correlation": corr,
                               "disagreement_rate": float(np.mean(decisions[a] != decisions[b])),
                               "a_correct_b_wrong": float(np.mean(~wrong_a & wrong_b)),
                               "a_wrong_b_correct": float(np.mean(wrong_a & ~wrong_b)),
                               "both_wrong": float(np.mean(wrong_a & wrong_b)),
                               "both_correct": float(np.mean(~wrong_a & ~wrong_b))})
    if (len(family), len(conditional), len(errors)) != (140, 112, 280):
        raise AssertionError("force extension analysis coverage changed")
    outputs = {"FAMILY": family, "CONDITIONAL": conditional, "ERROR": errors}
    paths = {name: ROOT / f"{PREFIX}_{name}.csv" for name in outputs}
    for name, table in outputs.items():
        write_csv(paths[name], table)
    audit = {"status": "ok", "parent_result_sha256": sha256(result_path),
             "parent_prediction_sha256": sha256(prediction_path),
             "source_trial_count": len(result["split_trial_ids"]["validation"]["source_trials"]),
             "native_target_trials": 1176,
             "rows": {name: len(table) for name, table in outputs.items()},
             "sha256": {name: sha256(path) for name, path in paths.items()},
             "boundary": "Public instructed intensity, source-trained models, held-user native trial comparisons; final users previously inspected and descriptive."}
    (ROOT / f"{PREFIX}_ANALYSIS_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"force_extension_analysis_rows": audit["rows"]}), flush=True)
    return audit


if __name__ == "__main__":
    analyze()
