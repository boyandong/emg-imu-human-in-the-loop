"""Read-back matched new-v1 quality family, increment and error cells."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2.family_screen import _score
from benchmarks.new_bank_v2.force_v1_extension_analysis import write_csv
from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parent
PREFIX = "ROAM_V1_QUALITY"
BASE = "F0v2"
CLASSES = ["0", "1", "2"]


def analyze() -> dict:
    result_path = ROOT / f"{PREFIX}_RESULTS.json"
    prediction_path = ROOT / f"{PREFIX}_PREDICTIONS.csv"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    arms = result["protocol"]["arms"]
    if (arms != [BASE, "F0v2+scale_pattern", "F0v2+ring_lag",
                 "F0v2+correlation_spectrum", "F0v2+frequency_direction"]
            or result["prediction_sha256"] != sha256(prediction_path)
            or result["baseline_replay_max_abs_error"] > 1e-10):
        raise AssertionError("frozen ROAM new-v1 quality source changed")
    with prediction_path.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    if len(raw) != 6300:
        raise AssertionError("quality five-arm target prediction count changed")
    indexed = {(r["phase"], r["condition"], r["trial_id"], r["arm"]): r for r in raw}
    if len(indexed) != len(raw):
        raise AssertionError("duplicate quality prediction")
    family, conditional, errors = [], [], []
    replayed_scores = 0
    for phase in ("validation", "final"):
        ids = result["split_trial_ids"][phase]
        if len(ids) != 45 or len(set(ids)) != 45:
            raise AssertionError("native quality target inventory changed")
        users = result["protocol"][f"{phase}_subjects"]
        for condition in result["protocol"]["conditions"]:
            if {r["trial_id"] for r in raw if r["phase"] == phase
                    and r["condition"] == condition} != set(ids):
                raise AssertionError("quality paired target bout set changed")
            for trial_id in ids:
                metadata = {(indexed[(phase, condition, trial_id, arm)]["subject"],
                             indexed[(phase, condition, trial_id, arm)]["label"])
                            for arm in arms}
                if len(metadata) != 1:
                    raise AssertionError("quality paired label or user changed")
            groups = [("pooled", "ALL", ids)]
            groups.extend(("subject", str(user), [trial_id for trial_id in ids
                if int(indexed[(phase, condition, trial_id, BASE)]["subject"]) == user])
                for user in users)
            for scope, subject, group_ids in groups:
                if len(group_ids) != (45 if scope == "pooled" else 9):
                    raise AssertionError("quality group native bout count changed")
                context = {"phase": phase, "scope": scope, "subject": subject,
                           "condition": condition, "evaluation_trials": len(group_ids)}
                truth = np.asarray([int(indexed[(phase, condition, trial_id, BASE)]["label"])
                                    for trial_id in group_ids])
                scores, decisions = {}, {}
                for arm in arms:
                    rows = [indexed[(phase, condition, trial_id, arm)] for trial_id in group_ids]
                    measured = _score(rows, CLASSES)
                    scores[arm] = measured
                    probability = np.asarray([[float(row[f"p_{c}"]) for c in CLASSES]
                                              for row in rows])
                    decisions[arm] = probability.argmax(axis=1)
                    saved = result["scores"][phase][arm][condition]
                    if scope == "pooled":
                        for metric in ("macro_f1", "accuracy", "log_loss", "brier"):
                            if abs(measured[metric] - saved[metric]) > 1e-10:
                                raise AssertionError(f"quality pooled replay changed: {phase}/{condition}/{arm}")
                    elif abs(measured["macro_f1"] - saved["per_subject_macro_f1"][subject]) > 1e-10:
                        raise AssertionError("quality subject macro-F1 replay changed")
                    replayed_scores += 1
                    family.append({**context, "feature_family": arm,
                        "feature_dimension": sum(result["feature_dimensions"][name]
                                                 for name in arm.split("+")),
                        **{metric: measured[metric] for metric in
                           ("macro_f1", "accuracy", "log_loss", "brier", "ece",
                            "per_class_f1_json")}})
                core = scores[BASE]
                for arm in arms[1:]:
                    current = scores[arm]
                    conditional.append({**context, "core_bank": BASE,
                        "added_family": arm.removeprefix("F0v2+"),
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
    expected = {"FAMILY": 840, "CONDITIONAL": 672, "ERROR": 1680}
    outputs = {"FAMILY": family, "CONDITIONAL": conditional, "ERROR": errors}
    if {name: len(rows) for name, rows in outputs.items()} != expected or replayed_scores != 840:
        raise AssertionError("quality analysis coverage changed")
    for name, rows in outputs.items():
        write_csv(ROOT / f"{PREFIX}_{name}.csv", rows)
    audit = {"status": "ok", "parent_result_sha256": sha256(result_path),
             "parent_prediction_sha256": sha256(prediction_path),
             "saved_score_groups_replayed": replayed_scores,
             "native_target_bouts": 90, "paired_fault_evaluations": 1260,
             "rows": expected,
             "sha256": {name: sha256(ROOT / f"{PREFIX}_{name}.csv") for name in outputs},
             "boundary": result["scope"]}
    (ROOT / f"{PREFIX}_ANALYSIS_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"roam_v1_quality_analysis_rows": expected}), flush=True)
    return audit


if __name__ == "__main__":
    analyze()
