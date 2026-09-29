"""Matched family/increment/error read-back for the wearing new-v1 screen."""
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
PREFIX = "WEARING_V1_EXTENSION"
CLASSES = [str(c) for c in range(5)]
BASE = "F0v2"


def analyze() -> dict:
    result_path = ROOT / f"{PREFIX}_RESULTS.json"
    prediction_path = ROOT / f"{PREFIX}_PREDICTIONS.csv"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    arms = result["protocol"]["arms"]
    if (arms != [BASE, "F0v2+scale_pattern", "F0v2+ring_lag",
                 "F0v2+correlation_spectrum", "F0v2+frequency_direction"]
            or result["prediction_sha256"] != sha256(prediction_path)):
        raise AssertionError("frozen wearing extension changed")
    with prediction_path.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    if len(raw) != 1200:
        raise AssertionError("five-arm wearing prediction coverage changed")
    indexed = {(r["phase"], r["trial_id"], r["arm"]): r for r in raw}
    if len(indexed) != len(raw):
        raise AssertionError("duplicate wearing arm/trial prediction")
    family, conditional, errors = [], [], []
    for phase in ("validation", "final"):
        subjects = result["protocol"][f"{phase}_subjects"]
        ids = [trial_id for subject in subjects for trial_id in
               result["split_trial_ids"][f"{phase}_{subject}"]["target"]]
        if len(ids) != 120 or len(set(ids)) != 120:
            raise AssertionError("native wearing target inventory changed")
        for trial_id in ids:
            metadata = {(indexed[(phase, trial_id, arm)]["subject"],
                         indexed[(phase, trial_id, arm)]["domain"],
                         indexed[(phase, trial_id, arm)]["label"]) for arm in arms}
            if len(metadata) != 1:
                raise AssertionError("wearing paired metadata changed")
        groups = [("pooled", "ALL", "ALL", ids)]
        for subject in subjects:
            groups.append(("subject", str(subject), "ALL", [trial_id for trial_id in ids
                if int(indexed[(phase, trial_id, BASE)]["subject"]) == subject]))
        for domain in result["protocol"]["target_domains"]:
            groups.append(("domain", "ALL", domain, [trial_id for trial_id in ids
                if indexed[(phase, trial_id, BASE)]["domain"] == domain]))
        for scope, subject, domain, trial_ids in groups:
            if not trial_ids:
                raise AssertionError("empty wearing analysis cell")
            context = {"phase": phase, "scope": scope, "subject": subject,
                       "condition": domain, "evaluation_trials": len(trial_ids)}
            truth = np.asarray([int(indexed[(phase, trial_id, BASE)]["label"])
                                for trial_id in trial_ids])
            scores, decisions = {}, {}
            for arm in arms:
                rows = [indexed[(phase, trial_id, arm)] for trial_id in trial_ids]
                normalized = [{"label": row["label"],
                               **{f"p_{c}": row[f"p_{c}"] for c in CLASSES}} for row in rows]
                measured = _score(normalized, CLASSES)
                scores[arm] = measured
                probability = np.asarray([[float(row[f"p_{c}"]) for c in CLASSES]
                                          for row in rows])
                decisions[arm] = probability.argmax(axis=1)
                if scope == "pooled":
                    saved = result["scores"][phase][arm]["pooled"]
                    for metric in ("macro_f1", "accuracy", "log_loss", "brier"):
                        if abs(measured[metric] - saved[metric]) > 1e-10:
                            raise AssertionError(f"wearing score changed: {phase}/{arm}/{metric}")
                family.append({**context, "feature_family": arm,
                    "feature_dimension": sum(result["feature_dimensions"][name]
                                             for name in arm.split("+")),
                    **{name: measured[name] for name in
                       ("macro_f1", "accuracy", "log_loss", "brier", "ece", "per_class_f1_json")}})
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
    if (len(family), len(conditional), len(errors)) != (80, 64, 160):
        raise AssertionError("wearing extension analysis coverage changed")
    outputs = {"FAMILY": family, "CONDITIONAL": conditional, "ERROR": errors}
    paths = {name: ROOT / f"{PREFIX}_{name}.csv" for name in outputs}
    for name, table in outputs.items():
        write_csv(paths[name], table)
    audit = {"status": "ok", "parent_result_sha256": sha256(result_path),
             "parent_prediction_sha256": sha256(prediction_path),
             "native_target_trials": 240,
             "rows": {name: len(table) for name, table in outputs.items()},
             "sha256": {name: sha256(path) for name, path in paths.items()},
             "boundary": "Same-user public electrode-shift trial predictions, validation and previously inspected descriptive final users; source domain only fits state."}
    (ROOT / f"{PREFIX}_ANALYSIS_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wearing_extension_analysis_rows": audit["rows"]}), flush=True)
    return audit


if __name__ == "__main__":
    analyze()
