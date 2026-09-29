"""Read-back conditional value and paired errors across three frozen screens.

This is a post-hoc descriptive comparison of *saved* probabilities; it does
not refit models, select an arm on final labels, or imply independent trials.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, log_loss

from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parent
STUDIES = {
    "roam_posture": ("ROAM_V1_EXTENSION", [0, 1, 2], "label"),
    "grab_user": ("GRAB_V1_EXTENSION", [4, 15, 16, 17], "gesture"),
    "grab_day": ("GRAB_DAY_V1_EXTENSION", [4, 15, 16, 17], "gesture"),
}
FIELDS = ["study", "phase", "group_type", "group", "family", "n_trials",
          "base_macro_f1", "added_macro_f1", "delta_macro_f1",
          "base_log_loss", "added_log_loss", "log_loss_improvement",
          "base_brier", "added_brier", "brier_improvement",
          "prediction_disagreement", "base_wrong_added_right", "base_right_added_wrong",
          "both_wrong", "both_right"]


def metrics(y: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> tuple[float, float, float]:
    decision = classes[np.argmax(probability, axis=1)]
    one_hot = (y[:, None] == classes[None, :]).astype(float)
    return (float(f1_score(y, decision, labels=classes, average="macro", zero_division=0)),
            float(log_loss(y, probability, labels=classes)),
            float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))))


def analyze() -> dict:
    output = []
    sources = {}
    for study, (prefix, class_list, label) in STUDIES.items():
        prediction_path = ROOT / f"{prefix}_PREDICTIONS.csv"
        result_path = ROOT / f"{prefix}_RESULTS.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if sha256(prediction_path) != result["prediction_sha256"]:
            raise ValueError(f"{study} saved prediction hash changed")
        sources[study] = {"prediction_sha256": sha256(prediction_path),
                          "result_sha256": sha256(result_path)}
        with prediction_path.open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        classes = np.asarray(class_list)
        for phase in ("validation", "final"):
            phase_rows = [r for r in predictions if r["phase"] == phase]
            base = {r["trial_id"]: r for r in phase_rows if r["arm"] == "F0v2"}
            if len(base) != len(result[f"{phase}_trial_ids"]):
                raise AssertionError("baseline target count changed")
            groups = [("pooled", "all", list(base))]
            for subject in sorted({int(r["subject"]) for r in base.values()}):
                groups.append(("subject", str(subject),
                               [identity for identity, r in base.items()
                                if int(r["subject"]) == subject]))
            if study == "roam_posture":
                for posture in result["protocol"]["target_postures"]:
                    groups.append(("posture", posture,
                                   [identity for identity, r in base.items()
                                    if r["condition"] == posture]))
            for family in result["protocol"]["candidate_families"]:
                arm = f"F0v2+{family}"
                added = {r["trial_id"]: r for r in phase_rows if r["arm"] == arm}
                if base.keys() != added.keys():
                    raise AssertionError(f"{study} unmatched baseline/add-on IDs")
                for group_type, group, identities in groups:
                    y = np.asarray([int(base[key][label]) for key in identities])
                    if any(int(added[key][label]) != int(truth) for key, truth in zip(identities, y)):
                        raise AssertionError("paired truth changed")
                    p0 = np.asarray([[float(base[key][f"p_{c}"]) for c in classes]
                                     for key in identities])
                    p1 = np.asarray([[float(added[key][f"p_{c}"]) for c in classes]
                                     for key in identities])
                    f0, ll0, br0 = metrics(y, p0, classes)
                    f1, ll1, br1 = metrics(y, p1, classes)
                    d0, d1 = classes[np.argmax(p0, axis=1)], classes[np.argmax(p1, axis=1)]
                    output.append({"study": study, "phase": phase, "group_type": group_type,
                                   "group": group, "family": family, "n_trials": len(y),
                                   "base_macro_f1": f0, "added_macro_f1": f1,
                                   "delta_macro_f1": f1 - f0,
                                   "base_log_loss": ll0, "added_log_loss": ll1,
                                   "log_loss_improvement": ll0 - ll1,
                                   "base_brier": br0, "added_brier": br1,
                                   "brier_improvement": br0 - br1,
                                   "prediction_disagreement": int(np.sum(d0 != d1)),
                                   "base_wrong_added_right": int(np.sum((d0 != y) & (d1 == y))),
                                   "base_right_added_wrong": int(np.sum((d0 == y) & (d1 != y))),
                                   "both_wrong": int(np.sum((d0 != y) & (d1 != y))),
                                   "both_right": int(np.sum((d0 == y) & (d1 == y)))})
    path = ROOT / "V1_EXTENSION_PAIRED.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    audit = {"status": "ok", "analysis": "post_hoc_saved_prediction_readback",
             "source_hashes": sources, "rows": len(output),
             "group_counts": {study: sum(r["study"] == study for r in output)
                              for study in STUDIES},
             "paired_csv_sha256": sha256(path),
             "scope": "Conditional/add-on and error-complementarity descriptions of three frozen public screens; validation/final remain separate. Shared recordings are not independent replications."}
    (ROOT / "V1_EXTENSION_PAIRED_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"paired extension read-back: {len(output)} matched groups", flush=True)
    return audit


if __name__ == "__main__":
    analyze()
