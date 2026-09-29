"""Post-hoc native-target read-back for the three frozen new-v1 family screens."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.family_screen import _score
from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parent
STUDIES = {"roam_posture": ("ROAM_V1_EXTENSION", ["0", "1", "2"]),
           "grab_user": ("GRAB_V1_EXTENSION", ["4", "15", "16", "17"]),
           "grab_day": ("GRAB_DAY_V1_EXTENSION", ["4", "15", "16", "17"])}
FIELDS = ["study", "dataset", "phase", "scope", "subject", "condition",
          "evaluation_unit", "evaluation_trials", "feature_family", "feature_dimension",
          "macro_f1", "accuracy", "log_loss", "brier", "ece", "per_class_f1_json"]


def analyze() -> dict:
    output = []
    sources = {}
    for study, (prefix, classes) in STUDIES.items():
        result_path = ROOT / f"{prefix}_RESULTS.json"
        prediction_path = ROOT / f"{prefix}_PREDICTIONS.csv"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["prediction_sha256"] != sha256(prediction_path):
            raise ValueError(f"{study} saved source hash changed")
        sources[study] = {"result_sha256": sha256(result_path),
                          "prediction_sha256": sha256(prediction_path)}
        with prediction_path.open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        for phase in ("validation", "final"):
            phase_rows = [r for r in predictions if r["phase"] == phase]
            by_arm = {arm: {r["trial_id"]: r for r in phase_rows if r["arm"] == arm}
                      for arm in result["protocol"]["arms"]}
            identities = list(by_arm["F0v2"])
            if (set(identities) != set(result[f"{phase}_trial_ids"])
                    or any(set(part) != set(identities) for part in by_arm.values())):
                raise AssertionError("five-arm native target identity mismatch")
            groups = [("pooled", "ALL", "ALL", identities)]
            for subject in sorted({int(r["subject"]) for r in by_arm["F0v2"].values()}):
                groups.append(("subject", str(subject), "ALL",
                               [identity for identity in identities
                                if int(by_arm["F0v2"][identity]["subject"]) == subject]))
            if study == "roam_posture":
                for posture in result["protocol"]["target_postures"]:
                    groups.append(("posture", "ALL", posture,
                                   [identity for identity in identities
                                    if by_arm["F0v2"][identity]["condition"] == posture]))
            for scope, subject, condition, keys in groups:
                for arm in result["protocol"]["arms"]:
                    rows = [by_arm[arm][identity] for identity in keys]
                    normalized = [{"label": r["label" if study == "roam_posture" else "gesture"],
                                   **{f"p_{c}": r[f"p_{c}"] for c in classes}} for r in rows]
                    value = _score(normalized, classes)
                    if scope == "pooled":
                        saved = result["scores"][arm][phase]
                        for metric in ("macro_f1", "accuracy", "log_loss", "brier"):
                            if abs(value[metric] - saved[metric]) > 1e-12:
                                raise AssertionError(f"{study}/{phase}/{arm} score replay failed: {metric}")
                    dimensions = result["feature_dimensions"]
                    output.append({"study": study,
                                   "dataset": "roam_emg_static" if study == "roam_posture" else "grabmyo",
                                   "phase": phase, "scope": scope, "subject": subject,
                                   "condition": condition,
                                   "evaluation_unit": "native_label_bout" if study == "roam_posture" else "native_recording",
                                   "evaluation_trials": value["trials"], "feature_family": arm,
                                   "feature_dimension": sum(dimensions[name] for name in arm.split("+")),
                                   **{name: value[name] for name in ("macro_f1", "accuracy", "log_loss",
                                                                         "brier", "ece", "per_class_f1_json")}})
    if len(output) != 220:
        raise AssertionError("three-study family screen count changed")
    path = ROOT / "V1_EXTENSION_FAMILY_SCREEN.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    audit = {"status": "ok", "analysis": "post_hoc_five_arm_saved_prediction_readback",
             "sources": sources, "rows": len(output), "table_sha256": sha256(path),
             "study_rows": {study: sum(r["study"] == study for r in output) for study in STUDIES},
             "boundary": "Same split as each frozen experiment, no refit or final-based promotion; GRAB studies share recordings."}
    (ROOT / "V1_EXTENSION_FAMILY_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"independent family screen read-back: {len(output)} cells", flush=True)
    return audit


if __name__ == "__main__":
    analyze()
