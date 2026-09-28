"""Add verified new-v2 wearing calibration cells to the canonical source table."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASSIFIER = ROOT.parent.parent
SOURCE = CLASSIFIER / "feature_bank" / "results" / "calibration_curve.csv"
RUN_ID = "feature_bank_new_v2_wearing_cal_20260928"


def export() -> dict:
    verification = json.loads((ROOT / "WEARING_CAL_VERIFICATION.json").read_text(encoding="utf-8"))
    if verification.get("status") != "ok" or verification.get("prediction_rows") != 960:
        raise ValueError("one-shot prediction audit is unavailable")
    result_path = ROOT / "WEARING_CAL_RESULTS.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["protocol_sha256"] != hashlib.sha256((ROOT / "WEARING_CAL_PROTOCOL.json").read_bytes()).hexdigest():
        raise ValueError("one-shot protocol hash mismatch")
    if result["archive_sha256"] != json.loads((ROOT / "WEARING_RESULTS.json").read_text(encoding="utf-8"))["archive_sha256"]:
        raise ValueError("archive differs from zero-shot experiment")
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if fields is None:
            raise ValueError("source calibration curve has no header")
        old = [row for row in reader if row["run_id"] != RUN_ID]
    new = []
    for phase, arms in result["scores"].items():
        for arm, methods in arms.items():
            for method, score in methods.items():
                shots = 0 if method == "source_only" else 1
                groups = [("ALL", "wearing_trial_1_to_4", score["pooled"],
                           "pooled across three subjects and four domains")]
                groups += [(subject, "wearing_trial_1_to_4", item,
                            "four domains, reciprocal native-trial folds")
                           for subject, item in score["by_subject"].items()]
                groups += [("ALL", domain, item, "pooled across three subjects; reciprocal native-trial folds")
                           for domain, item in score["by_domain"].items()]
                for subject, domain, cell, aggregation in groups:
                    values = {"run_id": RUN_ID, "phase": phase,
                              "dataset": "libemg_electrode_shift", "subject": subject,
                              "condition": domain, "session/domain": domain,
                              "shots_per_class": shots, "feature_bank": arm,
                              "mode": method, "method": "source_logistic" if shots == 0 else
                              "source_logistic_plus_fixed_half_target_prototype",
                              "evaluation_trials": cell["trials"],
                              "macro_f1": cell["macro_f1"], "accuracy": cell["accuracy"],
                              "log_loss": cell["log_loss"], "brier": cell["brier"],
                              "calibration_trials": 0 if shots == 0 else 5,
                              "supported": "True", "reason": "",
                              "aggregation": aggregation,
                              "protocol": "one native target trial per class in each domain; reciprocal folds",
                              "model": "source-frozen balanced logistic; optional source-scaled target prototypes"}
                    new.append({name: str(values.get(name, "")) for name in fields})
    if len(new) != 64:
        raise AssertionError("expected two phases, arms, methods and eight summary groups")
    with SOURCE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(old + new)
    audit = {"run_id": RUN_ID, "rows": len(new),
             "source_results_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
             "source_predictions_sha256": hashlib.sha256((ROOT / "WEARING_CAL_PREDICTIONS.csv").read_bytes()).hexdigest(),
             "source_verification_sha256": hashlib.sha256((ROOT / "WEARING_CAL_VERIFICATION.json").read_bytes()).hexdigest(),
             "source_table_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
             "boundary": "matched same-domain public-device trial evidence; five calibration trials per decision"}
    (ROOT / "WEARING_CAL_DELIVERY_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n",
                                                          encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "rows": len(new)}))
    return audit


if __name__ == "__main__":
    export()
