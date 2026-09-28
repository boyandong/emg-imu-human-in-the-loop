"""Export independently verified matched 0/1/2-shot force curves."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent.parent / "feature_bank" / "results" / "calibration_curve.csv"
RUN_ID = "feature_bank_new_v2_force_cal_20260928"


def export() -> dict:
    verification_path = ROOT / "FORCE_CAL_VERIFICATION.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != "ok" or verification.get("prediction_rows") != 3360:
        raise ValueError("independent force calibration verification is unavailable")
    result_path = ROOT / "FORCE_CAL_RESULTS.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["protocol_sha256"] != hashlib.sha256((ROOT / "FORCE_CAL_PROTOCOL.json").read_bytes()).hexdigest():
        raise ValueError("force calibration protocol hash changed")
    if result["parent_predictions_sha256"] != hashlib.sha256(
            (ROOT / "FORCE_TRIAL_PREDICTIONS.csv").read_bytes()).hexdigest():
        raise ValueError("force calibration parent predictions changed")
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields:
            raise ValueError("source calibration table has no header")
        retained = [row for row in reader if row["run_id"] != RUN_ID]
    added = []
    for phase, arms in result["scores"].items():
        for arm, methods in arms.items():
            for method, score in methods.items():
                shots = {"source_only": 0, "one_shot": 1, "two_shot": 2}[method]
                groups = [("ALL", "ten_conditions_except_MVC", score["pooled"], 20,
                           "pooled over two users and ten intensity conditions")]
                groups += [(subject, "ten_conditions_except_MVC", cell, 10,
                            "ten conditions within one user")
                           for subject, cell in score["by_subject"].items()]
                groups += [("ALL", condition, cell, 2,
                            "two target users within one intensity condition")
                           for condition, cell in score["by_condition"].items()]
                for subject, condition, cell, n_cells, aggregation in groups:
                    values = {"run_id": RUN_ID, "dataset": "libemg_contraction_intensity",
                              "phase": phase, "subject": subject, "condition": condition,
                              "session/domain": condition, "shots_per_class": shots,
                              "feature_bank": arm, "mode": method,
                              "method": "source_logistic" if shots == 0 else
                              "source_logistic_plus_fixed_half_target_prototype",
                              "evaluation_trials": cell["trials"],
                              "macro_f1": cell["macro_f1"], "accuracy": cell["accuracy"],
                              "log_loss": cell["log_loss"], "brier": cell["brier"],
                              "calibration_trials": 7 * shots * n_cells,
                              "supported": "True", "reason": "", "aggregation": aggregation,
                              "protocol": "native repetitions 1-2 calibration; identical repetitions 3-4 evaluation; MVC excluded",
                              "model": "source-frozen balanced logistic; optional source-scaled target prototypes"}
                    if not set(values) <= set(fields):
                        raise ValueError("calibration source table lacks required export field")
                    added.append({name: str(values.get(name, "")) for name in fields})
    if len(added) != 156:
        raise AssertionError("expected two phases, two arms, three budgets and thirteen summary groups")
    with SOURCE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained + added)
    audit = {"run_id": RUN_ID, "rows": len(added),
             "source_results_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
             "source_predictions_sha256": hashlib.sha256((ROOT / "FORCE_CAL_PREDICTIONS.csv").read_bytes()).hexdigest(),
             "source_verification_sha256": hashlib.sha256(verification_path.read_bytes()).hexdigest(),
             "source_table_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
             "boundary": "matched native public-device intensity trials; MVC and own-device live performance excluded"}
    (ROOT / "FORCE_CAL_DELIVERY_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n",
                                                      encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "rows": len(added)}))
    return audit


if __name__ == "__main__":
    export()
