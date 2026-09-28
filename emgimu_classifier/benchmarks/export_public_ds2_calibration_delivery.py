"""Append verified public DS2 v9 calibration scores to Feature Bank source delivery."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "benchmarks/discovery/public_ds2_force_v9"
SOURCE = ROOT / "feature_bank/results/calibration_curve.csv"
RUN_ID = "feature_bank_public_ds2_v9_personal_force_20260928"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export() -> int:
    study = json.loads((STUDY / "CALIBRATION_RESULTS.json").read_text(encoding="utf-8"))
    audit = json.loads((STUDY / "CALIBRATION_VERIFICATION.json").read_text(encoding="utf-8"))
    if audit["status"] != "pass" or audit["score_cells_recomputed"] != 336:
        raise ValueError("DS2 calibration verification is incomplete")
    for name, expected in audit["input_sha256"].items():
        if sha(STUDY / name) != expected:
            raise ValueError(f"DS2 calibration source changed: {name}")
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        existing = [row for row in reader if row["run_id"] == RUN_ID]
    rows = []
    for mode in ("unseen_high", "product_all"):
        for arm in ("F0", "F1", "F0_plus_F1"):
            for split, people in (("validation", range(13, 17)),
                                  ("final_descriptive", range(17, 21))):
                for shot in (0, 1, 2, 5):
                    grouped = study["scores"][mode][arm][split][str(shot)]
                    for subject in ("ALL", *(str(person) for person in people)):
                        score = grouped["all"] if subject == "ALL" else grouped["by_subject"][subject]
                        row = dict.fromkeys(fields, "")
                        row.update({
                            "run_id": RUN_ID, "phase": split,
                            "dataset": "public_ds2_v9_3ch_1500hz",
                            "subject": subject, "session/domain": "held_out_subject",
                            "condition": "unseen_high_force2" if mode == "unseen_high" else "product_all_force012",
                            "shots_per_class": shot, "feature_bank": arm,
                            "mode": mode, "method": "source_model_plus_fixed_personal_prototype_blend",
                            "evaluation_trials": score["trials"], "macro_f1": score["macro_f1"],
                            "accuracy": score["accuracy"], "log_loss": score["log_loss"],
                            "brier": score["brier"], "calibration_trials": shot * 4,
                            "supported": True, "aggregation": "whole_native_trial_mean",
                            "protocol": "benchmarks/discovery/public_ds2_force_v9/CALIBRATION_PROTOCOL.json",
                            "calibration_force": "low_average_only" if mode == "unseen_high" else "mixed_low_average_high",
                            "target_force_calibration": mode == "product_all",
                            "model": "source_balanced_logistic_plus_source_scaled_target_prototypes",
                        })
                        rows.append(row)
    if len(rows) != 240:
        raise ValueError("Unexpected canonical calibration row count")
    if existing:
        if existing != [{key: str(value) for key, value in row.items()} for row in rows]:
            raise ValueError("Existing DS2 calibration delivery differs from verified results")
        return len(rows)
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\r\n")
    writer.writerows(rows)
    with SOURCE.open("ab") as handle:
        handle.write(text.getvalue().encode("utf-8"))
    return len(rows)


if __name__ == "__main__":
    print(f"DS2 calibration source rows: {export()}")
