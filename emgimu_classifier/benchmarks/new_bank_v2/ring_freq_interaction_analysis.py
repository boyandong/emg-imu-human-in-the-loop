"""Read-back factorial interaction of ring-lag and spectral-direction additions."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2.roam_posture_run import sha256
from benchmarks.new_bank_v2.v1_extension_paired import metrics

ROOT = Path(__file__).resolve().parent
ARMS = ("F0v2", "F0v2+ring_lag", "F0v2+frequency_direction",
        "F0v2+ring_lag+frequency_direction")
FIELDS = ["study", "phase", "group_type", "group", "n_trials",
          "base_macro_f1", "ring_macro_f1", "frequency_macro_f1", "joint_macro_f1",
          "joint_minus_base_macro_f1", "S_macro_f1",
          "base_log_loss", "ring_log_loss", "frequency_log_loss", "joint_log_loss",
          "joint_log_loss_improvement", "S_negative_logloss",
          "joint_brier_improvement", "S_negative_brier"]


def analyze() -> dict:
    prediction_path = ROOT / "RING_FREQ_INTERACTION_PREDICTIONS.csv"
    result_path = ROOT / "RING_FREQ_INTERACTION_RESULTS.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (result["protocol_sha256"] != sha256(ROOT / "RING_FREQ_INTERACTION_PROTOCOL.json")
            or result["prediction_sha256"] != sha256(prediction_path)
            or result["prediction_rows"] != 3680):
        raise ValueError("interaction source hashes or row count changed")
    with prediction_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    output = []
    for study in result["studies"]:
        classes = np.asarray([0, 1, 2] if study == "roam_posture" else [4, 15, 16, 17])
        for phase in ("validation", "final"):
            phase_rows = [r for r in rows if r["study"] == study and r["phase"] == phase]
            by_arm = {arm: {r["trial_id"]: r for r in phase_rows if r["arm"] == arm}
                      for arm in ARMS}
            identities = list(by_arm[ARMS[0]])
            if (len(identities) != result["studies"][study][f"{phase}_trials"]
                    or any(set(part) != set(identities) for part in by_arm.values())):
                raise AssertionError("four-arm native trial matching failed")
            groups = [("pooled", "all", identities)]
            for subject in sorted({int(r["subject"]) for r in by_arm[ARMS[0]].values()}):
                groups.append(("subject", str(subject),
                               [key for key in identities if int(by_arm[ARMS[0]][key]["subject"]) == subject]))
            if study == "roam_posture":
                for posture in ("resting", "hanging", "unsupported", "reaching"):
                    groups.append(("posture", posture,
                                   [key for key in identities if by_arm[ARMS[0]][key]["condition"] == posture]))
            for group_type, group, keys in groups:
                y = np.asarray([int(by_arm[ARMS[0]][key]["label"]) for key in keys])
                values = {}
                for arm in ARMS:
                    if any(int(by_arm[arm][key]["label"]) != int(truth)
                           for key, truth in zip(keys, y)):
                        raise AssertionError("four-arm truth changed")
                    p = np.asarray([[float(by_arm[arm][key][f"p_{c}"]) for c in classes]
                                    for key in keys])
                    values[arm] = metrics(y, p, classes)
                f0, l0, b0 = values[ARMS[0]]
                fr, lr, br = values[ARMS[1]]
                ff, lf, bf = values[ARMS[2]]
                fj, lj, bj = values[ARMS[3]]
                output.append({"study": study, "phase": phase, "group_type": group_type,
                               "group": group, "n_trials": len(keys),
                               "base_macro_f1": f0, "ring_macro_f1": fr,
                               "frequency_macro_f1": ff, "joint_macro_f1": fj,
                               "joint_minus_base_macro_f1": fj - f0,
                               "S_macro_f1": fj - fr - ff + f0,
                               "base_log_loss": l0, "ring_log_loss": lr,
                               "frequency_log_loss": lf, "joint_log_loss": lj,
                               "joint_log_loss_improvement": l0 - lj,
                               "S_negative_logloss": lr + lf - lj - l0,
                               "joint_brier_improvement": b0 - bj,
                               "S_negative_brier": br + bf - bj - b0})
    if len(output) != 44:
        raise AssertionError("interaction group count changed")
    path = ROOT / "RING_FREQ_INTERACTION.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    audit = {"status": "ok", "analysis": "four_arm_matched_native_trial_readback",
             "prediction_sha256": sha256(prediction_path), "result_sha256": sha256(result_path),
             "rows": len(output), "table_sha256": sha256(path),
             "pooled": [r for r in output if r["group_type"] == "pooled"],
             "boundary": "Positive interaction alone does not establish absolute joint gain; final is descriptive and GRAB studies share recordings."}
    (ROOT / "RING_FREQ_INTERACTION_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"ring × frequency interaction: {len(output)} matched groups", flush=True)
    return audit


if __name__ == "__main__":
    analyze()
