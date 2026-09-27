"""Read-back check of the frozen signed-axis arm trial comparison."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

from benchmarks.song_real8_study import parse_label


ROOT = Path(__file__).resolve().parent


def run() -> dict:
    results = json.loads((ROOT / "SONG_ARM_SIGNED_RESULTS.json").read_text(encoding="utf-8"))
    protocol = ROOT / "SONG_ARM_SIGNED_PROTOCOL.json"
    baseline = ROOT / "SONG_ARM_CAL_TRIAL_PREDICTIONS.csv"
    predictions = ROOT / "SONG_ARM_SIGNED_TRIAL_PREDICTIONS.csv"
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != results["protocol_sha256"]:
        raise ValueError("protocol changed after scoring")
    if hashlib.sha256(baseline.read_bytes()).hexdigest() != results["baseline_sha256"]:
        raise ValueError("frozen baseline changed after scoring")
    if hashlib.sha256(predictions.read_bytes()).hexdigest() != results["prediction_csv_sha256"]:
        raise ValueError("saved predictions changed after scoring")
    classes = results["joint_classes"]
    arms = results["arm_classes"]
    with baseline.open(newline="", encoding="utf-8") as stream:
        old = {(row["phase"], row["trial_id"]): row for row in csv.DictReader(stream)
               if row["arm"] == "source_arm_x_same_hand"}
    with predictions.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != results["prediction_rows"] or len(rows) != len(old):
        raise ValueError("trial count differs")
    identities = set()
    checks = {}
    for phase, old_phase in (("validation", "validation"), ("final_descriptive", "final")):
        selected = [row for row in rows if row["phase"] == phase]
        expected = results["sessions"][phase]
        if len(selected) != expected["trials"] or {row["session"] for row in selected} != {expected["session"]}:
            raise ValueError(f"wrong session/trial count: {phase}")
        y = [row["truth"] for row in selected]
        for row in selected:
            key = (old_phase, row["trial_id"])
            if key in identities or key not in old:
                raise ValueError(f"duplicate or unknown trial: {key}")
            identities.add(key)
            original = old[key]
            if row["truth"] != original["label"] or row["truth_arm"] != parse_label(row["truth"])[0]:
                raise ValueError(f"truth mismatch: {key}")
            p_old = np.asarray([float(original[f"p_{label}"]) for label in classes])
            p_saved = np.asarray([float(row[f"p_baseline_{label}"]) for label in classes])
            p_new = np.asarray([float(row[f"p_candidate_{label}"]) for label in classes])
            if not np.array_equal(p_old, p_saved) or not np.isfinite(p_new).all() or not np.isclose(p_new.sum(), 1, atol=1e-12):
                raise ValueError(f"invalid or changed probabilities: {key}")
            for name, probability in (("baseline", p_saved), ("candidate", p_new)):
                joint = classes[int(np.argmax(probability))]
                arm_probability = [sum(probability[i] for i, label in enumerate(classes)
                                       if parse_label(label)[0] == arm) for arm in arms]
                arm = arms[int(np.argmax(arm_probability))]
                if joint != row[f"{name}_joint"] or arm != row[f"{name}_arm"]:
                    raise ValueError(f"hard decision mismatch: {key}/{name}")
        for name in ("baseline", "candidate"):
            probability = np.asarray([[float(row[f"p_{name}_{label}"]) for label in classes]
                                      for row in selected])
            predicted = [classes[int(i)] for i in np.argmax(probability, axis=1)]
            score = expected[name]
            observed = {"accuracy": accuracy_score(y, predicted),
                        "macro_f1": f1_score(y, predicted, labels=classes, average="macro", zero_division=0),
                        "log_loss": log_loss(y, probability, labels=classes)}
            for metric, value in observed.items():
                if not np.isclose(value, score[metric], rtol=0, atol=1e-12):
                    raise ValueError(f"metric differs: {phase}/{name}/{metric}")
        for arm in arms:
            truth_rows = [row for row in selected if row["truth_arm"] == arm]
            record = expected["arm_recall"][arm]
            if record["support"] != len(truth_rows):
                raise ValueError(f"arm support differs: {phase}/{arm}")
            for name in ("baseline", "candidate"):
                recall = sum(row[f"{name}_arm"] == arm for row in truth_rows) / len(truth_rows)
                if not np.isclose(recall, record[name], rtol=0, atol=1e-12):
                    raise ValueError(f"arm recall differs: {phase}/{arm}/{name}")
        checks[phase] = {"trials": len(selected), "baseline_accuracy": expected["baseline"]["accuracy"],
                         "candidate_accuracy": expected["candidate"]["accuracy"],
                         "candidate_macro_f1": expected["candidate"]["macro_f1"]}
    if identities != set(old):
        raise ValueError("frozen baseline has unverified trials")
    output = {"status": "verified", "trials": len(rows), "checks": checks,
              "boundary": "Read-back verifies saved trial scores and frozen baseline identity, not physical-device or new-day accuracy."}
    (ROOT / "SONG_ARM_SIGNED_VERIFICATION.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
