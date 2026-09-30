"""Matched MANUS session test of document CSP and SPD tangent increments."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v1.run import fit_predict
from benchmarks.new_bank_v2.manus_rest_transfer_run import check_protocol, external_rest
from benchmarks.new_bank_v2.manus_spatial_run import extract, score
from benchmarks.new_bank_v2.roam_posture_run import sha256
from emgimu.datasets.semg_manus import PATH_RE, load_semg_manus_windows
from emgimu.feature_bank.document_signal import DocumentCspFamily
from emgimu.feature_bank.families import SpdTangentFamily
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "F2_MANUS_CANDIDATES_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
PARENT_PATH = ROOT / "MANUS_REST_TRANSFER_PROTOCOL.json"
PARENT = json.loads(PARENT_PATH.read_text(encoding="utf-8"))
CLASSES = np.arange(6)


def evaluate() -> dict:
    check_protocol()
    for filename, key in (("MANUS_REST_TRANSFER_PROTOCOL.json", "parent_protocol_sha256"),
                          ("MANUS_REST_TRANSFER_RESULTS.json", "parent_result_sha256"),
                          ("MANUS_REST_TRANSFER_PREDICTIONS.csv", "parent_prediction_sha256")):
        if sha256(ROOT / filename) != PROTOCOL[key]:
            raise AssertionError(f"frozen MANUS parent changed: {filename}")
    if PROTOCOL["arms"] != ["F0v2", "F0v2+F2b_document", "F0v2+F2c_spd"]:
        raise ValueError("F2 candidate arms changed")
    parent_result = json.loads((ROOT / "MANUS_REST_TRANSFER_RESULTS.json").read_text(encoding="utf-8"))
    archive = Path(PARENT["manus_archive"])
    if sha256(archive).lower() != PARENT["manus_archive_sha256"].lower():
        raise AssertionError("MANUS native archive changed")
    rest = external_rest()
    arguments = dict(users=PARENT["users"], gestures=PARENT["gestures"],
                     speeds=PARENT["conditions"], window_ms=PARENT["window_ms"],
                     maximum_windows_per_trial=PARENT["maximum_windows_per_trial"])
    source = load_semg_manus_windows(archive, sessions=(PARENT["source_session"],), **arguments)
    if (source.batch.channels != 8 or source.batch.sample_rate_hz != 200
            or source.batch.emg.shape[1] != 40 or set(source.labels) != set(CLASSES)):
        raise AssertionError("MANUS source contract changed")
    families = {
        "F0v2": RestNoiseDetailV2(rest_label=0).fit(rest, np.zeros(rest.windows, dtype=int)),
        "F2b_document": DocumentCspFamily().fit(source.batch, source.labels),
        "F2c_spd": SpdTangentFamily(shrinkage=0.05).fit(source.batch),
    }
    source_x, (source_y, source_users, source_sessions, _, source_trials) = extract(source, families)
    if len(source_trials) != 108 or set(source_sessions) != {1}:
        raise AssertionError("MANUS source native inventory changed")
    rows, splits = [], {}
    for phase in ("validation", "final"):
        session = PARENT[f"{phase}_session"]
        target = load_semg_manus_windows(archive, sessions=(session,), **arguments)
        target_x, (target_y, target_users, target_sessions, target_speeds, target_trials) = extract(
            target, families)
        if (len(target_trials) != 108 or set(target_sessions) != {session}
                or set(source_trials) & set(target_trials)
                or source_trials.tolist() != parent_result["split_trial_ids"][phase]["source_trials"]
                or target_trials.tolist() != parent_result["split_trial_ids"][phase]["target_trials"]):
            raise AssertionError("MANUS parent/new split differs")
        splits[phase] = {"source_trials": source_trials.tolist(),
                         "target_trials": target_trials.tolist()}
        for arm in PROTOCOL["arms"]:
            names = arm.split("+")
            x = np.concatenate([source_x[name] for name in names], axis=1)
            xt = np.concatenate([target_x[name] for name in names], axis=1)
            classes, probabilities = fit_predict(x, source_y, xt)
            np.testing.assert_array_equal(classes, CLASSES)
            for trial, user, speed, label, probability in zip(
                    target_trials, target_users, target_speeds, target_y, probabilities):
                native = PATH_RE.fullmatch(str(trial))
                if (native is None or int(native["user"]) != user
                        or int(native["session"]) != session or native["speed"] != speed
                        or native["gesture"] != PARENT["gestures"][label]):
                    raise AssertionError("MANUS native target metadata changed")
                rows.append({"phase": phase, "subject": int(user), "condition": str(speed),
                             "arm": arm, "trial_id": str(trial), "label": int(label),
                             **{f"p_{c}": float(value) for c, value in zip(CLASSES, probability)}})
        print(f"F2 MANUS {phase}: {len(target_trials)} native trials", flush=True)

    with (ROOT / "MANUS_REST_TRANSFER_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        parent_rows = {(row["phase"], row["trial_id"]): row for row in csv.DictReader(stream)
                       if row["arm"] == "F0v2"}
    baseline = [row for row in rows if row["arm"] == "F0v2"]
    if len(baseline) != 216 or len(parent_rows) != 216:
        raise AssertionError("MANUS baseline coverage changed")
    max_error = 0.
    for row in baseline:
        old = parent_rows[(row["phase"], row["trial_id"])]
        if (row["subject"], row["condition"], row["label"]) != (
                int(old["subject"]), old["condition"], int(old["label"])):
            raise AssertionError("MANUS parent/new baseline identity changed")
        max_error = max(max_error, max(abs(row[f"p_{c}"] - float(old[f"p_{c}"])) for c in CLASSES))
    if max_error > 1e-8:
        raise AssertionError(f"F0v2 replay error {max_error}")
    scores = {phase: {arm: score(rows, phase, arm) for arm in PROTOCOL["arms"]}
              for phase in ("validation", "final")}
    path = ROOT / "F2_MANUS_CANDIDATES_PREDICTIONS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol_sha256": sha256(PROTOCOL_PATH),
              "parent_result_sha256": PROTOCOL["parent_result_sha256"],
              "parent_prediction_sha256": PROTOCOL["parent_prediction_sha256"],
              "archive_sha256": sha256(archive), "rest_archive_sha256": sha256(Path(PARENT["rest_archive"])),
              "prediction_sha256": sha256(path), "prediction_rows": len(rows),
              "source_windows": int(source.batch.windows),
              "feature_dimensions": {name: len(family.feature_names) for name, family in families.items()},
              "split_trial_ids": splits, "f0v2_parent_replay_max_abs_error": max_error,
              "scores": scores, "scope": PROTOCOL["boundary"]}
    (ROOT / "F2_MANUS_CANDIDATES_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n",
                                                              encoding="utf-8")
    for arm in PROTOCOL["arms"]:
        print(f"{arm}: validation F1={scores['validation'][arm]['pooled']['macro_f1']:.4f}, "
              f"final F1={scores['final'][arm]['pooled']['macro_f1']:.4f}", flush=True)
    return result


if __name__ == "__main__":
    evaluate()
