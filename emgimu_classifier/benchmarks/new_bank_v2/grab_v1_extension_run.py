"""Source-locked GRABMyo cross-user screen of four independent new-v1 families."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.grabmyo_crossday import run as grab
from benchmarks.new_bank_v2.grab_user_run import check_files
from benchmarks.new_bank_v2.roam_posture_run import sha256
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v1 import NEW_BANK_V1
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "GRAB_V1_EXTENSION_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
CLASSES = np.asarray(grab.GESTURES)
ARMS = tuple(PROTOCOL["arms"])


def check_protocol() -> None:
    parent = json.loads((ROOT / "GRAB_USER_PROTOCOL.json").read_text(encoding="utf-8"))
    if (sha256(ROOT / "GRAB_USER_PROTOCOL.json") != PROTOCOL["parent_user_protocol_sha256"]
            or sha256(ROOT / "GRAB_USER_PREDICTIONS.csv") != PROTOCOL["parent_user_prediction_sha256"]
            or PROTOCOL["source_subjects"] != parent["source_subjects"]
            or PROTOCOL["validation_subjects"] != parent["validation_subjects"]
            or PROTOCOL["final_subjects"] != parent["final_subjects"]
            or PROTOCOL["channels"] != parent["channels"]
            or PROTOCOL["sample_rate_hz"] != parent["sample_rate_hz"]
            or PROTOCOL["window_samples"] != parent["window_samples"]
            or PROTOCOL["day"] != parent["day"]
            or PROTOCOL["gesture_codes"] != parent["gesture_codes"]
            or PROTOCOL["rest_code"] != parent["rest_code"]
            or PROTOCOL["candidate_families"] != list(NEW_BANK_V1)
            or ARMS != ("F0v2", *[f"F0v2+{name}" for name in NEW_BANK_V1])):
        raise ValueError("frozen GRAB new-v1 extension protocol changed")


def replay_base(rows: list[dict]) -> None:
    with (ROOT / "GRAB_USER_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        original = {(row["phase"], row["trial_id"]): row for row in csv.DictReader(stream)
                    if row["arm"] == "F0v2"}
    baseline = [row for row in rows if row["arm"] == "F0v2"]
    if len(original) != 112 or len(baseline) != 112:
        raise AssertionError("frozen GRAB baseline count changed")
    for row in baseline:
        prior = original.get((row["phase"], row["trial_id"]))
        if prior is None or int(prior["gesture"]) != row["gesture"]:
            raise AssertionError("frozen GRAB baseline identity changed")
        np.testing.assert_allclose([row[f"p_{c}"] for c in CLASSES],
                                   [float(prior[f"p_{c}"]) for c in CLASSES],
                                   rtol=0, atol=1e-10)


def evaluate() -> dict:
    check_protocol()
    data_root = Path(PROTOCOL["data_root"])
    records = [row for row in grab.records() if row["session"] == PROTOCOL["day"]]
    if len(records) != 224:
        raise ValueError("Day1 native recording count changed")
    official_manifest_hash = check_files(data_root, records)
    source_set = set(PROTOCOL["source_subjects"])
    rest_windows = np.concatenate([grab.read_record(data_root, row) for row in records
                                   if row["subject"] in source_set and
                                   row["gesture"] == PROTOCOL["rest_code"]])
    rest_batch = FeatureBatch(rest_windows, grab.RATE)
    if rest_batch.windows != 560:
        raise ValueError("source Rest windows changed")
    families = {"F0v2": RestNoiseDetailV2(rest_label=17).fit(
                    rest_batch, np.full(rest_batch.windows, 17)),
                **{name: constructor().fit(rest_batch)
                   for name, constructor in NEW_BANK_V1.items()}}
    vectors = {name: [] for name in families}
    for index, record in enumerate(records, 1):
        batch = FeatureBatch(grab.read_record(data_root, record), grab.RATE)
        for name, family in families.items():
            vectors[name].append(grab.aggregate(family.transform(batch)))
        if index % 50 == 0 or index == len(records):
            print(f"GRAB extended families extracted {index}/{len(records)} recordings", flush=True)
    vectors = {name: np.stack(values) for name, values in vectors.items()}
    subjects = np.asarray([row["subject"] for row in records])
    y = np.asarray([row["gesture"] for row in records])
    source = np.isin(subjects, PROTOCOL["source_subjects"])
    if source.sum() != 112:
        raise ValueError("source native trial count changed")
    rows, scores = [], {}
    for arm in ARMS:
        x = np.concatenate([vectors[name] for name in arm.split("+")], axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, random_state=20260924))
        model.fit(x[source], y[source])
        np.testing.assert_array_equal(model[-1].classes_, CLASSES)
        if np.max(model[-1].n_iter_) >= 2000:
            raise RuntimeError(f"source classifier did not converge: {arm}")
        scores[arm] = {"source_trials": int(source.sum()), "feature_dimensions": int(x.shape[1])}
        for phase in ("validation", "final"):
            mask = np.isin(subjects, PROTOCOL[f"{phase}_subjects"])
            if mask.sum() != 56 or np.any(mask & source):
                raise ValueError("target subject split changed")
            probability = model.predict_proba(x[mask])
            scores[arm][phase] = {"trials": int(mask.sum()),
                                  **grab.score(y[mask], probability, CLASSES, subjects[mask])}
            for identity, p in zip(np.asarray(records, dtype=object)[mask], probability):
                rows.append({"arm": arm, "phase": phase, "trial_id": identity["stem"],
                             "subject": identity["subject"], "gesture": identity["gesture"],
                             **{f"p_{c}": float(value) for c, value in zip(CLASSES, p)}})
        print(f"{arm}: validation F1={scores[arm]['validation']['macro_f1']:.4f}, "
              f"final F1={scores[arm]['final']['macro_f1']:.4f}", flush=True)
    replay_base(rows)
    selected = min(ARMS, key=lambda name: (-scores[name]["validation"]["macro_f1"],
                                            scores[name]["validation"]["log_loss"]))
    path = ROOT / "GRAB_V1_EXTENSION_PREDICTIONS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol": PROTOCOL, "protocol_sha256": sha256(PROTOCOL_PATH),
              "official_sha256_manifest_sha256": official_manifest_hash,
              "prediction_sha256": sha256(path), "source_rest_windows": int(rest_batch.windows),
              "feature_dimensions": {name: int(x.shape[1]) for name, x in vectors.items()},
              "baseline_replay": "all 112 held-out F0v2 vectors within absolute 1e-10",
              "source_trial_ids": [row["stem"] for row in records if row["subject"] in source_set],
              "validation_trial_ids": [row["stem"] for row in records
                                       if row["subject"] in PROTOCOL["validation_subjects"]],
              "final_trial_ids": [row["stem"] for row in records
                                  if row["subject"] in PROTOCOL["final_subjects"]],
              "scores": scores, "validation_selected_arm": selected, "scope": PROTOCOL["scope"]}
    (ROOT / "GRAB_V1_EXTENSION_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    evaluate()
