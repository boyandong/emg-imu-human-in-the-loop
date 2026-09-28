"""Frozen Day1 subject-disjoint GRABMyo new-v2 spatial comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.grabmyo_crossday import run as grab
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v2 import (
    RestNoiseDetailV2, RingRelativeCovarianceV2, TraceCovarianceV2,
)

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "GRAB_USER_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
ARMS = tuple(PROTOCOL["arms"])
CLASSES = np.asarray(grab.GESTURES)


def check_protocol() -> None:
    if (PROTOCOL["day"] != 1
            or PROTOCOL["source_subjects"] != [1, 2, 3, 4]
            or PROTOCOL["validation_subjects"] != [5, 6]
            or PROTOCOL["final_subjects"] != [7, 8]
            or tuple(PROTOCOL["gesture_codes"]) != grab.GESTURES
            or tuple(PROTOCOL["channels"]) != grab.CHANNELS
            or PROTOCOL["sample_rate_hz"] != grab.RATE
            or PROTOCOL["window_samples"] != grab.WINDOW
            or tuple(PROTOCOL["trials_per_subject_gesture"]) != grab.TRIALS
            or PROTOCOL["rest_code"] != 17
            or ARMS != ("F0v2", "F0v2+F2a", "F0v2+F3c", "F0v2+F2a+F3c")):
        raise ValueError("cross-user protocol does not match verified GRABMyo subset")


def check_files(data_root: Path, records: list[dict]) -> str:
    checksums = grab.official_checksums(data_root)
    for row in records:
        for suffix in ("hea", "dat"):
            name = grab.relative_file(row, suffix)
            path = data_root / name
            if name not in checksums or not path.exists():
                raise ValueError(f"missing official Day1 file: {name}")
            if grab.sha256_bytes(path.read_bytes()) != checksums[name]:
                raise ValueError(f"official Day1 SHA-256 mismatch: {name}")
    return hashlib.sha256((data_root / "SHA256SUMS.txt").read_bytes()).hexdigest()


def evaluate(data_root: Path) -> dict:
    check_protocol()
    records = [row for row in grab.records() if row["session"] == PROTOCOL["day"]]
    if len(records) != 224:
        raise ValueError("expected 224 same-day native recordings")
    manifest_sha = check_files(data_root, records)
    source_set = set(PROTOCOL["source_subjects"])
    rest_windows = np.concatenate([grab.read_record(data_root, row) for row in records
                                   if row["subject"] in source_set and
                                   row["gesture"] == PROTOCOL["rest_code"]])
    rest_batch = FeatureBatch(rest_windows, grab.RATE)
    families = {
        "F0v2": RestNoiseDetailV2(rest_label=17).fit(
            rest_batch, np.full(rest_batch.windows, 17)),
        "F2a": TraceCovarianceV2(shrinkage=0.05).fit(rest_batch),
        "F3c": RingRelativeCovarianceV2(shrinkage=0.05).fit(rest_batch),
    }
    vectors = {name: [] for name in families}
    for index, record in enumerate(records, 1):
        batch = FeatureBatch(grab.read_record(data_root, record), grab.RATE)
        for name, family in families.items():
            vectors[name].append(grab.aggregate(family.transform(batch)))
        if index % 50 == 0 or index == len(records):
            print(f"GRAB user axis extracted {index}/{len(records)} recordings", flush=True)
    vectors = {name: np.stack(values) for name, values in vectors.items()}
    subjects = np.asarray([row["subject"] for row in records])
    y = np.asarray([row["gesture"] for row in records])
    train = np.isin(subjects, PROTOCOL["source_subjects"])
    if train.sum() != 112 or rest_batch.windows != 560:
        raise ValueError("source-user trial or Rest count changed")
    predictions = []
    scores = {}
    for arm in ARMS:
        features = np.concatenate([vectors[name] for name in arm.split("+")], axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, random_state=20260924))
        model.fit(features[train], y[train])
        np.testing.assert_array_equal(model[-1].classes_, CLASSES)
        if np.max(model[-1].n_iter_) >= 2000:
            raise RuntimeError(f"source classifier did not converge: {arm}")
        arm_scores = {"train_trials": int(train.sum()), "feature_dimensions": int(features.shape[1])}
        for phase in ("validation", "final"):
            mask = np.isin(subjects, PROTOCOL[f"{phase}_subjects"])
            if mask.sum() != 56 or np.any(mask & train):
                raise ValueError("subject-disjoint target split failed")
            probability = model.predict_proba(features[mask])
            arm_scores[phase] = {"trials": int(mask.sum()),
                                 **grab.score(y[mask], probability, CLASSES, subjects[mask])}
            for row, p in zip(np.asarray(records, dtype=object)[mask], probability):
                predictions.append({"arm": arm, "phase": phase, "trial_id": row["stem"],
                                    "subject": row["subject"], "gesture": row["gesture"],
                                    **{f"p_{label}": float(value)
                                       for label, value in zip(CLASSES, p)}})
        scores[arm] = arm_scores
        print(f"{arm}: unseen-user validation F1={arm_scores['validation']['macro_f1']:.4f}, "
              f"descriptive final F1={arm_scores['final']['macro_f1']:.4f}", flush=True)
    selected = min(ARMS, key=lambda name: (
        -scores[name]["validation"]["macro_f1"],
        scores[name]["validation"]["log_loss"]))
    prediction_path = ROOT / "GRAB_USER_PREDICTIONS.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    result = {"protocol": PROTOCOL,
              "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
              "official_sha256_manifest_sha256": manifest_sha,
              "verified_day1_files": 448,
              "source_rest_windows": int(rest_batch.windows),
              "feature_dimensions": {name: int(vector.shape[1]) for name, vector in vectors.items()},
              "source_trial_ids": [row["stem"] for row in records if row["subject"] in source_set],
              "validation_trial_ids": [row["stem"] for row in records if row["subject"] in PROTOCOL["validation_subjects"]],
              "final_trial_ids": [row["stem"] for row in records if row["subject"] in PROTOCOL["final_subjects"]],
              "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
              "scores": scores, "validation_selected_arm": selected,
              "scope": PROTOCOL["scope"]}
    (ROOT / "GRAB_USER_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"GRAB user axis validation-selected arm: {selected}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=Path("D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1"))
    evaluate(parser.parse_args().data_root)
