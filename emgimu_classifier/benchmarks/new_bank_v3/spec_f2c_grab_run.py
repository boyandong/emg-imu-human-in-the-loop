"""Source-only F2c V3 native increment on the frozen GRAB three-day split."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.grabmyo_crossday import run as grab
from benchmarks.new_bank_v1.grabmyo_run import aggregate
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2
from emgimu.feature_bank.spec_spatial_v3 import SpecSpdTangentV3


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "SPEC_F2C_GRAB_PROTOCOL.json"
PREDICTIONS_PATH = ROOT / "SPEC_F2C_GRAB_PREDICTIONS.csv"
RESULT_PATH = ROOT / "SPEC_F2C_GRAB_RESULTS.json"
PARENT_PROTOCOL = ROOT / "SPEC_SPATIAL_GRAB_PROTOCOL.json"
PARENT_PREDICTIONS = ROOT / "SPEC_SPATIAL_GRAB_PREDICTIONS.csv"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
CLASSES = np.asarray(grab.GESTURES)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(data_root: Path) -> dict:
    if (sha256(PARENT_PROTOCOL) != PROTOCOL["parent_spatial_protocol_sha256"]
            or sha256(PARENT_PREDICTIONS) != PROTOCOL["parent_spatial_prediction_sha256"]):
        raise AssertionError("Frozen V3 parent protocol or probabilities changed")
    if (tuple(PROTOCOL["channels"]) != grab.CHANNELS
            or PROTOCOL["sample_rate_hz"] != grab.RATE
            or PROTOCOL["window_samples"] != grab.WINDOW
            or PROTOCOL["arm"] != "F0+F2c_spec"):
        raise AssertionError("Native F2c protocol changed")
    grab.check_all_files(data_root)
    records = list(grab.records())
    if len(records) != 672:
        raise AssertionError("Frozen trial inventory changed")
    source_records = [row for row in records if row["session"] == 1]
    source_windows = np.concatenate([grab.read_record(data_root, row)
                                     for row in source_records])
    source_batch = FeatureBatch(source_windows, grab.RATE)
    rest = np.asarray([row["gesture"] == 17 for row in source_records])
    if source_batch.windows != 4480 or int(rest.sum()) != 56:
        raise AssertionError("Frozen source-window inventory changed")
    rest_windows = np.concatenate([grab.read_record(data_root, row)
                                   for row in source_records if row["gesture"] == 17])
    rest_batch = FeatureBatch(rest_windows, grab.RATE)
    f0 = RestNoiseDetailV2(rest_label=17).fit(rest_batch, np.full(rest_batch.windows, 17))
    f2c = SpecSpdTangentV3().fit(source_batch)
    print(f"spec F2c GRAB source reference fitted on {source_batch.windows} Day1 windows", flush=True)
    features = []
    for index, row in enumerate(records, 1):
        batch = FeatureBatch(grab.read_record(data_root, row), grab.RATE)
        features.append(np.concatenate([aggregate(f0.transform(batch)),
                                        aggregate(f2c.transform(batch))]))
        if index % 100 == 0 or index == len(records):
            print(f"spec F2c GRAB extracted {index}/{len(records)} trials", flush=True)
    features = np.stack(features)
    sessions = np.asarray([row["session"] for row in records])
    labels = np.asarray([row["gesture"] for row in records])
    subjects = np.asarray([row["subject"] for row in records])
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1., max_iter=2000, random_state=20260924))
    model.fit(features[sessions == 1], labels[sessions == 1])
    np.testing.assert_array_equal(model[-1].classes_, CLASSES)
    if np.max(model[-1].n_iter_) >= 2000:
        raise RuntimeError("F2c classifier did not converge")
    rows = []
    scores = {}
    for split, day in (("validation", 2), ("final", 3)):
        mask = sessions == day
        probabilities = model.predict_proba(features[mask])
        scores[split] = grab.score(labels[mask], probabilities, CLASSES, subjects[mask])
        for record, vector in zip(np.asarray(records, dtype=object)[mask], probabilities):
            rows.append({"arm": PROTOCOL["arm"], "split": split,
                         "trial_id": record["stem"], "subject": record["subject"],
                         "gesture": record["gesture"],
                         **{f"p_{c}": float(p) for c, p in zip(CLASSES, vector)}})
        print(f"{split} F1={scores[split]['macro_f1']:.4f}; "
              f"log loss={scores[split]['log_loss']:.4f}", flush=True)
    with PARENT_PREDICTIONS.open(newline="", encoding="utf-8") as stream:
        parent = {(row["split"], row["trial_id"]): row for row in csv.DictReader(stream)
                  if row["arm"] == "F0"}
    if len(parent) != 448 or len(rows) != 448 or {
            (row["split"], row["trial_id"]) for row in rows} != set(parent):
        raise AssertionError("F2c held-out trials do not match frozen F0")
    with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol_sha256": sha256(PROTOCOL_PATH),
              "parent_prediction_sha256": PROTOCOL["parent_spatial_prediction_sha256"],
              "prediction_sha256": sha256(PREDICTIONS_PATH),
              "official_manifest_sha256": sha256(data_root / "SHA256SUMS.txt"),
              "source_reference_windows": source_batch.windows,
              "prediction_rows": len(rows), "feature_dimension": int(features.shape[1]),
              "scores": scores, "boundary": PROTOCOL["boundary"]}
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    evaluate(Path("D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1"))
