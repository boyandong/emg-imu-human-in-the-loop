"""Frozen three-day GRABMyo evaluation of new v2 F2a/F3c features."""
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
from benchmarks.new_bank_v1.grabmyo_run import aggregate, verify_reproduced_baseline
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v2 import (
    RestNoiseDetailV2, RingRelativeCovarianceV2, TraceCovarianceV2,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "GRABMYO_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
ARMS = tuple(PROTOCOL["arms"])
CLASSES = np.asarray(grab.GESTURES)


def check_protocol() -> None:
    if (tuple(PROTOCOL["subjects"]) != grab.SUBJECTS
            or PROTOCOL["sessions"] != grab.PROTOCOL["sessions"]
            or tuple(PROTOCOL["gesture_codes"]) != grab.GESTURES
            or tuple(PROTOCOL["channels"]) != grab.CHANNELS
            or PROTOCOL["sample_rate_hz"] != grab.RATE
            or PROTOCOL["window_samples"] != grab.WINDOW
            or ARMS != ("F0", "F0+F2a", "F0+F3c", "F0+F2a+F3c")):
        raise ValueError("new v2 protocol does not match verified GRABMyo subset")


def evaluate(data_root: Path) -> dict:
    check_protocol()
    grab.check_all_files(data_root)
    records = list(grab.records())
    if len(records) != 672:
        raise ValueError("frozen record count changed")
    rest_windows = np.concatenate([grab.read_record(data_root, row) for row in records
                                   if row["session"] == 1 and row["gesture"] == 17])
    rest_batch = FeatureBatch(rest_windows, grab.RATE)
    families = {
        "F0": RestNoiseDetailV2(rest_label=17).fit(rest_batch, np.full(rest_batch.windows, 17)),
        "F2a": TraceCovarianceV2(shrinkage=0.05).fit(rest_batch),
        "F3c": RingRelativeCovarianceV2(shrinkage=0.05).fit(rest_batch),
    }
    vectors = {name: [] for name in families}
    for count, record in enumerate(records, 1):
        batch = FeatureBatch(grab.read_record(data_root, record), grab.RATE)
        for name, family in families.items():
            vectors[name].append(aggregate(family.transform(batch)))
        if count % 100 == 0 or count == len(records):
            print(f"new v2 GRABMyo extracted {count}/{len(records)} recordings", flush=True)
    vectors = {name: np.stack(values) for name, values in vectors.items()}
    sessions = np.asarray([row["session"] for row in records])
    y = np.asarray([row["gesture"] for row in records])
    subjects = np.asarray([row["subject"] for row in records])
    train = sessions == 1
    if any((sessions == day).sum() != 224 for day in (1, 2, 3)):
        raise ValueError("day split counts changed")
    predictions = []
    result = {
        "protocol": PROTOCOL,
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "official_sha256_manifest_sha256": hashlib.sha256((data_root / "SHA256SUMS.txt").read_bytes()).hexdigest(),
        "source_files_verified": 1344,
        "training_rest_windows": int(rest_batch.windows),
        "feature_dimensions": {name: int(values.shape[1]) for name, values in vectors.items()},
        "arms": {},
    }
    for arm in ARMS:
        parts = arm.split("+")
        features = np.concatenate([vectors[name] for name in parts], axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, random_state=20260924))
        model.fit(features[train], y[train])
        np.testing.assert_array_equal(model[-1].classes_, CLASSES)
        if np.max(model[-1].n_iter_) >= 2000:
            raise RuntimeError(f"classifier did not converge: {arm}")
        arm_result = {"train_trials": int(train.sum()), "feature_dimensions": int(features.shape[1])}
        for split, day in (("validation", 2), ("final", 3)):
            mask = sessions == day
            probability = model.predict_proba(features[mask])
            arm_result[split] = {"trials": int(mask.sum()),
                                 **grab.score(y[mask], probability, CLASSES, subjects[mask])}
            for row, values in zip(np.asarray(records, dtype=object)[mask], probability):
                predictions.append({"arm": arm, "split": split, "trial_id": row["stem"],
                                    "subject": row["subject"], "gesture": row["gesture"],
                                    **{f"p_{label}": float(value)
                                       for label, value in zip(CLASSES, values)}})
        result["arms"][arm] = arm_result
        print(f"{arm}: Day2 F1={arm_result['validation']['macro_f1']:.4f}, "
              f"Day3 F1={arm_result['final']['macro_f1']:.4f}", flush=True)
    verify_reproduced_baseline(predictions)
    result["baseline_reproduced"] = "old verified F0 probabilities within 1e-10 for all 448 held-out recordings"
    result["validation_selected_arm"] = min(ARMS, key=lambda name:
        (-result["arms"][name]["validation"]["macro_f1"],
         result["arms"][name]["validation"]["log_loss"]))
    (ROOT / "GRABMYO_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "GRABMYO_TRIAL_PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=Path("D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1"))
    evaluate(parser.parse_args().data_root)
