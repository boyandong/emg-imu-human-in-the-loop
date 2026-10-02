"""Frozen formula-correction replay on the existing GRAB three-day split."""
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
from emgimu.feature_bank.spec_spatial_v3 import SpecRingRelativeCovarianceV3, SpecTraceCovarianceV3


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / "new_bank_v2"
PROTOCOL_PATH = ROOT / "SPEC_SPATIAL_GRAB_PROTOCOL.json"
PREDICTIONS_PATH = ROOT / "SPEC_SPATIAL_GRAB_PREDICTIONS.csv"
RESULT_PATH = ROOT / "SPEC_SPATIAL_GRAB_RESULTS.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
CLASSES = np.asarray(grab.GESTURES)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(data_root: Path) -> dict:
    if sha256(PARENT / "GRABMYO_PROTOCOL.json") != PROTOCOL["parent_grab_protocol_sha256"]:
        raise AssertionError("Parent GRAB protocol changed")
    if sha256(PARENT / "GRABMYO_TRIAL_PREDICTIONS.csv") != PROTOCOL["parent_f0_prediction_sha256"]:
        raise AssertionError("Parent F0 predictions changed")
    if sha256(PARENT / "F3C_DOCUMENT_GRAB_PREDICTIONS.csv") != PROTOCOL["prior_uncentered_f3c_prediction_sha256"]:
        raise AssertionError("Prior uncentered F3c predictions changed")
    if (tuple(PROTOCOL["arms"]) != ("F0", "F0+F2a_spec", "F0+F3c_spec")
            or tuple(PROTOCOL["channels"]) != grab.CHANNELS
            or PROTOCOL["sample_rate_hz"] != grab.RATE
            or PROTOCOL["window_samples"] != grab.WINDOW):
        raise AssertionError("Frozen native protocol changed")
    grab.check_all_files(data_root)
    records = list(grab.records())
    if len(records) != 672:
        raise AssertionError("Trial inventory changed")
    rest = np.concatenate([grab.read_record(data_root, row) for row in records
                           if row["session"] == 1 and row["gesture"] == 17])
    rest_batch = FeatureBatch(rest, grab.RATE)
    families = {
        "F0": RestNoiseDetailV2(rest_label=17).fit(rest_batch, np.full(len(rest), 17)),
        "F2a_spec": SpecTraceCovarianceV3().fit(rest_batch),
        "F3c_spec": SpecRingRelativeCovarianceV3(ring_topology=True).fit(rest_batch),
    }
    vectors = {name: [] for name in families}
    for index, row in enumerate(records, 1):
        batch = FeatureBatch(grab.read_record(data_root, row), grab.RATE)
        for name, family in families.items():
            vectors[name].append(aggregate(family.transform(batch)))
        if index % 100 == 0 or index == len(records):
            print(f"spec spatial GRAB extracted {index}/{len(records)} trials", flush=True)
    vectors = {name: np.stack(value) for name, value in vectors.items()}
    sessions = np.asarray([row["session"] for row in records])
    labels = np.asarray([row["gesture"] for row in records])
    subjects = np.asarray([row["subject"] for row in records])
    train = sessions == 1
    if [int((sessions == day).sum()) for day in (1, 2, 3)] != [224, 224, 224]:
        raise AssertionError("Day split changed")
    predictions = []
    scores = {}
    for arm in PROTOCOL["arms"]:
        features = np.concatenate([vectors[name] for name in arm.split("+")], axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1., max_iter=2000, random_state=20260924))
        model.fit(features[train], labels[train])
        np.testing.assert_array_equal(model[-1].classes_, CLASSES)
        if np.max(model[-1].n_iter_) >= 2000:
            raise RuntimeError(f"Classifier did not converge: {arm}")
        scores[arm] = {"dimension": int(features.shape[1])}
        for split, day in (("validation", 2), ("final", 3)):
            mask = sessions == day
            probability = model.predict_proba(features[mask])
            scores[arm][split] = grab.score(labels[mask], probability, CLASSES, subjects[mask])
            for record, vector in zip(np.asarray(records, dtype=object)[mask], probability):
                predictions.append({"arm": arm, "split": split, "trial_id": record["stem"],
                                    "subject": record["subject"], "gesture": record["gesture"],
                                    **{f"p_{c}": float(p) for c, p in zip(CLASSES, vector)}})
        print(f"{arm}: Day2 F1={scores[arm]['validation']['macro_f1']:.4f}; "
              f"Day3 F1={scores[arm]['final']['macro_f1']:.4f}", flush=True)
    with (PARENT / "GRABMYO_TRIAL_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        old = {(row["split"], row["trial_id"]): row for row in csv.DictReader(stream)
               if row["arm"] == "F0"}
    f0 = [row for row in predictions if row["arm"] == "F0"]
    assert len(old) == len(f0) == 448
    for row in f0:
        reference = old[(row["split"], row["trial_id"])]
        np.testing.assert_allclose([row[f"p_{c}"] for c in CLASSES],
                                   [float(reference[f"p_{c}"]) for c in CLASSES],
                                   atol=1e-8, rtol=0)
    with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(predictions)
    result = {"protocol_sha256": sha256(PROTOCOL_PATH),
              "prediction_sha256": sha256(PREDICTIONS_PATH),
              "parent_f0_sha256": PROTOCOL["parent_f0_prediction_sha256"],
              "official_manifest_sha256": sha256(data_root / "SHA256SUMS.txt"),
              "prediction_rows": len(predictions),
              "dimensions": {name: int(value.shape[1]) for name, value in vectors.items()},
              "scores": scores, "boundary": PROTOCOL["boundary"]}
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    evaluate(Path("D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1"))
