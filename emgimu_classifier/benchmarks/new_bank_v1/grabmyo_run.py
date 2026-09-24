"""Frozen GRABMyo cross-day check of the independent v1 ring families."""
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
from emgimu.feature_bank.document_signal import RestNoiseLocalDetailFamily
from emgimu.feature_bank.new_bank_v1 import CorrelationSpectrumV1, RingLagV1


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "GRABMYO_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
ARMS = tuple(PROTOCOL["feature_arms"])
CLASSES = np.asarray(grab.GESTURES)


def aggregate(matrix: np.ndarray) -> np.ndarray:
    return np.r_[matrix.mean(axis=0), matrix.std(axis=0)].astype(np.float64)


def verify_protocol() -> None:
    if (tuple(PROTOCOL["subjects"]) != grab.SUBJECTS
            or PROTOCOL["sessions"] != grab.PROTOCOL["sessions"]
            or tuple(PROTOCOL["gesture_codes"]) != grab.GESTURES
            or tuple(PROTOCOL["channels"]) != grab.CHANNELS
            or PROTOCOL["sample_rate_hz"] != grab.RATE
            or PROTOCOL["window_samples"] != grab.WINDOW
            or ARMS != ("F0", "F0+ring_lag", "F0+correlation_spectrum",
                        "F0+ring_lag+correlation_spectrum")):
        raise ValueError("frozen new-bank and GRABMyo subset protocols disagree")


def verify_reproduced_baseline(predictions: list[dict]) -> None:
    path = ROOT.parent / "grabmyo_crossday" / "TRIAL_PREDICTIONS.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        old = {(row["split"], row["trial_id"]): row for row in csv.DictReader(stream)
               if row["arm"] == "F0"}
    current = {(row["split"], row["trial_id"]): row for row in predictions if row["arm"] == "F0"}
    if len(old) != 448 or old.keys() != current.keys():
        raise ValueError("F0 comparison lacks matched GRABMyo records")
    for key, row in current.items():
        reference = old[key]
        if (row["gesture"] != int(reference["gesture"])
                or row["subject"] != int(reference["subject"])):
            raise ValueError(f"F0 label or subject mismatch: {key}")
        for label in CLASSES:
            field = f"p_{label}"
            if abs(row[field] - float(reference[field])) > 1e-10:
                raise ValueError(f"F0 numerical mismatch: {key} {field}")


def evaluate(data_root: Path) -> None:
    verify_protocol()
    grab.check_all_files(data_root)
    records = list(grab.records())
    if len(records) != 672:
        raise ValueError("frozen GRABMyo record count changed")

    train_session = PROTOCOL["sessions"]["train"]
    rest = [grab.read_record(data_root, record) for record in records
            if record["session"] == train_session and record["gesture"] == 17]
    rest_batch = FeatureBatch(np.concatenate(rest), grab.RATE)
    families = {
        "F0": RestNoiseLocalDetailFamily(rest_label=17).fit(
            rest_batch, np.full(rest_batch.windows, 17)),
        "ring_lag": RingLagV1().fit(rest_batch),
        "correlation_spectrum": CorrelationSpectrumV1().fit(rest_batch),
    }
    vectors = {name: [] for name in families}
    for index, record in enumerate(records, 1):
        batch = FeatureBatch(grab.read_record(data_root, record), grab.RATE)
        for name, family in families.items():
            vectors[name].append(aggregate(family.transform(batch)))
        if index % 100 == 0 or index == len(records):
            print(f"GRABMyo new bank extracted {index}/{len(records)} trials", flush=True)
    vectors = {name: np.stack(items) for name, items in vectors.items()}
    sessions = np.asarray([record["session"] for record in records])
    labels = np.asarray([record["gesture"] for record in records])
    subjects = np.asarray([record["subject"] for record in records])
    train = sessions == train_session
    if train.sum() != 224 or any((sessions == number).sum() != 224 for number in (2, 3)):
        raise ValueError("native day split is incomplete")
    outputs: list[dict] = []
    result = {"protocol": PROTOCOL,
              "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
              "source_checksum_manifest_sha256": hashlib.sha256(
                  (data_root / "SHA256SUMS.txt").read_bytes()).hexdigest(),
              "source_file_verification": "all 1344 selected files match official SHA256SUMS",
              "training_rest_windows": int(rest_batch.windows),
              "feature_dimensions": {name: int(values.shape[1]) for name, values in vectors.items()},
              "arms": {}}
    for arm in ARMS:
        components = arm.split("+")
        features = np.concatenate([vectors[name] for name in components], axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, random_state=20260924))
        model.fit(features[train], labels[train])
        np.testing.assert_array_equal(model[-1].classes_, CLASSES)
        if np.max(model[-1].n_iter_) >= 2000:
            raise RuntimeError(f"classifier did not converge: {arm}")
        arm_result = {"train_trials": int(train.sum()),
                      "feature_dimensions": int(features.shape[1])}
        for split in ("validation", "final"):
            mask = sessions == PROTOCOL["sessions"][split]
            probability = model.predict_proba(features[mask])
            arm_result[split] = {"trials": int(mask.sum()),
                                 **grab.score(labels[mask], probability, CLASSES, subjects[mask])}
            for record, values in zip(np.asarray(records, dtype=object)[mask], probability):
                outputs.append({"arm": arm, "split": split, "trial_id": record["stem"],
                                "subject": record["subject"], "gesture": record["gesture"],
                                **{f"p_{label}": float(value)
                                   for label, value in zip(CLASSES, values)}})
        result["arms"][arm] = arm_result
        print(f"{arm}: Day2 F1={arm_result['validation']['macro_f1']:.4f}, "
              f"Day3 F1={arm_result['final']['macro_f1']:.4f}", flush=True)
    verify_reproduced_baseline(outputs)
    result["validation_selected_arm"] = min(ARMS, key=lambda name:
        (-result["arms"][name]["validation"]["macro_f1"],
         result["arms"][name]["validation"]["log_loss"]))
    (ROOT / "GRABMYO_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "GRABMYO_TRIAL_PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(outputs[0]))
        writer.writeheader()
        writer.writerows(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=Path("D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1"))
    evaluate(parser.parse_args().data_root)
