"""Acquire and evaluate the frozen public GRABMyo cross-day subset.

Raw records stay outside Git. The official SHA256SUMS verifies every selected
record before any feature extraction; no validation/final trials fit features.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.document_signal import RestNoiseLocalDetailFamily
from emgimu.feature_bank.families import SpectralStateFamily, TraceCovarianceFamily


ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "PROTOCOL.json").read_text(encoding="utf-8"))
BASE = "https://physionet.org/files/grabmyo/1.1.0/"
GESTURES = tuple(int(x) for x in PROTOCOL["gesture_codes"])
SESSIONS = tuple(PROTOCOL["sessions"].values())
SUBJECTS = tuple(PROTOCOL["subjects"])
TRIALS = tuple(PROTOCOL["trials_per_subject_session_gesture"])
RATE = int(PROTOCOL["sample_rate_hz"])
WINDOW = int(PROTOCOL["window_samples"])
CHANNELS = tuple(PROTOCOL["channels"])
GAIN_PATTERN = re.compile(r"^([+\-\deE.]+)\(([+\-\d]+)\)/([^\s]+)$")


def records():
    for session in SESSIONS:
        for subject in SUBJECTS:
            for gesture in GESTURES:
                for trial in TRIALS:
                    stem = f"session{session}_participant{subject}_gesture{gesture}_trial{trial}"
                    folder = f"Session{session}/session{session}_participant{subject}"
                    yield {"session": session, "subject": subject, "gesture": gesture,
                           "trial": trial, "stem": stem, "folder": folder}


def relative_file(record, suffix):
    return f"{record['folder']}/{record['stem']}.{suffix}"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def official_checksums(data_root):
    path = data_root / "SHA256SUMS.txt"
    if not path.exists():
        data = urllib.request.urlopen(BASE + "SHA256SUMS.txt", timeout=90).read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    checksums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        checksums[name.removeprefix("*")] = digest.lower()
    return checksums


def acquire_one(data_root, relative, expected):
    target = data_root / relative
    if target.exists() and sha256_bytes(target.read_bytes()) == expected:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    last_error = None
    for _ in range(3):
        try:
            request = urllib.request.Request(BASE + relative, headers={"User-Agent": "GRABMyo-research-audit/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()
            if sha256_bytes(data) != expected:
                raise ValueError(f"official checksum mismatch: {relative}")
            temporary.write_bytes(data)
            os.replace(temporary, target)
            return True
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to acquire {relative}: {last_error}")


def acquire(data_root, workers):
    checksums = official_checksums(data_root)
    selected = [relative_file(record, suffix) for record in records() for suffix in ("hea", "dat")]
    missing = [name for name in selected if name not in checksums]
    if missing:
        raise ValueError(f"{len(missing)} selected files absent from official SHA256SUMS, first={missing[0]}")
    completed = 0
    fetched = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(acquire_one, data_root, name, checksums[name]): name for name in selected}
        for future in as_completed(futures):
            fetched += int(future.result())
            completed += 1
            if completed % 50 == 0 or completed == len(selected):
                print(f"GRABMyo verified {completed}/{len(selected)} files; fetched {fetched}", flush=True)
    return {"source": PROTOCOL["source"], "source_version": PROTOCOL["source_version"],
            "checksum_manifest_sha256": sha256_bytes((data_root / "SHA256SUMS.txt").read_bytes()),
            "selected_files": len(selected), "fetched_now": fetched,
            "selected_records": len(selected) // 2}


def read_record(data_root, record):
    header = (data_root / relative_file(record, "hea")).read_text(encoding="utf-8").splitlines()
    stem, n_channels, sample_rate, n_samples = header[0].split()[:4]
    if (stem != record["stem"] or int(n_channels) != 32 or int(sample_rate) != RATE or
            int(n_samples) != 10240 or len(header) < 33):
        raise ValueError(f"unexpected WFDB header: {record['stem']}")
    gains = []
    baselines = []
    for index, expected_name in enumerate(CHANNELS):
        fields = header[index + 1].split()
        match = GAIN_PATTERN.fullmatch(fields[2])
        if fields[1] != "16" or fields[-1] != expected_name or not match or match.group(3) != "mV":
            raise ValueError(f"unexpected channel encoding: {record['stem']} {expected_name}")
        gains.append(float(match.group(1)))
        baselines.append(int(match.group(2)))
    raw = np.fromfile(data_root / relative_file(record, "dat"), dtype="<i2")
    if raw.size != 32 * int(n_samples):
        raise ValueError(f"unexpected signal length: {record['stem']}")
    physical = (raw.reshape(int(n_samples), 32)[:, :8].astype(np.float64)
                - np.asarray(baselines)) / np.asarray(gains)
    if not np.all(np.isfinite(physical)):
        raise ValueError(f"nonfinite physical signal: {record['stem']}")
    return physical.reshape(-1, WINDOW, len(CHANNELS))


def check_all_files(data_root):
    checksums = official_checksums(data_root)
    for record in records():
        for suffix in ("hea", "dat"):
            relative = relative_file(record, suffix)
            path = data_root / relative
            if not path.exists() or sha256_bytes(path.read_bytes()) != checksums[relative]:
                raise ValueError(f"missing or checksum-failing source file: {relative}")


def aggregate(matrix):
    return np.concatenate((matrix.mean(axis=0), matrix.std(axis=0))).astype(np.float64)


def features_for_record(windows, families):
    batch = FeatureBatch(windows, RATE)
    f0 = aggregate(families["F0"].transform(batch))
    f2a = aggregate(families["F2a"].transform(batch))
    f4 = aggregate(families["F4"].transform(batch))
    return {"F0": f0, "F0+F2a": np.r_[f0, f2a], "F0+F4": np.r_[f0, f4]}


def score(y, probabilities, classes, subjects):
    predicted = classes[np.argmax(probabilities, axis=1)]
    one_hot = (y[:, None] == classes[None, :]).astype(float)
    user_f1 = {str(subject): float(f1_score(y[subjects == subject], predicted[subjects == subject],
                                            labels=classes, average="macro", zero_division=0))
               for subject in sorted(set(subjects))}
    return {"accuracy": float(accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, labels=classes, average="macro", zero_division=0)),
            "log_loss": float(log_loss(y, probabilities, labels=classes)),
            "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
            "per_class_recall": {str(label): float(value) for label, value in zip(
                classes, recall_score(y, predicted, labels=classes, average=None, zero_division=0))},
            "per_subject_macro_f1": user_f1,
            "minimum_subject_macro_f1": min(user_f1.values())}


def evaluate(data_root, output):
    check_all_files(data_root)
    all_records = list(records())
    if len(all_records) != 672:
        raise ValueError("protocol record count changed")
    training_rest = [read_record(data_root, row) for row in all_records
                     if row["session"] == PROTOCOL["sessions"]["train"] and row["gesture"] == 17]
    rest_batch = FeatureBatch(np.concatenate(training_rest), RATE)
    families = {"F0": RestNoiseLocalDetailFamily(rest_label=17).fit(
        rest_batch, np.full(rest_batch.windows, 17)),
        "F2a": TraceCovarianceFamily().fit(rest_batch),
        "F4": SpectralStateFamily().fit(rest_batch)}
    rows = []
    vectors = {arm: [] for arm in PROTOCOL["feature_arms"]}
    for index, record in enumerate(all_records, 1):
        row = dict(record)
        row["trial_id"] = row["stem"]
        rows.append(row)
        features = features_for_record(read_record(data_root, record), families)
        for arm in vectors:
            vectors[arm].append(features[arm])
        if index % 100 == 0 or index == len(all_records):
            print(f"GRABMyo extracted {index}/{len(all_records)} trials", flush=True)
    if len({row["trial_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate trial identity")
    session = np.asarray([row["session"] for row in rows])
    y = np.asarray([row["gesture"] for row in rows])
    subjects = np.asarray([row["subject"] for row in rows])
    classes = np.asarray(GESTURES)
    result = {"protocol": PROTOCOL, "source_file_verification": "all 1344 selected files match official SHA256SUMS",
              "f0_rest_training_windows": int(rest_batch.windows), "arms": {}}
    predictions = []
    for arm, items in vectors.items():
        x = np.stack(items)
        train = session == PROTOCOL["sessions"]["train"]
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=20260924))
        model.fit(x[train], y[train])
        if not np.array_equal(model[-1].classes_, classes):
            raise ValueError("class order mismatch")
        arm_result = {"train_trials": int(train.sum()), "feature_dimensions": int(x.shape[1])}
        for split in ("validation", "final"):
            mask = session == PROTOCOL["sessions"][split]
            probability = model.predict_proba(x[mask])
            arm_result[split] = {"trials": int(mask.sum()), **score(y[mask], probability, classes, subjects[mask])}
            for row, prob in zip(np.asarray(rows, dtype=object)[mask], probability):
                predictions.append({"arm": arm, "split": split, "trial_id": row["trial_id"],
                                    "subject": row["subject"], "gesture": row["gesture"],
                                    **{f"p_{label}": float(value) for label, value in zip(classes, prob)}})
        result["arms"][arm] = arm_result
        print(f"{arm}: S2 macro-F1={arm_result['validation']['macro_f1']:.4f}, "
              f"S3 macro-F1={arm_result['final']['macro_f1']:.4f}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "RESULTS.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "TRIAL_PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("acquire", "evaluate"))
    parser.add_argument("--data-root", type=Path, default=Path("D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.phase == "acquire":
        print(json.dumps(acquire(args.data_root, args.workers), indent=2), flush=True)
    else:
        evaluate(args.data_root, ROOT)


if __name__ == "__main__":
    main()
