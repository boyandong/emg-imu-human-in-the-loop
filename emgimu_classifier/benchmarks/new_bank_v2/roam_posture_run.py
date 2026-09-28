"""Frozen, subject-disjoint ROAM-EMG static-posture new-v2 comparison."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.grabmyo_crossday.run import aggregate, score
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v2 import (
    RestNoiseDetailV2, RingRelativeCovarianceV2, TraceCovarianceV2,
)

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "ROAM_POSTURE_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
CLASSES = np.asarray([0, 1, 2])
ARMS = tuple(PROTOCOL["arms"])


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_protocol() -> None:
    if (PROTOCOL["source_subjects"] != list(range(1, 19))
            or PROTOCOL["validation_subjects"] != list(range(19, 24))
            or PROTOCOL["final_subjects"] != list(range(24, 29))
            or PROTOCOL["target_postures"] != ["resting", "hanging", "unsupported", "reaching"]
            or PROTOCOL["source_posture"] != "resting"
            or PROTOCOL["window_samples"] != 40
            or PROTOCOL["bout_edge_exclusion_samples"] != 40
            or PROTOCOL["sample_rate_hz"] != 200
            or ARMS != ("F0v2", "F0v2+F2a", "F0v2+F3c", "F0v2+F2a+F3c")):
        raise ValueError("frozen ROAM posture protocol changed")


def read_native(stream, name: str) -> tuple[np.ndarray, np.ndarray]:
    reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"))
    channels = PROTOCOL["channels"]
    if not set(["gt", "time_elapsed", *channels]) <= set(reader.fieldnames or []):
        raise ValueError(f"missing native CSV columns: {name}")
    labels, values, times = [], [], []
    for row in reader:
        labels.append(int(row["gt"]))
        values.append([float(row[channel]) for channel in channels])
        times.append(float(row["time_elapsed"]))
    y = np.asarray(labels, dtype=np.int64)
    emg = np.asarray(values, dtype=np.float32)
    if (len(y) < 7000 or emg.shape != (len(y), 8)
            or not set(y) <= set(CLASSES) or not np.isfinite(emg).all()
            or not 0.8 * len(y) / 200 <= times[-1] - times[0] <= 1.2 * len(y) / 200):
        raise ValueError(f"native channel, label, length or timing contract failed: {name}")
    return emg, y


def extract_archive(path: Path):
    windows: list[np.ndarray] = []
    labels: list[int] = []
    identities: list[dict] = []
    slices: list[tuple[int, int]] = []
    counts = {}
    with zipfile.ZipFile(path) as archive:
        native = {item.filename for item in archive.infolist() if not item.is_dir()}
        for subject in range(1, 29):
            for posture in PROTOCOL["target_postures"]:
                name = f"data/ROAM_EMG/s{subject}/s{subject}_static_{posture}.csv"
                if name not in native:
                    raise ValueError(f"missing native static recording: {name}")
                with archive.open(name) as stream:
                    emg, y = read_native(stream, name)
                changes = np.r_[0, np.flatnonzero(np.diff(y)) + 1, len(y)]
                sequence = [int(y[changes[i]]) for i in range(len(changes) - 1)]
                if sequence != PROTOCOL["expected_sequence"]:
                    raise ValueError(f"native gesture sequence differs: {name}: {sequence}")
                counts[name] = {"samples": len(y), "bouts": len(sequence)}
                for bout in range(len(sequence)):
                    left = int(changes[bout]) + PROTOCOL["bout_edge_exclusion_samples"]
                    right = int(changes[bout + 1]) - PROTOCOL["bout_edge_exclusion_samples"]
                    size = PROTOCOL["window_samples"]
                    n = (right - left) // size
                    if n < 10:
                        raise ValueError(f"too few uncontaminated windows: {name}, bout {bout}")
                    start = len(windows)
                    for j in range(n):
                        windows.append(emg[left + j * size:left + (j + 1) * size])
                    slices.append((start, len(windows)))
                    label = sequence[bout]
                    labels.append(label)
                    identities.append({"trial_id": f"s{subject}_static_{posture}_bout{bout + 1}",
                                       "subject": subject, "condition": posture, "label": label,
                                       "native_file": name, "bout": bout + 1, "windows": n})
            if subject % 7 == 0:
                print(f"ROAM static extracted {subject}/28 subjects", flush=True)
    return FeatureBatch(np.stack(windows), 200), np.asarray(labels), identities, slices, counts


def evaluate() -> dict:
    check_protocol()
    archive_path = Path(PROTOCOL["archive"])
    if sha256(archive_path).lower() != PROTOCOL["archive_sha256"].lower():
        raise ValueError("ROAM source archive SHA-256 mismatch")
    batch, y, identities, slices, counts = extract_archive(archive_path)
    if len(identities) != 1008 or len(counts) != 112:
        raise ValueError("native recording/bout count differs")
    subjects = np.asarray([row["subject"] for row in identities])
    postures = np.asarray([row["condition"] for row in identities])
    source = np.isin(subjects, PROTOCOL["source_subjects"]) & (postures == "resting")
    rest_windows = np.concatenate([batch.emg[a:b] for i, (a, b) in enumerate(slices)
                                   if source[i] and y[i] == 0])
    rest_batch = FeatureBatch(rest_windows, 200)
    families = {"F0v2": RestNoiseDetailV2(rest_label=0).fit(
                    rest_batch, np.zeros(rest_batch.windows, dtype=int)),
                "F2a": TraceCovarianceV2(shrinkage=0.05).fit(rest_batch),
                "F3c": RingRelativeCovarianceV2(shrinkage=0.05).fit(rest_batch)}
    vectors = {}
    for name, family in families.items():
        transformed = family.transform(batch)
        vectors[name] = np.stack([aggregate(transformed[a:b]) for a, b in slices])
    if source.sum() != 162 or rest_batch.windows < 1000:
        raise ValueError("source bout or Rest calibration contract failed")
    predictions, scores = [], {}
    for arm in ARMS:
        x = np.concatenate([vectors[name] for name in arm.split("+")], axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=20260929))
        model.fit(x[source], y[source])
        np.testing.assert_array_equal(model[-1].classes_, CLASSES)
        if np.max(model[-1].n_iter_) >= 2000:
            raise RuntimeError(f"source classifier did not converge: {arm}")
        arm_scores = {"source_bouts": int(source.sum()), "feature_dimensions": x.shape[1]}
        for phase in ("validation", "final"):
            mask = np.isin(subjects, PROTOCOL[f"{phase}_subjects"])
            if mask.sum() != 180 or np.any(mask & source):
                raise ValueError("target subject split changed")
            prob = model.predict_proba(x[mask])
            pooled = score(y[mask], prob, CLASSES, subjects[mask])
            by_posture = {posture: score(y[mask & (postures == posture)],
                                         model.predict_proba(x[mask & (postures == posture)]),
                                         CLASSES, subjects[mask & (postures == posture)])
                          for posture in PROTOCOL["target_postures"]}
            arm_scores[phase] = {**pooled, "bouts": int(mask.sum()),
                                 "by_posture": by_posture,
                                 "minimum_posture_macro_f1": min(
                                     row["macro_f1"] for row in by_posture.values())}
            for identity, probability in zip(np.asarray(identities, dtype=object)[mask], prob):
                predictions.append({"arm": arm, "phase": phase,
                                    "trial_id": identity["trial_id"],
                                    "subject": identity["subject"],
                                    "condition": identity["condition"],
                                    "label": identity["label"],
                                    **{f"p_{c}": float(value) for c, value in zip(CLASSES, probability)}})
        scores[arm] = arm_scores
        print(f"{arm}: validation F1={arm_scores['validation']['macro_f1']:.4f}, "
              f"final F1={arm_scores['final']['macro_f1']:.4f}", flush=True)
    chosen = min(ARMS, key=lambda arm: (-scores[arm]["validation"]["macro_f1"],
                                      scores[arm]["validation"]["log_loss"]))
    prediction_path = ROOT / "ROAM_POSTURE_PREDICTIONS.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    result = {"protocol": PROTOCOL,
              "protocol_sha256": sha256(PROTOCOL_PATH),
              "archive_sha256": sha256(archive_path),
              "prediction_sha256": sha256(prediction_path),
              "native_file_counts": counts,
              "source_rest_windows": int(rest_batch.windows),
              "total_windows": int(batch.windows),
              "feature_dimensions": {name: int(vector.shape[1]) for name, vector in vectors.items()},
              "source_trial_ids": [row["trial_id"] for i, row in enumerate(identities) if source[i]],
              "validation_trial_ids": [row["trial_id"] for row in identities if row["subject"] in PROTOCOL["validation_subjects"]],
              "final_trial_ids": [row["trial_id"] for row in identities if row["subject"] in PROTOCOL["final_subjects"]],
              "validation_selected_arm": chosen, "scores": scores, "scope": PROTOCOL["scope"]}
    (ROOT / "ROAM_POSTURE_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"ROAM posture validation-selected arm: {chosen}", flush=True)
    return result


if __name__ == "__main__":
    evaluate()
