"""Frozen synthetic-quality stress grid for source-only ROAM new-v2 arms."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.grabmyo_crossday.run import aggregate, score
from benchmarks.new_bank_v2.roam_posture_run import extract_archive, sha256
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v2 import (
    RestNoiseDetailV2, RingRelativeCovarianceV2, TraceCovarianceV2,
)

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "ROAM_QUALITY_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
CLASSES = np.asarray([0, 1, 2])
ARMS = tuple(PROTOCOL["arms"])


def check_protocol() -> None:
    posture = json.loads((ROOT / "ROAM_POSTURE_PROTOCOL.json").read_text(encoding="utf-8"))
    if (sha256(ROOT / "ROAM_POSTURE_PROTOCOL.json") != PROTOCOL["parent_posture_protocol_sha256"]
            or sha256(ROOT / "ROAM_POSTURE_PREDICTIONS.csv") != PROTOCOL["parent_posture_prediction_sha256"]
            or PROTOCOL["source_subjects"] != posture["source_subjects"]
            or PROTOCOL["validation_subjects"] != posture["validation_subjects"]
            or PROTOCOL["final_subjects"] != posture["final_subjects"]
            or PROTOCOL["posture"] != posture["source_posture"]
            or PROTOCOL["sample_rate_hz"] != 200 or PROTOCOL["window_samples"] != 40
            or ARMS != tuple(posture["arms"])
            or PROTOCOL["conditions"] != ["clean", *[f"dropout_ch{i}" for i in range(8)],
                                          "gain_half_all", "clip_q95_all", "line50_half_rms_all",
                                          "baseline_ramp_half_rms_all", "burst_ch0_2q95"]):
        raise ValueError("frozen ROAM quality protocol changed")


def fault(windows: np.ndarray, name: str, q95: np.ndarray, rms: np.ndarray) -> np.ndarray:
    """Return independent test-only fault copy; references derive from source only."""
    x = np.asarray(windows, dtype=np.float32).copy()
    if name == "clean":
        return x
    if name.startswith("dropout_ch"):
        channel = int(name.removeprefix("dropout_ch"))
        if channel not in range(8):
            raise ValueError("invalid synthetic dropout channel")
        x[:, :, channel] = 0
    elif name == "gain_half_all":
        x *= 0.5
    elif name == "clip_q95_all":
        np.clip(x, -q95, q95, out=x)
    elif name == "line50_half_rms_all":
        t = np.arange(x.shape[1], dtype=np.float32) / 200.0
        x += np.sin(2 * np.pi * 50 * t)[None, :, None] * (0.5 * rms)[None, None, :]
    elif name == "baseline_ramp_half_rms_all":
        ramp = np.linspace(-0.5, 0.5, x.shape[1], dtype=np.float32)
        x += ramp[None, :, None] * rms[None, None, :]
    elif name == "burst_ch0_2q95":
        x[:, 16:24, 0] += (2 * q95[0] * np.hanning(8))[None, :]
    else:
        raise ValueError(f"unknown synthetic corruption: {name}")
    return x


def packed(batch: FeatureBatch, slices: list[tuple[int, int]], indices: np.ndarray):
    windows = []
    intervals = []
    for i in indices:
        a, b = slices[int(i)]
        start = len(windows)
        windows.extend(batch.emg[a:b])
        intervals.append((start, len(windows)))
    return np.stack(windows), intervals


def aggregate_families(windows: np.ndarray, intervals, families) -> dict[str, np.ndarray]:
    batch = FeatureBatch(windows, 200)
    return {name: np.stack([aggregate(values[a:b]) for a, b in intervals])
            for name, family in families.items()
            for values in [family.transform(batch)]}


def replay_clean(predictions: list[dict]) -> None:
    with (ROOT / "ROAM_POSTURE_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        original = list(csv.DictReader(stream))
    original = {(r["arm"], r["phase"], r["trial_id"]): r for r in original
                if r["condition"] == "resting"}
    if len(original) != 360 or len(predictions) != 360:
        raise AssertionError("clean source-model replay count changed")
    for row in predictions:
        key = (row["arm"], row["phase"], row["trial_id"])
        if key not in original or int(original[key]["label"]) != row["label"]:
            raise AssertionError("clean source-model replay identity mismatch")
        actual = np.asarray([row[f"p_{c}"] for c in CLASSES], dtype=float)
        expected = np.asarray([float(original[key][f"p_{c}"]) for c in CLASSES])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def evaluate() -> dict:
    check_protocol()
    archive = Path(PROTOCOL["archive"])
    if sha256(archive).lower() != PROTOCOL["archive_sha256"].lower():
        raise ValueError("ROAM archive hash mismatch")
    batch, y, identities, slices, counts = extract_archive(archive)
    subjects = np.asarray([row["subject"] for row in identities])
    postures = np.asarray([row["condition"] for row in identities])
    source = np.isin(subjects, PROTOCOL["source_subjects"]) & (postures == "resting")
    source_indices = np.flatnonzero(source)
    source_windows, source_slices = packed(batch, slices, source_indices)
    rest_indices = [i for i, bout in enumerate(source_indices) if y[bout] == 0]
    rest_windows = np.concatenate([source_windows[source_slices[i][0]:source_slices[i][1]]
                                   for i in rest_indices])
    rest_batch = FeatureBatch(rest_windows, 200)
    families = {"F0v2": RestNoiseDetailV2(rest_label=0).fit(
                    rest_batch, np.zeros(rest_batch.windows, dtype=int)),
                "F2a": TraceCovarianceV2(shrinkage=0.05).fit(rest_batch),
                "F3c": RingRelativeCovarianceV2(shrinkage=0.05).fit(rest_batch)}
    source_vectors = aggregate_families(source_windows, source_slices, families)
    source_y = y[source_indices]
    if len(source_y) != 162 or rest_batch.windows != 1828:
        raise ValueError("source state and Rest-window count differ from frozen posture study")
    q95 = np.quantile(np.abs(source_windows.astype(np.float64)), 0.95, axis=(0, 1)).astype(np.float32)
    rms = np.sqrt(np.mean(source_windows.astype(np.float64) ** 2, axis=(0, 1))).astype(np.float32)
    if np.any(q95 <= 0) or np.any(rms <= 0):
        raise ValueError("source quality references invalid")
    models = {}
    for arm in ARMS:
        matrix = np.concatenate([source_vectors[name] for name in arm.split("+")], axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=20260929))
        model.fit(matrix, source_y)
        np.testing.assert_array_equal(model[-1].classes_, CLASSES)
        if np.max(model[-1].n_iter_) >= 2000:
            raise RuntimeError(f"source classifier did not converge: {arm}")
        models[arm] = model
    prediction_rows = []
    scores: dict = {arm: {} for arm in ARMS}
    for phase in ("validation", "final"):
        mask = np.isin(subjects, PROTOCOL[f"{phase}_subjects"]) & (postures == "resting")
        indices = np.flatnonzero(mask)
        if len(indices) != 45 or np.any(source[indices]):
            raise ValueError("held-out native-bout identity contract failed")
        target_windows, target_slices = packed(batch, slices, indices)
        for condition in PROTOCOL["conditions"]:
            corrupted = fault(target_windows, condition, q95, rms)
            features = aggregate_families(corrupted, target_slices, families)
            for arm in ARMS:
                matrix = np.concatenate([features[name] for name in arm.split("+")], axis=1)
                probability = models[arm].predict_proba(matrix)
                scores[arm].setdefault(phase, {})[condition] = score(
                    y[indices], probability, CLASSES, subjects[indices])
                for i, p in zip(indices, probability):
                    identity = identities[int(i)]
                    prediction_rows.append({"arm": arm, "phase": phase, "condition": condition,
                                            "trial_id": identity["trial_id"],
                                            "subject": identity["subject"],
                                            "label": identity["label"],
                                            **{f"p_{c}": float(value)
                                               for c, value in zip(CLASSES, p)}})
            print(f"ROAM quality {phase} {condition}", flush=True)
    replay_clean([row for row in prediction_rows if row["condition"] == "clean"])
    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for phase in ("validation", "final"):
            cells = scores[arm][phase]
            coordinates = {
                "dropout_mean": float(np.mean([cells[f"dropout_ch{i}"]["macro_f1"] for i in range(8)])),
                **{name: cells[name]["macro_f1"] for name in PROTOCOL["conditions"][9:]}}
            quality_mean = float(np.mean(list(coordinates.values())))
            minimum = min(coordinates, key=coordinates.get)
            summary[arm][phase] = {"clean_macro_f1": cells["clean"]["macro_f1"],
                                   "family_macro_f1": coordinates,
                                   "synthetic_quality_mean_macro_f1": quality_mean,
                                   "synthetic_quality_min_macro_f1": float(coordinates[minimum]),
                                   "synthetic_quality_min_family": minimum,
                                   "mean_minus_clean_macro_f1": quality_mean - cells["clean"]["macro_f1"],
                                   "minimum_named_fault_macro_f1": min(
                                       cells[name]["macro_f1"] for name in PROTOCOL["conditions"][1:])}
    path = ROOT / "ROAM_QUALITY_PREDICTIONS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    result = {"protocol": PROTOCOL, "protocol_sha256": sha256(PROTOCOL_PATH),
              "archive_sha256": sha256(archive), "prediction_sha256": sha256(path),
              "source_rest_windows": int(rest_batch.windows),
              "source_reference_q95": q95.tolist(), "source_reference_rms": rms.tolist(),
              "source_trial_ids": [identities[int(i)]["trial_id"] for i in source_indices],
              "validation_trial_ids": [row["trial_id"] for row in identities
                                       if row["subject"] in PROTOCOL["validation_subjects"] and row["condition"] == "resting"],
              "final_trial_ids": [row["trial_id"] for row in identities
                                  if row["subject"] in PROTOCOL["final_subjects"] and row["condition"] == "resting"],
              "native_static_files_verified": len(counts),
              "clean_source_model_replay": "all 360 arm-phase-bout vectors within absolute 1e-10",
              "scores": scores, "summary": summary, "scope": PROTOCOL["scope"]}
    (ROOT / "ROAM_QUALITY_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for arm in ARMS:
        print(f"{arm}: validation quality mean={summary[arm]['validation']['synthetic_quality_mean_macro_f1']:.4f}, "
              f"final={summary[arm]['final']['synthetic_quality_mean_macro_f1']:.4f}", flush=True)
    return result


if __name__ == "__main__":
    evaluate()
