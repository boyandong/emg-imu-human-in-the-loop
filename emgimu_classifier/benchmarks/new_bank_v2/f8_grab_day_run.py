"""Reproduce frozen GRAB day providers and compare fixed F8 routing on held-out trials."""
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
from emgimu.feature_bank.calibration import SessionSignature
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v1 import FrequencyDirectionV1
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "F8_GRAB_DAY_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
PARENT = ROOT / "GRAB_DAY_V1_EXTENSION_PREDICTIONS.csv"
CLASSES = np.asarray(grab.GESTURES)
PROVIDERS = tuple(PROTOCOL["providers"])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_protocol(data_root: Path) -> None:
    checks = (
        (ROOT / "GRAB_DAY_V1_EXTENSION_PROTOCOL.json", "parent_protocol_sha256"),
        (PARENT, "parent_predictions_sha256"),
        (data_root / "SHA256SUMS.txt", "official_sha256_manifest_sha256"),
    )
    for path, key in checks:
        if digest(path) != PROTOCOL[key]:
            raise ValueError(f"frozen F8 parent changed: {key}")
    if (PROVIDERS != ("F0v2", "F0v2+frequency_direction")
            or list(PROTOCOL["population_weights"]) != [0.5, 0.5]
            or PROTOCOL["calibration_trial_number_per_class"] != 1
            or PROTOCOL["evaluation_trial_numbers_per_class"] != [2, 3, 4, 5, 6, 7]):
        raise ValueError("unsupported F8 routing protocol")


def fixed_weights(signatures: dict[str, SessionSignature],
                  scaled: dict[str, np.ndarray], labels: np.ndarray,
                  calibration: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    weights = np.asarray(PROTOCOL["population_weights"], dtype=float)
    vectors = {}
    for i, name in enumerate(PROVIDERS):
        before = signatures[name].long_term_.copy()
        vector = signatures[name].from_session_calibration(
            scaled[name][calibration], labels[calibration])
        np.testing.assert_array_equal(before, signatures[name].long_term_)
        n = len(CLASSES)
        weights[i] *= float(np.clip((vector[n:2*n].mean() + 1.0) / 2.0, 0.05, 1.0))
        vectors[name] = vector.tolist()
    weights /= weights.sum()
    return weights, vectors


def run(data_root: Path = Path("D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1")) -> dict:
    check_protocol(data_root)
    grab.check_all_files(data_root)
    records = list(grab.records())
    if len(records) != 672:
        raise ValueError("frozen native recording count changed")
    rest = np.concatenate([grab.read_record(data_root, row) for row in records
                           if row["session"] == 1 and row["gesture"] == 17])
    rest_batch = FeatureBatch(rest, grab.RATE)
    families = {
        "F0v2": RestNoiseDetailV2(rest_label=17).fit(
            rest_batch, np.full(rest_batch.windows, 17)),
        "frequency_direction": FrequencyDirectionV1().fit(rest_batch),
    }
    values = {name: [] for name in families}
    for index, record in enumerate(records, 1):
        batch = FeatureBatch(grab.read_record(data_root, record), grab.RATE)
        for name, family in families.items():
            values[name].append(grab.aggregate(family.transform(batch)))
        if index % 100 == 0 or index == len(records):
            print(f"F8 GRAB extracted {index}/{len(records)} native recordings", flush=True)
    features = {name: np.stack(rows) for name, rows in values.items()}
    features[PROVIDERS[1]] = np.concatenate(
        (features[PROVIDERS[0]], features["frequency_direction"]), axis=1)
    session = np.asarray([row["session"] for row in records])
    subject = np.asarray([row["subject"] for row in records])
    labels = np.asarray([row["gesture"] for row in records])
    trial_number = np.asarray([row["trial"] for row in records])
    source = session == PROTOCOL["source_day"]
    predictions, scaled = {}, {}
    for name in PROVIDERS:
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, random_state=20260924))
        model.fit(features[name][source], labels[source])
        np.testing.assert_array_equal(model[-1].classes_, CLASSES)
        if np.max(model[-1].n_iter_) >= 2000:
            raise RuntimeError(f"source classifier failed to converge: {name}")
        predictions[name] = model.predict_proba(features[name])
        scaled[name] = model[0].transform(features[name])

    with PARENT.open(newline="", encoding="utf-8") as stream:
        frozen = {(row["arm"], row["trial_id"]): row for row in csv.DictReader(stream)
                  if row["arm"] in PROVIDERS}
    if len(frozen) != 896:
        raise ValueError("frozen provider row count changed")
    replay_error = 0.0
    for i, record in enumerate(records):
        if source[i]:
            continue
        for name in PROVIDERS:
            row = frozen.get((name, record["stem"]))
            if row is None or int(row["gesture"]) != labels[i] or int(row["subject"]) != subject[i]:
                raise AssertionError("frozen GRAB parent identity changed")
            p = np.asarray([float(row[f"p_{c}"]) for c in CLASSES])
            replay_error = max(replay_error, float(np.max(np.abs(p - predictions[name][i]))))
    if replay_error > 1e-8:
        raise AssertionError(f"frozen parent probability replay failed: {replay_error}")

    rows, blocks, scores = [], [], {}
    for phase, day in (("validation", 2), ("descriptive_final", 3)):
        scored = {"uniform": [], "f8_cosine": []}
        eval_indices = []
        for user in grab.SUBJECTS:
            user_source = np.flatnonzero(source & (subject == user))
            calibration = np.flatnonzero((session == day) & (subject == user) &
                                          (trial_number == 1))
            evaluation = np.flatnonzero((session == day) & (subject == user) &
                                         np.isin(trial_number, PROTOCOL["evaluation_trial_numbers_per_class"]))
            if len(user_source) != 28 or len(calibration) != 4 or len(evaluation) != 24:
                raise AssertionError("unexpected native source/calibration/evaluation split")
            if set(labels[calibration]) != set(CLASSES):
                raise AssertionError("calibration does not cover all classes")
            if set(calibration) & set(evaluation) or set(user_source) & (set(calibration) | set(evaluation)):
                raise AssertionError("trial leakage")
            signatures = {name: SessionSignature().fit_long_term(
                scaled[name][user_source], labels[user_source]) for name in PROVIDERS}
            weights, vectors = fixed_weights(signatures, scaled, labels, calibration)
            blocks.append({"phase": phase, "subject": user,
                           "source_trial_ids": [records[i]["stem"] for i in user_source],
                           "calibration_trial_ids": [records[i]["stem"] for i in calibration],
                           "evaluation_trial_ids": [records[i]["stem"] for i in evaluation],
                           "f8_weights": weights.tolist(), "f8_vectors": vectors})
            base = np.stack([predictions[name][evaluation] for name in PROVIDERS])
            for method, weight in (("uniform", np.asarray([0.5, 0.5])),
                                   ("f8_cosine", weights)):
                probability = np.tensordot(weight, base, axes=(0, 0))
                scored[method].append(probability)
                for i, p in zip(evaluation, probability):
                    rows.append({"phase": phase, "method": method,
                                 "trial_id": records[i]["stem"], "subject": user,
                                 "gesture": int(labels[i]), "calibration_shots_per_class": 1,
                                 **{f"p_{c}": float(v) for c, v in zip(CLASSES, p)}})
            eval_indices.extend(evaluation)
        eval_indices = np.asarray(eval_indices, dtype=int)
        if len(eval_indices) != 192 or len(set(eval_indices)) != 192:
            raise AssertionError("held-out trial count changed")
        scores[phase] = {}
        for method, parts in scored.items():
            p = np.concatenate(parts)
            scores[phase][method] = grab.score(labels[eval_indices], p, CLASSES,
                                                subject[eval_indices])
        print(f"F8 GRAB {phase}: uniform F1={scores[phase]['uniform']['macro_f1']:.4f}, "
              f"F8 F1={scores[phase]['f8_cosine']['macro_f1']:.4f}", flush=True)

    predictions_path = ROOT / "F8_GRAB_DAY_PREDICTIONS.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol_sha256": digest(PROTOCOL_PATH),
              "parent_prediction_sha256": digest(PARENT),
              "official_manifest_sha256": digest(data_root / "SHA256SUMS.txt"),
              "prediction_sha256": digest(predictions_path),
              "parent_probability_max_abs_error": replay_error,
              "native_prediction_rows": len(rows), "source_recordings": int(source.sum()),
              "calibration_recordings_per_day": 32, "evaluation_recordings_per_day": 192,
              "scores": scores, "blocks": blocks, "scope": PROTOCOL["scope"]}
    (ROOT / "F8_GRAB_DAY_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
