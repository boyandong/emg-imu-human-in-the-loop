"""Source-frozen new-v1 feature-space personal anchors on GRABMyo days."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.grabmyo_crossday import run as grab
from benchmarks.new_bank_v2.grab_score_calibration import csv_text, score
from benchmarks.new_bank_v2.roam_posture_run import sha256
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v1 import NEW_BANK_V1
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "V1_FEATURE_ANCHOR_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
CLASSES = tuple(PROTOCOL["classes"])
ARMS = tuple(PROTOCOL["arms"])
PREFIX = "GRAB_DAY_V1_EXTENSION"


def check_protocol() -> None:
    for suffix, key in (("PROTOCOL.json", "parent_protocol_sha256"),
                        ("RESULTS.json", "parent_result_sha256"),
                        ("PREDICTIONS.csv", "parent_prediction_sha256")):
        if sha256(ROOT / f"{PREFIX}_{suffix}") != PROTOCOL[key]:
            raise AssertionError(f"frozen parent changed: {suffix}")
    parent = json.loads((ROOT / f"{PREFIX}_PROTOCOL.json").read_text(encoding="utf-8"))
    if (tuple(PROTOCOL["subjects"]) != tuple(parent["subjects"])
            or tuple(PROTOCOL["classes"]) != tuple(parent["gesture_codes"])
            or tuple(PROTOCOL["arms"]) != tuple(parent["arms"])
            or PROTOCOL["target_days"] != {"validation": 2, "final": 3}
            or PROTOCOL["source_day"] != 1 or PROTOCOL["budgets"] != [0, 1, 2, 5]
            or PROTOCOL["evaluation_repetitions"] != [6, 7]
            or tuple(NEW_BANK_V1) != tuple(parent["candidate_families"])):
        raise AssertionError("personal feature-space protocol disagrees with frozen parent")


def frozen_probabilities() -> dict:
    with (ROOT / f"{PREFIX}_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    frozen = {}
    for row in rows:
        key = (row["arm"], row["trial_id"])
        if key in frozen:
            raise AssertionError("duplicate frozen parent trial")
        frozen[key] = np.asarray([float(row[f"p_{c}"]) for c in CLASSES])
    if len(frozen) != 2240:
        raise AssertionError("frozen target prediction count changed")
    return frozen


def anchor_probability(query: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    dimensions = query.size
    squared = np.sum((prototypes - query[None, :]) ** 2, axis=1) / dimensions
    pairwise = np.asarray([np.sum((prototypes[a] - prototypes[b]) ** 2) / dimensions
                           for a in range(len(CLASSES)) for b in range(a + 1, len(CLASSES))])
    positive = pairwise[pairwise > 1e-12]
    temperature = float(np.median(positive)) if positive.size else 1.0
    logits = -squared / max(temperature, 1e-12)
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def evaluate() -> dict:
    check_protocol()
    parent = json.loads((ROOT / f"{PREFIX}_PROTOCOL.json").read_text(encoding="utf-8"))
    data_root = Path(parent["data_root"])
    grab.check_all_files(data_root)
    records = list(grab.records())
    if len(records) != 672:
        raise AssertionError("native recording set changed")
    source_rest = np.concatenate([grab.read_record(data_root, row) for row in records
                                  if row["session"] == 1 and row["gesture"] == 17])
    rest_batch = FeatureBatch(source_rest, grab.RATE)
    if rest_batch.windows != 1120:
        raise AssertionError("source Rest windows changed")
    families = {"F0v2": RestNoiseDetailV2(rest_label=17).fit(
        rest_batch, np.full(rest_batch.windows, 17)),
        **{name: constructor().fit(rest_batch) for name, constructor in NEW_BANK_V1.items()}}
    vectors = {name: [] for name in families}
    for index, record in enumerate(records, 1):
        batch = FeatureBatch(grab.read_record(data_root, record), grab.RATE)
        for name, family in families.items():
            vectors[name].append(grab.aggregate(family.transform(batch)))
        if index % 100 == 0 or index == len(records):
            print(f"feature-anchor extracted {index}/{len(records)} recordings", flush=True)
    vectors = {name: np.stack(values) for name, values in vectors.items()}
    session = np.asarray([row["session"] for row in records])
    y = np.asarray([row["gesture"] for row in records])
    source = session == 1
    target_indices = np.flatnonzero(~source)
    target_ids = [records[i]["stem"] for i in target_indices]
    if len(target_indices) != 448 or len(set(target_ids)) != 448:
        raise AssertionError("target recording inventory changed")
    record_index = {(row["session"], row["subject"], row["gesture"], row["trial"]): index
                    for index, row in enumerate(records)}
    if len(record_index) != 672:
        raise AssertionError("duplicate native recording identity")
    frozen = frozen_probabilities()
    predictions, curve, assignments, feature_hashes = [], [], [], {}
    source_replay_errors = {}
    dimensions = {}
    for arm in ARMS:
        x = np.concatenate([vectors[name] for name in arm.split("+")], axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, random_state=20260924))
        model.fit(x[source], y[source])
        np.testing.assert_array_equal(model[-1].classes_, CLASSES)
        if np.max(model[-1].n_iter_) >= 2000:
            raise AssertionError("source logistic classifier failed to converge")
        standardized = model[0].transform(x)
        target_features = standardized[target_indices]
        feature_path = ROOT / f"V1_FEATURE_ANCHOR_{arm.replace('+', '_')}.npy"
        np.save(feature_path, target_features, allow_pickle=False)
        feature_hashes[arm] = sha256(feature_path)
        dimensions[arm] = int(x.shape[1])
        recomputed = model.predict_proba(x[target_indices])
        source_replay_errors[arm] = max(float(np.max(np.abs(p - frozen[(arm, trial_id)])))
                                        for p, trial_id in zip(recomputed, target_ids))
        if source_replay_errors[arm] > 1e-8:
            raise AssertionError(f"source model prediction drift: {arm} {source_replay_errors[arm]}")
        for phase, day in PROTOCOL["target_days"].items():
            for budget in PROTOCOL["budgets"]:
                phase_rows = []
                for subject in PROTOCOL["subjects"]:
                    calibration_ids = []
                    if budget:
                        prototypes = np.stack([np.mean([standardized[record_index[(day, subject, label, rep)]]
                                                        for rep in range(1, budget + 1)], axis=0)
                                               for label in CLASSES])
                        calibration_ids = [records[record_index[(day, subject, label, rep)]]["stem"]
                                           for label in CLASSES for rep in range(1, budget + 1)]
                    evaluation_ids = []
                    for label in CLASSES:
                        for rep in PROTOCOL["evaluation_repetitions"]:
                            index = record_index[(day, subject, label, rep)]
                            trial_id = records[index]["stem"]
                            evaluation_ids.append(trial_id)
                            p0 = frozen[(arm, trial_id)]
                            p = p0 if budget == 0 else (
                                0.5 * p0 + 0.5 * anchor_probability(standardized[index], prototypes))
                            if not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-12):
                                raise AssertionError("invalid feature-anchor probability")
                            phase_rows.append({"arm": arm, "phase": phase, "subject": subject,
                                               "shots_per_class": budget, "trial_id": trial_id,
                                               "label": label, "repetition": rep,
                                               **{f"p_{c}": float(value) for c, value in zip(CLASSES, p)}})
                    if set(calibration_ids) & set(evaluation_ids):
                        raise AssertionError("feature-anchor evaluation leakage")
                    assignments.append({"arm": arm, "phase": phase, "subject": subject,
                                        "shots_per_class": budget,
                                        "calibration_ids": calibration_ids,
                                        "evaluation_ids": evaluation_ids})
                predictions.extend(phase_rows)
                for group_subject in ("ALL", *PROTOCOL["subjects"]):
                    subset = phase_rows if group_subject == "ALL" else [
                        row for row in phase_rows if row["subject"] == group_subject]
                    metrics = score(subset)
                    curve.append({"arm": arm, "phase": phase, "subject": group_subject,
                                  "shots_per_class": budget,
                                  "signal_seconds": budget * len(CLASSES) * PROTOCOL["recording_seconds"],
                                  **{k: (json.dumps(v, sort_keys=True) if k == "per_class_f1" else v)
                                     for k, v in metrics.items()}})
            print(f"feature-anchor {arm}/{phase}: four budgets", flush=True)
    if len(predictions) != 2560 or len(curve) != 360 or len(assignments) != 320:
        raise AssertionError("feature-anchor output coverage changed")
    prediction_path = ROOT / "V1_FEATURE_ANCHOR_PREDICTIONS.csv"
    curve_path = ROOT / "V1_FEATURE_ANCHOR_CURVE.csv"
    prediction_path.write_text(csv_text(predictions), encoding="utf-8", newline="")
    curve_path.write_text(csv_text(curve), encoding="utf-8", newline="")
    result = {"status": "ok", "protocol_sha256": sha256(PROTOCOL_PATH),
              "parent_prediction_sha256": sha256(ROOT / f"{PREFIX}_PREDICTIONS.csv"),
              "official_manifest_sha256": sha256(data_root / "SHA256SUMS.txt"),
              "prediction_sha256": sha256(prediction_path), "curve_sha256": sha256(curve_path),
              "feature_sha256": feature_hashes, "target_trial_ids": target_ids,
              "source_replay_max_abs_error": source_replay_errors,
              "feature_dimensions": dimensions, "prediction_rows": len(predictions),
              "curve_rows": len(curve), "assignments": assignments,
              "scope": PROTOCOL["scope"]}
    (ROOT / "V1_FEATURE_ANCHOR_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"feature-anchor wrote {len(predictions)} predictions and {len(curve)} score cells", flush=True)
    return result


if __name__ == "__main__":
    evaluate()
