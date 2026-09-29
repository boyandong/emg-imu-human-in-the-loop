"""Source-locked cross-day screen of four independently defined new-v1 families."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.grabmyo_crossday import run as grab
from benchmarks.new_bank_v2.roam_posture_run import sha256
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v1 import NEW_BANK_V1
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "GRAB_DAY_V1_EXTENSION_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
CLASSES = np.asarray(grab.GESTURES)
ARMS = tuple(PROTOCOL["arms"])


def check_protocol() -> None:
    parent = json.loads((ROOT / "GRABMYO_PROTOCOL.json").read_text(encoding="utf-8"))
    if (sha256(ROOT / "GRABMYO_PROTOCOL.json") != PROTOCOL["parent_day_protocol_sha256"]
            or sha256(ROOT / "GRABMYO_TRIAL_PREDICTIONS.csv") != PROTOCOL["parent_day_prediction_sha256"]
            or PROTOCOL["subjects"] != parent["subjects"]
            or PROTOCOL["sessions"] != parent["sessions"]
            or PROTOCOL["channels"] != parent["channels"]
            or PROTOCOL["sample_rate_hz"] != parent["sample_rate_hz"]
            or PROTOCOL["window_samples"] != parent["window_samples"]
            or PROTOCOL["gesture_codes"] != parent["gesture_codes"]
            or PROTOCOL["rest_code"] != 17
            or PROTOCOL["candidate_families"] != list(NEW_BANK_V1)
            or ARMS != ("F0v2", *[f"F0v2+{name}" for name in NEW_BANK_V1])):
        raise ValueError("frozen GRAB cross-day extension protocol changed")


def replay_base(rows: list[dict]) -> None:
    with (ROOT / "GRABMYO_TRIAL_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        original = {(row["split"], row["trial_id"]): row for row in csv.DictReader(stream)
                    if row["arm"] == "F0"}
    baseline = [row for row in rows if row["arm"] == "F0v2"]
    if len(original) != 448 or len(baseline) != 448:
        raise AssertionError("frozen GRAB day baseline count changed")
    for row in baseline:
        prior = original.get((row["phase"], row["trial_id"]))
        if prior is None or int(prior["gesture"]) != row["gesture"]:
            raise AssertionError("frozen GRAB day baseline identity changed")
        np.testing.assert_allclose([row[f"p_{c}"] for c in CLASSES],
                                   [float(prior[f"p_{c}"]) for c in CLASSES],
                                   rtol=0, atol=1e-10)


def evaluate() -> dict:
    check_protocol()
    data_root = Path(PROTOCOL["data_root"])
    grab.check_all_files(data_root)
    records = list(grab.records())
    if len(records) != 672:
        raise ValueError("native recording count changed")
    rest_windows = np.concatenate([grab.read_record(data_root, row) for row in records
                                   if row["session"] == 1 and row["gesture"] == 17])
    rest_batch = FeatureBatch(rest_windows, grab.RATE)
    if rest_batch.windows != 1120:
        raise ValueError("source Rest window count changed")
    families = {"F0v2": RestNoiseDetailV2(rest_label=17).fit(
                    rest_batch, np.full(rest_batch.windows, 17)),
                **{name: constructor().fit(rest_batch)
                   for name, constructor in NEW_BANK_V1.items()}}
    vectors = {name: [] for name in families}
    for index, record in enumerate(records, 1):
        batch = FeatureBatch(grab.read_record(data_root, record), grab.RATE)
        for name, family in families.items():
            vectors[name].append(grab.aggregate(family.transform(batch)))
        if index % 100 == 0 or index == len(records):
            print(f"GRAB day extension extracted {index}/{len(records)} recordings", flush=True)
    vectors = {name: np.stack(values) for name, values in vectors.items()}
    sessions = np.asarray([row["session"] for row in records])
    subjects = np.asarray([row["subject"] for row in records])
    y = np.asarray([row["gesture"] for row in records])
    if any((sessions == day).sum() != 224 for day in (1, 2, 3)):
        raise ValueError("day partition counts changed")
    source = sessions == 1
    rows, scores = [], {}
    for arm in ARMS:
        x = np.concatenate([vectors[name] for name in arm.split("+")], axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, random_state=20260924))
        model.fit(x[source], y[source])
        np.testing.assert_array_equal(model[-1].classes_, CLASSES)
        if np.max(model[-1].n_iter_) >= 2000:
            raise RuntimeError(f"source classifier did not converge: {arm}")
        scores[arm] = {"source_trials": int(source.sum()), "feature_dimensions": int(x.shape[1])}
        for phase in ("validation", "final"):
            mask = sessions == PROTOCOL["sessions"][phase]
            probability = model.predict_proba(x[mask])
            scores[arm][phase] = {"trials": int(mask.sum()),
                                  **grab.score(y[mask], probability, CLASSES, subjects[mask])}
            for identity, p in zip(np.asarray(records, dtype=object)[mask], probability):
                rows.append({"arm": arm, "phase": phase, "trial_id": identity["stem"],
                             "subject": identity["subject"], "gesture": identity["gesture"],
                             **{f"p_{c}": float(value) for c, value in zip(CLASSES, p)}})
        print(f"{arm}: Day2 F1={scores[arm]['validation']['macro_f1']:.4f}, "
              f"Day3 F1={scores[arm]['final']['macro_f1']:.4f}", flush=True)
    replay_base(rows)
    selected = min(ARMS, key=lambda name: (-scores[name]["validation"]["macro_f1"],
                                            scores[name]["validation"]["log_loss"]))
    path = ROOT / "GRAB_DAY_V1_EXTENSION_PREDICTIONS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol": PROTOCOL, "protocol_sha256": sha256(PROTOCOL_PATH),
              "official_sha256_manifest_sha256": sha256(data_root / "SHA256SUMS.txt"),
              "prediction_sha256": sha256(path), "source_rest_windows": int(rest_batch.windows),
              "feature_dimensions": {name: int(x.shape[1]) for name, x in vectors.items()},
              "baseline_replay": "all 448 held-out F0v2 vectors within absolute 1e-10",
              "source_trial_ids": [row["stem"] for row in records if row["session"] == 1],
              "validation_trial_ids": [row["stem"] for row in records if row["session"] == 2],
              "final_trial_ids": [row["stem"] for row in records if row["session"] == 3],
              "scores": scores, "validation_selected_arm": selected, "scope": PROTOCOL["scope"]}
    (ROOT / "GRAB_DAY_V1_EXTENSION_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    evaluate()
