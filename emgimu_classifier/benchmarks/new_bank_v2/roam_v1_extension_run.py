"""Source-locked ROAM screening of four independent new-v1 family formulas."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.grabmyo_crossday.run import aggregate, score
from benchmarks.new_bank_v2.roam_posture_run import extract_archive, sha256
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v1 import NEW_BANK_V1
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "ROAM_V1_EXTENSION_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
CLASSES = np.asarray([0, 1, 2])
ARMS = tuple(PROTOCOL["arms"])


def check_protocol() -> None:
    parent = json.loads((ROOT / "ROAM_POSTURE_PROTOCOL.json").read_text(encoding="utf-8"))
    if (sha256(ROOT / "ROAM_POSTURE_PROTOCOL.json") != PROTOCOL["parent_posture_protocol_sha256"]
            or sha256(ROOT / "ROAM_POSTURE_PREDICTIONS.csv") != PROTOCOL["parent_posture_prediction_sha256"]
            or PROTOCOL["source_subjects"] != parent["source_subjects"]
            or PROTOCOL["validation_subjects"] != parent["validation_subjects"]
            or PROTOCOL["final_subjects"] != parent["final_subjects"]
            or PROTOCOL["target_postures"] != parent["target_postures"]
            or PROTOCOL["source_posture"] != parent["source_posture"]
            or PROTOCOL["sample_rate_hz"] != 200 or PROTOCOL["window_samples"] != 40
            or PROTOCOL["candidate_families"] != list(NEW_BANK_V1)
            or ARMS != ("F0v2", *[f"F0v2+{name}" for name in NEW_BANK_V1])):
        raise ValueError("frozen ROAM new-v1 extension protocol changed")


def replay_base(rows: list[dict]) -> None:
    with (ROOT / "ROAM_POSTURE_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        original = {(r["phase"], r["trial_id"]): r for r in csv.DictReader(stream)
                    if r["arm"] == "F0v2"}
    baseline = [r for r in rows if r["arm"] == "F0v2"]
    if len(original) != 360 or len(baseline) != 360:
        raise AssertionError("frozen ROAM baseline count changed")
    for row in baseline:
        prior = original.get((row["phase"], row["trial_id"]))
        if prior is None or int(prior["label"]) != row["label"]:
            raise AssertionError("frozen ROAM baseline identity changed")
        np.testing.assert_allclose(
            [row[f"p_{c}"] for c in CLASSES],
            [float(prior[f"p_{c}"]) for c in CLASSES], rtol=0, atol=1e-10)


def evaluate() -> dict:
    check_protocol()
    archive = Path(PROTOCOL["archive"])
    if sha256(archive).lower() != PROTOCOL["archive_sha256"].lower():
        raise ValueError("ROAM archive hash mismatch")
    batch, y, identities, slices, counts = extract_archive(archive)
    subjects = np.asarray([row["subject"] for row in identities])
    postures = np.asarray([row["condition"] for row in identities])
    source = np.isin(subjects, PROTOCOL["source_subjects"]) & (postures == "resting")
    rest_windows = np.concatenate([batch.emg[a:b] for i, (a, b) in enumerate(slices)
                                   if source[i] and y[i] == 0])
    rest_batch = FeatureBatch(rest_windows, 200)
    if source.sum() != 162 or rest_batch.windows != 1828 or len(counts) != 112:
        raise ValueError("source and native inventory contract changed")
    families = {"F0v2": RestNoiseDetailV2(rest_label=0).fit(
                    rest_batch, np.zeros(rest_batch.windows, dtype=int)),
                **{name: constructor().fit(rest_batch)
                   for name, constructor in NEW_BANK_V1.items()}}
    vectors = {}
    for name, family in families.items():
        transformed = family.transform(batch)
        vectors[name] = np.stack([aggregate(transformed[a:b]) for a, b in slices])
        print(f"ROAM extended family {name}: {vectors[name].shape[1]} bout features", flush=True)
    rows, scores = [], {}
    for arm in ARMS:
        x = np.concatenate([vectors[name] for name in arm.split("+")], axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=20260929))
        model.fit(x[source], y[source])
        np.testing.assert_array_equal(model[-1].classes_, CLASSES)
        if np.max(model[-1].n_iter_) >= 2000:
            raise RuntimeError(f"source classifier did not converge: {arm}")
        scores[arm] = {"source_bouts": int(source.sum()), "feature_dimensions": x.shape[1]}
        for phase in ("validation", "final"):
            mask = np.isin(subjects, PROTOCOL[f"{phase}_subjects"])
            if mask.sum() != 180 or np.any(mask & source):
                raise ValueError("target subject split changed")
            probability = model.predict_proba(x[mask])
            by_posture = {posture: score(y[mask & (postures == posture)],
                                         model.predict_proba(x[mask & (postures == posture)]),
                                         CLASSES, subjects[mask & (postures == posture)])
                          for posture in PROTOCOL["target_postures"]}
            scores[arm][phase] = {**score(y[mask], probability, CLASSES, subjects[mask]),
                                  "bouts": int(mask.sum()), "by_posture": by_posture,
                                  "minimum_posture_macro_f1": min(
                                      item["macro_f1"] for item in by_posture.values())}
            for identity, p in zip(np.asarray(identities, dtype=object)[mask], probability):
                rows.append({"arm": arm, "phase": phase, "trial_id": identity["trial_id"],
                             "subject": identity["subject"], "condition": identity["condition"],
                             "label": identity["label"],
                             **{f"p_{c}": float(value) for c, value in zip(CLASSES, p)}})
        print(f"{arm}: validation F1={scores[arm]['validation']['macro_f1']:.4f}, "
              f"final F1={scores[arm]['final']['macro_f1']:.4f}", flush=True)
    replay_base(rows)
    path = ROOT / "ROAM_V1_EXTENSION_PREDICTIONS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol": PROTOCOL, "protocol_sha256": sha256(PROTOCOL_PATH),
              "archive_sha256": sha256(archive), "prediction_sha256": sha256(path),
              "source_rest_windows": int(rest_batch.windows),
              "feature_dimensions": {name: int(x.shape[1]) for name, x in vectors.items()},
              "baseline_replay": "all 360 held-out F0v2 vectors within absolute 1e-10",
              "source_trial_ids": [row["trial_id"] for i, row in enumerate(identities) if source[i]],
              "validation_trial_ids": [row["trial_id"] for row in identities
                                       if row["subject"] in PROTOCOL["validation_subjects"]],
              "final_trial_ids": [row["trial_id"] for row in identities
                                  if row["subject"] in PROTOCOL["final_subjects"]],
              "scores": scores, "scope": PROTOCOL["scope"]}
    (ROOT / "ROAM_V1_EXTENSION_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    evaluate()
