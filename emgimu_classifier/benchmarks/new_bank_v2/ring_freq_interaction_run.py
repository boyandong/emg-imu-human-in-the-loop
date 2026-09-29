"""Frozen four-arm ring-lag by frequency-direction test on three public splits."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.grabmyo_crossday import run as grab
from benchmarks.new_bank_v2.grab_user_run import check_files
from benchmarks.new_bank_v2.roam_posture_run import extract_archive, sha256
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v1 import FrequencyDirectionV1, RingLagV1
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "RING_FREQ_INTERACTION_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
ARMS = tuple(PROTOCOL["arms"])
NAMES = ("F0v2", "ring_lag", "frequency_direction")
PARENTS = {"roam_posture": "ROAM_V1_EXTENSION", "grab_user": "GRAB_V1_EXTENSION",
           "grab_day": "GRAB_DAY_V1_EXTENSION"}


def check_protocol() -> None:
    if ARMS != ("F0v2", "F0v2+ring_lag", "F0v2+frequency_direction",
                "F0v2+ring_lag+frequency_direction"):
        raise ValueError("interaction arms changed")
    for study, prefix in PARENTS.items():
        reference = PROTOCOL["parent_studies"][study]
        if (reference["protocol"] != f"{prefix}_PROTOCOL.json"
                or sha256(ROOT / reference["protocol"]) != reference["protocol_sha256"]
                or sha256(ROOT / f"{prefix}_RESULTS.json") != reference["results_sha256"]):
            raise ValueError(f"frozen parent changed: {study}")


def load_roam() -> tuple[dict, np.ndarray, list[dict], dict, int]:
    parent = json.loads((ROOT / "ROAM_V1_EXTENSION_PROTOCOL.json").read_text(encoding="utf-8"))
    archive = Path(parent["archive"])
    if sha256(archive) != parent["archive_sha256"]:
        raise ValueError("ROAM source archive changed")
    batch, y, identities, slices, _ = extract_archive(archive)
    subjects = np.asarray([row["subject"] for row in identities])
    postures = np.asarray([row["condition"] for row in identities])
    source = np.isin(subjects, parent["source_subjects"]) & (postures == "resting")
    rest = np.concatenate([batch.emg[a:b] for index, (a, b) in enumerate(slices)
                           if source[index] and y[index] == 0])
    rest_batch = FeatureBatch(rest, 200)
    if source.sum() != 162 or rest_batch.windows != 1828:
        raise ValueError("ROAM source count changed")
    families = {"F0v2": RestNoiseDetailV2(rest_label=0).fit(
                    rest_batch, np.zeros(rest_batch.windows, dtype=int)),
                "ring_lag": RingLagV1().fit(rest_batch),
                "frequency_direction": FrequencyDirectionV1().fit(rest_batch)}
    from benchmarks.grabmyo_crossday.run import aggregate
    vectors = {name: np.stack([aggregate(family.transform(batch)[a:b]) for a, b in slices])
               for name, family in families.items()}
    masks = {phase: np.isin(subjects, parent[f"{phase}_subjects"])
             for phase in ("validation", "final")}
    if any(mask.sum() != 180 or np.any(mask & source) for mask in masks.values()):
        raise ValueError("ROAM target count changed")
    return vectors, y, identities, {"source": source, **masks}, int(rest_batch.windows)


def load_grab(study: str) -> tuple[dict, np.ndarray, list[dict], dict, int]:
    prefix = PARENTS[study]
    parent = json.loads((ROOT / f"{prefix}_PROTOCOL.json").read_text(encoding="utf-8"))
    data_root = Path(parent["data_root"])
    records = list(grab.records())
    if study == "grab_user":
        records = [row for row in records if row["session"] == 1]
        if len(records) != 224:
            raise ValueError("GRAB user inventory changed")
        check_files(data_root, records)
        source_users = set(parent["source_subjects"])
        rest_rows = [row for row in records if row["subject"] in source_users and
                     row["gesture"] == 17]
        masks = {"source": np.asarray([row["subject"] in source_users for row in records]),
                 **{phase: np.asarray([row["subject"] in parent[f"{phase}_subjects"]
                                       for row in records]) for phase in ("validation", "final")}}
        expected_rest = 560
        target_count = 56
    else:
        if len(records) != 672:
            raise ValueError("GRAB day inventory changed")
        grab.check_all_files(data_root)
        rest_rows = [row for row in records if row["session"] == 1 and row["gesture"] == 17]
        masks = {"source": np.asarray([row["session"] == 1 for row in records]),
                 **{phase: np.asarray([row["session"] == parent["sessions"][phase]
                                       for row in records]) for phase in ("validation", "final")}}
        expected_rest = 1120
        target_count = 224
    rest_batch = FeatureBatch(np.concatenate([grab.read_record(data_root, row)
                                              for row in rest_rows]), grab.RATE)
    if rest_batch.windows != expected_rest:
        raise ValueError("GRAB source Rest count changed")
    families = {"F0v2": RestNoiseDetailV2(rest_label=17).fit(
                    rest_batch, np.full(rest_batch.windows, 17)),
                "ring_lag": RingLagV1().fit(rest_batch),
                "frequency_direction": FrequencyDirectionV1().fit(rest_batch)}
    vectors = {name: [] for name in NAMES}
    for index, record in enumerate(records, 1):
        batch = FeatureBatch(grab.read_record(data_root, record), grab.RATE)
        for name, family in families.items():
            vectors[name].append(grab.aggregate(family.transform(batch)))
        if index % 200 == 0 or index == len(records):
            print(f"{study} interaction extracted {index}/{len(records)}", flush=True)
    vectors = {name: np.stack(values) for name, values in vectors.items()}
    if any(mask.sum() != target_count or np.any(mask & masks["source"])
           for phase, mask in masks.items() if phase != "source"):
        raise ValueError("GRAB target count changed")
    return vectors, np.asarray([row["gesture"] for row in records]), records, masks, rest_batch.windows


def replay_parent(study: str, rows: list[dict], classes: np.ndarray) -> float:
    prefix = PARENTS[study]
    with (ROOT / f"{prefix}_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        frozen = {(r["arm"], r["phase"], r["trial_id"]): r for r in csv.DictReader(stream)
                  if r["arm"] in ARMS[:3]}
    current = [r for r in rows if r["study"] == study and r["arm"] in ARMS[:3]]
    if len(current) != len(frozen):
        raise AssertionError(f"{study} parent single-arm count changed")
    maximum = 0.0
    for row in current:
        prior = frozen[(row["arm"], row["phase"], row["trial_id"])]
        if row["label"] != int(prior["label" if study == "roam_posture" else "gesture"]):
            raise AssertionError("parent truth changed")
        a = np.asarray([row[f"p_{c}"] for c in classes])
        b = np.asarray([float(prior[f"p_{c}"]) for c in classes])
        maximum = max(maximum, float(np.max(np.abs(a - b))))
    if maximum > 1e-8:
        raise AssertionError(f"{study} parent probability drift: {maximum}")
    return maximum


def evaluate() -> dict:
    check_protocol()
    all_rows = []
    studies = {}
    for study in PARENTS:
        vectors, y, identities, masks, rest_count = (load_roam() if study == "roam_posture"
                                                     else load_grab(study))
        classes = np.asarray([0, 1, 2] if study == "roam_posture" else grab.GESTURES)
        subjects = np.asarray([row["subject"] for row in identities])
        scores = {}
        for arm in ARMS:
            x = np.concatenate([vectors[name] for name in arm.split("+")], axis=1)
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, class_weight="balanced" if study == "roam_posture" else None,
                max_iter=2000, random_state=20260929 if study == "roam_posture" else 20260924))
            model.fit(x[masks["source"]], y[masks["source"]])
            np.testing.assert_array_equal(model[-1].classes_, classes)
            if np.max(model[-1].n_iter_) >= 2000:
                raise RuntimeError(f"interaction model did not converge: {study}/{arm}")
            scores[arm] = {}
            for phase in ("validation", "final"):
                mask = masks[phase]
                probability = model.predict_proba(x[mask])
                scores[arm][phase] = grab.score(y[mask], probability, classes, subjects[mask])
                for identity, truth, values in zip(np.asarray(identities, dtype=object)[mask],
                                                   y[mask], probability):
                    trial_id = identity["trial_id"] if study == "roam_posture" else identity["stem"]
                    all_rows.append({"study": study, "arm": arm, "phase": phase,
                                     "trial_id": trial_id, "subject": identity["subject"],
                                     "condition": identity["condition"] if study == "roam_posture" else "ALL",
                                     "label": int(truth),
                                     **{f"p_{c}": float(v) for c, v in zip(classes, values)},
                                     **{f"p_{c}": "" for c in ({0, 1, 2, 4, 15, 16, 17} - set(classes))}})
            print(f"{study}/{arm}: validation F1={scores[arm]['validation']['macro_f1']:.4f}, "
                  f"final F1={scores[arm]['final']['macro_f1']:.4f}", flush=True)
        maximum = replay_parent(study, all_rows, classes)
        studies[study] = {"source_rest_windows": int(rest_count), "scores": scores,
                          "parent_maximum_probability_difference": maximum,
                          "source_trials": int(masks["source"].sum()),
                          "validation_trials": int(masks["validation"].sum()),
                          "final_trials": int(masks["final"].sum())}
    path = ROOT / "RING_FREQ_INTERACTION_PREDICTIONS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    result = {"protocol": PROTOCOL, "protocol_sha256": sha256(PROTOCOL_PATH),
              "prediction_sha256": sha256(path), "prediction_rows": len(all_rows),
              "studies": studies, "scope": PROTOCOL["scope"]}
    (ROOT / "RING_FREQ_INTERACTION_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    evaluate()
