"""Source-subject OOF temperature fitting for five frozen new-family arms."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.grabmyo_crossday import run as grab
from benchmarks.new_bank_v2.family_screen import _score
from benchmarks.new_bank_v2.grab_user_run import check_files
from benchmarks.new_bank_v2.roam_posture_run import extract_archive, sha256
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v1 import NEW_BANK_V1
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "V1_SOURCE_OOF_CAL_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
STUDIES = {"roam_posture": "ROAM_V1_EXTENSION", "grab_user": "GRAB_V1_EXTENSION",
           "grab_day": "GRAB_DAY_V1_EXTENSION"}
CLASSES = {"roam_posture": np.asarray([0, 1, 2]),
           "grab_user": np.asarray([4, 15, 16, 17]),
           "grab_day": np.asarray([4, 15, 16, 17])}
ARMS = tuple(PROTOCOL["candidate_arms"])
TEMPERATURES = np.unique(np.r_[1.0, np.geomspace(0.5, 5.0, 61)])
ALL_CLASSES = (0, 1, 2, 4, 15, 16, 17)


def check_protocol() -> None:
    if ARMS != ("F0v2", *[f"F0v2+{name}" for name in NEW_BANK_V1]):
        raise ValueError("five-arm calibration protocol changed")
    if tuple(PROTOCOL["parent_studies"]) != tuple(STUDIES.values()):
        raise ValueError("parent study list changed")
    for prefix in STUDIES.values():
        for key, suffix in (("protocol", "PROTOCOL.json"),
                            ("results", "RESULTS.json"),
                            ("predictions", "PREDICTIONS.csv")):
            if sha256(ROOT / f"{prefix}_{suffix}") != PROTOCOL["parent_sha256"][prefix][key]:
                raise ValueError(f"frozen source changed: {prefix}/{key}")


def source_trials(study: str) -> tuple[list[FeatureBatch], np.ndarray, np.ndarray, list[str], int]:
    prefix = STUDIES[study]
    parent = json.loads((ROOT / f"{prefix}_PROTOCOL.json").read_text(encoding="utf-8"))
    if study == "roam_posture":
        archive = Path(parent["archive"])
        if sha256(archive) != parent["archive_sha256"]:
            raise ValueError("ROAM source archive changed")
        batch, labels, identities, slices, _ = extract_archive(archive)
        mask = np.asarray([r["subject"] in parent["source_subjects"] and
                           r["condition"] == "resting" for r in identities])
        selected = np.flatnonzero(mask)
        raw = [FeatureBatch(batch.emg[slices[i][0]:slices[i][1]], 200) for i in selected]
        y = labels[mask]
        subjects = np.asarray([identities[i]["subject"] for i in selected])
        ids = [identities[i]["trial_id"] for i in selected]
        rest_label = 0
    else:
        data_root = Path(parent["data_root"])
        records = [r for r in grab.records() if r["session"] == 1]
        if study == "grab_user":
            records = [r for r in records if r["subject"] in parent["source_subjects"]]
            check_files(data_root, records)
        else:
            grab.check_all_files(data_root)
        raw = [FeatureBatch(grab.read_record(data_root, r), grab.RATE) for r in records]
        y = np.asarray([r["gesture"] for r in records])
        subjects = np.asarray([r["subject"] for r in records])
        ids = [r["stem"] for r in records]
        rest_label = 17
    result = json.loads((ROOT / f"{prefix}_RESULTS.json").read_text(encoding="utf-8"))
    if set(ids) != set(result["source_trial_ids"]):
        raise AssertionError(f"{study} source native trial identity changed")
    expected = {"roam_posture": 162, "grab_user": 112, "grab_day": 224}[study]
    if len(ids) != expected:
        raise AssertionError(f"{study} source count changed")
    return raw, y, subjects, ids, rest_label


def temperature_transform(probability: np.ndarray, value: float) -> np.ndarray:
    if value == 1.0:
        return probability.copy()
    logits = np.log(np.clip(probability, 1e-12, 1.0)) / value
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def nll(y: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> float:
    index = np.searchsorted(classes, y)
    return float(np.mean(-np.log(np.clip(probability[np.arange(len(y)), index], 1e-12, 1.0))))


def score_rows(y: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> dict:
    rows = [{"label": str(label), **{f"p_{c}": float(p)
            for c, p in zip(classes, values)}} for label, values in zip(y, probability)]
    return _score(rows, [str(c) for c in classes])


def prediction_row(study: str, partition: str, arm: str, trial_id: str,
                   subject: int, condition: str, label: int, temperature: float,
                   raw: np.ndarray, calibrated: np.ndarray, classes: np.ndarray) -> dict:
    values = {"study": study, "partition": partition, "arm": arm,
              "trial_id": trial_id, "subject": subject, "condition": condition,
              "label": int(label), "temperature": float(temperature)}
    for c in ALL_CLASSES:
        if c in classes:
            index = int(np.where(classes == c)[0][0])
            values[f"raw_p_{c}"] = float(raw[index])
            values[f"cal_p_{c}"] = float(calibrated[index])
        else:
            values[f"raw_p_{c}"] = ""
            values[f"cal_p_{c}"] = ""
    return values


def evaluate() -> dict:
    check_protocol()
    predictions = []
    results = {}
    for study, prefix in STUDIES.items():
        raw, y, subjects, ids, rest_label = source_trials(study)
        classes = CLASSES[study]
        rate = raw[0].sample_rate_hz
        samples = raw[0].emg.shape[1]
        metadata = FeatureBatch(np.zeros((1, samples, 8), dtype=np.float32), rate)
        candidate = {name: constructor().fit(metadata) for name, constructor in NEW_BANK_V1.items()}
        vectors = {name: np.stack([grab.aggregate(family.transform(batch)) for batch in raw])
                   for name, family in candidate.items()}
        oof = {arm: np.full((len(y), len(classes)), np.nan) for arm in ARMS}
        fold_audit = []
        for held_subject in sorted(set(subjects)):
            train = subjects != held_subject
            held = ~train
            rest = np.concatenate([batch.emg for i, batch in enumerate(raw)
                                   if train[i] and y[i] == rest_label])
            rest_batch = FeatureBatch(rest, rate)
            family = RestNoiseDetailV2(rest_label=rest_label).fit(
                rest_batch, np.full(rest_batch.windows, rest_label))
            f0 = np.stack([grab.aggregate(family.transform(batch)) for batch in raw])
            for arm in ARMS:
                blocks = [f0 if name == "F0v2" else vectors[name] for name in arm.split("+")]
                x = np.concatenate(blocks, axis=1)
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=1.0, class_weight="balanced" if study == "roam_posture" else None,
                    max_iter=2000, random_state=20260929 if study == "roam_posture" else 20260924))
                model.fit(x[train], y[train])
                np.testing.assert_array_equal(model[-1].classes_, classes)
                if np.max(model[-1].n_iter_) >= 2000:
                    raise RuntimeError(f"source OOF model failed to converge: {study}/{arm}/{held_subject}")
                oof[arm][held] = model.predict_proba(x[held])
            fold_audit.append({"held_subject": int(held_subject),
                               "train_trials": int(train.sum()), "held_trials": int(held.sum()),
                               "train_rest_windows": int(rest_batch.windows),
                               "train_subjects": sorted(set(map(int, subjects[train]))),
                               "held_trial_ids": [ids[i] for i in np.flatnonzero(held)]})
            print(f"{study} source OOF subject {held_subject}: {held.sum()} held trials", flush=True)
        with (ROOT / f"{prefix}_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
            frozen = list(csv.DictReader(stream))
        results[study] = {"source_trials": len(y), "source_subjects": sorted(set(map(int, subjects))),
                          "source_folds": fold_audit, "arms": {}}
        for arm in ARMS:
            if not np.isfinite(oof[arm]).all() or np.any(oof[arm] <= 0):
                raise AssertionError(f"incomplete source OOF probabilities: {study}/{arm}")
            losses = [(nll(y, temperature_transform(oof[arm], float(t)), classes),
                       abs(float(np.log(t))), float(t)) for t in TEMPERATURES]
            chosen = min(losses)[2]
            source_cal = temperature_transform(oof[arm], chosen)
            arm_result = {"temperature": chosen,
                          "source_oof_raw": score_rows(y, oof[arm], classes),
                          "source_oof_calibrated": score_rows(y, source_cal, classes),
                          "target": {}}
            for identity, subject, label, p0, p1 in zip(ids, subjects, y, oof[arm], source_cal):
                predictions.append(prediction_row(study, "source_oof", arm, identity,
                                                  int(subject), "source", int(label), chosen,
                                                  p0, p1, classes))
            for phase in ("validation", "final"):
                target = [r for r in frozen if r["arm"] == arm and r["phase"] == phase]
                parent = json.loads((ROOT / f"{prefix}_RESULTS.json").read_text(encoding="utf-8"))
                if {r["trial_id"] for r in target} != set(parent[f"{phase}_trial_ids"]):
                    raise AssertionError(f"frozen target IDs changed: {study}/{arm}/{phase}")
                target_y = np.asarray([int(r["label" if study == "roam_posture" else "gesture"])
                                       for r in target])
                target_raw = np.asarray([[float(r[f"p_{c}"]) for c in classes] for r in target])
                target_cal = temperature_transform(target_raw, chosen)
                raw_score, cal_score = (score_rows(target_y, p, classes)
                                        for p in (target_raw, target_cal))
                if raw_score["macro_f1"] != cal_score["macro_f1"]:
                    raise AssertionError("temperature unexpectedly changed target decisions")
                arm_result["target"][phase] = {"raw": raw_score, "calibrated": cal_score}
                for r, truth, p0, p1 in zip(target, target_y, target_raw, target_cal):
                    predictions.append(prediction_row(study, phase, arm, r["trial_id"],
                        int(r["subject"]), r["condition"] if study == "roam_posture" else study,
                        int(truth), chosen, p0, p1, classes))
            results[study]["arms"][arm] = arm_result
            print(f"{study}/{arm}: T={chosen:.3f}; validation ΔLL="
                  f"{arm_result['target']['validation']['raw']['log_loss'] - arm_result['target']['validation']['calibrated']['log_loss']:+.4f}",
                  flush=True)
    path = ROOT / "V1_SOURCE_OOF_CAL_PREDICTIONS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    if len(predictions) != 7090:
        raise AssertionError("source OOF plus target prediction inventory changed")
    result = {"protocol": PROTOCOL, "protocol_sha256": sha256(PROTOCOL_PATH),
              "prediction_sha256": sha256(path), "prediction_rows": len(predictions),
              "temperature_candidates": [float(t) for t in TEMPERATURES],
              "studies": results, "scope": PROTOCOL["scope"]}
    (ROOT / "V1_SOURCE_OOF_CAL_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    evaluate()
