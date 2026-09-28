"""Frozen new-v2 eight-channel cross-user intensity experiment."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v1.run import fit_predict, metrics, sha256
from emgimu.datasets.libemg_force import FILE_RE, load_libemg_force_windows
from emgimu.feature_bank.families import LocalDetailFamily
from emgimu.feature_bank.force_full_fusion import aggregate
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2, RingRelativeCovarianceV2, TraceCovarianceV2

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "FORCE_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
ARMS = tuple(PROTOCOL["arms"])
CLASSES = np.arange(7)
RAW = Path("D:/emg-imu-benchmarks/data/raw/libemg_force/official/ContractionIntensity-main")
ARCHIVE = Path("D:/emg-imu-benchmarks/data/raw/libemg_force/contraction-intensity-main.zip")


def _score(rows: list[dict], phase: str, arm: str) -> dict:
    part = [row for row in rows if row["phase"] == phase and row["arm"] == arm]
    y = np.asarray([row["label"] for row in part])
    p = np.asarray([[row[f"p_{c}"] for c in CLASSES] for row in part])
    users = np.asarray([row["subject"] for row in part])
    conditions = np.asarray([row["condition"] for row in part])
    by_subject = {str(user): metrics(y[users == user], p[users == user], CLASSES)
                  for user in sorted(set(users))}
    by_condition = {str(condition): metrics(y[conditions == condition], p[conditions == condition], CLASSES)
                    for condition in sorted(set(conditions))}
    return {"pooled": metrics(y, p, CLASSES), "by_subject": by_subject,
            "by_condition": by_condition,
            "minimum_subject_macro_f1": min(value["macro_f1"] for value in by_subject.values()),
            "worst_condition_macro_f1": min(value["macro_f1"] for value in by_condition.values())}


def run() -> None:
    if (ARMS != ("F0", "F0v2", "F0v2+F2a", "F0v2+F3c", "F0v2+F2a+F3c")
            or PROTOCOL["rest_label"] != 0 or PROTOCOL["source_conditions"] != ["Ramp"]
            or PROTOCOL["source_subjects"] != [1, 2, 3, 4, 5, 6]):
        raise ValueError("frozen cross-user intensity protocol changed")
    source = load_libemg_force_windows(RAW, subjects=PROTOCOL["source_subjects"],
                                       conditions=PROTOCOL["source_conditions"])
    if source.batch.channels != 8 or source.batch.sample_rate_hz != 1000 or set(source.labels) != set(CLASSES):
        raise ValueError("unexpected source data contract")
    families = {
        "F0": LocalDetailFamily().fit(source.batch, source.labels),
        "F0v2": RestNoiseDetailV2(rest_label=0).fit(source.batch, source.labels),
        "F2a": TraceCovarianceV2(shrinkage=0.05).fit(source.batch),
        "F3c": RingRelativeCovarianceV2(shrinkage=0.05).fit(source.batch),
    }
    source_vectors = {}
    source_y = source_trials = source_users = None
    for name, family in families.items():
        x, y, users, trials = aggregate(family.transform(source.batch), source)
        if source_y is not None:
            np.testing.assert_array_equal(source_y, y)
            np.testing.assert_array_equal(source_trials, trials)
        source_vectors[name] = x
        source_y, source_users, source_trials = y, users, trials
    rows: list[dict] = []
    splits: dict[str, dict] = {}
    for phase in ("validation", "final"):
        target = load_libemg_force_windows(RAW, subjects=PROTOCOL[f"{phase}_subjects"],
                                           conditions=PROTOCOL["target_conditions"])
        target_vectors = {}
        target_y = target_users = target_trials = None
        for name, family in families.items():
            x, y, users, trials = aggregate(family.transform(target.batch), target)
            if target_y is not None:
                np.testing.assert_array_equal(target_y, y)
                np.testing.assert_array_equal(target_trials, trials)
            target_vectors[name] = x
            target_y, target_users, target_trials = y, users, trials
        if set(source_trials) & set(target_trials) or set(source_users) & set(target_users):
            raise ValueError("source and target subjects/trials overlap")
        splits[phase] = {"source_subjects": sorted(set(map(int, source_users))),
                         "target_subjects": sorted(set(map(int, target_users))),
                         "source_trials": source_trials.tolist(),
                         "target_trials": target_trials.tolist()}
        for arm in ARMS:
            names = arm.split("+")
            x = np.concatenate([source_vectors[name] for name in names], axis=1)
            xt = np.concatenate([target_vectors[name] for name in names], axis=1)
            classes, p = fit_predict(x, source_y, xt)
            np.testing.assert_array_equal(classes, CLASSES)
            for trial, user, label, probability in zip(target_trials, target_users, target_y, p):
                match = FILE_RE.fullmatch(str(trial) + ".csv")
                if match is None or int(match["subject"]) != user or int(match["label"]) - 1 != label:
                    raise ValueError("native target trial ID mismatch")
                rows.append({"phase": phase, "subject": int(user), "condition": match["condition"],
                             "arm": arm, "trial_id": str(trial), "label": int(label),
                             **{f"p_{c}": float(value) for c, value in zip(CLASSES, probability)}})
        print(f"force {phase}: {len(target_trials)} held-out native trials", flush=True)
    scores = {phase: {arm: _score(rows, phase, arm) for arm in ARMS}
              for phase in ("validation", "final")}
    selected = min(ARMS, key=lambda arm: (-scores["validation"][arm]["pooled"]["macro_f1"],
                                           scores["validation"][arm]["pooled"]["log_loss"]))
    result = {"protocol": PROTOCOL,
              "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
              "archive_sha256": sha256(ARCHIVE),
              "feature_dimensions": {name: len(family.feature_names) for name, family in families.items()},
              "source_rest_windows": int(np.sum(source.labels == 0)),
              "split_trial_ids": splits, "validation_selected_arm": selected,
              "scores": scores, "scope": PROTOCOL["scope"]}
    (ROOT / "FORCE_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "FORCE_TRIAL_PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"force validation-selected arm: {selected}", flush=True)


if __name__ == "__main__":
    run()
