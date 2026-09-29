"""Source-frozen independent new-v1 family screening on force intensity."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v1.run import fit_predict, sha256
from benchmarks.new_bank_v2.force_run import RAW, ARCHIVE, _score
from emgimu.datasets.libemg_force import FILE_RE, load_libemg_force_windows
from emgimu.feature_bank.force_full_fusion import aggregate
from emgimu.feature_bank.new_bank_v1 import NEW_BANK_V1
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "FORCE_V1_EXTENSION_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
PARENT = "FORCE"
CLASSES = np.asarray(PROTOCOL["classes"])
ARMS = tuple(PROTOCOL["arms"])


def check_protocol() -> dict:
    for suffix, key in (("PROTOCOL.json", "parent_protocol_sha256"),
                        ("RESULTS.json", "parent_result_sha256"),
                        ("TRIAL_PREDICTIONS.csv", "parent_prediction_sha256")):
        if sha256(ROOT / f"{PARENT}_{suffix}") != PROTOCOL[key]:
            raise AssertionError(f"frozen force parent changed: {suffix}")
    parent = json.loads((ROOT / "FORCE_PROTOCOL.json").read_text(encoding="utf-8"))
    if (PROTOCOL["source_subjects"] != parent["source_subjects"]
            or PROTOCOL["source_conditions"] != parent["source_conditions"]
            or PROTOCOL["validation_subjects"] != parent["validation_subjects"]
            or PROTOCOL["final_subjects"] != parent["final_subjects"]
            or PROTOCOL["target_conditions"] != parent["target_conditions"]
            or PROTOCOL["classes"] != list(range(7))
            or PROTOCOL["candidate_families"] != list(NEW_BANK_V1)
            or ARMS != ("F0v2", *[f"F0v2+{name}" for name in NEW_BANK_V1])):
        raise AssertionError("force extension disagrees with frozen parent")
    return parent


def fit_vectors(data, families: dict) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    vectors = {}
    first = None
    for name, family in families.items():
        x, y, users, trials = aggregate(family.transform(data.batch), data)
        if first is None:
            first = (y, users, trials)
        else:
            for observed, expected in zip((y, users, trials), first):
                np.testing.assert_array_equal(observed, expected)
        vectors[name] = x
    return vectors, *first


def frozen_base_replay(rows: list[dict]) -> float:
    with (ROOT / "FORCE_TRIAL_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        parent = {(row["phase"], row["trial_id"]): row for row in csv.DictReader(stream)
                  if row["arm"] == "F0v2"}
    base = [row for row in rows if row["arm"] == "F0v2"]
    if len(parent) != 1176 or len(base) != 1176:
        raise AssertionError("parent baseline prediction coverage changed")
    maximum = 0.0
    for row in base:
        previous = parent[(row["phase"], row["trial_id"])]
        if (row["subject"], row["condition"], row["label"]) != (
                int(previous["subject"]), previous["condition"], int(previous["label"])):
            raise AssertionError("frozen baseline native identity changed")
        current = np.asarray([row[f"p_{c}"] for c in CLASSES])
        original = np.asarray([float(previous[f"p_{c}"]) for c in CLASSES])
        maximum = max(maximum, float(np.max(np.abs(current - original))))
    if maximum > 1e-8:
        raise AssertionError(f"baseline numerical replay drift: {maximum}")
    return maximum


def evaluate() -> dict:
    check_protocol()
    source = load_libemg_force_windows(RAW, subjects=PROTOCOL["source_subjects"],
                                       conditions=PROTOCOL["source_conditions"])
    if (source.batch.channels != 8 or source.batch.sample_rate_hz != 1000
            or source.batch.emg.shape[1] != 200 or set(source.labels) != set(CLASSES)):
        raise AssertionError("source signal contract changed")
    families = {"F0v2": RestNoiseDetailV2(rest_label=0).fit(source.batch, source.labels),
                **{name: constructor().fit(source.batch, source.labels)
                   for name, constructor in NEW_BANK_V1.items()}}
    source_vectors, y, source_users, source_trials = fit_vectors(source, families)
    parent_result = json.loads((ROOT / "FORCE_RESULTS.json").read_text(encoding="utf-8"))
    if set(source_trials) != set(parent_result["split_trial_ids"]["validation"]["source_trials"]):
        raise AssertionError("frozen source trial identity changed")
    rows = []
    splits = {}
    for phase in ("validation", "final"):
        target = load_libemg_force_windows(RAW, subjects=PROTOCOL[f"{phase}_subjects"],
                                           conditions=PROTOCOL["target_conditions"])
        target_vectors, target_y, target_users, target_trials = fit_vectors(target, families)
        if (set(source_trials) & set(target_trials) or set(source_users) & set(target_users)
                or set(target_trials) != set(parent_result["split_trial_ids"][phase]["target_trials"])):
            raise AssertionError("target split changed or overlaps source")
        splits[phase] = {"source_subjects": sorted(map(int, set(source_users))),
                         "target_subjects": sorted(map(int, set(target_users))),
                         "source_trials": source_trials.tolist(),
                         "target_trials": target_trials.tolist()}
        for arm in ARMS:
            names = arm.split("+")
            x = np.concatenate([source_vectors[name] for name in names], axis=1)
            xt = np.concatenate([target_vectors[name] for name in names], axis=1)
            classes, probability = fit_predict(x, y, xt)
            np.testing.assert_array_equal(classes, CLASSES)
            for trial, user, label, p in zip(target_trials, target_users, target_y, probability):
                match = FILE_RE.fullmatch(str(trial) + ".csv")
                if (match is None or int(match["subject"]) != user
                        or int(match["label"]) - 1 != label):
                    raise AssertionError("native force trial metadata changed")
                rows.append({"phase": phase, "subject": int(user),
                             "condition": match["condition"], "arm": arm,
                             "trial_id": str(trial), "label": int(label),
                             **{f"p_{c}": float(value) for c, value in zip(CLASSES, p)}})
        print(f"force new-v1 {phase}: {len(target_trials)} held-out native trials", flush=True)
    baseline_error = frozen_base_replay(rows)
    scores = {phase: {arm: _score(rows, phase, arm) for arm in ARMS}
              for phase in ("validation", "final")}
    selected = min(ARMS, key=lambda arm: (-scores["validation"][arm]["pooled"]["macro_f1"],
                                           scores["validation"][arm]["pooled"]["log_loss"]))
    path = ROOT / "FORCE_V1_EXTENSION_PREDICTIONS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol": PROTOCOL, "protocol_sha256": sha256(PROTOCOL_PATH),
              "archive_sha256": sha256(ARCHIVE), "prediction_sha256": sha256(path),
              "feature_dimensions": {name: int(values.shape[1])
                                     for name, values in source_vectors.items()},
              "source_rest_windows": int(np.sum(source.labels == 0)),
              "baseline_replay_max_abs_error": baseline_error,
              "split_trial_ids": splits, "validation_selected_arm": selected,
              "scores": scores, "scope": PROTOCOL["scope"]}
    (ROOT / "FORCE_V1_EXTENSION_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"force new-v1 validation-selected arm: {selected}", flush=True)
    return result


if __name__ == "__main__":
    evaluate()
