"""Frozen matched 0/1/2-shot force calibration on the new eight-channel bank."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from benchmarks.new_bank_v1.run import metrics, sha256
from benchmarks.new_bank_v2.force_run import ARCHIVE, RAW
from emgimu.datasets.libemg_force import FILE_RE, load_libemg_force_windows
from emgimu.feature_bank.force_full_fusion import aggregate
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2, TraceCovarianceV2

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "FORCE_CAL_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
CLASSES = np.arange(7)
ARMS = ("F0v2", "F0v2+F2a")
METHODS = ("source_only", "one_shot", "two_shot")


def _score(rows: list[dict], phase: str, arm: str, method: str) -> dict:
    chosen = [r for r in rows if r["phase"] == phase and r["arm"] == arm and r["method"] == method]
    y = np.asarray([r["label"] for r in chosen])
    p = np.asarray([[r[f"p_{c}"] for c in CLASSES] for r in chosen])
    users = np.asarray([r["subject"] for r in chosen])
    conditions = np.asarray([r["condition"] for r in chosen])
    subjects = {str(user): metrics(y[users == user], p[users == user], CLASSES)
                for user in sorted(set(users))}
    domains = {str(condition): metrics(y[conditions == condition], p[conditions == condition], CLASSES)
               for condition in sorted(set(conditions))}
    return {"pooled": metrics(y, p, CLASSES), "by_subject": subjects,
            "by_condition": domains,
            "minimum_subject_macro_f1": min(v["macro_f1"] for v in subjects.values()),
            "worst_condition_macro_f1": min(v["macro_f1"] for v in domains.values())}


def _parent_rows() -> dict[tuple[str, str, str], dict[str, str]]:
    with (ROOT / "FORCE_TRIAL_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    selected = {(r["phase"], r["arm"], r["trial_id"]): r for r in records if r["arm"] in ARMS}
    if len(records) != 5880 or len(selected) != 2352:
        raise ValueError("frozen parent prediction rows changed")
    return selected


def run() -> None:
    if hashlib.sha256((ROOT / "FORCE_PROTOCOL.json").read_bytes()).hexdigest() != PROTOCOL["parent_protocol_sha256"]:
        raise ValueError("parent protocol changed")
    if (PROTOCOL["arms"] != list(ARMS) or PROTOCOL["shots_per_class"] != [0, 1, 2]
            or PROTOCOL["source_condition"] != "Ramp"
            or PROTOCOL["evaluation_repetitions"] != [3, 4]):
        raise ValueError("calibration protocol changed")
    parent = json.loads((ROOT / "FORCE_RESULTS.json").read_text(encoding="utf-8"))
    if parent["validation_selected_arm"] != ARMS[0] or parent["archive_sha256"] != sha256(ARCHIVE):
        raise ValueError("parent selection or dataset archive changed")
    frozen = _parent_rows()
    source = load_libemg_force_windows(RAW, subjects=PROTOCOL["source_subjects"],
                                       conditions=[PROTOCOL["source_condition"]])
    families = {"F0v2": RestNoiseDetailV2(rest_label=0).fit(source.batch, source.labels),
                "F2a": TraceCovarianceV2(shrinkage=0.05).fit(source.batch)}
    source_vectors: dict[str, np.ndarray] = {}
    source_y = source_trials = None
    for name, family in families.items():
        x, y, _, trials = aggregate(family.transform(source.batch), source)
        if source_y is not None:
            np.testing.assert_array_equal(source_y, y)
            np.testing.assert_array_equal(source_trials, trials)
        source_vectors[name] = x
        source_y, source_trials = y, trials
    rows: list[dict] = []
    assignments: dict[str, dict] = {}
    temperatures: dict[str, float] = {}
    for phase in ("validation", "final"):
        target = load_libemg_force_windows(RAW, subjects=PROTOCOL[f"{phase}_subjects"],
                                           conditions=PROTOCOL["target_conditions"])
        target_vectors: dict[str, np.ndarray] = {}
        target_y = target_users = target_trials = None
        for name, family in families.items():
            xt, y, users, trials = aggregate(family.transform(target.batch), target)
            if target_y is not None:
                np.testing.assert_array_equal(target_y, y)
                np.testing.assert_array_equal(target_trials, trials)
            target_vectors[name] = xt
            target_y, target_users, target_trials = y, users, trials
        if set(source_trials) & set(target_trials):
            raise ValueError("source and target trials overlap")
        meta = [FILE_RE.fullmatch(str(trial) + ".csv") for trial in target_trials]
        if any(match is None or int(match["subject"]) != user or
               int(match["label"]) - 1 != label
               for match, user, label in zip(meta, target_users, target_y)):
            raise ValueError("target trial filename identity invalid")
        for arm in ARMS:
            names = arm.split("+")
            x = np.concatenate([source_vectors[name] for name in names], axis=1)
            xt = np.concatenate([target_vectors[name] for name in names], axis=1)
            scaler = StandardScaler().fit(x)
            z, zt = scaler.transform(x), scaler.transform(xt)
            model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                       random_state=20260924).fit(z, source_y)
            np.testing.assert_array_equal(model.classes_, CLASSES)
            if np.max(model.n_iter_) >= 2000:
                raise RuntimeError("source classifier did not converge")
            p_source = model.predict_proba(zt)
            for trial, label, values in zip(target_trials, target_y, p_source):
                old = frozen[(phase, arm, str(trial))]
                if int(old["label"]) != label or not np.allclose(
                        values, [float(old[f"p_{c}"]) for c in CLASSES], atol=1e-12, rtol=0):
                    raise ValueError("source-only model failed parent probability replay")
            centers = np.stack([z[source_y == label].mean(axis=0) for label in CLASSES])
            within = np.mean((z - centers[source_y]) ** 2, axis=1)
            temperature = max(float(np.median(within)), 1e-6)
            temperatures[f"{phase}_{arm}"] = temperature
            for subject in PROTOCOL[f"{phase}_subjects"]:
                for condition in PROTOCOL["target_conditions"]:
                    context = np.asarray([user == subject and match["condition"] == condition
                                          for user, match in zip(target_users, meta)])
                    test = context & np.asarray([int(match["repetition"]) in (3, 4) for match in meta])
                    if test.sum() != 14 or set(target_y[test]) != set(CLASSES):
                        raise ValueError("matched evaluation repetitions incomplete")
                    assignment = f"{phase}_{subject}_{condition}"
                    for shot, method in zip((0, 1, 2), METHODS):
                        cal_reps = set(PROTOCOL["calibration_repetitions"][str(shot)])
                        cal = context & np.asarray([int(match["repetition"]) in cal_reps for match in meta])
                        if cal.sum() != 7 * shot or np.any(cal & test) or (shot and set(target_y[cal]) != set(CLASSES)):
                            raise ValueError("matched calibration repetitions incomplete")
                        if shot:
                            prototypes = np.stack([zt[cal][target_y[cal] == label].mean(axis=0)
                                                   for label in CLASSES])
                            distance = np.mean((zt[test, None, :] - prototypes[None, :, :]) ** 2, axis=2)
                            personal = softmax(-distance.astype(np.float64) / temperature, axis=1)
                            probabilities = 0.5 * p_source[test] + 0.5 * personal
                            probabilities /= probabilities.sum(axis=1, keepdims=True)
                        else:
                            probabilities = p_source[test]
                        assignments.setdefault(assignment, {"source": source_trials.tolist(),
                                                            "evaluation": target_trials[test].tolist(),
                                                            "calibration": {}})
                        assignments[assignment]["calibration"][str(shot)] = target_trials[cal].tolist()
                        for trial, label, values in zip(target_trials[test], target_y[test], probabilities):
                            rows.append({"phase": phase, "subject": subject, "condition": condition,
                                         "arm": arm, "method": method, "trial_id": str(trial),
                                         "label": int(label),
                                         **{f"p_{c}": float(v) for c, v in zip(CLASSES, values)}})
        print(f"force calibration {phase}: {len(target_trials)} target trials loaded", flush=True)
    if len(rows) != 1680 or len(assignments) != 40:
        raise AssertionError("unexpected force calibration output shape")
    scores = {phase: {arm: {method: _score(rows, phase, arm, method) for method in METHODS}
                      for arm in ARMS} for phase in ("validation", "final")}
    result = {"protocol": PROTOCOL, "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
              "archive_sha256": parent["archive_sha256"], "parent_predictions_sha256":
              hashlib.sha256((ROOT / "FORCE_TRIAL_PREDICTIONS.csv").read_bytes()).hexdigest(),
              "assignments": assignments, "source_only_temperatures": temperatures,
              "scores": scores, "boundary": PROTOCOL["boundary"]}
    (ROOT / "FORCE_CAL_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "FORCE_CAL_PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run()
