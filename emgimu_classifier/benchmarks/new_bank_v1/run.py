"""Frozen source-only force and wearing screens for the independent new bank."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emgimu.datasets.electrode_shift import PATH_RE
from emgimu.datasets.libemg_force import load_libemg_force_windows
from emgimu.feature_bank.families import LocalDetailFamily
from emgimu.feature_bank.force_full_fusion import aggregate
from emgimu.feature_bank.new_bank_v1 import NEW_BANK_V1
from emgimu.feature_bank.wearing_full_fusion import load as load_wearing


ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "PROTOCOL.json").read_text(encoding="utf-8"))
FACTORIES = {"F0": LocalDetailFamily, **NEW_BANK_V1}
RAW_FORCE = Path("D:/emg-imu-benchmarks/data/raw/libemg_force/official/ContractionIntensity-main")
ARCHIVE_FORCE = Path("D:/emg-imu-benchmarks/data/raw/libemg_force/contraction-intensity-main.zip")
RAW_WEARING = Path("D:/emg-imu-benchmarks/data/raw/libemg_electrode_shift/CIILData-main.zip")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(source, target, names):
    source_features, target_features = {}, {}
    source_y = target_y = source_users = target_users = source_trials = target_trials = None
    dimensions = {}
    for name in names:
        family = FACTORIES[name]().fit(source.batch, source.labels)
        a, y, users, trials = aggregate(family.transform(source.batch), source)
        b, yy, uu, tt = aggregate(family.transform(target.batch), target)
        if source_y is not None:
            np.testing.assert_array_equal(source_y, y)
            np.testing.assert_array_equal(target_y, yy)
            np.testing.assert_array_equal(source_trials, trials)
            np.testing.assert_array_equal(target_trials, tt)
        source_y, target_y = y, yy
        source_users, target_users = users, uu
        source_trials, target_trials = trials, tt
        source_features[name], target_features[name] = a, b
        dimensions[name] = int(a.shape[1])
    if set(source_trials) & set(target_trials):
        raise ValueError("source/target native trial overlap")
    return (source_features, target_features, source_y, target_y,
            source_users, target_users, source_trials, target_trials, dimensions)


def fit_predict(x, y, xt):
    classes = np.unique(y)
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=20260924))
    model.fit(x, y)
    np.testing.assert_array_equal(model[-1].classes_, classes)
    if np.max(model[-1].n_iter_) >= 2000:
        raise RuntimeError("logistic classifier did not converge")
    return classes, model.predict_proba(xt)


def append_predictions(rows, dataset, phase, subject, arms, extracted, target_conditions):
    source_x, target_x, source_y, target_y, _, target_users, source_trials, target_trials, dimensions = extracted
    if len(target_conditions) != len(target_trials):
        raise ValueError("trial conditions not aligned")
    for arm in arms:
        members = arm.split("+")
        x = np.concatenate([source_x[name] for name in members], axis=1)
        xt = np.concatenate([target_x[name] for name in members], axis=1)
        classes, probabilities = fit_predict(x, source_y, xt)
        for trial, label, user, condition, probability in zip(
                target_trials, target_y, target_users, target_conditions, probabilities):
            rows.append({"dataset": dataset, "phase": phase, "arm": arm, "subject": int(user),
                         "condition": str(condition), "trial_id": str(trial), "label": int(label),
                         **{f"p_{class_id}": float(value) for class_id, value in zip(classes, probability)}})
    return {"source_trials": source_trials.tolist(), "target_trials": target_trials.tolist(),
            "source_subject": subject, "feature_dimensions": dimensions}


def run_force(rows, split_manifest):
    spec = PROTOCOL["force"]
    source = load_libemg_force_windows(RAW_FORCE, subjects=spec["source_subjects"],
                                       conditions=spec["source_conditions"])
    for phase in ("validation", "final"):
        users = spec[f"{phase}_subjects"]
        target = load_libemg_force_windows(RAW_FORCE, subjects=users, conditions=spec["target_conditions"])
        names = tuple(dict.fromkeys(part for arm in spec["arms"] for part in arm.split("+")))
        extracted = extract(source, target, names)
        trials = extracted[7]
        conditions = np.array([np.unique(target.conditions[target.trials == trial]).item() for trial in trials])
        split_manifest[f"force_{phase}"] = append_predictions(
            rows, "force", phase, "pooled_source_1_to_6", spec["arms"], extracted, conditions)
        print(f"force {phase}: {len(trials)} held-out native trials", flush=True)


def run_wearing(rows, split_manifest):
    spec = PROTOCOL["wearing"]
    for phase in ("validation", "final"):
        for subject in spec[f"{phase}_subjects"]:
            source = load_wearing(RAW_WEARING, subject, (spec["source_domain"],))
            target = load_wearing(RAW_WEARING, subject, tuple(spec["target_domains"]))
            names = tuple(dict.fromkeys(part for arm in spec["arms"] for part in arm.split("+")))
            extracted = extract(source, target, names)
            conditions = np.array([PATH_RE.fullmatch(trial)["domain"] for trial in extracted[7]])
            split_manifest[f"wearing_{phase}_{subject}"] = append_predictions(
                rows, "wearing", phase, subject, spec["arms"], extracted, conditions)
            print(f"wearing {phase} subject {subject}: {len(extracted[7])} held-out native trials", flush=True)


def metrics(y, probabilities, classes):
    prediction = classes[probabilities.argmax(axis=1)]
    one_hot = (y[:, None] == classes[None, :]).astype(float)
    return {"trials": int(len(y)), "accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, labels=classes, average="macro", zero_division=0)),
            "log_loss": float(log_loss(y, probabilities, labels=classes)),
            "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
            "per_class_recall": {str(label): float(value) for label, value in zip(classes,
                recall_score(y, prediction, labels=classes, average=None, zero_division=0))}}


def summarize(rows, dataset, phase, arm):
    part = [row for row in rows if row["dataset"] == dataset and row["phase"] == phase and row["arm"] == arm]
    classes = np.arange(7 if dataset == "force" else 5)
    y = np.array([row["label"] for row in part])
    probabilities = np.array([[row[f"p_{i}"] for i in classes] for row in part])
    subjects = np.array([row["subject"] for row in part])
    conditions = np.array([row["condition"] for row in part])
    by_subject = {str(s): metrics(y[subjects == s], probabilities[subjects == s], classes)
                  for s in sorted(set(subjects))}
    by_condition = {str(c): metrics(y[conditions == c], probabilities[conditions == c], classes)
                    for c in sorted(set(conditions))}
    return {"pooled": metrics(y, probabilities, classes), "by_subject": by_subject,
            "by_condition": by_condition,
            "minimum_subject_macro_f1": min(x["macro_f1"] for x in by_subject.values()),
            "worst_condition_macro_f1": min(x["macro_f1"] for x in by_condition.values())}


def run():
    rows, split_manifest = [], {}
    run_force(rows, split_manifest)
    run_wearing(rows, split_manifest)
    results = {"protocol": PROTOCOL, "archive_sha256": {"force": sha256(ARCHIVE_FORCE),
        "wearing": sha256(RAW_WEARING)}, "split_trial_ids": split_manifest, "results": {}}
    for dataset in ("force", "wearing"):
        results["results"][dataset] = {}
        for phase in ("validation", "final"):
            arms = {arm: summarize(rows, dataset, phase, arm) for arm in PROTOCOL[dataset]["arms"]}
            chosen = min(arms, key=lambda arm: (-arms[arm]["pooled"]["macro_f1"],
                                           arms[arm]["pooled"]["log_loss"]))
            results["results"][dataset][phase] = {"arms": arms, "phase_best_descriptive_arm": chosen}
            for arm, value in arms.items():
                print(f"{dataset} {phase} {arm}: F1={value['pooled']['macro_f1']:.4f}", flush=True)
        results["results"][dataset]["validation_selected_arm"] = (
            results["results"][dataset]["validation"]["phase_best_descriptive_arm"])
    (ROOT / "RESULTS.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "TRIAL_PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
