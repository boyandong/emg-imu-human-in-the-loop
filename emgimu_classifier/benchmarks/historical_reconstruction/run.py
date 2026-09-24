"""Frozen public wearing screen for independently reconstructed ring features."""
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
from emgimu.feature_bank.families import LocalDetailFamily
from emgimu.feature_bank.force_full_fusion import aggregate
from emgimu.feature_bank.reconstructed_ring import ReconstructedCes, ReconstructedRlcs
from emgimu.feature_bank.wearing_full_fusion import load


ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "PROTOCOL.json").read_text(encoding="utf-8"))
FACTORIES = {"F0": LocalDetailFamily, "RLCS_reconstructed": ReconstructedRlcs,
             "CES_reconstructed": ReconstructedCes}
ARMS = {arm: tuple(arm.split("+")) for arm in PROTOCOL["arms"]}
CLASSES = np.arange(5)


def metrics(y, prob):
    pred = CLASSES[prob.argmax(axis=1)]
    one_hot = (y[:, None] == CLASSES[None, :]).astype(float)
    return {"trials": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
            "log_loss": float(log_loss(y, prob, labels=CLASSES)),
            "brier": float(np.mean(np.sum((prob - one_hot) ** 2, axis=1))),
            "per_class_recall": {name: float(value) for name, value in zip(PROTOCOL["labels"],
                recall_score(y, pred, labels=CLASSES, average=None, zero_division=0))}}


def run(archive: Path, output: Path):
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    predictions = []
    splits = {}
    dimensions = {}
    for phase, subjects in PROTOCOL["subjects"].items():
        for subject in subjects:
            source = load(archive, subject, (PROTOCOL["source_domain"],))
            target = load(archive, subject, tuple(PROTOCOL["target_domains"]))
            source_features, target_features = {}, {}
            source_y = target_y = source_trials = target_trials = None
            for name, factory in FACTORIES.items():
                family = factory().fit(source.batch, source.labels)
                source_features[name], y, _, trials = aggregate(family.transform(source.batch), source)
                target_features[name], yy, _, tt = aggregate(family.transform(target.batch), target)
                if source_y is not None:
                    np.testing.assert_array_equal(source_y, y)
                    np.testing.assert_array_equal(target_y, yy)
                    np.testing.assert_array_equal(source_trials, trials)
                    np.testing.assert_array_equal(target_trials, tt)
                source_y, target_y, source_trials, target_trials = y, yy, trials, tt
                dimensions[name] = int(source_features[name].shape[1])
            if set(source_trials) & set(target_trials):
                raise ValueError("source/target trial overlap")
            splits[str(subject)] = {"phase": phase, "source": source_trials.tolist(),
                                    "target": target_trials.tolist()}
            for arm, members in ARMS.items():
                x = np.concatenate([source_features[name] for name in members], axis=1)
                xt = np.concatenate([target_features[name] for name in members], axis=1)
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=1000, random_state=20260924))
                model.fit(x, source_y)
                np.testing.assert_array_equal(model[-1].classes_, CLASSES)
                probability = model.predict_proba(xt)
                for trial, label, prob in zip(target_trials, target_y, probability):
                    domain = PATH_RE.fullmatch(trial)["domain"]
                    predictions.append({"phase": phase, "subject": subject, "arm": arm,
                        "trial_id": trial, "domain": domain, "label": int(label),
                        **{f"p_{i}": float(value) for i, value in enumerate(prob)}})
            print(f"{phase} subject {subject}: {len(target_trials)} held-out trials", flush=True)
    results = {"protocol": PROTOCOL, "source_archive_sha256": archive_hash,
               "family_dimensions": dimensions, "source_target_trial_ids": splits, "arms": {}}
    for phase in PROTOCOL["subjects"]:
        results["arms"][phase] = {}
        for arm in ARMS:
            part = [row for row in predictions if row["phase"] == phase and row["arm"] == arm]
            y = np.array([row["label"] for row in part])
            prob = np.array([[row[f"p_{i}"] for i in CLASSES] for row in part])
            domain = np.array([row["domain"] for row in part])
            subject = np.array([row["subject"] for row in part])
            by_subject = {str(s): metrics(y[subject == s], prob[subject == s]) for s in sorted(set(subject))}
            by_domain = {str(d): metrics(y[domain == d], prob[domain == d]) for d in sorted(set(domain))}
            results["arms"][phase][arm] = {"pooled": metrics(y, prob), "by_subject": by_subject,
                "by_domain": by_domain, "minimum_subject_macro_f1": min(v["macro_f1"] for v in by_subject.values()),
                "worst_domain_macro_f1": min(v["macro_f1"] for v in by_domain.values())}
            print(f"{phase} {arm}: F1={results['arms'][phase][arm]['pooled']['macro_f1']:.4f}", flush=True)
    results["paired_correctness_changes_vs_F0"] = {}
    lookup = {(row["phase"], row["subject"], row["arm"], row["trial_id"]): row
              for row in predictions}
    for phase in PROTOCOL["subjects"]:
        controls = [row for row in predictions if row["phase"] == phase and row["arm"] == "F0"]
        comparisons = {}
        for arm in ARMS:
            if arm == "F0":
                continue
            counts = {"base_wrong_added_correct": 0, "base_correct_added_wrong": 0,
                      "both_wrong": 0, "both_correct": 0}
            for base in controls:
                other = lookup[(phase, base["subject"], arm, base["trial_id"])]
                if base["label"] != other["label"]:
                    raise ValueError("paired trial label mismatch")
                base_correct = int(np.argmax([base[f"p_{i}"] for i in CLASSES])) == base["label"]
                other_correct = int(np.argmax([other[f"p_{i}"] for i in CLASSES])) == base["label"]
                key = ("base_correct_added_wrong" if base_correct else "both_wrong") if not other_correct else (
                    "both_correct" if base_correct else "base_wrong_added_correct")
                counts[key] += 1
            if sum(counts.values()) != len(controls):
                raise ValueError("incomplete paired comparison")
            comparisons[arm] = counts
        results["paired_correctness_changes_vs_F0"][phase] = comparisons
    output.mkdir(parents=True, exist_ok=True)
    (output / "RESULTS.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (output / "TRIAL_PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path(
        "D:/emg-imu-benchmarks/data/raw/libemg_electrode_shift/CIILData-main.zip"))
    args = parser.parse_args()
    run(args.archive, ROOT)


if __name__ == "__main__":
    main()
