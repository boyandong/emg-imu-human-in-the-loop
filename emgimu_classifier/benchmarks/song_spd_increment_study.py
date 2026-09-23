"""Causal Song F2c/F7 SPD increments with source-frozen fits and matched trials.

The four sessions are one participant on one day; S04 is exploratory because
other project analyses have already inspected its outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.song_real8_study import HANDS, _join_batches, load_session
from emgimu.feature_bank import SpdTangentFamily, SpdTangentPersonalAnchor
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.families import LocalDetailFamily


def trial_probabilities(labels, probabilities, trial_ids, classes):
    """One vote per formal trial, preserving native trial identity."""
    ids, truth, averaged = [], [], []
    for trial in np.unique(trial_ids):
        selected = trial_ids == trial
        unique = np.unique(labels[selected])
        if len(unique) != 1:
            raise ValueError(f"mixed hand labels in {trial}")
        ids.append(trial)
        truth.append(unique[0])
        averaged.append(probabilities[selected].mean(axis=0))
    ids, truth, averaged = np.asarray(ids), np.asarray(truth), np.stack(averaged)
    if set(truth) != set(classes):
        raise ValueError("every formal hand class must be represented")
    return ids, truth, averaged


def metrics(truth, probability, classes):
    probability = np.asarray(probability, dtype=np.float64)
    if (probability.shape != (len(truth), len(classes)) or
            not np.isfinite(probability).all() or np.any(probability < 0) or
            np.any(probability.sum(axis=1) <= 0)):
        raise ValueError("probabilities must be finite aligned nonnegative rows")
    probability = probability / probability.sum(axis=1, keepdims=True)
    prediction = classes[np.argmax(probability, axis=1)]
    one_hot = truth[:, None] == classes[None, :]
    return {
        "trials": len(truth),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=classes,
                                   average="macro", zero_division=0)),
        "log_loss": float(log_loss(truth, probability, labels=classes)),
        "brier": float(np.mean((probability - one_hot) ** 2)),
        "recall": dict(zip(classes.tolist(), map(float, recall_score(
            truth, prediction, labels=classes, average=None, zero_division=0)))),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=classes).tolist(),
    }


def spd_trial_probabilities(family, data, budget, expected_ids, classes):
    if budget not in (1, 2):
        raise ValueError("calibration budget must be one or two blocks per class")
    selected = data["calibration_shot"] <= budget
    labels = data["calibration_hand"][selected]
    shots = data["calibration_shot"][selected]
    if any(np.count_nonzero(labels == hand) != 3 * budget for hand in HANDS):
        raise ValueError("calibration blocks incomplete or unbalanced")
    calibration_ids = np.asarray([f"cal:{label}:{shot}" for label, shot in zip(labels, shots)])
    if len(np.unique(calibration_ids)) != 4 * budget:
        raise ValueError("calibration block identity mismatch")
    calibration = FeatureBatch(data["calibration_batch"].emg[selected], 250.0)
    anchor = SpdTangentPersonalAnchor(family).fit_trials(calibration, labels, calibration_ids)
    if not np.array_equal(anchor.anchor_.classes_, classes):
        raise ValueError("SPD class order differs from source classifier")
    ids, coordinates = anchor.transform_trials(data["batch"], data["trial"])
    if not np.array_equal(ids, expected_ids):
        raise ValueError("SPD and F0 evaluation trial IDs differ")
    probability = softmax(-coordinates[:, :len(classes)] /
                          anchor.anchor_.similarity_scale_, axis=1)
    return probability, {
        "calibration_blocks": int(len(np.unique(calibration_ids))),
        "calibration_windows": int(len(labels)),
        "source_reference_sha256": anchor.source_reference_sha256_,
        "calibration_trial_ids": np.unique(calibration_ids).tolist(),
        "formal_trial_overlap": bool(set(calibration_ids) & set(ids)),
    }


def paired_changes(truth, base, candidate, classes):
    source_correct = classes[np.argmax(base, axis=1)] == truth
    candidate_correct = classes[np.argmax(candidate, axis=1)] == truth
    corrected = int(np.sum(~source_correct & candidate_correct))
    new_errors = int(np.sum(source_correct & ~candidate_correct))
    rng = np.random.default_rng(20260924)
    by_class = [np.flatnonzero(truth == label) for label in classes]
    differences = []
    source_pred = classes[np.argmax(base, axis=1)]
    candidate_pred = classes[np.argmax(candidate, axis=1)]
    for _ in range(4000):
        indices = np.concatenate([rng.choice(group, len(group), replace=True)
                                  for group in by_class])
        differences.append(f1_score(truth[indices], candidate_pred[indices], labels=classes,
                                    average="macro", zero_division=0) -
                           f1_score(truth[indices], source_pred[indices], labels=classes,
                                    average="macro", zero_division=0))
    return {
        "corrected_trials": corrected, "new_errors": new_errors,
        "paired_accuracy_discordance_exact_p": float(binomtest(
            corrected, corrected + new_errors, 0.5).pvalue) if corrected + new_errors else 1.0,
        "paired_macro_f1_delta_bootstrap_95pct": np.quantile(
            differences, [0.025, 0.975]).tolist(),
        "bootstrap": "4000 class-stratified paired trial resamples, seed 20260924; descriptive interval",
    }


def run(root: Path):
    print("[1/4] load causal Song source S01/S02 and validation S03", flush=True)
    data = {sid: load_session(root / f"2026-09-18_{sid}", sid, "causal")
            for sid in ("S01", "S02", "S03")}
    source_batch = _join_batches([data["S01"], data["S02"]])
    source_labels = np.concatenate([data[sid]["hand"] for sid in ("S01", "S02")])
    f0 = LocalDetailFamily().fit(source_batch)
    spd = SpdTangentFamily().fit(source_batch)
    features = {name: {sid: family.transform(data[sid]["batch"]) for sid in data}
                for name, family in (("F0", f0), ("SPD", spd))}
    print("[2/4] fit source-only F0 and F0+SPD models", flush=True)
    models = {}
    for arm, names in (("F0", ("F0",)), ("F0_plus_SPD", ("F0", "SPD"))):
        x_train = np.concatenate([np.concatenate([features[name][sid] for name in names], axis=1)
                                  for sid in ("S01", "S02")])
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=0))
        model.fit(x_train, source_labels)
        models[arm] = (model, names)
    classes = models["F0"][0][-1].classes_
    if classes.tolist() != ["fist", "index_pinch", "neutral", "open_hand"]:
        raise ValueError("unexpected source hand-class order")

    def session_source_scores(sid):
        outcome, probabilities = {}, {}
        for arm, (model, names) in models.items():
            x = np.concatenate([features[name][sid] for name in names], axis=1)
            ids, truth, p = trial_probabilities(data[sid]["hand"], model.predict_proba(x),
                                                data[sid]["trial"], classes)
            outcome[arm] = metrics(truth, p, classes)
            probabilities[arm] = p
        return ids, truth, probabilities, outcome

    val_ids, val_truth, val_p, val_scores = session_source_scores("S03")
    print("[3/4] choose 1/2-block SPD fusion weights on S03", flush=True)
    weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    selected_weights, validation_calibrated = {}, {}
    for budget in (1, 2):
        personal, audit = spd_trial_probabilities(spd, data["S03"], budget, val_ids, classes)
        candidates = {str(weight): metrics(val_truth,
                        (1 - weight) * val_p["F0"] + weight * personal, classes)
                      for weight in weights}
        selected = max(weights, key=lambda weight: (candidates[str(weight)]["macro_f1"],
                                                    candidates[str(weight)]["accuracy"], -weight))
        selected_weights[str(budget)] = selected
        validation_calibrated[str(budget)] = {"selected_weight": selected,
                                                "anchor_alone": metrics(val_truth, personal, classes),
                                                "candidates": candidates, "audit": audit}

    # No S04 array is loaded until both fusion weights are fixed from S03.
    print("[4/4] evaluate locked arms on S04", flush=True)
    data["S04"] = load_session(root / "2026-09-18_S04", "S04", "causal")
    for name, family in (("F0", f0), ("SPD", spd)):
        features[name]["S04"] = family.transform(data["S04"]["batch"])
    final_ids, final_truth, final_p, final_scores = session_source_scores("S04")
    final_calibrated = {}
    for budget in (1, 2):
        personal, audit = spd_trial_probabilities(spd, data["S04"], budget, final_ids, classes)
        weight = selected_weights[str(budget)]
        mixed = (1 - weight) * final_p["F0"] + weight * personal
        final_calibrated[str(budget)] = {
            "weight_fixed_on_S03": weight,
            "anchor_alone": metrics(final_truth, personal, classes),
            "F0_plus_SPD_anchor": metrics(final_truth, mixed, classes),
            "paired_vs_F0": paired_changes(final_truth, final_p["F0"], mixed, classes),
            "audit": audit,
        }
    return {
        "status": "exploratory_one_person_one_day_native_spd_increment",
        "filter_mode": "causal_continuous",
        "source_sessions": ["S01", "S02"], "selection_session": "S03",
        "final_session": "S04", "classes": classes.tolist(),
        "source_hdf5_sha256": {sid: item["audit"]["sha256"] for sid, item in data.items()},
        "source_spd_reference_sha256": SpdTangentPersonalAnchor(spd).source_reference_sha256_,
        "protocol": "Source-only logistic F0 versus concat F0+F2c SPD at zero shot; F7 SPD personal trial prototypes from 1/2 pre-formal blocks per class, fixed probability mixture with source F0; S03 alone chooses weight; each formal trial one vote.",
        "source_zero_shot": {"validation": val_scores, "final": final_scores},
        "source_zero_shot_paired_vs_F0": {
            "validation": paired_changes(val_truth, val_p["F0"], val_p["F0_plus_SPD"], classes),
            "final": paired_changes(final_truth, final_p["F0"], final_p["F0_plus_SPD"], classes),
        },
        "personal_anchor": {"validation": validation_calibrated,
                            "selected_weights": selected_weights, "final": final_calibrated},
        "boundary": "S01-S03 failed formal collection readiness; S04 passed. All sessions are one participant/day, with cue labels rather than verified physiological onset. S04 has been inspected by earlier project studies, so this is not untouched confirmation; no physical device live or cross-person/day inference. F2c is a source-log tangent approximation and F7 anchor is not exact affine-invariant geodesic or historical RLCS."
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"selected_weights": result["personal_anchor"]["selected_weights"],
                      "S04_F0_F1": result["source_zero_shot"]["final"]["F0"]["macro_f1"],
                      "S04_F0_SPD_F1": result["source_zero_shot"]["final"]["F0_plus_SPD"]["macro_f1"]}),
          flush=True)
