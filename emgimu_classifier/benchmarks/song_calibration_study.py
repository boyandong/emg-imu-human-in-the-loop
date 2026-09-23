"""Prespecified 0/1/2-shot calibration comparison on Song real-device sessions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.song_real8_study import HANDS, ROLES, _join_batches, _trial_metrics, load_session
from emgimu.feature_bank.families import LocalDetailFamily


def target_calibration(data, family, budget):
    if budget not in (1, 2):
        raise ValueError("budget must be one or two blocks per class")
    all_features = family.transform(data["calibration_batch"])
    available = data["calibration_shot"] <= budget
    labels = data["calibration_hand"][available]
    if set(labels) != set(HANDS) or any(np.count_nonzero(labels == hand) != 3 * budget for hand in HANDS):
        raise ValueError("calibration budget is incomplete or unbalanced")
    return all_features[available], labels


def predict_calibrated(model, formal_features, calibration_features, calibration_labels,
                       source_neutral, beta, weight):
    """Source-logistic plus target prototypes; all states fitted before formal trial scoring."""
    if not (0 <= beta <= 1 and 0 <= weight <= 1):
        raise ValueError("beta and weight must be in [0,1]")
    classes = model[-1].classes_
    target_neutral = calibration_features[calibration_labels == "neutral"].mean(axis=0)
    shift = beta * (target_neutral - source_neutral)
    adjusted = formal_features - shift
    source_probability = model.predict_proba(adjusted)
    if weight == 0:
        return source_probability
    scale = model.named_steps["standardscaler"]
    z_formal = scale.transform(adjusted)
    z_calibration = scale.transform(calibration_features - shift)
    prototypes = np.stack([z_calibration[calibration_labels == hand].mean(axis=0) for hand in classes])
    distance = np.mean((z_formal[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
    personal_probability = softmax(-distance, axis=1)
    return (1 - weight) * source_probability + weight * personal_probability


def paired_trial_audit(labels, trial_ids, base_probability, calibrated_probability, classes):
    true, baseline, calibrated = [], [], []
    for trial in dict.fromkeys(trial_ids):
        index = np.flatnonzero(trial_ids == trial)
        if len(set(labels[index])) != 1:
            raise ValueError(f"inconsistent trial labels: {trial}")
        true.append(labels[index[0]])
        baseline.append(classes[np.argmax(base_probability[index].mean(axis=0))])
        calibrated.append(classes[np.argmax(calibrated_probability[index].mean(axis=0))])
    true, baseline, calibrated = map(np.asarray, (true, baseline, calibrated))
    base_correct, cal_correct = baseline == true, calibrated == true
    gain = int(np.sum(~base_correct & cal_correct))
    loss = int(np.sum(base_correct & ~cal_correct))
    rng = np.random.default_rng(20260924)
    by_class = [np.flatnonzero(true == hand) for hand in classes]
    differences = []
    for _ in range(4000):
        resample = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in by_class])
        differences.append(f1_score(true[resample], calibrated[resample], labels=classes,
                                    average="macro", zero_division=0) -
                           f1_score(true[resample], baseline[resample], labels=classes,
                                    average="macro", zero_division=0))
    return {"trial_count": len(true), "corrected_trials": gain, "new_errors": loss,
            "changed_but_both_correct_or_wrong": int(np.sum((baseline != calibrated) & (base_correct == cal_correct))),
            "paired_macro_f1_delta_bootstrap_95pct": np.quantile(differences, [0.025, 0.975]).tolist(),
            "paired_accuracy_discordance_exact_p": float(binomtest(gain, gain + loss, 0.5).pvalue) if gain + loss else 1.0,
            "bootstrap": "4000 class-stratified trial resamples, seed 20260924; descriptive interval"}


def run(root: Path, filter_mode: str = "zero_phase"):
    data = {sid: load_session(root / f"2026-09-18_{sid}", sid, filter_mode) for sid in ROLES}
    family = LocalDetailFamily().fit(_join_batches([data["S01"], data["S02"]]))
    formal = {sid: family.transform(item["batch"]) for sid, item in data.items()}
    train_x = np.concatenate([formal[sid] for sid in ("S01", "S02")])
    train_y = np.concatenate([data[sid]["hand"] for sid in ("S01", "S02")])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=0))
    model.fit(train_x, train_y)
    source_neutral = np.concatenate([
        family.transform(data[sid]["calibration_batch"])[data[sid]["calibration_hand"] == "neutral"]
        for sid in ("S01", "S02")
    ]).mean(axis=0)
    classes = model[-1].classes_
    base = {}
    base_probability = {}
    for sid in ("S03", "S04"):
        base_probability[sid] = model.predict_proba(formal[sid])
        base[sid] = _trial_metrics(data[sid]["hand"], base_probability[sid], data[sid]["trial"], classes)
    outcome = {"status": "exploratory_actual_calibration_blocks_not_deployment_validated",
               "filter_mode": filter_mode,
               "source_hashes": {sid: data[sid]["audit"]["sha256"] for sid in ROLES},
               "source_roles": ROLES, "classes": classes.tolist(),
               "zero_shot": {"validation": base["S03"], "test": base["S04"]},
               "method": "F0 source-trained logistic; optional target neutral feature shift and standardized target class prototypes; S03 chooses beta/weight before S04 test",
               "calibration": {}}
    for budget in (1, 2):
        print(f"calibration budget {budget} block(s)/class: S03 select, S04 test", flush=True)
        val_x, val_y = target_calibration(data["S03"], family, budget)
        candidates = []
        for beta in (0.0, 0.5, 1.0):
            for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                proba = predict_calibrated(model, formal["S03"], val_x, val_y, source_neutral, beta, weight)
                metrics = _trial_metrics(data["S03"]["hand"], proba, data["S03"]["trial"], classes)
                candidates.append((metrics["macro_f1"], metrics["accuracy"], -weight, -beta, beta, weight, metrics))
        best = max(candidates)
        beta, weight = best[4], best[5]
        test_x, test_y = target_calibration(data["S04"], family, budget)
        test_proba = predict_calibrated(model, formal["S04"], test_x, test_y, source_neutral, beta, weight)
        test = _trial_metrics(data["S04"]["hand"], test_proba, data["S04"]["trial"], classes)
        outcome["calibration"][str(budget)] = {
            "blocks_per_class": budget, "calibration_blocks": 4 * budget,
            "calibration_windows": int(len(test_y)),
            "nominal_cue_duration_seconds": 14 if budget == 1 else 28,
            "S04_first_to_last_selected_block_elapsed_seconds": data["S04"]["calibration_elapsed_seconds"][str(budget)],
            "selected_beta": beta, "selected_prototype_weight": weight,
            "validation": best[6], "test": test,
            "paired_test_audit": paired_trial_audit(data["S04"]["hand"], data["S04"]["trial"],
                                                  base_probability["S04"], test_proba, classes),
            "test_delta_macro_f1_vs_zero": test["macro_f1"] - base["S04"]["macro_f1"],
            "test_delta_accuracy_vs_zero": test["accuracy"] - base["S04"]["accuracy"]}
    outcome["limitations"] = [
        "same person/day; no cross-day or cross-user inference",
        "S01-S03 collection readiness failed; use is exploratory",
        "calibration blocks precede formal trials but cue labels are not physiological onset verification",
        "offline trial-level stable-window scoring only; causal filtering does not validate the live UI/latency path",
        "S03 selected coefficients; S04 data used only for prescribed calibration and final scoring",
        "calibration cue durations exclude device preparation and UI operator time; observed span includes interleaved protocol blocks and uses nominal EMG sample rate"]
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--filter-mode", choices=("zero_phase", "causal"), default="zero_phase")
    args = parser.parse_args()
    result = run(args.source, args.filter_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
