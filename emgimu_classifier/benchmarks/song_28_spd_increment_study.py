"""Matched causal F2c SPD increment to Song's fixed 28-state F0+IMU endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.song_real8_study import ARMS, HANDS, _join_batches, load_session, parse_label
from benchmarks.song_spd_increment_study import metrics, trial_probabilities
from emgimu.feature_bank.families import BodyContextFamily, LocalDetailFamily, SpdTangentFamily


def detailed_metrics(truth, probability, classes):
    result = metrics(truth, probability, classes)
    prediction = classes[np.argmax(probability, axis=1)]
    true_parts = [parse_label(label) for label in truth]
    predicted_parts = [parse_label(label) for label in prediction]
    true_arm, true_hand = map(np.asarray, zip(*true_parts))
    predicted_arm, predicted_hand = map(np.asarray, zip(*predicted_parts))
    result["arm_accuracy"] = float(np.mean(true_arm == predicted_arm))
    result["hand_accuracy"] = float(np.mean(true_hand == predicted_hand))
    result["arm_support"] = {arm: int(np.sum(true_arm == arm)) for arm in ARMS}
    result["hand_support"] = {hand: int(np.sum(true_hand == hand)) for hand in HANDS}
    result["per_arm_joint_accuracy"] = {
        arm: float(np.mean(prediction[true_arm == arm] == truth[true_arm == arm]))
        if np.any(true_arm == arm) else None for arm in ARMS}
    result["per_hand_joint_accuracy"] = {
        hand: float(np.mean(prediction[true_hand == hand] == truth[true_hand == hand]))
        if np.any(true_hand == hand) else None for hand in HANDS}
    return result


def paired_changes(truth, base, candidate, classes):
    baseline_correct = classes[np.argmax(base, axis=1)] == truth
    candidate_correct = classes[np.argmax(candidate, axis=1)] == truth
    gain = int(np.sum(~baseline_correct & candidate_correct))
    loss = int(np.sum(baseline_correct & ~candidate_correct))
    return {"corrected_trials": gain, "new_errors": loss,
            "paired_accuracy_discordance_exact_p": float(binomtest(gain, gain + loss, 0.5).pvalue)
            if gain + loss else 1.0}


def run(root: Path):
    baseline_file = Path(__file__).resolve().parent / "song_real8/CAUSAL_RESULTS.json"
    baseline_record = json.loads(baseline_file.read_text(encoding="utf-8"))
    if baseline_record["filter_mode"] != "causal":
        raise ValueError("saved 28-state baseline does not use causal filtering")
    saved_runtime_file = Path(__file__).resolve().parent / "song_real8/SPD_28_STATE_RESULTS.json"
    if saved_runtime_file.is_file():
        saved_runtime = json.loads(saved_runtime_file.read_text(encoding="utf-8"))["runtime_versions"]
        current_runtime = {"python": sys.version.split()[0], "numpy": np.__version__,
                           "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
        if current_runtime != saved_runtime:
            raise RuntimeError(f"saved 28-state comparison requires runtime {saved_runtime}; "
                               f"current runtime is {current_runtime}")
    print("[1/3] load causal Song source S01/S02 and validation S03", flush=True)
    data = {sid: load_session(root / f"2026-09-18_{sid}", sid, "causal")
            for sid in ("S01", "S02", "S03")}
    source = _join_batches([data["S01"], data["S02"]])
    families = {"F0": LocalDetailFamily().fit(source),
                "IMU": BodyContextFamily().fit(source),
                "SPD": SpdTangentFamily().fit(source)}
    features = {name: {sid: family.transform(item["batch"]) for sid, item in data.items()}
                for name, family in families.items()}
    arms = {"F0_plus_IMU": ("F0", "IMU"),
            "F0_plus_IMU_plus_SPD": ("F0", "IMU", "SPD")}
    source_labels = np.concatenate([data[sid]["composite"] for sid in ("S01", "S02")])
    models = {}
    for name, family_names in arms.items():
        x = np.concatenate([np.concatenate([features[family][sid] for family in family_names], axis=1)
                            for sid in ("S01", "S02")])
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=0))
        model.fit(x, source_labels)
        models[name] = model
    classes = models["F0_plus_IMU"][-1].classes_
    if len(classes) != 28 or set(classes) != {f"{arm}_{hand}" for arm in ARMS for hand in HANDS}:
        raise ValueError("source model does not cover all 28 hand×arm labels")
    if not np.array_equal(classes, models["F0_plus_IMU_plus_SPD"][-1].classes_):
        raise ValueError("source models have different class order")

    def evaluate(sid):
        phase = {}
        trial_ids = truth = None
        for name, family_names in arms.items():
            x = np.concatenate([features[family][sid] for family in family_names], axis=1)
            ids, labels, p = trial_probabilities(data[sid]["composite"],
                                                 models[name].predict_proba(x),
                                                 data[sid]["trial"], classes)
            if trial_ids is None:
                trial_ids, truth = ids, labels
            elif not np.array_equal(ids, trial_ids) or not np.array_equal(labels, truth):
                raise ValueError("matched 28-state arms use different trials or labels")
            phase[name] = {"metrics": detailed_metrics(labels, p, classes), "probabilities": p}
        paired = paired_changes(truth, phase["F0_plus_IMU"]["probabilities"],
                                phase["F0_plus_IMU_plus_SPD"]["probabilities"], classes)
        return {"trial_count": len(trial_ids),
                "scores": {name: phase[name]["metrics"] for name in arms},
                "paired_vs_F0_plus_IMU": paired}

    print("[2/3] compare fixed source models on S03", flush=True)
    validation = evaluate("S03")
    for metric in ("accuracy", "macro_f1"):
        if not np.isclose(validation["scores"]["F0_plus_IMU"][metric],
                          baseline_record["secondary_28_state"]["validation"][metric],
                          rtol=0, atol=1e-12):
            raise ValueError(f"current S03 baseline differs from saved causal run: {metric}")
    # No S04 array is read until both source models and validation are fixed.
    print("[3/3] evaluate both fixed models on S04", flush=True)
    data["S04"] = load_session(root / "2026-09-18_S04", "S04", "causal")
    for name, family in families.items():
        features[name]["S04"] = family.transform(data["S04"]["batch"])
    final = evaluate("S04")
    for metric in ("accuracy", "macro_f1"):
        if not np.isclose(final["scores"]["F0_plus_IMU"][metric],
                          baseline_record["secondary_28_state"]["test"][metric],
                          rtol=0, atol=1e-12):
            raise ValueError(f"current S04 baseline differs from saved causal run: {metric}")
    saved_hashes = {item["session"]: item["sha256"] for item in baseline_record["source_audit"]}
    if saved_hashes != {sid: item["audit"]["sha256"] for sid, item in data.items()}:
        raise ValueError("source recording digests differ from saved causal baseline")
    return {
        "status": "exploratory_one_person_one_day_28_state_spd_increment",
        "source_sessions": ["S01", "S02"], "validation_session": "S03", "final_session": "S04",
        "filter_mode": "causal_continuous", "window_samples": 50, "imu_window_samples": 22,
        "source_hdf5_sha256": {sid: item["audit"]["sha256"] for sid, item in data.items()},
        "saved_baseline_sha256": hashlib.sha256(baseline_file.read_bytes()).hexdigest(),
        "runtime_versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                             "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "feature_dimensions": {name: len(family.feature_names) for name, family in families.items()},
        "protocol": "C=1 balanced logistic and source-only StandardScaler; fixed F0+real-IMU baseline versus same matched source model with F2c SPD tangent appended; S03 validation before S04 final; one vote per formal trial; no personal calibration or final-based tuning.",
        "validation": validation, "final": final,
        "boundary": "S01-S03 failed formal collection readiness; S04 passed, but all sessions are one participant/day and S04 was previously inspected. Cued stable-trial 28-state classification does not establish continuous live recognition, physiological onset, cross-person/day transfer or historical RLCS equivalence."
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"validation_F1": {key: value["macro_f1"] for key, value in
                                         result["validation"]["scores"].items()},
                      "final_F1": {key: value["macro_f1"] for key, value in
                                   result["final"]["scores"].items()},
                      "final_paired": result["final"]["paired_vs_F0_plus_IMU"]}), flush=True)
