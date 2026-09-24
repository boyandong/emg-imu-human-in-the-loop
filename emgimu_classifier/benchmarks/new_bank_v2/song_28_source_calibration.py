"""Source-session OOF temperature calibration for the frozen Song 28-state factorization."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.new_bank_v2.song_arm_calibration import joint_probabilities
from benchmarks.song_28_spd_increment_study import detailed_metrics
from benchmarks.song_real8_study import ARMS, HANDS, _join_batches, load_session, parse_label
from benchmarks.song_spd_increment_study import trial_probabilities
from emgimu.feature_bank.families import BodyContextFamily
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2, RingRelativeCovarianceV2, TraceCovarianceV2


ROOT = Path(__file__).resolve().parent
SOURCE = Path("E:/qxy/emg_meta/emg_meta/data/Song")
PROTOCOL_PATH = ROOT / "SONG_28_SOURCE_CAL_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def fit_branches(data: dict, source_ids: list[str], target_id: str) -> dict:
    batch = _join_batches([data[sid] for sid in source_ids])
    hand = np.concatenate([data[sid]["hand"] for sid in source_ids])
    arm = np.asarray([parse_label(str(label))[0] for sid in source_ids
                      for label in data[sid]["composite"]])
    families = (
        RestNoiseDetailV2(rest_label="neutral").fit(batch, hand),
        TraceCovarianceV2(shrinkage=0.05).fit(batch),
        RingRelativeCovarianceV2(shrinkage=0.05).fit(batch),
    )
    imu_family = BodyContextFamily().fit(batch)

    def hand_features(item: dict) -> np.ndarray:
        return np.concatenate([family.transform(item["batch"]) for family in families], axis=1)

    def classifier(features: np.ndarray, labels: np.ndarray):
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=20260924))
        model.fit(features, labels)
        return model

    source_hand = np.concatenate([hand_features(data[sid]) for sid in source_ids])
    source_arm = np.concatenate([imu_family.transform(data[sid]["batch"]) for sid in source_ids])
    hand_model = classifier(source_hand, hand)
    arm_model = classifier(source_arm, arm)
    hand_classes, arm_classes = hand_model[-1].classes_, arm_model[-1].classes_
    if set(hand_classes) != set(HANDS) or set(arm_classes) != set(ARMS):
        raise ValueError("source fold lacks hand or arm classes")
    target = data[target_id]
    trial = target["trial"]
    target_arm = np.asarray([parse_label(str(label))[0] for label in target["composite"]])
    hand_ids, hand_truth, hand_probability = trial_probabilities(
        target["hand"], hand_model.predict_proba(hand_features(target)), trial, hand_classes)
    arm_ids, arm_truth, arm_probability = trial_probabilities(
        target_arm, arm_model.predict_proba(imu_family.transform(target["batch"])),
        trial, arm_classes)
    if not np.array_equal(hand_ids, arm_ids):
        raise ValueError(f"hand and arm trial identities differ: {target_id}")
    return {"trial_ids": hand_ids, "hand_truth": hand_truth, "arm_truth": arm_truth,
            "hand_classes": hand_classes, "arm_classes": arm_classes,
            "hand_probability": hand_probability, "arm_probability": arm_probability}


def temperature_probability(probability: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    values = np.asarray(probability, dtype=np.float64)
    if (values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0)
            or not np.allclose(values.sum(axis=1), 1, atol=1e-9)):
        raise ValueError("probabilities must be finite normalized rows")
    return softmax(np.log(np.maximum(values, 1e-12)) / temperature, axis=1)


def select_temperature(truth: np.ndarray, probability: np.ndarray,
                       classes: np.ndarray, candidates: list[float]) -> tuple[float, list[dict]]:
    index = np.searchsorted(classes, truth)
    if np.any(index >= len(classes)) or not np.array_equal(classes[index], truth):
        raise ValueError("OOF truth not aligned with model classes")
    scores = []
    for candidate in candidates:
        calibrated = temperature_probability(probability, float(candidate))
        scores.append({"temperature": float(candidate),
                       "source_oof_trial_log_loss": float(-np.mean(np.log(
                           np.maximum(calibrated[np.arange(len(truth)), index], 1e-12))))})
    winner = min(scores, key=lambda row: (row["source_oof_trial_log_loss"],
                                          abs(row["temperature"] - 1.0), row["temperature"]))
    return winner["temperature"], scores


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def run(source: Path = SOURCE) -> dict:
    prior_path = ROOT / "SONG_ARM_CAL_RESULTS.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    old28 = json.loads((ROOT / "SONG_28_RESULTS.json").read_text(encoding="utf-8"))
    import scipy, sklearn, sys
    runtime = {"python": sys.version.split()[0], "numpy": np.__version__,
               "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    if (runtime != prior["runtime_versions"]
            or PROTOCOL["source_oof_folds"] != [["S01", "S02"], ["S02", "S01"]]
            or PROTOCOL["comparison"] != ["source_factorized_uncalibrated",
                                           "source_factorized_source_oof_temperature"]):
        raise ValueError("frozen source-calibration protocol or runtime differs")
    data = {sid: load_session(source / f"2026-09-18_{sid}", sid, "causal")
            for sid in ("S01", "S02", "S03", "S04")}
    hashes = {sid: data[sid]["audit"]["sha256"] for sid in data}
    if hashes != prior["source_hdf5_sha256"]:
        raise ValueError("source recordings changed")
    oof = []; oof_rows = []
    for source_id, heldout_id in PROTOCOL["source_oof_folds"]:
        fit = fit_branches(data, [source_id], heldout_id)
        oof.append(fit)
        for trial, hand, arm, ph, pa in zip(fit["trial_ids"], fit["hand_truth"], fit["arm_truth"],
                                            fit["hand_probability"], fit["arm_probability"]):
            oof_rows.append({"fit_session": source_id, "heldout_session": heldout_id,
                             "trial_id": str(trial), "hand_truth": str(hand), "arm_truth": str(arm),
                             **{f"hand_p_{name}": float(value)
                                for name, value in zip(fit["hand_classes"], ph)},
                             **{f"arm_p_{name}": float(value)
                                for name, value in zip(fit["arm_classes"], pa)}})
        print(f"Song 28 source OOF: fit {source_id}, scored {len(fit['trial_ids'])} {heldout_id} trials", flush=True)
    hand_classes, arm_classes = oof[0]["hand_classes"], oof[0]["arm_classes"]
    if any(not np.array_equal(item["hand_classes"], hand_classes)
           or not np.array_equal(item["arm_classes"], arm_classes) for item in oof):
        raise ValueError("source fold class order differs")
    candidates = PROTOCOL["temperature_candidates"]
    hand_temperature, hand_scores = select_temperature(
        np.concatenate([item["hand_truth"] for item in oof]),
        np.concatenate([item["hand_probability"] for item in oof]), hand_classes, candidates)
    arm_temperature, arm_scores = select_temperature(
        np.concatenate([item["arm_truth"] for item in oof]),
        np.concatenate([item["arm_probability"] for item in oof]), arm_classes, candidates)
    print(f"Song 28 source OOF temperatures: hand={hand_temperature}, arm={arm_temperature}", flush=True)
    result = {"protocol": PROTOCOL, "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
              "runtime_versions": runtime, "source_hdf5_sha256": hashes,
              "reference_arm_results_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
              "source_oof_trial_rows": len(oof_rows),
              "source_oof": {"hand": {"selected_temperature": hand_temperature, "scores": hand_scores},
                             "arm": {"selected_temperature": arm_temperature, "scores": arm_scores}},
              "sessions": {}}
    full_rows = []
    joint_classes = np.asarray(old28["classes"])
    baseline = { (row["phase"], row["trial_id"]): row for row in
                csv.DictReader((ROOT / "SONG_ARM_CAL_TRIAL_PREDICTIONS.csv").open(newline="", encoding="utf-8"))
                if row["arm"] == "source_arm_x_same_hand"}
    max_baseline_error = 0.0
    for phase, sid in (("validation", "S03"), ("final", "S04")):
        fit = fit_branches(data, PROTOCOL["source_sessions"], sid)
        if (not np.array_equal(fit["hand_classes"], hand_classes)
                or not np.array_equal(fit["arm_classes"], arm_classes)):
            raise ValueError("full source class order differs")
        truth = np.asarray([f"{arm}_{hand}" for arm, hand in
                            zip(fit["arm_truth"], fit["hand_truth"])])
        probabilities = {
            "source_factorized_uncalibrated": joint_probabilities(
                fit["hand_probability"], fit["arm_probability"], hand_classes, arm_classes, joint_classes),
            "source_factorized_source_oof_temperature": joint_probabilities(
                temperature_probability(fit["hand_probability"], hand_temperature),
                temperature_probability(fit["arm_probability"], arm_temperature),
                hand_classes, arm_classes, joint_classes),
        }
        if not np.array_equal(probabilities[PROTOCOL["comparison"][0]].argmax(axis=1),
                              probabilities[PROTOCOL["comparison"][1]].argmax(axis=1)):
            raise ValueError("temperature unexpectedly changed a factorized hard decision")
        for trial, label, p in zip(fit["trial_ids"], truth, probabilities[PROTOCOL["comparison"][0]]):
            reference = baseline.get((phase, str(trial)))
            if reference is None or reference["label"] != label:
                raise ValueError("frozen factorized baseline trial missing")
            max_baseline_error = max(max_baseline_error, max(
                abs(float(reference[f"p_{name}"]) - float(value)) for name, value in zip(joint_classes, p)))
        if max_baseline_error > 1e-9:
            raise ValueError(f"frozen factorized baseline probability changed: {max_baseline_error}")
        metrics = {}
        for name, probability in probabilities.items():
            metrics[name] = detailed_metrics(truth, probability, joint_classes)
            for trial, label, values in zip(fit["trial_ids"], truth, probability):
                full_rows.append({"phase": phase, "session": sid, "arm": name,
                                  "trial_id": str(trial), "label": str(label),
                                  **{f"p_{klass}": float(p) for klass, p in zip(joint_classes, values)}})
            print(f"{sid} {name}: LL={metrics[name]['log_loss']:.4f} F1={metrics[name]['macro_f1']:.4f}", flush=True)
        result["sessions"][phase] = {"session": sid, "trial_ids": fit["trial_ids"].tolist(),
                                     "arms": metrics}
    result["max_uncalibrated_baseline_probability_error"] = max_baseline_error
    result["prediction_rows"] = len(full_rows)
    write_csv(ROOT / "SONG_28_SOURCE_CAL_OOF.csv", oof_rows)
    write_csv(ROOT / "SONG_28_SOURCE_CAL_TRIAL_PREDICTIONS.csv", full_rows)
    (ROOT / "SONG_28_SOURCE_CAL_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
