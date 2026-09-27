"""One frozen source-only signed-device-axis arm candidate for Song."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.new_bank_v2.song_arm_calibration import joint_probabilities
from benchmarks.song_28_spd_increment_study import detailed_metrics
from benchmarks.song_real8_study import ARMS, HANDS, _join_batches, load_session, parse_label
from benchmarks.song_spd_increment_study import trial_probabilities
from emgimu.feature_bank.families import BodyContextFamily


ROOT = Path(__file__).resolve().parent
SOURCE = Path("E:/qxy/emg_meta/emg_meta/data/Song")
PROTOCOL_PATH = ROOT / "SONG_ARM_SIGNED_PROTOCOL.json"
BASELINE_PATH = ROOT / "SONG_ARM_CAL_TRIAL_PREDICTIONS.csv"


def signed_features(imu: np.ndarray) -> np.ndarray:
    """Six causal device-axis quantities from one 22-sample IMU window."""
    values = np.asarray(imu, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (22, 6) or not np.isfinite(values).all():
        raise ValueError("expected finite [window,22,6] recorded IMU")
    gyro_mean = values[:, :, 3:6].mean(axis=1)
    accel_change = values[:, -5:, :3].mean(axis=1) - values[:, :5, :3].mean(axis=1)
    return np.column_stack((gyro_mean, accel_change))


def frozen_baseline(classes: np.ndarray) -> dict[str, dict[str, tuple[str, np.ndarray]]]:
    result = {"validation": {}, "final": {}}
    with BASELINE_PATH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["arm"] != "source_arm_x_same_hand":
                continue
            phase, trial = row["phase"], row["trial_id"]
            if phase not in result or trial in result[phase]:
                raise ValueError(f"unexpected frozen baseline identity: {phase}/{trial}")
            p = np.asarray([float(row[f"p_{label}"]) for label in classes])
            if not np.isfinite(p).all() or not np.isclose(p.sum(), 1, atol=1e-12):
                raise ValueError(f"invalid frozen probability: {phase}/{trial}")
            result[phase][trial] = (row["label"], p)
    return result


def branch_marginals(joint: np.ndarray, classes: np.ndarray,
                     hand_classes: np.ndarray, arm_classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hand = np.asarray([joint[[parse_label(str(label))[1] == name for label in classes]].sum()
                       for name in hand_classes])
    arm = np.asarray([joint[[parse_label(str(label))[0] == name for label in classes]].sum()
                      for name in arm_classes])
    return hand, arm


def run(source: Path = SOURCE) -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if (protocol["source_train_sessions"] != ["S01", "S02"]
            or protocol["validation_session"] != "S03"
            or protocol["descriptive_final_session"] != "S04"):
        raise ValueError("frozen split changed")
    prior = json.loads((ROOT / "SONG_ARM_CAL_RESULTS.json").read_text(encoding="utf-8"))
    original = json.loads((ROOT / "SONG_28_RESULTS.json").read_text(encoding="utf-8"))
    if prior["source_hdf5_sha256"] != original["source_hdf5_sha256"]:
        raise ValueError("baseline source hashes disagree")
    import sys, scipy, sklearn
    runtime = {"python": sys.version.split()[0], "numpy": np.__version__,
               "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    if runtime != prior["runtime_versions"]:
        raise RuntimeError(f"frozen baseline runtime differs: {runtime}")
    data = {sid: load_session(source / f"2026-09-18_{sid}", sid, "causal")
            for sid in ("S01", "S02", "S03", "S04")}
    if {sid: item["audit"]["sha256"] for sid, item in data.items()} != prior["source_hdf5_sha256"]:
        raise ValueError("native recording changed")
    family = BodyContextFamily().fit(_join_batches([data["S01"], data["S02"]]))
    feature = {sid: np.column_stack((family.transform(item["batch"]),
                                    signed_features(item["batch"].imu)))
               for sid, item in data.items()}
    arm_train = np.asarray([parse_label(str(label))[0] for sid in ("S01", "S02")
                            for label in data[sid]["composite"]])
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=20260924))
    model.fit(np.concatenate([feature["S01"], feature["S02"]]), arm_train)
    arm_classes = model[-1].classes_
    hand_classes = np.asarray(sorted(HANDS))
    joint_classes = np.asarray(original["classes"])
    if set(arm_classes) != set(ARMS):
        raise ValueError("incomplete source arm labels")
    baseline = frozen_baseline(joint_classes)
    result = {"protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
              "baseline_sha256": hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest(),
              "source_hdf5_sha256": prior["source_hdf5_sha256"], "runtime_versions": runtime,
              "arm_classes": arm_classes.tolist(), "joint_classes": joint_classes.tolist(),
              "sessions": {}}
    rows = []
    for phase, sid in (("validation", "S03"), ("final_descriptive", "S04")):
        item = data[sid]
        arm_truth_window = np.asarray([parse_label(str(label))[0] for label in item["composite"]])
        ids, arm_truth, candidate_arm = trial_probabilities(
            arm_truth_window, model.predict_proba(feature[sid]), item["trial"], arm_classes)
        phase_baseline = baseline["validation" if sid == "S03" else "final"]
        if set(map(str, ids)) != set(phase_baseline):
            raise ValueError(f"frozen trial set differs: {sid}")
        truth_joint, baseline_joint, candidate_joint = [], [], []
        for trial, actual_arm, arm_p in zip(ids, arm_truth, candidate_arm):
            trial = str(trial)
            truth, frozen = phase_baseline[trial]
            if parse_label(truth)[0] != actual_arm:
                raise ValueError(f"frozen arm truth differs: {trial}")
            hand_p, frozen_arm = branch_marginals(frozen, joint_classes, hand_classes, arm_classes)
            rebuilt = joint_probabilities(hand_p[None, :], frozen_arm[None, :],
                                          hand_classes, arm_classes, joint_classes)[0]
            if not np.allclose(rebuilt, frozen, rtol=0, atol=2e-15):
                raise ValueError(f"baseline does not factorize: {trial}")
            candidate = joint_probabilities(hand_p[None, :], arm_p[None, :],
                                            hand_classes, arm_classes, joint_classes)[0]
            truth_joint.append(truth); baseline_joint.append(frozen); candidate_joint.append(candidate)
            rows.append({"phase": phase, "session": sid, "trial_id": trial, "truth": truth,
                         "truth_arm": actual_arm,
                         "baseline_arm": str(arm_classes[int(np.argmax(frozen_arm))]),
                         "candidate_arm": str(arm_classes[int(np.argmax(arm_p))]),
                         "baseline_joint": str(joint_classes[int(np.argmax(frozen))]),
                         "candidate_joint": str(joint_classes[int(np.argmax(candidate))]),
                         **{f"p_baseline_{label}": float(p) for label, p in zip(joint_classes, frozen)},
                         **{f"p_candidate_{label}": float(p) for label, p in zip(joint_classes, candidate)}})
        baseline_score = detailed_metrics(np.asarray(truth_joint), np.stack(baseline_joint), joint_classes)
        candidate_score = detailed_metrics(np.asarray(truth_joint), np.stack(candidate_joint), joint_classes)
        expected = prior["sessions"]["validation" if sid == "S03" else "final"]["arms"]["source_arm_x_same_hand"]
        for metric in ("accuracy", "macro_f1", "log_loss"):
            if not np.isclose(baseline_score[metric], expected[metric], atol=1e-12, rtol=0):
                raise ValueError(f"baseline metric mismatch: {sid}/{metric}")
        session_rows = [row for row in rows if row["session"] == sid]
        arm_recall = {arm: {"support": int(sum(row["truth_arm"] == arm for row in session_rows)),
                            "baseline": float(np.mean([row["baseline_arm"] == arm for row in session_rows
                                                        if row["truth_arm"] == arm])),
                            "candidate": float(np.mean([row["candidate_arm"] == arm for row in session_rows
                                                         if row["truth_arm"] == arm]))}
                      for arm in arm_classes}
        result["sessions"][phase] = {"session": sid, "trials": len(ids),
                                      "baseline": baseline_score, "candidate": candidate_score,
                                      "arm_recall": arm_recall,
                                      "baseline_arm_correct": int(sum(row["baseline_arm"] == row["truth_arm"] for row in session_rows)),
                                      "candidate_arm_correct": int(sum(row["candidate_arm"] == row["truth_arm"] for row in session_rows)),
                                      "candidate_corrected_arm_errors": int(sum(row["candidate_arm"] == row["truth_arm"] and row["baseline_arm"] != row["truth_arm"] for row in session_rows)),
                                      "candidate_new_arm_errors": int(sum(row["candidate_arm"] != row["truth_arm"] and row["baseline_arm"] == row["truth_arm"] for row in session_rows))}
        print(f"{sid}: baseline joint F1={baseline_score['macro_f1']:.4f}, signed F1={candidate_score['macro_f1']:.4f}", flush=True)
    csv_path = ROOT / "SONG_ARM_SIGNED_TRIAL_PREDICTIONS.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result["prediction_rows"] = len(rows)
    result["prediction_csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    (ROOT / "SONG_ARM_SIGNED_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
