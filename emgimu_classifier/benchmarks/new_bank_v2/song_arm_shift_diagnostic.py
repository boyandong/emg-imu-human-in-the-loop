"""Post-hoc, label-aware diagnosis of guided versus formal Song IMU descriptors.

This is an audit of the frozen arm-calibration comparison, not a model-selection
or calibration procedure. Formal labels are used only to explain its errors.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from benchmarks.song_real8_study import ARMS, load_session, parse_label
from benchmarks.new_bank_v2.song_arm_calibration import (
    ROOT, SOURCE, PROTOCOL, calibration_imu_windows, imu_features, prototypes,
)
from emgimu.feature_bank.families import BodyContextFamily


def trial_arm_decisions(path: Path, phase: str, joint_classes: list[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["phase"] != phase:
                continue
            trial = row["trial_id"]
            arm = row["arm"]
            predicted = joint_classes[int(np.argmax([float(row[f"p_{label}"])
                                                     for label in joint_classes]))]
            fields = result.setdefault(trial, {})
            if arm in fields or ("truth" in fields and fields["truth"] != row["label"]):
                raise ValueError(f"duplicate arm or inconsistent saved truth: {phase}/{trial}")
            fields["truth"] = row["label"]
            fields[arm] = parse_label(predicted)[0]
    return result


def normalized_direction(reference: np.ndarray, observed: np.ndarray, scale: np.ndarray) -> dict:
    a, b = reference / scale, observed / scale
    a_norm, b_norm = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return {"guided_norm": a_norm, "formal_norm": b_norm,
            "cosine": float(np.dot(a, b) / (a_norm * b_norm)) if a_norm and b_norm else None,
            "distance": float(np.linalg.norm(a - b))}


def run(source: Path = SOURCE) -> dict:
    prior = json.loads((ROOT / "SONG_ARM_CAL_RESULTS.json").read_text(encoding="utf-8"))
    old28 = json.loads((ROOT / "SONG_28_RESULTS.json").read_text(encoding="utf-8"))
    if prior["protocol"] != PROTOCOL or prior["source_hdf5_sha256"] != old28["source_hdf5_sha256"]:
        raise ValueError("frozen arm study or source hashes changed")
    classes = list(old28["classes"])
    arm_classes = np.asarray(sorted(ARMS))
    scale = np.asarray(prior["source_arm_coordinate_scale"])
    if np.any(scale <= 0) or len(scale) != 13:
        raise ValueError("invalid frozen source coordinate scale")
    family = BodyContextFamily().fit(load_session(source / "2026-09-18_S01", "S01", "causal")["batch"])
    result = {"status": "posthoc_diagnostic_only", "source_hdf5_sha256": {},
              "reference_arm_results_sha256": None,
              "reference_arm_probabilities_sha256": hashlib.sha256(
                  (ROOT / "SONG_ARM_CAL_TRIAL_PREDICTIONS.csv").read_bytes()).hexdigest(),
              "sessions": {}}
    result["reference_arm_results_sha256"] = hashlib.sha256(
        (ROOT / "SONG_ARM_CAL_RESULTS.json").read_bytes()).hexdigest()
    for phase, sid in (("validation", "S03"), ("final", "S04")):
        data = load_session(source / f"2026-09-18_{sid}", sid, "causal")
        if data["audit"]["sha256"] != prior["source_hdf5_sha256"][sid]:
            raise ValueError(f"source changed: {sid}")
        result["source_hdf5_sha256"][sid] = data["audit"]["sha256"]
        blocks, audit = calibration_imu_windows(source / f"2026-09-18_{sid}", sid)
        if audit != prior["calibration_blocks"][sid]:
            raise ValueError(f"calibration intervals changed: {sid}")
        rest, guided = prototypes(family, blocks, arm_classes)
        features = imu_features(family, data["batch"].imu)
        trial_ids = data["trial"]
        truth_window = np.asarray([parse_label(str(label))[0] for label in data["composite"]])
        decisions = trial_arm_decisions(ROOT / "SONG_ARM_CAL_TRIAL_PREDICTIONS.csv", phase, classes)
        trial_features = []
        trial_truth = []
        source_pred = []
        guide_pred = []
        for trial in np.unique(trial_ids):
            selected = trial_ids == trial
            true = np.unique(truth_window[selected])
            if len(true) != 1 or str(trial) not in decisions:
                raise ValueError(f"trial identity/arm label mismatch: {trial}")
            arm_decisions = decisions[str(trial)]
            joint_truth = np.unique(data["composite"][selected])
            if (len(joint_truth) != 1 or arm_decisions["truth"] != joint_truth[0]
                    or set(arm_decisions) != {*PROTOCOL["arms"], "truth"}):
                raise ValueError(f"missing saved arm decisions: {trial}")
            trial_features.append(np.median(features[selected], axis=0))
            trial_truth.append(true[0])
            source_pred.append(arm_decisions["source_arm_x_same_hand"])
            guide_pred.append(arm_decisions["guided_cal_arm_x_same_hand"])
        if (len(trial_truth) != len(prior["sessions"][phase]["trial_ids"])
                or set(decisions) != set(prior["sessions"][phase]["trial_ids"])):
            raise ValueError(f"formal trial count differs from frozen study: {sid}")
        trial_features = np.stack(trial_features)
        trial_truth = np.asarray(trial_truth)
        formal = np.stack([np.median(trial_features[trial_truth == arm], axis=0)
                           for arm in arm_classes])
        formal_relative = formal - rest
        d = np.linalg.norm((formal_relative[:, None, :] - guided[None, :, :]) /
                           scale[None, None, :], axis=2)
        true_index = np.arange(len(arm_classes))
        nearest = np.argmin(d, axis=1)
        per_arm = {}
        for i, arm in enumerate(arm_classes):
            ids = trial_truth == arm
            per_arm[str(arm)] = {
                "trials": int(np.sum(ids)),
                "source_correct": int(np.sum(np.asarray(source_pred)[ids] == arm)),
                "guided_correct": int(np.sum(np.asarray(guide_pred)[ids] == arm)),
                "guided_prediction_counts": dict(Counter(np.asarray(guide_pred)[ids].tolist())),
                "guided_corrected_source_arm_errors": int(np.sum(
                    (np.asarray(guide_pred)[ids] == arm) & (np.asarray(source_pred)[ids] != arm))),
                "guided_new_arm_errors": int(np.sum(
                    (np.asarray(guide_pred)[ids] != arm) & (np.asarray(source_pred)[ids] == arm))),
                "nearest_guided_centroid": str(arm_classes[nearest[i]]),
                "own_guided_distance": float(d[i, i]),
                "nearest_guided_distance": float(d[i, nearest[i]]),
                "direction_all_13": normalized_direction(guided[i], formal_relative[i], scale),
                "direction_gravity_3": normalized_direction(guided[i, 10:13],
                                                            formal_relative[i, 10:13], scale[10:13]),
            }
        for name, predictions in (("source_arm_x_same_hand", source_pred),
                                  ("guided_cal_arm_x_same_hand", guide_pred)):
            observed = float(np.mean(np.asarray(predictions) == trial_truth))
            expected = prior["sessions"][phase]["arms"][name]["arm_accuracy"]
            if not np.isclose(observed, expected, rtol=0, atol=1e-12):
                raise ValueError(f"arm decision replay disagrees with frozen result: {sid}/{name}")
        result["sessions"][phase] = {"session": sid, "formal_trials": len(trial_truth),
                                     "arm_order": arm_classes.tolist(),
                                     "formal_to_guided_distances": d.tolist(),
                                     "formal_arm_centroids_matching_own_guide": int(np.sum(nearest == true_index)),
                                     "per_arm": per_arm}
        print(f"{sid}: {sum(nearest == true_index)}/7 formal arm centroids nearest own guide", flush=True)
    path = ROOT / "SONG_ARM_SHIFT_DIAGNOSTIC.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
