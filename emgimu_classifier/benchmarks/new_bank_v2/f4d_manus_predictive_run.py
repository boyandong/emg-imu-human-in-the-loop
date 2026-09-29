"""Frozen-source MANUS test of F4d session spectral subtraction."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.grabmyo_crossday.run import score
from benchmarks.new_bank_v2.manus_rest_transfer_run import external_rest
from benchmarks.new_bank_v2.roam_posture_run import sha256
from emgimu.datasets.semg_manus import load_semg_manus_windows
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2
from emgimu.feature_bank.relative_spectrum import LogBandEnergyFamily, PersonalSessionSpectralShift

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "F4D_MANUS_PREDICTIVE_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
CLASSES = np.arange(6)


def aggregate_trials(values: np.ndarray, labels: np.ndarray, speeds: np.ndarray,
                     identities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trials, inverse = np.unique(identities, return_inverse=True)
    vectors, y, conditions = [], [], []
    for index in range(len(trials)):
        selected = inverse == index
        if len(set(labels[selected])) != 1 or len(set(speeds[selected])) != 1:
            raise AssertionError("MANUS native trial has mixed labels or speeds")
        vectors.append(np.mean(values[selected], axis=0))
        y.append(int(labels[selected][0]))
        conditions.append(str(speeds[selected][0]))
    return np.stack(vectors), np.asarray(y), np.asarray(conditions), trials


def classifier(x: np.ndarray, y: np.ndarray):
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=20260924))
    model.fit(x, y)
    np.testing.assert_array_equal(model[-1].classes_, CLASSES)
    if np.max(model[-1].n_iter_) >= 2000:
        raise RuntimeError("MANUS F4d source classifier failed to converge")
    return model


def evaluate() -> dict:
    parent_protocol_path = ROOT / "F4D_MANUS_SESSION_PROTOCOL.json"
    parent_result_path = ROOT / "F4D_MANUS_SESSION_RESULTS.json"
    if (sha256(parent_protocol_path) != PROTOCOL["parent_protocol_sha256"]
            or sha256(parent_result_path) != PROTOCOL["parent_result_sha256"]):
        raise AssertionError("frozen F4d diagnostic changed")
    parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    parent = json.loads(parent_result_path.read_text(encoding="utf-8"))
    source_protocol = json.loads((ROOT / "MANUS_REST_TRANSFER_PROTOCOL.json").read_text(encoding="utf-8"))
    archive = Path(parent_protocol["archive"])
    if sha256(archive).lower() != parent_protocol["archive_sha256"].lower():
        raise AssertionError("MANUS F4d archive changed")
    rest = external_rest()
    f0 = RestNoiseDetailV2(rest_label=0).fit(rest, np.zeros(rest.windows, dtype=int))
    data = load_semg_manus_windows(
        archive, users=parent_protocol["users"], sessions=(1, 2, 3),
        gestures=source_protocol["gestures"], speeds=source_protocol["conditions"],
        window_ms=parent_protocol["window_ms"],
        maximum_windows_per_trial=parent_protocol["maximum_windows_per_trial"])
    if len(set(data.trials)) != 324 or rest.windows != 1828:
        raise AssertionError("MANUS F4d predictive source inventory changed")
    rows, source_ids = [], {}
    for user in parent_protocol["users"]:
        source = (data.users == user) & (data.sessions == 1)
        source_batch = data.batch.take(np.flatnonzero(source))
        spectrum = LogBandEnergyFamily().fit(source_batch)
        source_spec = spectrum.transform(source_batch)
        source_f0 = f0.transform(source_batch)
        profile = PersonalSessionSpectralShift().fit_long_term(source_spec, data.trials[source])
        source_vectors, ys, _, trials = aggregate_trials(
            source_spec - profile.long_reference_, data.labels[source],
            data.speeds[source], data.trials[source])
        f0_vectors, f0_y, _, f0_trials = aggregate_trials(
            source_f0, data.labels[source], data.speeds[source], data.trials[source])
        np.testing.assert_array_equal(ys, f0_y)
        np.testing.assert_array_equal(trials, f0_trials)
        if set(trials) != set(parent["users"][str(user)]["long_trial_ids"]):
            raise AssertionError("MANUS F4d predictive source trials changed")
        source_ids[str(user)] = trials.tolist()
        f0_model = classifier(f0_vectors, ys)
        spectral_model = classifier(np.concatenate((f0_vectors, source_vectors), axis=1), ys)
        for phase, session in (("validation", 2), ("final", 3)):
            cal = ((data.users == user) & (data.sessions == session)
                   & (data.speeds == parent_protocol["current_calibration_speed"]))
            held = ((data.users == user) & (data.sessions == session)
                    & np.isin(data.speeds, parent_protocol["held_out_evaluation_speeds"]))
            frozen = parent["users"][str(user)]["sessions"][phase]
            if set(data.trials[cal]) != set(frozen["calibration_trial_ids"]):
                raise AssertionError("MANUS F4d predictive calibration changed")
            expected = {trial for group in frozen["held_out"].values()
                        for trial in group["native_trials"]}
            if set(data.trials[held]) != expected:
                raise AssertionError("MANUS F4d predictive held trials changed")
            profile.fit_session_calibration(
                spectrum.transform(data.batch.take(np.flatnonzero(cal))), data.trials[cal])
            coordinates = profile.transform_evaluation(
                spectrum.transform(data.batch.take(np.flatnonzero(held))), data.trials[held])
            np.testing.assert_allclose(coordinates["session_minus_long"],
                                       frozen["session_minus_long_vector"], rtol=0, atol=1e-6)
            raw_spec = coordinates["window_minus_long"]
            adjusted_spec = raw_spec - coordinates["session_minus_long"]
            held_f0 = f0.transform(data.batch.take(np.flatnonzero(held)))
            raw, y, speed, held_trials = aggregate_trials(
                raw_spec, data.labels[held], data.speeds[held], data.trials[held])
            adjusted, adjusted_y, adjusted_speed, adjusted_trials = aggregate_trials(
                adjusted_spec, data.labels[held], data.speeds[held], data.trials[held])
            f0_target, f0_y, f0_speed, f0_trials = aggregate_trials(
                held_f0, data.labels[held], data.speeds[held], data.trials[held])
            np.testing.assert_array_equal(y, adjusted_y)
            np.testing.assert_array_equal(y, f0_y)
            np.testing.assert_array_equal(speed, adjusted_speed)
            np.testing.assert_array_equal(speed, f0_speed)
            np.testing.assert_array_equal(held_trials, adjusted_trials)
            np.testing.assert_array_equal(held_trials, f0_trials)
            predictions = {
                "F0v2": f0_model.predict_proba(f0_target),
                "F0v2+F4d_long": spectral_model.predict_proba(
                    np.concatenate((f0_target, raw), axis=1)),
                "F0v2+F4d_session": spectral_model.predict_proba(
                    np.concatenate((f0_target, adjusted), axis=1)),
            }
            for arm, probabilities in predictions.items():
                for trial, label, condition, p in zip(held_trials, y, speed, probabilities):
                    rows.append({"phase": phase, "user": user, "arm": arm,
                                 "trial_id": str(trial), "label": int(label),
                                 "speed": str(condition),
                                 **{f"p_{c}": float(value) for c, value in zip(CLASSES, p)}})
            print(f"F4d predictive user {user} {phase}: {len(held_trials)} held trials", flush=True)
    scores = {}
    for phase in ("validation", "final"):
        scores[phase] = {}
        for arm in PROTOCOL["arms"]:
            part = [r for r in rows if r["phase"] == phase and r["arm"] == arm]
            y = np.asarray([r["label"] for r in part])
            p = np.asarray([[r[f"p_{c}"] for c in CLASSES] for r in part])
            users = np.asarray([r["user"] for r in part])
            speed = np.asarray([r["speed"] for r in part])
            scores[phase][arm] = {"pooled": score(y, p, CLASSES, users),
                                  "by_speed": {name: score(y[speed == name], p[speed == name],
                                                           CLASSES, users[speed == name])
                                               for name in parent_protocol["held_out_evaluation_speeds"]}}
    path = ROOT / "F4D_MANUS_PREDICTIVE_PREDICTIONS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol_sha256": sha256(PROTOCOL_PATH),
              "parent_protocol_sha256": PROTOCOL["parent_protocol_sha256"],
              "parent_result_sha256": PROTOCOL["parent_result_sha256"],
              "archive_sha256": sha256(archive),
              "rest_windows": rest.windows, "prediction_sha256": sha256(path),
              "prediction_rows": len(rows), "source_trial_ids": source_ids,
              "scores": scores, "scope": PROTOCOL["boundary"]}
    (ROOT / "F4D_MANUS_PREDICTIVE_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for phase in ("validation", "final"):
        print(f"F4d predictive {phase}: " + ", ".join(
            f"{arm} F1={scores[phase][arm]['pooled']['macro_f1']:.4f}"
            for arm in PROTOCOL["arms"]), flush=True)
    return result


if __name__ == "__main__":
    evaluate()
