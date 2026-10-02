"""Matched-speed leave-one-native-trial-out F4d session correction control."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.grabmyo_crossday.run import score
from benchmarks.new_bank_v2.f4d_manus_predictive_run import aggregate_trials, classifier
from benchmarks.new_bank_v2.manus_rest_transfer_run import external_rest
from benchmarks.new_bank_v2.roam_posture_run import sha256
from emgimu.datasets.semg_manus import load_semg_manus_windows
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2
from emgimu.feature_bank.relative_spectrum import LogBandEnergyFamily, PersonalSessionSpectralShift

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "F4D_MANUS_SAME_SPEED_PROTOCOL.json"
CLASSES = np.arange(6)
ARMS = ("F0v2", "F0v2+F4d_long", "F0v2+F4d_same_speed")


def evaluate() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    predictive_path = ROOT / protocol["parent_predictive_protocol"]
    if sha256(predictive_path) != protocol["parent_predictive_protocol_sha256"]:
        raise AssertionError("frozen predictive protocol changed")
    parent_protocol = json.loads((ROOT / "F4D_MANUS_SESSION_PROTOCOL.json").read_text(encoding="utf-8"))
    parent = json.loads((ROOT / "F4D_MANUS_SESSION_RESULTS.json").read_text(encoding="utf-8"))
    source_protocol = json.loads((ROOT / "MANUS_REST_TRANSFER_PROTOCOL.json").read_text(encoding="utf-8"))
    archive = Path(parent_protocol["archive"])
    if sha256(archive).lower() != parent_protocol["archive_sha256"].lower():
        raise AssertionError("MANUS archive changed")
    rest = external_rest()
    f0 = RestNoiseDetailV2(rest_label=0).fit(rest, np.zeros(rest.windows, dtype=int))
    data = load_semg_manus_windows(
        archive, users=parent_protocol["users"], sessions=(1, 2, 3),
        gestures=source_protocol["gestures"], speeds=source_protocol["conditions"],
        window_ms=parent_protocol["window_ms"],
        maximum_windows_per_trial=parent_protocol["maximum_windows_per_trial"])
    if len(set(data.trials)) != 324 or rest.windows != 1828:
        raise AssertionError("MANUS inventory changed")

    rows = []
    for user in parent_protocol["users"]:
        source = (data.users == user) & (data.sessions == 1)
        source_batch = data.batch.take(np.flatnonzero(source))
        spectrum = LogBandEnergyFamily().fit(source_batch)
        source_spec = spectrum.transform(source_batch)
        source_f0 = f0.transform(source_batch)
        profile = PersonalSessionSpectralShift().fit_long_term(source_spec, data.trials[source])
        spectral_x, y, _, source_trials = aggregate_trials(
            source_spec - profile.long_reference_, data.labels[source],
            data.speeds[source], data.trials[source])
        f0_x, f0_y, _, f0_trials = aggregate_trials(
            source_f0, data.labels[source], data.speeds[source], data.trials[source])
        np.testing.assert_array_equal(y, f0_y)
        np.testing.assert_array_equal(source_trials, f0_trials)
        if set(source_trials) != set(parent["users"][str(user)]["long_trial_ids"]):
            raise AssertionError("source trial identities changed")
        f0_model = classifier(f0_x, y)
        spectral_model = classifier(np.concatenate((f0_x, spectral_x), axis=1), y)

        for phase, session in (("validation", 2), ("final", 3)):
            for speed in parent_protocol["held_out_evaluation_speeds"]:
                selection = ((data.users == user) & (data.sessions == session)
                             & (data.speeds == speed))
                trial_ids = sorted(set(data.trials[selection]))
                expected = parent["users"][str(user)]["sessions"][phase]["held_out"][speed]["native_trials"]
                if len(trial_ids) != 6 or set(trial_ids) != set(expected):
                    raise AssertionError("one native trial per gesture and speed required")
                for held_trial in trial_ids:
                    calibration = selection & (data.trials != held_trial)
                    held = selection & (data.trials == held_trial)
                    cal_ids = sorted(set(data.trials[calibration]))
                    if len(cal_ids) != 5 or held_trial in cal_ids:
                        raise AssertionError("held trial leaked into calibration")
                    profile.fit_session_calibration(
                        spectrum.transform(data.batch.take(np.flatnonzero(calibration))),
                        data.trials[calibration])
                    coordinates = profile.transform_evaluation(
                        spectrum.transform(data.batch.take(np.flatnonzero(held))),
                        data.trials[held])
                    raw = coordinates["window_minus_long"]
                    adjusted = raw - coordinates["session_minus_long"]
                    held_f0 = f0.transform(data.batch.take(np.flatnonzero(held)))
                    raw_x = np.mean(raw, axis=0)
                    adjusted_x = np.mean(adjusted, axis=0)
                    f0_x = np.mean(held_f0, axis=0)
                    label = int(np.unique(data.labels[held]).item())
                    predictions = {
                        "F0v2": f0_model.predict_proba(f0_x[None, :])[0],
                        "F0v2+F4d_long": spectral_model.predict_proba(
                            np.concatenate((f0_x, raw_x))[None, :])[0],
                        "F0v2+F4d_same_speed": spectral_model.predict_proba(
                            np.concatenate((f0_x, adjusted_x))[None, :])[0],
                    }
                    for arm, probabilities in predictions.items():
                        rows.append({"phase": phase, "user": user, "speed": speed,
                                     "trial_id": held_trial, "label": label, "arm": arm,
                                     "calibration_trial_ids": "|".join(cal_ids),
                                     **{f"p_{c}": float(value) for c, value in zip(CLASSES, probabilities)}})
            print(f"F4d same-speed user {user} {phase}: 12 held trials", flush=True)

    scores = {}
    for phase in ("validation", "final"):
        scores[phase] = {}
        for arm in ARMS:
            part = [row for row in rows if row["phase"] == phase and row["arm"] == arm]
            y = np.asarray([row["label"] for row in part])
            p = np.asarray([[row[f"p_{c}"] for c in CLASSES] for row in part])
            users = np.asarray([row["user"] for row in part])
            speeds = np.asarray([row["speed"] for row in part])
            scores[phase][arm] = {
                "pooled": score(y, p, CLASSES, users),
                "by_speed": {speed: score(y[speeds == speed], p[speeds == speed],
                                          CLASSES, users[speeds == speed])
                             for speed in parent_protocol["held_out_evaluation_speeds"]},
            }
    predictions_path = ROOT / "F4D_MANUS_SAME_SPEED_PREDICTIONS.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "parent_predictive_protocol_sha256": sha256(predictive_path),
        "parent_result_sha256": sha256(ROOT / "F4D_MANUS_SESSION_RESULTS.json"),
        "archive_sha256": sha256(archive), "rest_windows": rest.windows,
        "prediction_sha256": sha256(predictions_path), "prediction_rows": len(rows),
        "scores": scores, "scope": protocol["boundary"],
    }
    (ROOT / "F4D_MANUS_SAME_SPEED_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for phase in ("validation", "final"):
        print(f"F4d same-speed {phase}: " + ", ".join(
            f"{arm} F1={scores[phase][arm]['pooled']['macro_f1']:.4f}"
            for arm in ARMS), flush=True)
    return result


if __name__ == "__main__":
    evaluate()
