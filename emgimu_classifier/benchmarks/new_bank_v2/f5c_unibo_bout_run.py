"""Frozen full-bout UniBo G5 versus optional order-two path signature."""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2.roam_posture_run import sha256
from emgimu.feature_bank.force_nested_oof import fit_temperature, temperature_probability
from emgimu.feature_bank.temporal import PathSignatureFamily
from emgimu.feature_bank.unibo_full_fusion import classifier
from emgimu.feature_bank.unibo_sequence_temporal import (
    bout_weights, complete_paths, g5_features, load_bouts,
)
from emgimu.feature_bank.unibo_study import _metrics


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "F5C_UNIBO_BOUT_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _features(bouts, family):
    return family.transform(complete_paths(bouts)).astype(np.float64)


def _fit(features, labels, weights):
    scaler, model = classifier(features, labels, weights)
    if np.max(model.n_iter_) >= 1000:
        raise RuntimeError("source bout classifier did not converge")
    return scaler, model


def _probability(state, values):
    scaler, model = state
    return model.predict_proba(scaler.transform(values))


def _score(rows, phase, arm):
    selected = [row for row in rows if row["phase"] == phase and row["arm"] == arm]
    y = np.asarray([row["label"] for row in selected], dtype=int)
    p = np.asarray([[row[f"p_{c}"] for c in range(4)] for row in selected], dtype=float)
    users = np.asarray([row["user"] for row in selected])
    days = np.asarray([row["day"] for row in selected], dtype=int)
    posture = np.asarray([row["posture"] for row in selected])
    # The native equal-user/trial/class/bout weighting is frozen in the parent.
    weights = np.asarray([row["weight"] for row in selected], dtype=float)
    pooled = _metrics(y, p, weights)
    by_user = {str(user): _metrics(y[users == user], p[users == user], weights[users == user])
               for user in sorted(set(users))}
    by_day = {str(day): _metrics(y[days == day], p[days == day], weights[days == day])
              for day in sorted(set(days))}
    by_posture = {str(value): _metrics(y[posture == value], p[posture == value],
                                       weights[posture == value]) for value in sorted(set(posture))}
    return {"pooled": pooled, "by_user": by_user, "by_day": by_day,
            "by_posture": by_posture,
            "minimum_user_macro_f1": min(item["macro_f1"] for item in by_user.values()),
            "minimum_day_macro_f1": min(item["macro_f1"] for item in by_day.values()),
            "minimum_posture_macro_f1": min(item["macro_f1"] for item in by_posture.values())}


def evaluate() -> dict:
    parent = Path(PROTOCOL["frozen_parent_source"])
    final_parent = Path(PROTOCOL["frozen_parent_final"])
    for path, key in ((parent / "fitted_states.pkl", "parent_state_sha256"),
                      (parent / "heldout_predictions.npz", "parent_validation_prediction_sha256"),
                      (parent / "bout_metadata.json", "parent_validation_bout_metadata_sha256"),
                      (final_parent / "heldout_predictions.npz", "parent_final_prediction_sha256"),
                      (final_parent / "bout_metadata.json", "parent_final_bout_metadata_sha256")):
        if sha256(path) != PROTOCOL[key]:
            raise AssertionError(f"frozen UniBo parent changed: {path}")
    if (PROTOCOL["arms"] != ["G5", "F5c", "G5+F5c"]
            or PROTOCOL["source_days"] != [1, 2, 3, 4, 5]
            or PROTOCOL["inner_fit_days"] != [1, 2, 3, 4]
            or PROTOCOL["source_temperature_day"] != 5
            or PROTOCOL["validation_day"] != 6
            or PROTOCOL["final_days"] != [7, 8]):
        raise ValueError("F5c source/target protocol changed")
    dataset = Path(PROTOCOL["dataset_root"])
    print("[1/3] loading complete UniBo source, validation and final bouts", flush=True)
    previous = load_bouts(dataset, days=range(1, 7))
    current_meta = [{k: v for k, v in bout.items() if k not in ("path", "windows")}
                    for bout in previous]
    if current_meta != json.loads((parent / "bout_metadata.json").read_text()):
        raise AssertionError("source/validation native bout metadata changed")
    final = load_bouts(dataset, days=(7, 8))
    final_meta = [{k: v for k, v in bout.items() if k not in ("path", "windows")}
                  for bout in final]
    if final_meta != json.loads((final_parent / "bout_metadata.json").read_text()):
        raise AssertionError("final native bout metadata changed")
    states, parent_temperatures = pickle.loads((parent / "fitted_states.pkl").read_bytes())
    if {bout["user"] for bout in previous + final} != set(states):
        raise AssertionError("frozen personal-source user coverage changed")
    phase_bouts = {"validation": [], "final": []}
    probabilities = {phase: {arm: [] for arm in PROTOCOL["arms"]}
                     for phase in phase_bouts}
    learned_temperatures = {}
    splits = {}
    for user in sorted(states):
        inner = [bout for bout in previous if bout["user"] == user and bout["day"] <= 4]
        cal = [bout for bout in previous if bout["user"] == user and bout["day"] == 5]
        train = inner + cal
        ev = {"validation": [bout for bout in previous if bout["user"] == user and bout["day"] == 6],
              "final": [bout for bout in final if bout["user"] == user]}
        groups = [inner, cal, ev["validation"], ev["final"]]
        trial_sets = [{bout["trial"] for bout in group} for group in groups]
        if any(first & second for index, first in enumerate(trial_sets)
               for second in trial_sets[index+1:]):
            raise AssertionError("source/calibration/evaluation trial leakage")
        if not all(group for group in groups) or {bout["label"] for bout in inner} != set(range(4)):
            raise AssertionError("source/target complete-bout coverage changed")
        splits[user] = {"inner_source": sorted(trial_sets[0]), "temperature": sorted(trial_sets[1]),
                        "validation": sorted(trial_sets[2]), "final": sorted(trial_sets[3])}
        inner_g5 = states[user][0]["g5"][0]
        full_g5, full_scaler, full_model = states[user][1]["g5"]
        inner_signature = PathSignatureFamily().fit(complete_paths(inner))
        inner_f5c, cal_f5c = _features(inner, inner_signature), _features(cal, inner_signature)
        inner_g5_x, cal_g5_x = g5_features(inner_g5, inner), g5_features(inner_g5, cal)
        yi = np.asarray([bout["label"] for bout in inner])
        yc = np.asarray([bout["label"] for bout in cal])
        weights = bout_weights(inner)
        temperatures = {}
        for arm, x, xc in (("F5c", inner_f5c, cal_f5c),
                           ("G5+F5c", np.c_[inner_g5_x, inner_f5c],
                            np.c_[cal_g5_x, cal_f5c])):
            fitted = _fit(x, yi, weights)
            temperatures[arm] = fit_temperature(_probability(fitted, xc), yc)
        learned_temperatures[user] = temperatures
        full_signature = PathSignatureFamily().fit(complete_paths(train))
        full_f5c = _features(train, full_signature)
        full_g5_x = g5_features(full_g5, train)
        yt = np.asarray([bout["label"] for bout in train])
        wt = bout_weights(train)
        full_models = {"F5c": _fit(full_f5c, yt, wt),
                       "G5+F5c": _fit(np.c_[full_g5_x, full_f5c], yt, wt)}
        for phase in phase_bouts:
            target = ev[phase]
            f5c = _features(target, full_signature)
            g5 = g5_features(full_g5, target)
            g5_raw = full_model.predict_proba(full_scaler.transform(g5))
            p = {"G5": temperature_probability(g5_raw, parent_temperatures[user]["G5"]),
                 "F5c": temperature_probability(_probability(full_models["F5c"], f5c),
                                                temperatures["F5c"]),
                 "G5+F5c": temperature_probability(
                     _probability(full_models["G5+F5c"], np.c_[g5, f5c]),
                     temperatures["G5+F5c"])}
            for arm in p:
                probabilities[phase][arm].append(p[arm])
            phase_bouts[phase].extend(target)
        print(f"[2/3] {user}: source={len(train)} Day6={len(ev['validation'])} "
              f"Day7-8={len(ev['final'])}", flush=True)

    parent_error = {}
    for phase, path in (("validation", parent / "heldout_predictions.npz"),
                        ("final", final_parent / "heldout_predictions.npz")):
        with np.load(path, allow_pickle=False) as old:
            ordered = phase_bouts[phase]
            np.testing.assert_array_equal(old["bout_ids"], [bout["id"] for bout in ordered])
            np.testing.assert_array_equal(old["labels"], [bout["label"] for bout in ordered])
            new = np.concatenate(probabilities[phase]["G5"])
            error = float(np.max(np.abs(old["G5"] - new)))
            if error > 1e-10:
                raise AssertionError(f"frozen {phase} G5 replay differs by {error}")
            parent_error[phase] = error
    rows = []
    for phase, bouts in phase_bouts.items():
        weights = bout_weights(bouts)
        for arm in PROTOCOL["arms"]:
            p = np.concatenate(probabilities[phase][arm])
            for bout, weight, row in zip(bouts, weights, p):
                rows.append({"phase": phase, "arm": arm, "bout_id": bout["id"],
                             "trial_id": bout["trial"], "user": bout["user"],
                             "day": bout["day"], "posture": bout["posture"],
                             "label": bout["label"], "weight": float(weight),
                             **{f"p_{c}": float(value) for c, value in enumerate(row)}})
    scores = {phase: {arm: _score(rows, phase, arm) for arm in PROTOCOL["arms"]}
              for phase in phase_bouts}
    output = ROOT / "F5C_UNIBO_BOUT_PREDICTIONS.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol_sha256": sha256(PROTOCOL_PATH),
              "parent_state_sha256": PROTOCOL["parent_state_sha256"],
              "parent_validation_prediction_sha256": PROTOCOL["parent_validation_prediction_sha256"],
              "parent_final_prediction_sha256": PROTOCOL["parent_final_prediction_sha256"],
              "prediction_sha256": sha256(output), "prediction_rows": len(rows),
              "parent_g5_replay_max_abs_error": parent_error,
              "signature_dimension": 20, "g5_dimension": len(full_g5.feature_names),
              "source_temperatures": learned_temperatures, "split_trial_ids": splits,
              "bout_counts": {phase: len(value) for phase, value in phase_bouts.items()},
              "scores": scores, "scope": PROTOCOL["boundary"]}
    (ROOT / "F5C_UNIBO_BOUT_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n",
                                                      encoding="utf-8")
    for phase in phase_bouts:
        print(f"[3/3] {phase}: " + ", ".join(
            f"{arm} F1={scores[phase][arm]['pooled']['macro_f1']:.4f}"
            for arm in PROTOCOL["arms"]), flush=True)
    return result


if __name__ == "__main__":
    evaluate()
