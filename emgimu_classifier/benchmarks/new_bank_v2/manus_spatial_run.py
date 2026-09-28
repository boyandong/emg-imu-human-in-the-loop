"""Frozen reduced-bank MANUS Session 1 -> Sessions 2/3 native-trial study."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v1.run import fit_predict, metrics, sha256
from emgimu.datasets.semg_manus import PATH_RE, load_semg_manus_windows
from emgimu.feature_bank.families import LocalDetailFamily
from emgimu.feature_bank.manus_study import _aggregate
from emgimu.feature_bank.new_bank_v2 import RingRelativeCovarianceV2, TraceCovarianceV2

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "MANUS_SPATIAL_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
ARCHIVE = Path(PROTOCOL["archive"])
CLASSES = np.arange(len(PROTOCOL["gestures"]))


def score(rows: list[dict], phase: str, arm: str) -> dict:
    chosen = [row for row in rows if row["phase"] == phase and row["arm"] == arm]
    y = np.asarray([row["label"] for row in chosen], dtype=int)
    p = np.asarray([[row[f"p_{c}"] for c in CLASSES] for row in chosen], dtype=float)
    users = np.asarray([row["subject"] for row in chosen], dtype=int)
    speeds = np.asarray([row["condition"] for row in chosen])
    by_user = {str(user): metrics(y[users == user], p[users == user], CLASSES)
               for user in PROTOCOL["users"]}
    by_speed = {speed: metrics(y[speeds == speed], p[speeds == speed], CLASSES)
                for speed in PROTOCOL["conditions"]}
    return {"pooled": metrics(y, p, CLASSES), "by_user": by_user,
            "by_speed": by_speed,
            "minimum_user_macro_f1": min(v["macro_f1"] for v in by_user.values()),
            "minimum_speed_macro_f1": min(v["macro_f1"] for v in by_speed.values())}


def extract(data, families):
    vectors = {}
    identity = None
    for name, family in families.items():
        vector, labels, users, sessions, speeds, trials = _aggregate(family.transform(data.batch), data)
        current = (labels, users, sessions, speeds, trials)
        if identity is not None:
            for previous, value in zip(identity, current):
                np.testing.assert_array_equal(previous, value)
        identity = current
        vectors[name] = vector
    return vectors, identity


def run() -> None:
    if (PROTOCOL["arms"] != ["F0", "F0+F2a", "F0+F3c", "F0+F2a+F3c"]
            or PROTOCOL["users"] != [3, 4, 5, 6, 7, 8]
            or PROTOCOL["source_session"] != 1
            or PROTOCOL["validation_session"] != 2
            or PROTOCOL["final_session"] != 3):
        raise ValueError("frozen reduced-bank protocol changed")
    if sha256(ARCHIVE).lower() != PROTOCOL["archive_sha256"].lower():
        raise ValueError("MANUS archive hash differs from frozen source")
    arguments = {"users": PROTOCOL["users"], "gestures": PROTOCOL["gestures"],
                 "speeds": PROTOCOL["conditions"], "window_ms": PROTOCOL["window_ms"],
                 "maximum_windows_per_trial": PROTOCOL["maximum_windows_per_trial"]}
    source = load_semg_manus_windows(ARCHIVE, sessions=(1,), **arguments)
    if (source.batch.channels != 8 or source.batch.sample_rate_hz != 200
            or set(source.labels) != set(CLASSES) or source.batch.emg.shape[1] != 40):
        raise ValueError("unexpected native source data contract")
    families = {"F0": LocalDetailFamily().fit(source.batch, source.labels),
                "F2a": TraceCovarianceV2(shrinkage=0.05).fit(source.batch),
                "F3c": RingRelativeCovarianceV2(shrinkage=0.05).fit(source.batch)}
    source_x, (source_y, source_users, source_sessions, _, source_trials) = extract(source, families)
    if set(source_sessions) != {1} or len(source_trials) != 108:
        raise ValueError("unexpected native source trial count")
    rows = []
    splits = {}
    for phase in ("validation", "final"):
        session = PROTOCOL[f"{phase}_session"]
        target = load_semg_manus_windows(ARCHIVE, sessions=(session,), **arguments)
        target_x, (target_y, target_users, target_sessions, target_speeds, target_trials) = extract(target, families)
        if (len(target_trials) != 108 or set(target_sessions) != {session}
                or set(source_trials) & set(target_trials)):
            raise ValueError("target session/trial identity contract failed")
        splits[phase] = {"source_trials": source_trials.tolist(),
                         "target_trials": target_trials.tolist(),
                         "source_users": sorted(set(map(int, source_users))),
                         "target_users": sorted(set(map(int, target_users)))}
        for arm in PROTOCOL["arms"]:
            members = arm.split("+")
            x = np.concatenate([source_x[name] for name in members], axis=1)
            xt = np.concatenate([target_x[name] for name in members], axis=1)
            classes, probabilities = fit_predict(x, source_y, xt)
            np.testing.assert_array_equal(classes, CLASSES)
            for trial, user, speed, label, probability in zip(
                    target_trials, target_users, target_speeds, target_y, probabilities):
                native = PATH_RE.fullmatch(str(trial))
                if (native is None or int(native["user"]) != user
                        or int(native["session"]) != session
                        or native["speed"] != speed
                        or native["gesture"] != PROTOCOL["gestures"][label]):
                    raise ValueError("native MANUS trial metadata mismatch")
                rows.append({"phase": phase, "subject": int(user), "condition": str(speed),
                             "arm": arm, "trial_id": str(trial), "label": int(label),
                             **{f"p_{c}": float(value) for c, value in zip(CLASSES, probability)}})
        print(f"MANUS {phase}: {len(target_trials)} held-out native trials", flush=True)
    scores = {phase: {arm: score(rows, phase, arm) for arm in PROTOCOL["arms"]}
              for phase in ("validation", "final")}
    selected = min(PROTOCOL["arms"], key=lambda arm: (
        -scores["validation"][arm]["pooled"]["macro_f1"],
        scores["validation"][arm]["pooled"]["log_loss"]))
    predictions_path = ROOT / "MANUS_SPATIAL_PREDICTIONS.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol": PROTOCOL,
              "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
              "archive_sha256": sha256(ARCHIVE),
              "prediction_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
              "feature_dimensions": {name: len(family.feature_names) for name, family in families.items()},
              "source_windows": int(source.batch.windows), "split_trial_ids": splits,
              "validation_selected_arm": selected, "scores": scores,
              "scope": PROTOCOL["scope"]}
    (ROOT / "MANUS_SPATIAL_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"MANUS validation-selected arm: {selected}", flush=True)


if __name__ == "__main__":
    run()
