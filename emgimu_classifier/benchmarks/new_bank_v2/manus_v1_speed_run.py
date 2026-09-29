"""Independent new-v1 family MANUS speed screen with frozen external Rest."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v1.run import fit_predict
from benchmarks.new_bank_v2.manus_rest_transfer_run import external_rest
from benchmarks.new_bank_v2.manus_spatial_run import extract, score
from benchmarks.new_bank_v2.roam_posture_run import sha256
from emgimu.datasets.semg_manus import PATH_RE, load_semg_manus_windows
from emgimu.feature_bank.new_bank_v1 import NEW_BANK_V1
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "MANUS_V1_SPEED_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
PARENT = "MANUS_REST_TRANSFER"
ARMS = tuple(PROTOCOL["arms"])
CLASSES = np.asarray(PROTOCOL["classes"])


def check_protocol() -> dict:
    for suffix, key in (("PROTOCOL.json", "parent_protocol_sha256"),
                        ("RESULTS.json", "parent_result_sha256"),
                        ("PREDICTIONS.csv", "parent_prediction_sha256")):
        if sha256(ROOT / f"{PARENT}_{suffix}") != PROTOCOL[key]:
            raise AssertionError(f"frozen MANUS parent changed: {suffix}")
    parent = json.loads((ROOT / f"{PARENT}_PROTOCOL.json").read_text(encoding="utf-8"))
    if (PROTOCOL["users"] != parent["users"] or PROTOCOL["speeds"] != parent["conditions"]
            or PROTOCOL["source_session"] != parent["source_session"]
            or PROTOCOL["validation_session"] != parent["validation_session"]
            or PROTOCOL["final_session"] != parent["final_session"]
            or PROTOCOL["candidate_families"] != list(NEW_BANK_V1)
            or ARMS != ("F0v2", *[f"F0v2+{name}" for name in NEW_BANK_V1])):
        raise AssertionError("MANUS extension disagrees with frozen parent")
    return parent


def baseline_replay(rows: list[dict]) -> float:
    with (ROOT / f"{PARENT}_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        parent = {(r["phase"], r["trial_id"]): r for r in csv.DictReader(stream)
                  if r["arm"] == "F0v2"}
    base = [r for r in rows if r["arm"] == "F0v2"]
    if len(base) != 216 or len(parent) != 216:
        raise AssertionError("MANUS baseline trial count changed")
    maximum = 0.0
    for row in base:
        prior = parent[(row["phase"], row["trial_id"])]
        if (row["subject"], row["condition"], row["label"]) != (
                int(prior["subject"]), prior["condition"], int(prior["label"])):
            raise AssertionError("MANUS baseline trial identity changed")
        p = np.asarray([row[f"p_{c}"] for c in CLASSES])
        q = np.asarray([float(prior[f"p_{c}"]) for c in CLASSES])
        maximum = max(maximum, float(np.max(np.abs(p - q))))
    if maximum > 1e-8:
        raise AssertionError(f"MANUS baseline numerical replay drift: {maximum}")
    return maximum


def evaluate() -> dict:
    parent = check_protocol()
    rest = external_rest()
    families = {"F0v2": RestNoiseDetailV2(rest_label=0).fit(
        rest, np.zeros(rest.windows, dtype=int)),
        **{name: constructor().fit(rest) for name, constructor in NEW_BANK_V1.items()}}
    archive = Path(parent["manus_archive"])
    arguments = {"users": parent["users"], "gestures": parent["gestures"],
                 "speeds": parent["conditions"], "window_ms": parent["window_ms"],
                 "maximum_windows_per_trial": parent["maximum_windows_per_trial"]}
    source = load_semg_manus_windows(archive, sessions=(1,), **arguments)
    if (source.batch.channels != 8 or source.batch.sample_rate_hz != 200
            or source.batch.emg.shape[1] != 40 or set(source.labels) != set(CLASSES)):
        raise AssertionError("MANUS source signal contract changed")
    xs, (ys, source_users, source_sessions, _, source_trials) = extract(source, families)
    parent_result = json.loads((ROOT / f"{PARENT}_RESULTS.json").read_text(encoding="utf-8"))
    if (set(source_sessions) != {1} or len(source_trials) != 108
            or set(source_trials) != set(parent_result["split_trial_ids"]["validation"]["source_trials"])):
        raise AssertionError("MANUS source trial identity changed")
    rows, splits = [], {}
    for phase in ("validation", "final"):
        session = PROTOCOL[f"{phase}_session"]
        target = load_semg_manus_windows(archive, sessions=(session,), **arguments)
        xt, (yt, target_users, target_sessions, target_speeds, target_trials) = extract(target, families)
        if (set(target_sessions) != {session} or len(target_trials) != 108
                or set(source_trials) & set(target_trials)
                or set(target_trials) != set(parent_result["split_trial_ids"][phase]["target_trials"])):
            raise AssertionError("MANUS target trial identity changed")
        splits[phase] = {"source_trials": source_trials.tolist(),
                         "target_trials": target_trials.tolist(),
                         "source_users": sorted(map(int, set(source_users))),
                         "target_users": sorted(map(int, set(target_users)))}
        for arm in ARMS:
            names = arm.split("+")
            x = np.concatenate([xs[name] for name in names], axis=1)
            x_target = np.concatenate([xt[name] for name in names], axis=1)
            classes, probability = fit_predict(x, ys, x_target)
            np.testing.assert_array_equal(classes, CLASSES)
            for trial_id, user, speed, label, p in zip(
                    target_trials, target_users, target_speeds, yt, probability):
                native = PATH_RE.fullmatch(str(trial_id))
                if (native is None or int(native["user"]) != user
                        or int(native["session"]) != session or native["speed"] != speed
                        or native["gesture"] != parent["gestures"][label]):
                    raise AssertionError("MANUS native trial metadata changed")
                rows.append({"phase": phase, "subject": int(user),
                             "condition": str(speed), "arm": arm,
                             "trial_id": str(trial_id), "label": int(label),
                             **{f"p_{c}": float(value) for c, value in zip(CLASSES, p)}})
        print(f"MANUS new-v1 {phase}: {len(target_trials)} held-out native trials", flush=True)
    replay_error = baseline_replay(rows)
    scores = {phase: {arm: score(rows, phase, arm) for arm in ARMS}
              for phase in ("validation", "final")}
    selected = min(ARMS, key=lambda arm: (-scores["validation"][arm]["pooled"]["macro_f1"],
                                           scores["validation"][arm]["pooled"]["log_loss"]))
    path = ROOT / "MANUS_V1_SPEED_PREDICTIONS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol": PROTOCOL, "protocol_sha256": sha256(PROTOCOL_PATH),
              "manus_archive_sha256": sha256(archive),
              "rest_archive_sha256": sha256(Path(parent["rest_archive"])),
              "external_rest_windows": rest.windows,
              "source_windows": source.batch.windows,
              "prediction_sha256": sha256(path),
              "feature_dimensions": {name: len(family.feature_names)
                                     for name, family in families.items()},
              "split_trial_ids": splits, "baseline_replay_max_abs_error": replay_error,
              "validation_selected_arm": selected, "scores": scores,
              "scope": PROTOCOL["scope"]}
    (ROOT / "MANUS_V1_SPEED_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"MANUS new-v1 validation-selected arm: {selected}", flush=True)
    return result


if __name__ == "__main__":
    evaluate()
