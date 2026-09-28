"""Source-only external Rest prior for the exact new-v2 MANUS speed study."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v1.run import fit_predict
from benchmarks.new_bank_v2.manus_spatial_run import extract, score
from benchmarks.new_bank_v2.roam_posture_run import extract_archive, sha256
from emgimu.datasets.semg_manus import PATH_RE, load_semg_manus_windows
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v2 import (
    RestNoiseDetailV2, RingRelativeCovarianceV2, TraceCovarianceV2,
)

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "MANUS_REST_TRANSFER_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
CLASSES = np.arange(6)


def check_protocol() -> None:
    manus = json.loads((ROOT / "MANUS_SPATIAL_PROTOCOL.json").read_text(encoding="utf-8"))
    roam = json.loads((ROOT / "ROAM_POSTURE_PROTOCOL.json").read_text(encoding="utf-8"))
    if (sha256(ROOT / "MANUS_SPATIAL_PROTOCOL.json") != PROTOCOL["parent_manus_protocol_sha256"]
            or sha256(ROOT / "ROAM_POSTURE_PROTOCOL.json") != PROTOCOL["parent_roam_protocol_sha256"]
            or PROTOCOL["users"] != manus["users"]
            or PROTOCOL["gestures"] != manus["gestures"]
            or PROTOCOL["conditions"] != manus["conditions"]
            or PROTOCOL["source_session"] != manus["source_session"]
            or PROTOCOL["validation_session"] != manus["validation_session"]
            or PROTOCOL["final_session"] != manus["final_session"]
            or PROTOCOL["window_ms"] != manus["window_ms"]
            or PROTOCOL["maximum_windows_per_trial"] != manus["maximum_windows_per_trial"]
            or PROTOCOL["sample_rate_hz"] != 200 or PROTOCOL["channels"] != 8
            or PROTOCOL["arms"] != ["F0v2", "F0v2+F2a", "F0v2+F3c", "F0v2+F2a+F3c"]
            or roam["source_subjects"] != list(range(1, 19))):
        raise ValueError("frozen MANUS external-Rest transfer protocol changed")


def external_rest() -> FeatureBatch:
    archive = Path(PROTOCOL["rest_archive"])
    if sha256(archive).lower() != PROTOCOL["rest_archive_sha256"].lower():
        raise ValueError("external Rest archive hash mismatch")
    batch, labels, identities, slices, _ = extract_archive(archive)
    windows = np.concatenate([batch.emg[a:b] for i, (a, b) in enumerate(slices)
                              if identities[i]["subject"] in range(1, 19)
                              and identities[i]["condition"] == "resting" and labels[i] == 0])
    result = FeatureBatch(windows, 200)
    if result.windows != 1828 or result.channels != 8 or result.emg.shape[1] != 40:
        raise ValueError("external source Rest contract changed")
    return result


def evaluate() -> dict:
    check_protocol()
    archive = Path(PROTOCOL["manus_archive"])
    if sha256(archive).lower() != PROTOCOL["manus_archive_sha256"].lower():
        raise ValueError("MANUS source archive hash mismatch")
    rest = external_rest()
    families = {"F0v2": RestNoiseDetailV2(rest_label=0).fit(
                    rest, np.zeros(rest.windows, dtype=int)),
                "F2a": TraceCovarianceV2(shrinkage=0.05).fit(rest),
                "F3c": RingRelativeCovarianceV2(shrinkage=0.05).fit(rest)}
    arguments = {"users": PROTOCOL["users"], "gestures": PROTOCOL["gestures"],
                 "speeds": PROTOCOL["conditions"], "window_ms": PROTOCOL["window_ms"],
                 "maximum_windows_per_trial": PROTOCOL["maximum_windows_per_trial"]}
    source = load_semg_manus_windows(archive, sessions=(PROTOCOL["source_session"],), **arguments)
    if (source.batch.channels != 8 or source.batch.sample_rate_hz != 200
            or source.batch.emg.shape[1] != 40 or set(source.labels) != set(CLASSES)):
        raise ValueError("unexpected native MANUS source contract")
    source_x, (source_y, source_users, source_sessions, _, source_trials) = extract(source, families)
    if set(source_sessions) != {1} or len(source_trials) != 108:
        raise ValueError("MANUS source trial identity changed")
    rows = []
    splits = {}
    for phase in ("validation", "final"):
        session = PROTOCOL[f"{phase}_session"]
        target = load_semg_manus_windows(archive, sessions=(session,), **arguments)
        target_x, (target_y, target_users, target_sessions, target_speeds, target_trials) = extract(
            target, families)
        if (len(target_trials) != 108 or set(target_sessions) != {session}
                or set(source_trials) & set(target_trials)):
            raise ValueError("MANUS target trial identity changed")
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
                        or int(native["session"]) != session or native["speed"] != speed
                        or native["gesture"] != PROTOCOL["gestures"][label]):
                    raise ValueError("native MANUS trial metadata mismatch")
                rows.append({"phase": phase, "subject": int(user), "condition": str(speed),
                             "arm": arm, "trial_id": str(trial), "label": int(label),
                             **{f"p_{c}": float(value) for c, value in zip(CLASSES, probability)}})
        print(f"MANUS external-Rest {phase}: {len(target_trials)} held-out native trials", flush=True)
    scores = {phase: {arm: score(rows, phase, arm) for arm in PROTOCOL["arms"]}
              for phase in ("validation", "final")}
    selected = min(PROTOCOL["arms"], key=lambda arm: (
        -scores["validation"][arm]["pooled"]["macro_f1"],
        scores["validation"][arm]["pooled"]["log_loss"]))
    prediction_path = ROOT / "MANUS_REST_TRANSFER_PREDICTIONS.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol": PROTOCOL, "protocol_sha256": sha256(PROTOCOL_PATH),
              "manus_archive_sha256": sha256(archive),
              "rest_archive_sha256": sha256(Path(PROTOCOL["rest_archive"])),
              "external_rest_windows": int(rest.windows),
              "source_windows": int(source.batch.windows),
              "prediction_sha256": sha256(prediction_path),
              "feature_dimensions": {name: len(family.feature_names)
                                     for name, family in families.items()},
              "split_trial_ids": splits, "validation_selected_arm": selected,
              "scores": scores, "scope": PROTOCOL["scope"]}
    (ROOT / "MANUS_REST_TRANSFER_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for arm in PROTOCOL["arms"]:
        print(f"{arm}: Session2 F1={scores['validation'][arm]['pooled']['macro_f1']:.4f}, "
              f"Session3 F1={scores['final'][arm]['pooled']['macro_f1']:.4f}", flush=True)
    print(f"MANUS external-Rest validation-selected arm: {selected}", flush=True)
    return result


if __name__ == "__main__":
    evaluate()
