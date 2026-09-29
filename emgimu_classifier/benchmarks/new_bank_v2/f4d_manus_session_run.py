"""Multi-user native F4d context diagnostic with disjoint session calibration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2.roam_posture_run import sha256
from emgimu.datasets.semg_manus import load_semg_manus_windows
from emgimu.feature_bank.relative_spectrum import LogBandEnergyFamily, PersonalSessionSpectralShift

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "F4D_MANUS_SESSION_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def evaluate() -> dict:
    parent_path = ROOT / PROTOCOL["parent_protocol"]
    if sha256(parent_path) != PROTOCOL["parent_protocol_sha256"]:
        raise AssertionError("frozen MANUS native protocol changed")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    archive = Path(PROTOCOL["archive"])
    if (sha256(archive).lower() != PROTOCOL["archive_sha256"].lower()
            or PROTOCOL["users"] != parent["users"]
            or PROTOCOL["window_ms"] != parent["window_ms"]
            or PROTOCOL["maximum_windows_per_trial"] != parent["maximum_windows_per_trial"]):
        raise AssertionError("MANUS F4d source contract changed")
    data = load_semg_manus_windows(
        archive, users=PROTOCOL["users"], sessions=(1, 2, 3),
        gestures=parent["gestures"], speeds=parent["conditions"],
        window_ms=PROTOCOL["window_ms"],
        maximum_windows_per_trial=PROTOCOL["maximum_windows_per_trial"])
    if (data.batch.channels != 8 or data.batch.sample_rate_hz != 200
            or data.batch.emg.shape[1] != 40 or len(set(data.trials)) != 324):
        raise AssertionError("MANUS F4d native trial inventory changed")
    output = {}
    for user in PROTOCOL["users"]:
        long_mask = (data.users == user) & (data.sessions == PROTOCOL["long_term_session"])
        long_ids = data.trials[long_mask]
        if len(set(long_ids)) != 18 or set(data.labels[long_mask]) != set(range(6)):
            raise AssertionError("MANUS long-term source is incomplete")
        family = LogBandEnergyFamily().fit(data.batch.take(np.flatnonzero(long_mask)))
        if len(family.feature_names) != 32:
            raise AssertionError("MANUS F4d source log-band dimension changed")
        long_spectrum = family.transform(data.batch.take(np.flatnonzero(long_mask)))
        user_sessions = {}
        for phase, session in (("validation", PROTOCOL["validation_session"]),
                               ("final", PROTOCOL["final_session"])):
            cal_mask = ((data.users == user) & (data.sessions == session)
                        & (data.speeds == PROTOCOL["current_calibration_speed"]))
            eval_mask = ((data.users == user) & (data.sessions == session)
                         & np.isin(data.speeds, PROTOCOL["held_out_evaluation_speeds"]))
            cal_ids, eval_ids = data.trials[cal_mask], data.trials[eval_mask]
            if (len(set(cal_ids)) != 6 or len(set(eval_ids)) != 12
                    or set(cal_ids) & set(eval_ids) or set(long_ids) & (set(cal_ids) | set(eval_ids))
                    or set(data.labels[cal_mask]) != set(range(6))):
                raise AssertionError("MANUS F4d calibration/evaluation trial split changed")
            cal_spectrum = family.transform(data.batch.take(np.flatnonzero(cal_mask)))
            eval_spectrum = family.transform(data.batch.take(np.flatnonzero(eval_mask)))
            profile = PersonalSessionSpectralShift().fit_long_term(long_spectrum, long_ids)
            profile.fit_session_calibration(cal_spectrum, cal_ids)
            coordinates = profile.transform_evaluation(eval_spectrum, eval_ids)
            session_delta = coordinates["session_minus_long"]
            if (session_delta.shape != (32,)
                    or coordinates["window_minus_long"].shape != (len(eval_ids), 32)
                    or not np.isfinite(session_delta).all()
                    or not np.isfinite(coordinates["window_minus_long"]).all()):
                raise AssertionError("MANUS F4d native coordinate contract failed")
            held = {}
            for speed in PROTOCOL["held_out_evaluation_speeds"]:
                mask = data.speeds[eval_mask] == speed
                if len(set(eval_ids[mask])) != 6:
                    raise AssertionError("MANUS F4d held speed has missing gestures")
                mean_minus_long = coordinates["window_minus_long"][mask].mean(axis=0)
                held[speed] = {"native_trials": sorted(set(eval_ids[mask])),
                               "windows": int(mask.sum()),
                               "held_mean_minus_long_vector": mean_minus_long.tolist(),
                               "held_mean_minus_session_l2": float(np.linalg.norm(
                                   mean_minus_long - session_delta))}
            user_sessions[phase] = {"session": session,
                                    "calibration_trial_ids": sorted(set(cal_ids)),
                                    "calibration_windows": int(cal_mask.sum()),
                                    "session_minus_long_vector": session_delta.tolist(),
                                    "session_minus_long_mean_abs": float(np.mean(np.abs(session_delta))),
                                    "held_out": held}
            print(f"F4d MANUS user {user} {phase}: 6 calibration, 12 held trials", flush=True)
        output[str(user)] = {"long_trial_ids": sorted(set(long_ids)),
                             "long_windows": int(long_mask.sum()),
                             "bands_hz": [list(band) for band in family.bands_],
                             "sessions": user_sessions}
    result = {"protocol_sha256": sha256(PROTOCOL_PATH),
              "parent_protocol_sha256": PROTOCOL["parent_protocol_sha256"],
              "archive_sha256": sha256(archive),
              "users": output,
              "total_source_trials": 6 * 18,
              "total_current_calibration_trials": 6 * 2 * 6,
              "total_held_out_trials": 6 * 2 * 12,
              "scope": PROTOCOL["scope"]}
    (ROOT / "F4D_MANUS_SESSION_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    evaluate()
