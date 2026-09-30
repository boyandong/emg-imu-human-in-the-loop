"""One-person Song F8 Rest-noise shift from isolated calibration blocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from benchmarks.song_real8_study import load_session
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.session_shift_summary import FamilySessionShiftSummary


def trial_balanced_noise(batch: FeatureBatch, labels: np.ndarray,
                         trials: np.ndarray, rest_label: str) -> np.ndarray:
    """Independent read-back of equal-trial-mass median adjacent differences."""
    chosen = np.asarray(labels) == rest_label
    if not np.any(chosen):
        raise ValueError("Rest calibration is required")
    differences = np.median(np.abs(np.diff(np.asarray(batch.emg[chosen], dtype=np.float64), axis=1)), axis=1)
    ids = np.asarray(trials)[chosen]
    return np.mean([differences[ids == ident].mean(axis=0) for ident in np.unique(ids)], axis=0)


def run(source: Path, output: Path) -> dict:
    data = {session: load_session(source / f"2026-09-18_{session}", session,
                                  filter_mode="causal")
            for session in ("S01", "S02", "S03", "S04")}
    source_batch = FeatureBatch(np.concatenate([data[s]["batch"].emg for s in ("S01", "S02")]), 250.)
    source_labels = np.concatenate([data[s]["hand"] for s in ("S01", "S02")])
    source_ids = np.concatenate([data[s]["trial"] for s in ("S01", "S02")])
    family = FamilySessionShiftSummary().fit_long_term(
        source_batch, source_labels, source_ids, rest_label="neutral")
    frozen = pickle.dumps(family)
    source_noise = trial_balanced_noise(source_batch, source_labels, source_ids, "neutral")
    result = {
        "status": "one_person_same_day_f8_rest_noise_observation_not_deployed",
        "source_sessions": ["S01", "S02"], "validation_session": "S03",
        "descriptive_final_session": "S04", "filter_mode": "causal_continuous",
        "hdf5_sha256": {s: data[s]["audit"]["sha256"] for s in data},
        "source_state_sha256": hashlib.sha256(frozen).hexdigest(),
        "source_formal_windows": len(source_batch.emg),
        "source_neutral_adjacent_diff_median_counts": source_noise.tolist(),
        "target": {},
        "boundary": "Only source S01/S02 formal windows fit the long-term reference. S03/S04 summaries use their separately recorded labeled calibration blocks at 1 or 2 shots per class, never formal evaluation windows. Same participant and calendar day, source/S03 readiness failures and prior S04 inspection make this exploratory. The result is a session descriptor, not predictive value, measured force, re-donning evidence or a deployed adaptation rule.",
    }
    for session in ("S03", "S04"):
        target = data[session]
        result["target"][session] = {}
        for shots in (1, 2):
            allowed = np.asarray(target["calibration_shot"]) <= shots
            calibration = target["calibration_batch"].take(np.flatnonzero(allowed))
            labels = np.asarray(target["calibration_hand"])[allowed]
            shot = np.asarray(target["calibration_shot"])[allowed]
            ids = np.asarray([f"{session}:cal:{int(n)}:{label}"
                              for n, label in zip(shot, labels)])
            if set(ids) & set(source_ids) or set(ids) & set(target["trial"]):
                raise AssertionError("calibration/source/formal trial identities overlap")
            if set(labels) != set(source_labels):
                raise AssertionError("calibration class inventory changed")
            summary = family.from_calibration(calibration, labels, ids)
            direct = np.log(trial_balanced_noise(calibration, labels, ids, "neutral") / source_noise)
            if not summary["rest_noise_shift_available"] or not np.allclose(
                    summary["rest_noise_log_ratio_per_channel"], direct, atol=1e-10):
                raise AssertionError("F8 Rest-noise summary fails native direct read-back")
            result["target"][session][str(shots)] = {
                "calibration_trial_ids": sorted(set(ids)),
                "calibration_windows": len(ids),
                "formal_evaluation_windows_used": 0,
                "readiness": target["audit"]["readiness"],
                "rest_noise_log_ratio_per_channel": summary["rest_noise_log_ratio_per_channel"],
                "rest_noise_shift_norm": summary["rest_noise_shift_norm"],
                "direct_readback_max_abs_error": float(np.max(np.abs(
                    np.asarray(summary["rest_noise_log_ratio_per_channel"]) - direct))),
            }
            print(f"F8 {session} {shots}-shot: Rest-noise shift norm="
                  f"{summary['rest_noise_shift_norm']:.4f}", flush=True)
    if frozen != pickle.dumps(family):
        raise AssertionError("target calibration mutated long-term source state")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path("E:/qxy/emg_meta/emg_meta/data/Song"))
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/song_real8/F8_REST_NOISE_SHIFT.json"))
    args = parser.parse_args()
    run(args.source, args.output)
