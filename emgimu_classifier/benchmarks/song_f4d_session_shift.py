"""Source-only F4d long baseline and Song session-calibration diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from benchmarks.song_real8_study import _hash, _text, window_starts
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.relative_spectrum import LogBandEnergyFamily, PersonalSessionSpectralShift


def _load(source: Path, session: str):
    file = source / f"2026-09-18_{session}" / "session.h5"
    readiness = json.loads((file.parent / "SESSION_COLLECTION_READINESS.json").read_text(encoding="utf-8"))
    sha = _hash(file)
    if sha != readiness["hdf5_sha256"]:
        raise ValueError(f"{session} source differs from readiness digest")
    with h5py.File(file) as handle:
        attrs = handle["meta"].attrs
        if (int(attrs["num_emg_channels"]) != 8 or int(attrs["emg_nominal_rate_hz"]) != 250 or
                _text(attrs["session_id"]) != session):
            raise ValueError(f"{session} has incompatible sensor metadata")
        raw = handle["streams/emg/raw"][:]
        indices = handle["streams/emg/sample_index"][:]
        calibration = handle["calibration_blocks"][:]
        trials = handle["trials"][:]
    if not np.array_equal(indices, np.arange(len(raw))):
        raise ValueError(f"{session} has noncontiguous EMG sample indices")
    return sha, raw, calibration, trials


def _windows(raw, rows, session, *, label, kind):
    windows, identities = [], []
    for row in rows:
        if (_text(row["label"]) != label or _text(row["trial_kind"]) != kind or
                not bool(row["valid"]) or _text(row["completion_status"]) != "completed"):
            continue
        start, end = int(row["stable_start_sample"]), int(row["stable_end_sample"])
        if not (0 <= start < end <= len(raw)):
            raise ValueError("stable interval outside recorded EMG")
        starts = np.arange(start, end - 49, 50) if kind == "calibration" else window_starts(start, end)
        for point in starts:
            windows.append(raw[int(point):int(point) + 50])
            identities.append(f"{session}:{int(row['trial_id'])}")
    if not windows:
        raise ValueError(f"{session} has no usable {label} windows")
    return np.stack(windows), identities


def run(source: Path, output: Path) -> dict:
    loaded = {session: _load(source, session) for session in ("S01", "S02", "S03", "S04")}
    long_windows, long_ids = [], []
    for session in ("S01", "S02"):
        _, raw, blocks, _ = loaded[session]
        windows, ids = _windows(raw, blocks, session, label="calibration_rest_initial", kind="calibration")
        long_windows.extend(windows)
        long_ids.extend(ids)
    long_batch = FeatureBatch(np.stack(long_windows), 250.0)
    family = LogBandEnergyFamily().fit(long_batch)
    long_spectrum = family.transform(long_batch)
    result = {
        "status": "native_same_person_day_f4d_context_diagnostic_only",
        "source_sha256": {name: values[0] for name, values in loaded.items()},
        "bands_hz": [list(band) for band in family.bands_],
        "features": len(family.feature_names),
        "long_source": {"sessions": ["S01", "S02"], "rest_windows": len(long_ids),
                        "rest_trial_ids": sorted(set(long_ids)), "equal_trial_mass": True},
        "current_sessions": {},
        "boundary": "Long source uses only initial-rest calibration blocks from S01/S02; each current session uses its own separate initial-rest block and evaluates distinct still-neutral formal trials. All four sessions are one person on one day; S01-S03 failed whole-session collection readiness, though individually valid blocks are used with original file hashes. Relative spectral coordinates describe context, not fatigue, generalization, live adaptation or improved accuracy.",
    }
    for session in ("S03", "S04"):
        _, raw, blocks, trials = loaded[session]
        cal_windows, cal_ids = _windows(raw, blocks, session, label="calibration_rest_initial", kind="calibration")
        eval_windows, eval_ids = _windows(raw, trials, session, label="still_neutral", kind="formal")
        cal_spectrum = family.transform(FeatureBatch(cal_windows, 250.0))
        eval_spectrum = family.transform(FeatureBatch(eval_windows, 250.0))
        profile = PersonalSessionSpectralShift().fit_long_term(long_spectrum, long_ids)
        profile.fit_session_calibration(cal_spectrum, cal_ids)
        coordinates = profile.transform_evaluation(eval_spectrum, eval_ids)
        session_delta = coordinates["session_minus_long"]
        eval_mean = np.mean(coordinates["window_minus_long"], axis=0)
        result["current_sessions"][session] = {
            "calibration_rest_windows": len(cal_ids),
            "calibration_trial_ids": sorted(set(cal_ids)),
            "held_out_still_neutral_windows": len(eval_ids),
            "held_out_formal_trial_count": len(set(eval_ids)),
            "session_minus_long_log_band_vector": session_delta.tolist(),
            "session_minus_long_mean_abs_log_energy": float(np.mean(np.abs(session_delta))),
            "held_out_still_neutral_mean_minus_long_log_band_vector": eval_mean.tolist(),
            "held_out_mean_minus_session_residual_l2": float(np.linalg.norm(eval_mean - session_delta)),
        }
        print(json.dumps({"session": session,
                          "calibration_rest_windows": len(cal_ids),
                          "held_out_still_neutral_windows": len(eval_ids),
                          "session_minus_long_mean_abs_log_energy": result["current_sessions"][session][
                              "session_minus_long_mean_abs_log_energy"]}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
