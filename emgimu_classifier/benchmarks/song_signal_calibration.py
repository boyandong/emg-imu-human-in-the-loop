"""Calibration-only channel-scale update for the frozen Song F0+SPD bundle.

Pre-formal native calibration blocks supply the target profile. Synthetic
common-gain scenarios are diagnostic and do not represent a new wearing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.song_gain_sensitivity import GAINS, median_window_rms
from benchmarks.song_real8_study import load_session
from benchmarks.song_spd_increment_study import metrics, trial_probabilities


def correction(source_windows: np.ndarray, target_windows: np.ndarray) -> np.ndarray:
    source = np.asarray(median_window_rms(source_windows), dtype=np.float64)
    target = np.asarray(median_window_rms(target_windows), dtype=np.float64)
    if np.any(source <= 0) or np.any(target <= 0):
        raise ValueError("Channel-scale calibration requires positive per-channel RMS")
    return np.clip(source / target, .25, 4.0)


def native_calibration_windows(data: dict, budget: int) -> np.ndarray:
    if budget not in (1, 2):
        raise ValueError("Only one or two pre-formal blocks per class exist")
    selected = data["calibration_shot"] <= budget
    hands = data["calibration_hand"][selected]
    if any(np.sum(hands == hand) != 3 * budget
           for hand in ("neutral", "index_pinch", "fist", "open_hand")):
        raise ValueError("Calibration blocks are incomplete or unbalanced")
    return data["calibration_batch"].emg[selected]


def evaluate(runtime, data: dict, gain: float, scale: np.ndarray, labels: np.ndarray) -> dict:
    if scale.shape != (8,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Expected eight positive finite channel scales")
    probability = np.stack([
        runtime.predict_filtered_window(window * gain * scale)
        for window in data["batch"].emg
    ])
    ids, truth, averaged = trial_probabilities(
        data["hand"], probability, data["trial"], labels)
    return {"formal_trials": len(ids), **metrics(truth, averaged, labels)}


def run(source: Path, collection_root: Path, reference: Path, output: Path) -> dict:
    sys.path.insert(0, str(collection_root.resolve()))
    from emgforce.inference.song_local import LABELS, SongLocalRuntime

    frozen = json.loads(reference.read_text(encoding="utf-8"))["sessions"]
    data = {
        session: load_session(source / f"2026-09-18_{session}", session, filter_mode="causal")
        for session in ("S01", "S02", "S03")
    }
    runtime = SongLocalRuntime(collection_root / "models" / "song_real8_f0_spd")
    source_profile = np.concatenate([
        native_calibration_windows(data[session], 2) for session in ("S01", "S02")
    ])
    source_formal = np.concatenate([data[session]["batch"].emg
                                    for session in ("S01", "S02")])
    labels = np.asarray(LABELS)
    result = {
        "status": "exploratory_native_preformal_signal_calibration_not_deployed",
        "bundle_sha256": runtime.sha256,
        "source_sha256": {session: data[session]["audit"]["sha256"]
                          for session in ("S01", "S02")},
        "source_median_calibration_rms_by_channel_adc_counts": median_window_rms(source_profile),
        "source_median_formal_rms_by_channel_adc_counts": median_window_rms(source_formal),
        "fixed_gains": list(GAINS),
        "budgets": [0, 1, 2],
        "method": "Source per-channel median 200ms RMS is computed from S01/S02 pre-formal 2-block-per-class calibration windows. At each target session, exactly 1 or 2 pre-formal blocks per class estimate channel RMS. Multiply each target formal window by source_RMS/target_RMS clipped to [0.25,4]. Synthetic gain, when present, affects both target calibration and formal windows. The classifier, SPD reference, thresholds and source scaler remain frozen; no target formal labels or samples estimate the correction.",
        "validation": None,
        "final": None,
        "boundary": "One participant/day; S01-S03 collection readiness failed and S04 had already been inspected in prior studies. Cued-stable 200ms formal windows are not continuous live decoding. Synthetic uniform gain does not reproduce electrode geometry, ADC saturation, per-channel noise or another wearing. No method was selected from S04, and no app default changed.",
    }
    for session in ("S03", "S04"):
        if session == "S04":
            # S04 stays unopened until the method and S03 diagnostic are fixed.
            data[session] = load_session(source / "2026-09-18_S04", session,
                                         filter_mode="causal")
        target = data[session]
        expected = frozen[session]["models"]["F0+SPD"]
        if (target["audit"]["sha256"] != frozen[session]["source_hdf5_sha256"] or
                runtime.sha256 != expected["bundle_sha256"]):
            raise ValueError(f"Song source/model digest drift: {session}")
        cases = []
        for gain in GAINS:
            rows = []
            for budget in (0, 1, 2):
                scale = (np.ones(8) if budget == 0 else correction(
                    source_profile, native_calibration_windows(target, budget) * gain))
                score = evaluate(runtime, target, gain, scale, labels)
                rows.append({"budget_blocks_per_class": budget,
                             "channel_scale": scale.tolist(), **score})
            cases.append({"gain": gain, "budgets": rows})
        baseline = next(case for case in cases if case["gain"] == 1.0)["budgets"][0]
        for new_key, old_key in (
            ("accuracy", "replayed_accuracy"), ("macro_f1", "replayed_macro_f1"),
            ("log_loss", "replayed_log_loss"), ("brier", "replayed_brier"),
        ):
            if abs(baseline[new_key] - expected[old_key]) > 1e-5:
                raise ValueError(f"uncalibrated baseline drift: {session}/{new_key}")
        result["validation" if session == "S03" else "final"] = {
            "session": session, "source_sha256": target["audit"]["sha256"],
            "readiness": target["audit"]["readiness"],
            "calibration_elapsed_seconds": target["calibration_elapsed_seconds"],
            "posthoc_amplitude_context_not_used_for_correction": {
                "formal_median_rms_by_channel_adc_counts":
                    median_window_rms(target["batch"].emg),
                "calibration_1_median_rms_by_channel_adc_counts":
                    median_window_rms(native_calibration_windows(target, 1)),
                "calibration_2_median_rms_by_channel_adc_counts":
                    median_window_rms(native_calibration_windows(target, 2)),
            },
            "cases": cases,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for key in ("validation", "final"):
        cases = result[key]["cases"]
        for gain in (.25, 1.0):
            rows = next(case for case in cases if case["gain"] == gain)["budgets"]
            print(json.dumps({"session": result[key]["session"], "gain": gain,
                              "macro_f1_0_1_2": [row["macro_f1"] for row in rows],
                              "log_loss_0_1_2": [row["log_loss"] for row in rows]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.collection_root, args.reference, args.output)
