"""Source-frozen F9 observations on actual Song raw ADC windows.

No quality threshold is adopted by the app from this one-person/day audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import h5py
import numpy as np

from benchmarks.song_real8_study import WINDOW, _text, load_session, window_starts
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.quality_observability import QualityObservabilityFamily


def raw_formal_windows(folder: Path, session: str, expected_ids: np.ndarray) -> np.ndarray:
    """Use exactly the already-audited, IMU-eligible formal stable windows."""
    with h5py.File(folder / "session.h5") as handle:
        raw = handle["streams/emg/raw"][:]
        imu_indices = handle["streams/imu/emg_sample_index"][:]
        trials = handle["trials"][:]
    windows, ids = [], []
    for row in trials:
        if (_text(row["trial_kind"]) != "formal" or not bool(row["valid"]) or
                _text(row["completion_status"]) != "completed"):
            continue
        start, end = int(row["stable_start_sample"]), int(row["stable_end_sample"])
        if not 0 <= start < end <= len(raw):
            continue
        for left in window_starts(start, end):
            right = int(left) + WINDOW
            last_imu = int(np.searchsorted(imu_indices, right - 1, side="right"))
            if last_imu < 22:
                continue
            windows.append(raw[left:right])
            ids.append(f"{session}:{int(row['trial_id'])}")
    if not np.array_equal(np.asarray(ids), expected_ids):
        raise ValueError(f"raw and frozen causal window identities differ: {session}")
    result = np.asarray(windows)
    if result.shape != (len(expected_ids), 50, 8):
        raise ValueError(f"unexpected Song raw shape: {session}")
    return result


def quality_summary(features: np.ndarray, names: tuple[str, ...], labels: np.ndarray) -> dict:
    def column(name):
        return np.asarray(features[:, names.index(name)], dtype=np.float64)

    mean = column("F9.mean_quality")
    minimum = column("F9.min_quality")
    bad = column("F9.bad_channel_count")
    if len(labels) != len(mean):
        raise ValueError("quality rows and hand labels differ")
    hands = ("neutral", "index_pinch", "fist", "open_hand")
    if set(labels) != set(hands):
        raise ValueError("all four Song hand classes must be observed")
    return {
        "windows": len(labels),
        "mean_quality": float(np.mean(mean)),
        "min_quality_median": float(np.median(minimum)),
        "any_bad_channel_fraction": float(np.mean(bad >= 1)),
        "hypothetical_min_quality_below_0_5_fraction": float(np.mean(minimum < .5)),
        "by_hand": {
            label: {
                "windows": int(np.sum(labels == label)),
                "mean_quality": float(np.mean(mean[labels == label])),
                "hypothetical_min_quality_below_0_5_fraction":
                    float(np.mean(minimum[labels == label] < .5)),
            }
            for label in hands
        },
    }


def component_summary(features: np.ndarray, names: tuple[str, ...]) -> dict:
    def channels(prefix: str) -> np.ndarray:
        return np.asarray(features[:, [names.index(f"{prefix}.ch{c}") for c in range(1, 9)]],
                          dtype=np.float64)

    amplitude = np.any(np.abs(channels("F9.amplitude_z")) > 3.0, axis=1)
    zero = np.any(channels("F9.zero_fraction") > .5, axis=1)
    flat = np.any(channels("F9.flatline_fraction") > .5, axis=1)
    clip = np.any(channels("F9.clip_fraction") > .5, axis=1)
    hardware = np.any(channels("F9v2.longest_flatline_ratio") > .9, axis=1) | np.any(
        channels("F9.clip_fraction") > .01, axis=1)
    legacy = np.asarray(features[:, names.index("F9.min_quality")]) < .5
    return {
        "legacy_quality_flag_fraction": float(np.mean(legacy)),
        "amplitude_z_over_3_fraction": float(np.mean(amplitude)),
        "zero_fraction_over_0_5_fraction": float(np.mean(zero)),
        "flatline_fraction_over_0_5_fraction": float(np.mean(flat)),
        "clip_fraction_over_0_5_fraction": float(np.mean(clip)),
        "candidate_hardware_only_flag_fraction": float(np.mean(hardware)),
        "legacy_flag_not_explained_by_components": int(np.sum(legacy & ~(amplitude | zero | flat | clip))),
    }


def run(source: Path, output: Path) -> dict:
    data, raw = {}, {}
    for session in ("S01", "S02", "S03", "S04"):
        folder = source / f"2026-09-18_{session}"
        data[session] = load_session(folder, session, filter_mode="causal")
        raw[session] = raw_formal_windows(folder, session, data[session]["trial"])
    source_batch = FeatureBatch(np.concatenate((raw["S01"], raw["S02"])), 250.0)
    family = QualityObservabilityFamily(
        adc_min=-8388608, adc_max=8388607, line_frequency_hz=50.0,
        pre_highpass_available=True, ring_topology=False,
    ).fit(source_batch)
    source_state = pickle.dumps(family)
    names = family.feature_names
    result = {
        "status": "raw_adc_same_person_day_F9_observation_not_deployed",
        "source_sessions": [
            {"session": session, "sha256": data[session]["audit"]["sha256"],
             "readiness": data[session]["audit"]["readiness"],
             "formal_windows": len(raw[session])}
            for session in ("S01", "S02")
        ],
        "source_f9_state_sha256": hashlib.sha256(source_state).hexdigest(),
        "adc_range_counts": [-8388608, 8388607],
        "sampling_rate_hz": 250,
        "window_samples": 50,
        "availability": family.availability_,
        "feature_dimension": len(names),
        "validation_and_final": {},
        "boundary": "All F9 references fit on S01/S02 formal stable raw-count windows only. S03/S04 remain read-only; their labels summarize potential gesture-dependent rejection rather than fit quality rules. A constant-channel synthetic fault tests detection, not observed hardware failure. The <0.5 minimum-quality gate and separate hardware-only flatline/clipping candidate are hypothetical, not chosen or deployed. Legacy F9 quality does not include line-noise, low-frequency or covariance scores in its quality aggregate. Source sessions and S03 failed collection readiness; S04 was previously inspected; one participant/day cannot validate another wearing, live recognition, or fault prevalence.",
    }
    for session in ("S03", "S04"):
        values = raw[session]
        batch = FeatureBatch(values, 250.0)
        clean = family.transform(batch)
        fault = values.copy()
        fault[:, :, 0] = fault[:, :1, 0]
        corrupted = family.transform(FeatureBatch(fault, 250.0))
        clean_flat = clean[:, names.index("F9v2.longest_flatline_ratio.ch1")]
        fault_flat = corrupted[:, names.index("F9v2.longest_flatline_ratio.ch1")]
        clean_min = clean[:, names.index("F9.min_quality")]
        fault_min = corrupted[:, names.index("F9.min_quality")]
        result["validation_and_final"][session] = {
            "sha256": data[session]["audit"]["sha256"],
            "readiness": data[session]["audit"]["readiness"],
            "raw_formal_windows": len(values),
            "observed": quality_summary(clean, names, data[session]["hand"]),
            "observed_quality_flag_components": component_summary(clean, names),
            "synthetic_ch1_constant": {
                "longest_flatline_median_clean": float(np.median(clean_flat)),
                "longest_flatline_median_fault": float(np.median(fault_flat)),
                "min_quality_below_0_5_fraction_clean": float(np.mean(clean_min < .5)),
                "min_quality_below_0_5_fraction_fault": float(np.mean(fault_min < .5)),
                "detected_fault_fraction": float(np.mean(fault_flat > .9)),
                "candidate_hardware_only_flag_fraction":
                    component_summary(corrupted, names)["candidate_hardware_only_flag_fraction"],
            },
        }
    if pickle.dumps(family) != source_state:
        raise AssertionError("F9 source state changed during target observation")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({session: {
        "observed_any_bad_channel": result["validation_and_final"][session]["observed"]["any_bad_channel_fraction"],
        "synthetic_fault_detected": result["validation_and_final"][session]["synthetic_ch1_constant"]["detected_fault_fraction"],
    } for session in ("S03", "S04")}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
