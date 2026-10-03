"""Source-frozen F9 V3 observation replay on existing Song raw ADC windows."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from benchmarks.song_quality_observability import raw_formal_windows
from benchmarks.song_real8_study import load_session
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.document_quality_v3 import DocumentQualityObservationsV3


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE / "F9_DOCUMENT_V3_PROTOCOL.json"
OUTPUT = HERE / "F9_DOCUMENT_V3_RESULTS.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group(values: np.ndarray, names: tuple[str, ...], metric: str) -> np.ndarray:
    return values[:, [names.index(f"F9v3.{metric}.ch{channel}") for channel in range(1, 9)]]


def run(source: Path) -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent = HERE / "QUALITY_OBSERVABILITY.json"
    if sha(parent) != protocol["parent_quality_result_sha256"]:
        raise ValueError("parent Song source inventory changed")
    raw = {}
    trial_ids = {}
    for session in protocol["source_sessions"] + protocol["evaluation_sessions"]:
        folder = source / f"2026-09-18_{session}"
        data = load_session(folder, session, filter_mode="causal")
        if (data["audit"]["sha256"] != protocol["expected_session_sha256"][session]
                or len(data["trial"]) != protocol["expected_formal_windows"][session]):
            raise ValueError(f"Song source session changed: {session}")
        raw[session] = raw_formal_windows(folder, session, data["trial"])
        trial_ids[session] = set(data["trial"].tolist())
        print(f"F9 V3 loaded {session}: {len(raw[session])} windows", flush=True)
    if any(trial_ids[a] & trial_ids[b] for a in protocol["source_sessions"]
           for b in protocol["evaluation_sessions"]):
        raise ValueError("source/evaluation trial identities overlap")
    training = np.concatenate([raw[session] for session in protocol["source_sessions"]])
    if training.shape[1:] != (protocol["window_samples"], protocol["channels"]):
        raise ValueError("Song raw window contract changed")
    family = DocumentQualityObservationsV3(
        adc_range=tuple(protocol["adc_range_counts"]),
        line_frequency_hz=protocol["line_frequency_hz"],
        pre_highpass_available=protocol["pre_highpass_available"],
        ring_order=protocol["ring_order"],
    ).fit(FeatureBatch(training, protocol["emg_sample_rate_hz"]))
    names = family.feature_names
    frozen = pickle.dumps(family)
    results = {}
    for session in protocol["evaluation_sessions"]:
        values = raw[session]
        observed = family.transform(FeatureBatch(values, protocol["emg_sample_rate_hz"]))
        fault = values.copy()
        fault[:, :, 0] = fault[:, :1, 0]
        artificial = family.transform(FeatureBatch(fault, protocol["emg_sample_rate_hz"]))
        if not np.isfinite(observed).all() or not np.isfinite(artificial).all():
            raise ValueError("nonfinite Song quality observation")
        flat = group(observed, names, "longest_flatline_ratio")
        fault_flat = group(artificial, names, "longest_flatline_ratio")
        flags = {key: float(observed[0, names.index(f"F9v3.available.{key}")])
                 for key in ("adc", "line", "low_frequency", "ring")}
        if flags != {"adc": 1.0, "line": 0.0, "low_frequency": 1.0, "ring": 0.0}:
            raise ValueError("unexpected Song metadata availability")
        results[session] = {
            "formal_windows": len(values),
            "feature_dimension": observed.shape[1],
            "availability": flags,
            "median_zero_fraction": float(np.median(group(observed, names, "zero_fraction"))),
            "median_longest_flatline_ratio": float(np.median(flat)),
            "median_absolute_robust_amplitude_z": float(np.median(np.abs(group(observed, names, "amplitude_z")))),
            "median_covariance_distance": float(np.median(observed[:, names.index("F9v3.covariance_distance")])),
            "synthetic_constant_ch1_median_flatline": float(np.median(fault_flat[:, 0])),
            "synthetic_constant_ch1_detected_fraction_at_0_9": float(np.mean(fault_flat[:, 0] > .9)),
            "natural_ch1_above_0_9_fraction": float(np.mean(flat[:, 0] > .9)),
        }
        print(f"F9 V3 summarized {session}: {len(values)} windows", flush=True)
    if pickle.dumps(family) != frozen:
        raise ValueError("held-out Song transform changed source state")
    result = {"status": "native_observation_only", "protocol_sha256": sha(PROTOCOL),
              "parent_quality_result_sha256": sha(parent),
              "source_sessions": protocol["source_sessions"],
              "evaluation_sessions": results,
              "source_formal_windows": len(training),
              "feature_dimension": len(names),
              "boundary": protocol["boundary"]}
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    run(args.source)
