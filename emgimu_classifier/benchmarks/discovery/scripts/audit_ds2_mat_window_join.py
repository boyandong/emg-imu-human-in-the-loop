"""Prove the public DS2 raw-MAT to MAV-window order without inventing force IDs.

The publication specifies 1500 Hz, a 2 s initial rest, 6 s action, 250 ms
windows and 80% overlap: 3000 + 75*k, width 375, k=0..115.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


TRIALS = 2863
WINDOWS = 116
CHANNELS = 3
SAMPLES = 15000
STARTS = 3000 + 75 * np.arange(WINDOWS)
WIDTH = 375
TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trial_mav(raw_trial: np.ndarray) -> np.ndarray:
    x = np.asarray(raw_trial, dtype=np.float64)
    if x.shape != (CHANNELS, SAMPLES) or not np.isfinite(x).all():
        raise ValueError("DS2 raw trial must be finite 3x15000")
    prefix = np.concatenate((np.zeros((CHANNELS, 1)),
                             np.cumsum(np.abs(x), axis=1)), axis=1)
    return ((prefix[:, STARTS + WIDTH] - prefix[:, STARTS]) / WIDTH).T


def audit(raw_path: Path, mav_path: Path, label_path: Path,
          native_audit: Path, output: Path) -> dict:
    prior = json.loads(native_audit.read_text(encoding="utf-8"))
    hashes = {"raw_mat_sha256": sha256(raw_path),
              "mav_mat_sha256": sha256(mav_path),
              "gesture_label_mat_sha256": sha256(label_path)}
    for name in ("raw_mat_sha256", "gesture_label_mat_sha256"):
        if hashes[name] != prior[name]:
            raise ValueError(f"Public DS2 source digest changed: {name}")
    raw = loadmat(raw_path, variable_names=["data_final_all"])["data_final_all"]
    mav = loadmat(mav_path, variable_names=["matrix_all_MAV"])["matrix_all_MAV"]
    labels = loadmat(label_path, variable_names=["y"])["y"].reshape(-1)
    if raw.shape != (TRIALS, CHANNELS, SAMPLES) or mav.shape != (
            TRIALS * WINDOWS, CHANNELS) or labels.shape != (TRIALS * WINDOWS,):
        raise ValueError("DS2 MAT array shapes do not support the publication window contract")
    if not np.isfinite(mav).all() or not np.isfinite(labels).all():
        raise ValueError("Nonfinite DS2 features or labels")
    if np.any(labels != np.floor(labels)) or not set(labels.astype(int)).issubset(range(5)):
        raise ValueError("Unexpected DS2 gesture label value")
    blocks = labels.astype(np.int8).reshape(TRIALS, WINDOWS)
    rows = []
    global_max_error = 0.0
    mismatched_values = 0
    uniform_trials = 0
    mixed = []
    for trial in range(TRIALS):
        expected = trial_mav(raw[trial])
        observed = mav[trial * WINDOWS:(trial + 1) * WINDOWS]
        error = np.abs(expected - observed)
        max_error = float(np.max(error))
        global_max_error = max(global_max_error, max_error)
        mismatch = int(np.sum(error > TOLERANCE))
        mismatched_values += mismatch
        unique, counts = np.unique(blocks[trial], return_counts=True)
        uniform = len(unique) == 1
        uniform_trials += uniform
        if not uniform:
            mixed.append({"raw_trial_index_zero_based": trial,
                          "label_counts": {str(int(k)): int(v)
                                           for k, v in zip(unique, counts)},
                          "label_change_window_zero_based": np.flatnonzero(
                              np.diff(blocks[trial]) != 0).astype(int).tolist()})
        rows.append({"raw_trial_index_zero_based": trial,
                     "first_window_index_zero_based": trial * WINDOWS,
                     "last_window_index_zero_based": (trial + 1) * WINDOWS - 1,
                     "mav_max_abs_error": max_error,
                     "mav_values_over_tolerance": mismatch,
                     "uniform_gesture_label": bool(uniform),
                     "gesture_label_if_uniform": int(unique[0]) if uniform else "N/A"})
    if mismatched_values:
        raise ValueError(f"MAV/raw ordinal join fails at {mismatched_values} values")
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "DS2_MAT_TRIAL_WINDOW_JOIN.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "public_ds2_raw_to_mav_window_order_verified_gesture_labels_partial",
        "publication": "https://www.mdpi.com/2306-5729/10/12/194",
        "source_sha256": hashes,
        "native_audit_sha256": sha256(native_audit),
        "window_contract": {"sample_rate_hz": 1500, "raw_samples_per_trial": SAMPLES,
                            "action_start_sample_zero_based": 3000,
                            "action_samples": 9000, "window_samples": WIDTH,
                            "hop_samples": 75, "windows_per_trial": WINDOWS},
        "raw_trials": TRIALS,
        "mav_window_rows": len(mav),
        "mav_values_compared": int(mav.size),
        "mav_values_over_1e_minus_10": mismatched_values,
        "global_max_mav_abs_error": global_max_error,
        "uniform_gesture_label_trials": uniform_trials,
        "mixed_gesture_label_trials": mixed,
        "trial_join_csv": csv_path.name,
        "trial_join_csv_sha256": sha256(csv_path),
        "boundary": "The exact MAV reconstruction proves ordinal alignment of public raw MAT trials with all 116-window feature blocks. Uniform label blocks support a gesture code for 2862 trials only; the mixed block is retained as ambiguous. No subject ID, force level, TDMS segment identity or old B0/X1-H/X2 experiment-input identity is inferred. Gesture code names require the publication's ordering and independent validation before use as named classes.",
    }
    (output / "DS2_MAT_TRIAL_WINDOW_JOIN_AUDIT.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"mav_values_compared": result["mav_values_compared"],
                      "mav_values_over_tolerance": mismatched_values,
                      "global_max_mav_abs_error": global_max_error,
                      "uniform_gesture_label_trials": uniform_trials,
                      "mixed_gesture_label_trials": mixed}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-mat", required=True, type=Path)
    parser.add_argument("--mav-mat", required=True, type=Path)
    parser.add_argument("--label-mat", required=True, type=Path)
    parser.add_argument("--native-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit(args.raw_mat, args.mav_mat, args.label_mat, args.native_audit, args.output)
