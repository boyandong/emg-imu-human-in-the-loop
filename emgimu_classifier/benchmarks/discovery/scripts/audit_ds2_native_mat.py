"""Read the public DS2 v8 raw MAT after extraction; avoid invented trial labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(raw_path: Path, label_path: Path, output: Path):
    raw = loadmat(raw_path, variable_names=["data_final_all"])["data_final_all"]
    labels = loadmat(label_path, variable_names=["y"])["y"].reshape(-1)
    if raw.shape != (2863, 3, 15000) or raw.dtype != np.float64:
        raise ValueError(f"unexpected public DS2 raw MAT shape/type: {raw.shape}/{raw.dtype}")
    if labels.shape != (332108,) or not set(np.unique(labels)).issubset(set(range(5))):
        raise ValueError("unexpected public DS2 gesture window labels")
    if not np.isfinite(raw).all():
        raise ValueError("public DS2 raw MAT contains NaN/Inf")
    sample_ids = np.random.default_rng(20260924).choice(len(raw), 6, replace=False)
    samples = []
    for index in sample_ids:
        trial = raw[int(index)]
        rms = np.sqrt(np.mean(trial * trial, axis=1))
        samples.append({"raw_trial_index_zero_based": int(index), "channels": 3,
                        "samples_per_channel": 15000, "finite": True,
                        "zero_fraction": float(np.mean(trial == 0)),
                        "rms_per_channel": rms.tolist(),
                        "min_per_channel": trial.min(axis=1).tolist(),
                        "max_per_channel": trial.max(axis=1).tolist()})
    result = {"status": "native_public_v8_mat_shape_and_samples_checked_historical_identity_unproven",
              "raw_mat_sha256": sha256(raw_path), "gesture_label_mat_sha256": sha256(label_path),
              "raw_trials": int(raw.shape[0]), "raw_channels": 3,
              "samples_per_trial_per_channel": 15000,
              "raw_all_finite": True,
              "gesture_window_label_count": len(labels),
              "gesture_window_label_distribution": {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
              "sampled_raw_trials": samples,
              "boundary": "Raw MAT lacks verified per-trial subject, gesture and force mapping. The separate window-label MAT cannot be joined to these raw trials by shape. Sample rate and physical units are not independently encoded or verified here. No old B0/X1-H/X2 input identity or reproduction claim."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "raw_trials", "raw_channels", "gesture_window_label_count")}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_mat", type=Path)
    parser.add_argument("gesture_label_mat", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.raw_mat, args.gesture_label_mat, args.output)
