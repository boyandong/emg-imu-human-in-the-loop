"""Export the source-only causal Song F0+SPD model without raw recordings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.song_real8_study import _join_batches, _trial_metrics, load_session
from emgimu.feature_bank.families import LocalDetailFamily, SpdTangentFamily


def export(source: Path, output: Path) -> dict:
    data = {sid: load_session(source / f"2026-09-18_{sid}", sid, "causal")
            for sid in ("S01", "S02", "S03")}
    print("fit source F0 and SPD from Song S01/S02", flush=True)
    train = _join_batches([data["S01"], data["S02"]])
    f0 = LocalDetailFamily().fit(train)
    spd = SpdTangentFamily().fit(train)

    def matrix(sid):
        return np.concatenate((f0.transform(data[sid]["batch"]),
                               spd.transform(data[sid]["batch"])), axis=1)

    source_x = np.concatenate([matrix(sid) for sid in ("S01", "S02")])
    source_y = np.concatenate([data[sid]["hand"] for sid in ("S01", "S02")])
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=0))
    model.fit(source_x, source_y)
    classes = model[-1].classes_.tolist()
    if classes != ["fist", "index_pinch", "neutral", "open_hand"]:
        raise ValueError("unexpected source Song class order")
    validation = _trial_metrics(data["S03"]["hand"], model.predict_proba(matrix("S03")),
                                data["S03"]["trial"], model[-1].classes_)
    reference = np.asarray(spd.reference_, dtype=np.float64)
    reference_sha = hashlib.sha256(np.ascontiguousarray(reference).tobytes()).hexdigest()
    payload = {
        "format_version": 1, "model_kind": "song_real8_causal_f0_spd_logistic",
        "sample_rate_hz": 250, "channels": 8, "window_samples": 50, "hop_samples": 25,
        "filter": {"highpass_hz": 40.0, "highpass_order": 4,
                   "notches_hz": [50.0, 100.0], "notch_q": 30.0,
                   "implementation": "causal_sosfilt_zero_initial_state_continuous"},
        "classes": classes, "f0_thresholds": f0.thresholds_.tolist(),
        "spd_shrinkage": 0.05, "spd_reference": reference.tolist(),
        "standard_scaler_mean": model.named_steps["standardscaler"].mean_.tolist(),
        "standard_scaler_scale": model.named_steps["standardscaler"].scale_.tolist(),
        "logistic_coef": model[-1].coef_.tolist(),
        "logistic_intercept": model[-1].intercept_.tolist(),
    }
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "song_f0_model.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "format_version": 1, "model_id": "Song real8 causal F0+SPD · exploratory",
        "algorithm_id": "song_real8_local_v1", "artifact": artifact.name,
        "sha256": digest, "source_sessions": ["S01", "S02"],
        "validation_session": "S03", "source_spd_reference_sha256": reference_sha,
        "source_hdf5_sha256": {sid: data[sid]["audit"]["sha256"] for sid in data},
        "validation_trial_accuracy": validation["accuracy"],
        "validation_trial_macro_f1": validation["macro_f1"],
        "model_status": "exploratory_one_person_one_day_not_formal_frozen",
        "limitations": "S01-S03 collection readiness failed; same-person cue-labelled stable trials are not continuous live accuracy",
    }
    (output / "song_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"bundle": str(output), "sha256": digest,
                      "S03_trial_macro_f1": validation["macro_f1"]}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.source, args.output)
