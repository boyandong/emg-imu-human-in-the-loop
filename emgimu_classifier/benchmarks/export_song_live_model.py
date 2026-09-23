"""Export the source-trained causal Song F0 model as a small, data-free JSON bundle."""
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
from emgimu.feature_bank.families import LocalDetailFamily


def export(source: Path, output: Path):
    data = {sid: load_session(source / f"2026-09-18_{sid}", sid, "causal")
            for sid in ("S01", "S02", "S03")}
    print("fitting causal F0 from S01/S02", flush=True)
    family = LocalDetailFamily().fit(_join_batches([data["S01"], data["S02"]]))
    train_x = np.concatenate([family.transform(data[sid]["batch"]) for sid in ("S01", "S02")])
    train_y = np.concatenate([data[sid]["hand"] for sid in ("S01", "S02")])
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=0))
    model.fit(train_x, train_y)
    classes = model[-1].classes_.tolist()
    if classes != ["fist", "index_pinch", "neutral", "open_hand"]:
        raise ValueError(f"unexpected class order: {classes}")
    val_x = family.transform(data["S03"]["batch"])
    validation = _trial_metrics(data["S03"]["hand"], model.predict_proba(val_x),
                                data["S03"]["trial"], model[-1].classes_)
    payload = {
        "format_version": 1, "model_kind": "song_real8_causal_f0_logistic",
        "sample_rate_hz": 250, "channels": 8, "window_samples": 50, "hop_samples": 25,
        "filter": {"highpass_hz": 40.0, "highpass_order": 4,
                   "notches_hz": [50.0, 100.0], "notch_q": 30.0,
                   "implementation": "causal_sosfilt_zero_initial_state_continuous"},
        "classes": classes, "f0_thresholds": family.thresholds_.tolist(),
        "standard_scaler_mean": model.named_steps["standardscaler"].mean_.tolist(),
        "standard_scaler_scale": model.named_steps["standardscaler"].scale_.tolist(),
        "logistic_coef": model[-1].coef_.tolist(),
        "logistic_intercept": model[-1].intercept_.tolist(),
    }
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "song_f0_model.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {"format_version": 1, "model_id": "Song real8 causal F0 · exploratory",
                "algorithm_id": "song_real8_local_v1", "artifact": artifact.name,
                "sha256": digest, "source_sessions": ["S01", "S02"],
                "validation_session": "S03",
                "source_hdf5_sha256": {sid: data[sid]["audit"]["sha256"] for sid in data},
                "validation_trial_accuracy": validation["accuracy"],
                "validation_trial_macro_f1": validation["macro_f1"],
                "model_status": "exploratory_one_person_one_day_not_formal_frozen",
                "limitations": "S01-S03 collection readiness failed; cue-labelled stable-window validation is not continuous live accuracy"}
    (output / "song_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"bundle": str(output), "sha256": digest,
                      "S03_trial_accuracy": validation["accuracy"],
                      "S03_trial_macro_f1": validation["macro_f1"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.source, args.output)
