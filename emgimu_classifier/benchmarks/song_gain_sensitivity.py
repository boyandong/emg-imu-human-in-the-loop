"""Frozen Song-model sensitivity to synthetic, common-channel EMG gain changes.

This is a same-person/day, cued-stable diagnostic; it is not new-wearing data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.song_real8_study import load_session
from benchmarks.song_spd_increment_study import metrics, trial_probabilities


GAINS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
MODELS = ("F0", "F0+SPD")


def summarize(truth: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> dict:
    result = metrics(truth, probability, classes)
    predicted = classes[np.argmax(probability, axis=1)]
    active = truth != "neutral"
    result["predicted_neutral_fraction"] = float(np.mean(predicted == "neutral"))
    result["active_to_neutral_error_fraction"] = float(np.mean(predicted[active] == "neutral"))
    result["predicted_support"] = {
        str(label): int(np.sum(predicted == label)) for label in classes
    }
    return result


def median_window_rms(windows: np.ndarray) -> list[float]:
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (50, 8) or not np.isfinite(values).all():
        raise ValueError("Expected finite causal 50x8 EMG windows")
    return np.median(np.sqrt(np.mean(values ** 2, axis=1)), axis=0).tolist()


def run(source: Path, collection_root: Path, reference: Path, output: Path) -> dict:
    sys.path.insert(0, str(collection_root.resolve()))
    from emgforce.inference.song_local import LABELS, SongLocalRuntime

    frozen = json.loads(reference.read_text(encoding="utf-8"))["sessions"]
    classes = np.asarray(LABELS)
    source_data = [
        load_session(source / f"2026-09-18_{session}", session, filter_mode="causal")
        for session in ("S01", "S02")
    ]
    source_rms = median_window_rms(np.concatenate([d["batch"].emg for d in source_data]))
    result = {
        "status": "synthetic_common_gain_same_person_day_not_deployed",
        "fixed_gains": list(GAINS),
        "method": "Multiply all eight causal-filtered EMG channels by the same gain before frozen-model inference; all three non-overlapping stable windows remain assigned to their original formal trial, with one averaged-probability vote per trial.",
        "source_sessions": [
            {"session": d["audit"]["session"], "source_sha256": d["audit"]["sha256"],
             "collection_readiness": d["audit"]["readiness"]} for d in source_data
        ],
        "source_median_stable_window_rms_by_channel_adc_counts": source_rms,
        "sessions": {},
        "boundary": "This linear filtered-signal perturbation approximates a common analog gain difference only. It does not reproduce changed electrode geometry, per-channel gain, ADC saturation/quantization, physical USB acquisition, uncued transitions or another user/day. S03 failed collection readiness and S04 had already been inspected in earlier work. No model, threshold or product default is selected from these curves.",
    }
    for session in ("S03", "S04"):
        data = load_session(source / f"2026-09-18_{session}", session, filter_mode="causal")
        if data["audit"]["sha256"] != frozen[session]["source_hdf5_sha256"]:
            raise ValueError(f"source digest changed for {session}")
        recorded = {
            "source_sha256": data["audit"]["sha256"],
            "collection_readiness": data["audit"]["readiness"],
            "median_stable_window_rms_by_channel_adc_counts": median_window_rms(data["batch"].emg),
            "models": {},
        }
        recorded["median_rms_ratio_to_source_by_channel"] = (
            np.asarray(recorded["median_stable_window_rms_by_channel_adc_counts"]) /
            np.asarray(source_rms)
        ).tolist()
        for name in MODELS:
            directory = collection_root / "models" / (
                "song_real8_f0_spd" if name == "F0+SPD" else "song_real8_f0"
            )
            runtime = SongLocalRuntime(directory)
            expected = frozen[session]["models"][name]
            if runtime.sha256 != expected["bundle_sha256"]:
                raise ValueError(f"model digest changed for {session}/{name}")
            rows = []
            for gain in GAINS:
                window_probability = np.stack([
                    runtime.predict_filtered_window(window * gain)
                    for window in data["batch"].emg
                ])
                ids, truth, averaged = trial_probabilities(
                    data["hand"], window_probability, data["trial"], classes
                )
                score = summarize(truth, averaged, classes)
                rows.append({"gain": gain, "formal_trials": len(ids), **score})
            baseline = next(row for row in rows if row["gain"] == 1.0)
            for new_key, old_key in (
                ("accuracy", "replayed_accuracy"),
                ("macro_f1", "replayed_macro_f1"),
                ("log_loss", "replayed_log_loss"),
                ("brier", "replayed_brier"),
            ):
                if abs(baseline[new_key] - expected[old_key]) > 1e-5:
                    raise ValueError(f"frozen baseline disagrees with prior audit: {session}/{name}/{new_key}")
            recorded["models"][name] = {"bundle_sha256": runtime.sha256, "gains": rows}
        result["sessions"][session] = recorded
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for session, value in result["sessions"].items():
        for name, model in value["models"].items():
            print(json.dumps({"session": session, "model": name,
                              "gain_0_5_active_to_neutral": model["gains"][1]["active_to_neutral_error_fraction"],
                              "gain_1_active_to_neutral": model["gains"][3]["active_to_neutral_error_fraction"],
                              "gain_0_5_macro_f1": model["gains"][1]["macro_f1"],
                              "gain_1_macro_f1": model["gains"][3]["macro_f1"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.collection_root, args.reference, args.output)
