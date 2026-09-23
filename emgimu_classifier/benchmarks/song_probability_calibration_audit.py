"""Trial-level reliability of already-exported Song F0 and F0+SPD bundles."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.song_real8_study import load_session
from benchmarks.song_spd_increment_study import metrics, trial_probabilities


def calibration_bins(truth, probability, classes, bins: int = 10):
    y = np.asarray(truth)
    p = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(classes)
    if (bins < 2 or p.ndim != 2 or p.shape != (len(y), len(labels)) or len(y) == 0 or
            not np.isfinite(p).all() or np.any(p < 0) or
            not np.allclose(p.sum(axis=1), 1, atol=1e-5) or
            not np.isin(y, labels).all()):
        raise ValueError("Finite normalized probabilities and aligned known labels required")
    peak = p.max(axis=1)
    correct = labels[p.argmax(axis=1)] == y
    indices = np.minimum((peak * bins).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        selected = indices == index
        count = int(selected.sum())
        mean_confidence = float(peak[selected].mean()) if count else None
        accuracy = float(correct[selected].mean()) if count else None
        rows.append({
            "bin": index,
            "lower_inclusive": index / bins,
            "upper_inclusive_only_for_final_bin": (index + 1) / bins,
            "trials": count,
            "mean_confidence": mean_confidence,
            "observed_accuracy": accuracy,
            "absolute_gap": abs(accuracy - mean_confidence) if count else None,
        })
    ece = sum(row["trials"] * row["absolute_gap"] for row in rows if row["trials"]) / len(y)
    if sum(row["trials"] for row in rows) != len(y):
        raise AssertionError("calibration bins omitted trials")
    return rows, float(ece)


def run(source: Path, collection_root: Path, bundles: dict[str, Path],
        baseline: Path, output: Path, csv_output: Path) -> dict:
    sys.path.insert(0, str(collection_root.resolve()))
    from emgforce.inference.song_local import LABELS, SongLocalRuntime

    saved = json.loads(baseline.read_text(encoding="utf-8"))
    classes = np.asarray(LABELS)
    result = {
        "status": "held_out_cued_stable_trial_reliability_not_live_calibration",
        "source_result": str(baseline),
        "binning": "10 equal-width bins of maximum four-class trial-mean probability; ECE weights absolute confidence/accuracy gap by trial count; no bin fitting or temperature refit",
        "sessions": {},
        "boundary": "S01/S02 source-trained bundles are unchanged. S03 validation and previously inspected S04 final contain one person's same-day completed cued stable trials only. Trial confidence is mean of up to three nonoverlapping 200 ms window probabilities. Reliability here cannot certify live human/device calibration, cross-person/day transfer or clinical use.",
    }
    csv_rows = []
    for session, phase in (("S03", "validation"), ("S04", "final")):
        data = load_session(source / f"2026-09-18_{session}", session, filter_mode="causal")
        result["sessions"][session] = {
            "source_hdf5_sha256": data["audit"]["sha256"], "models": {},
        }
        if data["audit"]["sha256"] != saved["source_hdf5_sha256"][session]:
            raise ValueError("Saved SPD study and current Song file differ")
        paired_predictions = {}
        for name, bundle in bundles.items():
            runtime = SongLocalRuntime(bundle)
            window_probability = np.stack([
                runtime.predict_filtered_window(window) for window in data["batch"].emg
            ])
            ids, truth, trial_probability = trial_probabilities(
                data["hand"], window_probability, data["trial"], classes)
            scored = metrics(truth, trial_probability, classes)
            saved_name = "F0" if name == "F0" else "F0_plus_SPD"
            reference = saved["source_zero_shot"][phase][saved_name]
            for key in ("trials", "accuracy", "macro_f1", "log_loss", "brier"):
                if abs(scored[key] - reference[key]) > 1e-5:
                    raise AssertionError(f"{session}/{name} {key} differs from frozen SPD study")
            bins, ece = calibration_bins(truth, trial_probability, classes)
            paired_predictions[name] = (ids.copy(), truth.copy(), classes[np.argmax(trial_probability, axis=1)])
            result["sessions"][session]["models"][name] = {
                "bundle_sha256": runtime.sha256,
                "formal_trial_count": len(ids),
                "replayed_accuracy": scored["accuracy"],
                "replayed_macro_f1": scored["macro_f1"],
                "replayed_log_loss": scored["log_loss"],
                "replayed_brier": scored["brier"],
                "top_label_ece_10_bins": ece,
                "bins": bins,
            }
            for row in bins:
                csv_rows.append({"dataset": "Song_real8_one_person_one_day",
                                 "session": session, "phase": phase,
                                 "feature_bank": name, **row})
            print(json.dumps({"session": session, "model": name,
                              "trials": len(ids), "ece": ece}), flush=True)
        if "F0" in paired_predictions and "F0+SPD" in paired_predictions:
            left_ids, left_truth, left_pred = paired_predictions["F0"]
            right_ids, right_truth, right_pred = paired_predictions["F0+SPD"]
            if not np.array_equal(left_ids, right_ids) or not np.array_equal(left_truth, right_truth):
                raise AssertionError("Song model trial identities or labels differ")
            left_error = left_pred != left_truth
            right_error = right_pred != right_truth
            corrected = int(np.sum(left_error & ~right_error))
            new_errors = int(np.sum(~left_error & right_error))
            frozen = saved["source_zero_shot_paired_vs_F0"][phase]
            if corrected != frozen["corrected_trials"] or new_errors != frozen["new_errors"]:
                raise AssertionError("Song paired decisions differ from frozen SPD study")
            error_correlation = (float(np.corrcoef(left_error.astype(float), right_error.astype(float))[0, 1])
                                 if np.any(left_error) and np.any(~left_error) and
                                 np.any(right_error) and np.any(~right_error) else None)
            result["sessions"][session]["paired_F0_vs_SPD"] = {
                "trials": len(left_ids),
                "prediction_disagreement_rate": float(np.mean(left_pred != right_pred)),
                "error_correlation": error_correlation,
                "F0_correct_SPD_wrong": new_errors,
                "F0_wrong_SPD_correct": corrected,
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--f0-bundle", required=True, type=Path)
    parser.add_argument("--spd-bundle", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.collection_root,
        {"F0": args.f0_bundle, "F0+SPD": args.spd_bundle},
        args.baseline, args.output, args.csv_output)
