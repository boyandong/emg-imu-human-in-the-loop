"""Source-frozen F9 candidate gate on matched unseen-user GRABMyo trials."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from benchmarks.grabmyo_crossday import run as grab
from benchmarks.new_bank_v2 import grab_user_run as parent_run
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.quality_mask_v1 import SourceCalibratedQualityMask
from emgimu.feature_bank.quality_observability import QualityObservabilityFamily


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "F9_GRAB_GATE_PROTOCOL.json"
RESULT_PATH = ROOT / "F9_GRAB_GATE_RESULTS.json"
ROWS_PATH = ROOT / "F9_GRAB_GATE_TRIALS.csv"
CLASSES = tuple(grab.GESTURES)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    accepted = [row for row in rows if not row["rejected"]]
    rejected = [row for row in rows if row["rejected"]]
    error = lambda row: not row["f0_correct"]
    return {
        "trials": total,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "coverage": len(accepted) / total if total else None,
        "baseline_errors": sum(map(error, rows)),
        "errors_rejected": sum(map(error, rejected)),
        "correct_rejected": sum(row["f0_correct"] for row in rejected),
        "accepted_error_rate": sum(map(error, accepted)) / len(accepted) if accepted else None,
        "rejection_by_gesture": {str(gesture): sum(row["rejected"] and row["gesture"] == gesture
                                          for row in rows) for gesture in CLASSES},
    }


def evaluate(data_root: Path) -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    parent_run.check_protocol()
    for filename, key in (("GRAB_USER_PROTOCOL.json", "parent_protocol_sha256"),
                          ("GRAB_USER_RESULTS.json", "parent_result_sha256"),
                          ("GRAB_USER_PREDICTIONS.csv", "parent_prediction_sha256")):
        if sha(ROOT / filename) != protocol[key]:
            raise AssertionError(f"frozen GRAB parent changed: {filename}")
    parent = json.loads((ROOT / "GRAB_USER_RESULTS.json").read_text(encoding="utf-8"))
    records = [row for row in grab.records() if row["session"] == parent_run.PROTOCOL["day"]]
    if len(records) != 224:
        raise AssertionError("GRAB Day1 native inventory changed")
    manifest_sha = parent_run.check_files(data_root, records)
    if manifest_sha != parent["official_sha256_manifest_sha256"]:
        raise AssertionError("GRAB official source manifest changed")
    with (ROOT / "GRAB_USER_PREDICTIONS.csv").open(encoding="utf-8", newline="") as stream:
        predictions = {(row["phase"], row["trial_id"]): row for row in csv.DictReader(stream)
                       if row["arm"] == "F0v2"}
    if len(predictions) != 112 or protocol["source_quantile"] != .995:
        raise AssertionError("frozen F0v2 coverage or F9 rule changed")
    source_records = [row for row in records if row["subject"] in parent_run.PROTOCOL["source_subjects"]]
    source_windows = np.concatenate([grab.read_record(data_root, row) for row in source_records])
    if source_windows.shape != (2240, grab.WINDOW, len(grab.CHANNELS)):
        raise AssertionError("GRAB F9 source window topology changed")
    source = FeatureBatch(source_windows, grab.RATE)
    family = QualityObservabilityFamily(pre_highpass_available=False,
        line_frequency_available=False, ring_topology=False).fit(source)
    mask = SourceCalibratedQualityMask(source_quantile=protocol["source_quantile"]).fit(
        family.transform(source), family.feature_names)
    expected_availability = {"adc_clipping": False, "line_noise": False,
                             "low_frequency_pre_highpass": False}
    if family.availability_ != expected_availability or mask.available_ != expected_availability:
        raise AssertionError("unknown GRAB quality metadata was treated as available")
    frozen = pickle.dumps((family, mask))
    rows = []
    for phase in ("validation", "final"):
        target_ids = set(parent[f"{phase}_trial_ids"])
        target_records = [row for row in records if row["stem"] in target_ids]
        if len(target_records) != 56 or {row["stem"] for row in target_records} != target_ids:
            raise AssertionError(f"GRAB {phase} trial identities changed")
        for count, record in enumerate(target_records, 1):
            current = FeatureBatch(grab.read_record(data_root, record), grab.RATE)
            quality = mask.transform(family.transform(current), family.feature_names)
            if quality.shape != (20, 12):
                raise AssertionError("GRAB twenty-window F9 output changed")
            previous = predictions[(phase, record["stem"])]
            if (int(previous["subject"]) != record["subject"] or
                    int(previous["gesture"]) != record["gesture"]):
                raise AssertionError("frozen F0v2 trial identity/label mismatch")
            probability = np.array([float(previous[f"p_{label}"]) for label in CLASSES])
            if not np.isfinite(probability).all() or abs(probability.sum() - 1.) > 1e-5:
                raise AssertionError("frozen F0v2 probability invalid")
            rows.append({"phase": phase, "trial_id": record["stem"],
                         "subject": record["subject"], "gesture": record["gesture"],
                         "f0_correct": CLASSES[int(np.argmax(probability))] == record["gesture"],
                         "min_quality": float(np.min(quality[:, -2])),
                         "mean_quality": float(np.mean(quality[:, -3])),
                         "rejected": bool(np.min(quality[:, -2]) < .5)})
            if count % 14 == 0:
                print(f"GRAB F9 {phase}: {count}/{len(target_records)} trials", flush=True)
    if frozen != pickle.dumps((family, mask)):
        raise AssertionError("target transforms changed source-fitted F9 state")
    if len(rows) != 112 or {(row["phase"], row["trial_id"]) for row in rows} != set(predictions):
        raise AssertionError("matched GRAB F0/F9 target coverage changed")
    with ROWS_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "public_unseen_user_f9_diagnostic_gate_not_deployed",
        "protocol_sha256": sha(PROTOCOL_PATH),
        "parent_prediction_sha256": protocol["parent_prediction_sha256"],
        "official_sha256_manifest_sha256": manifest_sha,
        "source_state_sha256": hashlib.sha256(frozen).hexdigest(),
        "source_trials": len(source_records), "source_windows": source.windows,
        "availability": family.availability_,
        "thresholds": {key: value.tolist() for key, value in mask.thresholds_.items()},
        "trial_rows_sha256": sha(ROWS_PATH),
        "phases": {phase: {"pooled": summarize([row for row in rows if row["phase"] == phase]),
                           "subjects": {str(subject): summarize([row for row in rows
                                                                  if row["phase"] == phase and row["subject"] == subject])
                                        for subject in parent_run.PROTOCOL[f"{phase}_subjects"]}}
                   for phase in ("validation", "final")},
        "scope": "Public 8-channel 2048 Hz same-day unseen-user quality-gate diagnostic against unchanged F0v2 predictions. Unknown ADC rail, mains frequency and pre-highpass status are explicitly masked. No target refit or gate selection, 250 Hz own-device transfer, physical-fault labels or live safety claim; final subjects are descriptive.",
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for phase in ("validation", "final"):
        summary = result["phases"][phase]["pooled"]
        print(f"GRAB F9 {phase}: coverage={summary['coverage']:.3f}; "
              f"errors rejected={summary['errors_rejected']}; "
              f"correct rejected={summary['correct_rejected']}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=Path("D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1"))
    args = parser.parse_args()
    evaluate(args.data_root)
