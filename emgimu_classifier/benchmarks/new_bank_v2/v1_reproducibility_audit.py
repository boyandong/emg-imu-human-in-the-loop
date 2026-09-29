"""Recheck saved protocol, split, model and prediction provenance for seven screens."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parent
PREFIXES = ("ROAM_V1_EXTENSION", "GRAB_V1_EXTENSION", "GRAB_DAY_V1_EXTENSION",
            "FORCE_V1_EXTENSION", "WEARING_V1_EXTENSION", "MANUS_V1_SPEED",
            "ROAM_V1_QUALITY")
EXPECTED_ROWS = (1800, 560, 2240, 5880, 1200, 1080, 6300)
FAMILIES = ("scale_pattern", "ring_lag", "correlation_spectrum", "frequency_direction")


def identity_sets(result: dict, prefix: str) -> tuple[set, set, set]:
    if prefix in PREFIXES[:3]:
        return (set(result["source_trial_ids"]), set(result["validation_trial_ids"]),
                set(result["final_trial_ids"]))
    split = result["split_trial_ids"]
    if prefix == "WEARING_V1_EXTENSION":
        source = {trial for item in split.values() for trial in item["source"]}
        validation = {trial for name, item in split.items() if name.startswith("validation_")
                      for trial in item["target"]}
        final = {trial for name, item in split.items() if name.startswith("final_")
                 for trial in item["target"]}
        return source, validation, final
    if prefix == "ROAM_V1_QUALITY":
        return (set(result["source_trial_ids"]), set(split["validation"]),
                set(split["final"]))
    return (set(split["validation"]["source_trials"]),
            set(split["validation"]["target_trials"]),
            set(split["final"]["target_trials"]))


def build() -> dict:
    rows = []
    for prefix, expected in zip(PREFIXES, EXPECTED_ROWS):
        protocol_path = ROOT / f"{prefix}_PROTOCOL.json"
        result_path = ROOT / f"{prefix}_RESULTS.json"
        prediction_path = ROOT / f"{prefix}_PREDICTIONS.csv"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["protocol"] != protocol or result["protocol_sha256"] != sha256(protocol_path):
            raise AssertionError(f"embedded protocol changed: {prefix}")
        if result["prediction_sha256"] != sha256(prediction_path):
            raise AssertionError(f"prediction hash changed: {prefix}")
        channel_origin = "current_protocol"
        channel_contract = protocol.get("channels")
        if channel_contract is None and prefix == "ROAM_V1_QUALITY":
            posture = json.loads((ROOT / "ROAM_POSTURE_PROTOCOL.json").read_text(encoding="utf-8"))
            channel_contract = posture["channels"]
            channel_origin = "frozen_ROAM_posture_loader_protocol"
        channel_count = (len(channel_contract) if isinstance(channel_contract, list)
                         else channel_contract)
        if channel_count != 8:
            raise AssertionError(f"channel contract changed: {prefix}")
        if (protocol["sample_rate_hz"] <= 0 or protocol["window_samples"] <= 0
                or tuple(protocol["candidate_families"]) != FAMILIES
                or protocol["arms"] != ["F0v2", *[f"F0v2+{name}" for name in FAMILIES]]):
            raise AssertionError(f"family or sampling contract changed: {prefix}")
        model_text = protocol.get("classifier", protocol.get("source_fit", ""))
        if "LogisticRegression" not in model_text or "random_state=" not in model_text:
            raise AssertionError(f"classifier seed not explicit: {prefix}")
        with prediction_path.open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        if len(predictions) != expected:
            raise AssertionError(f"prediction coverage changed: {prefix}")
        source, validation, final = identity_sets(result, prefix)
        if (not source or not validation or not final or source & validation
                or source & final or validation & final):
            raise AssertionError(f"source/validation/final trial overlap: {prefix}")
        for phase, target in (("validation", validation), ("final", final)):
            observed = {row["trial_id"] for row in predictions if row["phase"] == phase}
            if observed != target:
                raise AssertionError(f"target prediction trial coverage changed: {prefix}/{phase}")
        rows.append({"experiment": prefix, "protocol_sha256": sha256(protocol_path),
                     "result_sha256": sha256(result_path),
                     "prediction_sha256": sha256(prediction_path),
                     "prediction_rows": len(predictions), "source_trials": len(source),
                     "validation_trials": len(validation), "final_trials": len(final),
                     "channels": 8, "sample_rate_hz": protocol["sample_rate_hz"],
                     "channel_contract_origin": channel_origin,
                     "window_samples": protocol["window_samples"],
                     "model": model_text,
                     "source_state": protocol.get("source_state", protocol.get("source_fit", "")),
                     "selection": protocol["selection"]})
    audit = {"status": "saved_new_v1_experiments_reproducibility_checked",
             "screens": len(rows), "prediction_rows": sum(row["prediction_rows"] for row in rows),
             "experiments": rows,
             "boundary": "Checks the seven saved versioned screens, exact protocol/result/prediction binding, explicit model seeds, source/target trial separation and row coverage. It does not rerun all raw archives, audit every older feature-bank run, prove numerical environment identity, or verify physical-device performance."}
    target = ROOT / "V1_REPRODUCIBILITY_AUDIT.json"
    target.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"screens": len(rows), "prediction_rows": audit["prediction_rows"]}), flush=True)
    return audit


if __name__ == "__main__":
    build()
