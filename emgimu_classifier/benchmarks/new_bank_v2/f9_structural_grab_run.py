"""Exploratory structural F9 gate replay against frozen GRABMyo F0v2 trials."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from benchmarks.grabmyo_crossday import run as grab
from benchmarks.new_bank_v2 import f9_grab_gate_run as previous
from benchmarks.new_bank_v2 import grab_user_run
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.quality_mask_v1 import SourceCalibratedQualityMask
from emgimu.feature_bank.quality_observability import QualityObservabilityFamily


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "F9_STRUCTURAL_GRAB_PROTOCOL.json"
RESULT = ROOT / "F9_STRUCTURAL_GRAB_RESULTS.json"
TRIALS = ROOT / "F9_STRUCTURAL_GRAB_TRIALS.csv"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(data_root: Path) -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for filename, key in (("F9_GRAB_GATE_PROTOCOL.json", "parent_protocol_sha256"),
                          ("F9_GRAB_GATE_RESULTS.json", "parent_results_sha256"),
                          ("GRAB_USER_PREDICTIONS.csv", "frozen_f0_predictions_sha256")):
        if _sha(ROOT / filename) != protocol[key]:
            raise AssertionError(f"frozen parent changed: {filename}")
    prior = json.loads((ROOT / "F9_GRAB_GATE_RESULTS.json").read_text(encoding="utf-8"))
    if _sha(ROOT / "F9_GRAB_GATE_TRIALS.csv") != prior["trial_rows_sha256"]:
        raise AssertionError("parent F9 trial rows changed")
    with (ROOT / "F9_GRAB_GATE_TRIALS.csv").open(encoding="utf-8", newline="") as stream:
        prior_rows = {(row["phase"], row["trial_id"]): row for row in csv.DictReader(stream)}
    with (ROOT / "GRAB_USER_PREDICTIONS.csv").open(encoding="utf-8", newline="") as stream:
        predictions = {(row["phase"], row["trial_id"]): row for row in csv.DictReader(stream)
                       if row["arm"] == "F0v2"}
    if len(prior_rows) != len(predictions) or len(predictions) != 112:
        raise AssertionError("frozen F0/F9 trial inventory changed")

    grab_user_run.check_protocol()
    records = [row for row in grab.records() if row["session"] == grab_user_run.PROTOCOL["day"]]
    if len(records) != 224 or grab_user_run.check_files(data_root, records) != prior["official_sha256_manifest_sha256"]:
        raise AssertionError("official GRAB records or checksums changed")
    source_records = [row for row in records if row["subject"] in grab_user_run.PROTOCOL["source_subjects"]]
    source = FeatureBatch(np.concatenate([grab.read_record(data_root, row)
                                          for row in source_records]), grab.RATE)
    family = QualityObservabilityFamily(pre_highpass_available=False,
        line_frequency_available=False, ring_topology=False).fit(source)
    mask = SourceCalibratedQualityMask(source_quantile=.995).fit(
        family.transform(source), family.feature_names)
    frozen = pickle.dumps((family, mask))
    if hashlib.sha256(frozen).hexdigest() != prior["source_state_sha256"]:
        raise AssertionError("source-fitted F9 state changed")
    if family.availability_ != prior["availability"] or mask.available_ != prior["availability"]:
        raise AssertionError("quality availability changed")

    rows = []
    for phase in ("validation", "final"):
        target_ids = set(json.loads((ROOT / "GRAB_USER_RESULTS.json").read_text(encoding="utf-8"))
                         [f"{phase}_trial_ids"])
        target = [row for row in records if row["stem"] in target_ids]
        if len(target) != 56 or {row["stem"] for row in target} != target_ids:
            raise AssertionError("target trial inventory changed")
        for count, record in enumerate(target, 1):
            observation = family.transform(FeatureBatch(grab.read_record(data_root, record), grab.RATE))
            structural = mask.structural_invalid(observation, family.feature_names)
            if structural.shape != (20, 8):
                raise AssertionError("fixed twenty-window eight-channel topology changed")
            key = (phase, record["stem"])
            old, prediction = prior_rows[key], predictions[key]
            correct = grab.GESTURES[int(np.argmax([float(prediction[f"p_{label}"])
                                                   for label in grab.GESTURES]))] == record["gesture"]
            if (int(old["subject"]) != record["subject"] or
                    int(old["gesture"]) != record["gesture"] or
                    (old["f0_correct"] == "True") != correct):
                raise AssertionError("frozen F0 prediction and native trial mismatch")
            rows.append({"phase": phase, "trial_id": record["stem"],
                         "subject": record["subject"], "gesture": record["gesture"],
                         "f0_correct": correct, "parent_rejected": old["rejected"] == "True",
                         "rejected": bool(np.any(structural)),
                         "structural_windows": int(np.any(structural, axis=1).sum())})
            if count % 14 == 0:
                print(f"GRAB structural F9 {phase}: {count}/{len(target)} trials", flush=True)
    if frozen != pickle.dumps((family, mask)):
        raise AssertionError("target transform changed source-fitted F9 state")
    if {(row["phase"], row["trial_id"]) for row in rows} != set(prior_rows):
        raise AssertionError("new structural rows do not cover the parent trials")
    with TRIALS.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {"status": "exploratory_structural_diagnostic_not_deployed",
              "protocol_sha256": _sha(PROTOCOL),
              "parent_results_sha256": protocol["parent_results_sha256"],
              "source_state_sha256": prior["source_state_sha256"],
              "official_sha256_manifest_sha256": prior["official_sha256_manifest_sha256"],
              "availability": mask.available_,
              "trial_rows_sha256": _sha(TRIALS),
              "phases": {phase: {"pooled": previous.summarize([r for r in rows if r["phase"] == phase]),
                                 "subjects": {str(subject): previous.summarize([r for r in rows
                                                                                  if r["phase"] == phase and r["subject"] == subject])
                                              for subject in grab_user_run.PROTOCOL[f"{phase}_subjects"]}}
                         for phase in ("validation", "final")},
              "scope": protocol["limits"]}
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for phase in ("validation", "final"):
        item = result["phases"][phase]["pooled"]
        print(f"GRAB structural F9 {phase}: coverage={item['coverage']:.3f}; "
              f"errors rejected={item['errors_rejected']}; correct rejected={item['correct_rejected']}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=Path("D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1"))
    run(parser.parse_args().data_root)
