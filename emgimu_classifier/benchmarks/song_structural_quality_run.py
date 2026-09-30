"""Exploratory severe-structure F9 replay on one-person Song raw ADC sessions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from benchmarks.song_quality_observability import raw_formal_windows
from benchmarks.song_real8_study import load_session
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.quality_mask_v1 import SourceCalibratedQualityMask
from emgimu.feature_bank.quality_observability import QualityObservabilityFamily


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "song_real8"
PROTOCOL = ROOT / "F9_STRUCTURAL_SONG_PROTOCOL.json"
RESULT = ROOT / "F9_STRUCTURAL_SONG_RESULTS.json"
TRIALS = ROOT / "F9_STRUCTURAL_SONG_TRIALS.csv"
CLASSES = ("fist", "index_pinch", "neutral", "open_hand")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict]) -> dict:
    accepted = [r for r in rows if not r["rejected"]]
    rejected = [r for r in rows if r["rejected"]]
    return {"trials": len(rows), "accepted": len(accepted),
            "rejected": len(rejected), "coverage": len(accepted) / len(rows),
            "baseline_errors": sum(not r["f0_correct"] for r in rows),
            "errors_rejected": sum(not r["f0_correct"] for r in rejected),
            "correct_rejected": sum(r["f0_correct"] for r in rejected),
            "synthetic_constant_channel_trials_detected":
                sum(r["synthetic_constant_ch1_detected"] for r in rows)}


def evaluate(source: Path) -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for filename, key in (("QUALITY_MASK_V1.json", "parent_quality_result_sha256"),
                          ("F4_INCREMENT_TRIAL_PREDICTIONS.csv", "frozen_f0_predictions_sha256"),
                          ("F4_INCREMENT_RESULTS.json", "frozen_f0_manifest_sha256")):
        if sha(ROOT / filename) != protocol[key]:
            raise AssertionError(f"frozen Song parent changed: {filename}")
    parent = json.loads((ROOT / "QUALITY_MASK_V1.json").read_text(encoding="utf-8"))
    frozen_f0 = json.loads((ROOT / "F4_INCREMENT_RESULTS.json").read_text(encoding="utf-8"))
    with (ROOT / "F4_INCREMENT_TRIAL_PREDICTIONS.csv").open(encoding="utf-8", newline="") as stream:
        predictions = {(row["session"], row["trial_id"]): row for row in csv.DictReader(stream)
                       if row["arm"] == "F0" and row["session"] in ("S03", "S04")}
    data, raw = {}, {}
    for session in ("S01", "S02", "S03", "S04"):
        folder = source / f"2026-09-18_{session}"
        data[session] = load_session(folder, session, filter_mode="causal")
        raw[session] = raw_formal_windows(folder, session, data[session]["trial"])
        if data[session]["audit"]["sha256"] != frozen_f0["source_hdf5_sha256"][session]:
            raise AssertionError(f"frozen HDF5 changed for {session}")
    if ({s: data[s]["audit"]["sha256"] for s in ("S01", "S02")} != parent["source_sha256"]
            or {s: data[s]["audit"]["sha256"] for s in ("S03", "S04")} != parent["target_sha256"]):
        raise AssertionError("Song F9 parent session checksums changed")
    source_batch = FeatureBatch(np.concatenate((raw["S01"], raw["S02"])), 250.)
    family = QualityObservabilityFamily(adc_min=-8388608, adc_max=8388607,
        line_frequency_hz=50., pre_highpass_available=True, ring_topology=False).fit(source_batch)
    mask = SourceCalibratedQualityMask(source_quantile=.995).fit(
        family.transform(source_batch), family.feature_names)
    frozen = pickle.dumps((family, mask))
    if (source_batch.windows != parent["source_windows"] or
            mask.available_ != parent["availability"] or
            list(mask.feature_names) != parent["quality_feature_names"] or
            set(mask.thresholds_) != set(parent["thresholds"])):
        raise AssertionError("reconstructed source-only Song F9 rule differs from parent")
    for component, values in mask.thresholds_.items():
        if not np.array_equal(values, np.asarray(parent["thresholds"][component])):
            raise AssertionError(f"parent Song F9 source threshold changed: {component}")

    rows = []
    for session in ("S03", "S04"):
        batch = FeatureBatch(raw[session], 250.)
        invalid = mask.structural_invalid(family.transform(batch), family.feature_names)
        if invalid.shape != (len(raw[session]), 8):
            raise AssertionError("Song raw-ADC window topology changed")
        fault = raw[session].copy()
        fault[:, :, 0] = fault[:, :1, 0]
        synthetic = mask.structural_invalid(
            family.transform(FeatureBatch(fault, 250.)), family.feature_names)
        trials = np.asarray(data[session]["trial"]).astype(str)
        labels = np.asarray(data[session]["hand"]).astype(str)
        if len(trials) != len(invalid) or len(labels) != len(invalid):
            raise AssertionError("Song native trial/window alignment changed")
        for trial in sorted(set(trials)):
            selected = trials == trial
            if len(set(labels[selected])) != 1:
                raise AssertionError("one native Song trial has mixed hand truth")
            label = labels[selected][0]
            prediction = predictions[(session, trial)]
            if prediction["truth"] != label or label not in CLASSES:
                raise AssertionError("frozen Song F0 prediction/truth mismatch")
            probabilities = np.array([float(prediction[f"p_{hand}"]) for hand in CLASSES])
            if not np.isfinite(probabilities).all() or abs(probabilities.sum() - 1.) > 1e-5:
                raise AssertionError("frozen F0 probabilities invalid")
            rows.append({"session": session, "trial_id": trial, "truth": label,
                         "f0_correct": CLASSES[int(probabilities.argmax())] == label,
                         "windows": int(selected.sum()),
                         "structural_windows": int(np.any(invalid[selected], axis=1).sum()),
                         "rejected": bool(np.any(invalid[selected])),
                         "synthetic_constant_ch1_detected": bool(np.all(synthetic[selected, 0]))})
        print(f"Song structural F9 {session}: {len(set(trials))} trials", flush=True)
    if frozen != pickle.dumps((family, mask)):
        raise AssertionError("target transformed source-fitted Song F9 state")
    if len(rows) != len(predictions) or {(r["session"], r["trial_id"]) for r in rows} != set(predictions):
        raise AssertionError("Song structural rows do not cover all frozen F0 trials")
    with TRIALS.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {"status": "one_person_exploratory_structural_rule_not_deployed",
              "protocol_sha256": sha(PROTOCOL),
              "parent_quality_result_sha256": protocol["parent_quality_result_sha256"],
              "frozen_f0_predictions_sha256": protocol["frozen_f0_predictions_sha256"],
              "reconstructed_source_state_sha256": hashlib.sha256(frozen).hexdigest(),
              "parent_source_state_sha256": parent["source_state_sha256"],
              "source_windows": parent["source_windows"],
              "availability": mask.available_,
              "trial_rows_sha256": sha(TRIALS),
              "sessions": {session: summarize([r for r in rows if r["session"] == session])
                           for session in ("S03", "S04")},
              "scope": protocol["limits"]}
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for session, item in result["sessions"].items():
        print(f"Song structural F9 {session}: coverage={item['coverage']:.3f}; "
              f"errors rejected={item['errors_rejected']}; correct rejected={item['correct_rejected']}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path("E:/qxy/emg_meta/emg_meta/data/Song"))
    evaluate(parser.parse_args().source)
