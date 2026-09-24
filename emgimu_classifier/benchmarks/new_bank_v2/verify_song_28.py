"""Read back every saved 28-state probability and verify the study outputs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.song_28_spd_increment_study import detailed_metrics


ROOT = Path(__file__).resolve().parent


def _same(actual, stated, location: str) -> None:
    if isinstance(actual, dict):
        if not isinstance(stated, dict) or actual.keys() != stated.keys():
            raise ValueError(f"metric keys differ: {location}")
        for key in actual:
            _same(actual[key], stated[key], f"{location}/{key}")
    elif isinstance(actual, list):
        if not isinstance(stated, list) or len(actual) != len(stated):
            raise ValueError(f"metric list differs: {location}")
        for index, (a, b) in enumerate(zip(actual, stated)):
            _same(a, b, f"{location}/{index}")
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if not np.isclose(actual, stated, rtol=0, atol=1e-12):
            raise ValueError(f"metric differs: {location}: {actual} versus {stated}")
    elif actual != stated:
        raise ValueError(f"metric differs: {location}: {actual!r} versus {stated!r}")


def verify() -> dict:
    result = json.loads((ROOT / "SONG_28_RESULTS.json").read_text(encoding="utf-8"))
    protocol_path = ROOT / "SONG_28_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if result["protocol"] != protocol or result["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        raise ValueError("protocol content/hash mismatch")
    baseline_path = ROOT.parent / "song_real8" / "CAUSAL_RESULTS.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if result["saved_causal_baseline_sha256"] != hashlib.sha256(baseline_path.read_bytes()).hexdigest():
        raise ValueError("saved baseline hash mismatch")
    if result["source_hdf5_sha256"] != {item["session"]: item["sha256"] for item in baseline["source_audit"]}:
        raise ValueError("source file hashes differ from baseline")
    prior_runtime = json.loads((ROOT.parent / "song_real8" / "SPD_28_STATE_RESULTS.json").read_text(encoding="utf-8"))["runtime_versions"]
    if result["runtime_versions"] != prior_runtime:
        raise ValueError("28-state study runtime differs from prior reproduced baseline runtime")
    with (ROOT / "SONG_28_TRIAL_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    classes = np.asarray(result["classes"])
    arms = (protocol["reference_arm"], *protocol["arms"])
    trial_count = sum(len(result["sessions"][phase]["trial_ids"])
                      for phase in ("validation", "final"))
    if len(rows) != len(arms) * trial_count or len(rows) != result["prediction_rows"]:
        raise ValueError("prediction row count mismatch")
    paired = {}
    for phase, sid in (("validation", "S03"), ("final", "S04")):
        expected_ids = result["sessions"][phase]["trial_ids"]
        grouped = {}
        for arm in arms:
            subset = [r for r in rows if r["phase"] == phase and r["arm"] == arm]
            if ([r["trial_id"] for r in subset] != expected_ids
                    or any(r["session"] != sid for r in subset)):
                raise ValueError(f"trial coverage/order mismatch: {phase}/{arm}")
            truth = np.asarray([r["label"] for r in subset])
            probability = np.asarray([[float(r[f"p_{label}"]) for label in classes] for r in subset])
            if (not np.all(np.isfinite(probability)) or np.any(probability < 0)
                    or not np.allclose(probability.sum(axis=1), 1, atol=1e-12)):
                raise ValueError(f"invalid probabilities: {phase}/{arm}")
            scores = detailed_metrics(truth, probability, classes)
            _same(scores, result["sessions"][phase]["arms"][arm], f"{phase}/{arm}")
            grouped[arm] = (truth, classes[probability.argmax(axis=1)])
        truth, base = grouped["F0v2+IMU"]
        if any(not np.array_equal(truth, item[0]) for item in grouped.values()):
            raise ValueError(f"truth differs among arms: {phase}")
        prior = baseline["secondary_28_state"]["validation" if phase == "validation" else "test"]
        for key in ("trials", "accuracy", "macro_f1"):
            _same(result["sessions"][phase]["arms"][protocol["reference_arm"]][key], prior[key], f"prior/{phase}/{key}")
        paired[phase] = {}
        for arm in protocol["arms"][1:]:
            pred = grouped[arm][1]
            paired[phase][arm] = {
                "corrected_F0v2_IMU_errors": int(np.sum((pred == truth) & (base != truth))),
                "new_errors_from_F0v2_IMU": int(np.sum((pred != truth) & (base == truth))),
            }
    scores = result["sessions"]["validation"]["arms"]
    selected = min(protocol["arms"], key=lambda arm: (
        -scores[arm]["macro_f1"], scores[arm]["log_loss"], protocol["arms"].index(arm)))
    if selected != result["validation_selected_arm"]:
        raise ValueError("validation arm selection mismatch")
    verification = {"status": "verified", "prediction_rows": len(rows),
                    "heldout_native_trials": trial_count, "metric_groups_read_back": len(arms) * 2,
                    "baseline_metrics_reproduced": True,
                    "baseline_runtime_reproduced": result["runtime_versions"],
                    "validation_selected_arm": selected,
                    "paired_vs_F0v2_plus_IMU": paired}
    (ROOT / "SONG_28_VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    return verification


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
