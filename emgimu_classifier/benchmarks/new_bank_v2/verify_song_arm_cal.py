"""Read back paired Song calibration probabilities and check their provenance."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.song_28_spd_increment_study import detailed_metrics
from benchmarks.song_real8_study import parse_label


ROOT = Path(__file__).resolve().parent


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify() -> dict:
    result = json.loads((ROOT / "SONG_ARM_CAL_RESULTS.json").read_text(encoding="utf-8"))
    protocol_path = ROOT / "SONG_ARM_CAL_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if result["protocol"] != protocol or result["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        raise ValueError("protocol content/hash mismatch")
    prior_path = ROOT / "SONG_28_RESULTS.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if (result["reference_28_results_sha256"] != hashlib.sha256(prior_path.read_bytes()).hexdigest()
            or result["runtime_versions"] != prior["runtime_versions"]
            or result["source_hdf5_sha256"] != prior["source_hdf5_sha256"]):
        raise ValueError("previous 28-state result, runtime or source hashes changed")
    for sid, blocks in result["calibration_blocks"].items():
        if (set(blocks) != set(protocol["calibration_blocks"])
                or not all(item["preformal"] and item["descriptor_windows"] >= 1
                           for item in blocks.values())):
            raise ValueError(f"calibration coverage/preformal audit failed: {sid}")
        values = list(blocks.values())
        if any(not 0 <= item["trial_start_sample"] <= item["emg_stable_start"]
               < item["emg_stable_end"] <= item["trial_end_sample"]
               for item in values):
            raise ValueError(f"calibration interval ordering failed: {sid}")
        guided = [item for name, item in blocks.items() if name.startswith("calibration_arm_")]
        for key, group in (("selected_eight_block_span_seconds", values),
                           ("six_guided_arm_block_span_seconds", guided)):
            expected_span = (max(item["trial_end_sample"] for item in group)
                             - min(item["trial_start_sample"] for item in group)) / 250.0
            if not np.isclose(expected_span, result["calibration_burden"][sid][key], atol=1e-12):
                raise ValueError(f"calibration time burden mismatch: {sid}/{key}")
    rows = read_csv(ROOT / "SONG_ARM_CAL_TRIAL_PREDICTIONS.csv")
    old28 = read_csv(ROOT / "SONG_28_TRIAL_PREDICTIONS.csv")
    old4 = read_csv(ROOT / "SONG_TRIAL_PREDICTIONS.csv")
    classes = np.asarray(prior["classes"])
    hand_classes = np.asarray(sorted({parse_label(label)[1] for label in classes}))
    old4_lookup = {(r["phase"], r["trial_id"]): r for r in old4 if r["arm"] == "F0v2+F2a+F3c"}
    old28_lookup = {(r["phase"], r["trial_id"]): r for r in old28 if r["arm"] == "F0v2+IMU"}
    expected = len(protocol["arms"]) * sum(len(result["sessions"][phase]["trial_ids"])
                                            for phase in ("validation", "final"))
    if len(rows) != expected or len(rows) != result["prediction_rows"]:
        raise ValueError("prediction row count mismatch")
    paired = {}; max_hand_replay_error = 0.0
    for phase, sid in (("validation", "S03"), ("final", "S04")):
        ids = result["sessions"][phase]["trial_ids"]
        arm_values = {}
        for arm in protocol["arms"]:
            subset = [r for r in rows if r["phase"] == phase and r["arm"] == arm]
            if ([r["trial_id"] for r in subset] != ids
                    or any(r["session"] != sid for r in subset)):
                raise ValueError(f"trial identities/order differ: {phase}/{arm}")
            truth = np.asarray([r["label"] for r in subset])
            probability = np.asarray([[float(r[f"p_{label}"]) for label in classes] for r in subset])
            if (not np.all(np.isfinite(probability)) or np.any(probability < 0)
                    or not np.allclose(probability.sum(axis=1), 1, atol=1e-12)):
                raise ValueError(f"invalid probabilities: {phase}/{arm}")
            actual = detailed_metrics(truth, probability, classes)
            stated = result["sessions"][phase]["arms"][arm]
            for key in ("trials", "accuracy", "macro_f1", "log_loss", "brier", "arm_accuracy", "hand_accuracy"):
                if not np.isclose(actual[key], stated[key], rtol=0, atol=1e-12):
                    raise ValueError(f"metric mismatch: {phase}/{arm}/{key}")
            if actual["confusion_matrix"] != stated["confusion_matrix"]:
                raise ValueError(f"confusion matrix mismatch: {phase}/{arm}")
            for cue, score in actual["per_arm_joint_accuracy"].items():
                if not np.isclose(score, stated["per_arm_joint_accuracy"][cue], atol=1e-12):
                    raise ValueError(f"per-arm score mismatch: {phase}/{arm}/{cue}")
            hand_marginal = np.stack([
                probability[:, [i for i, label in enumerate(classes)
                                if parse_label(label)[1] == hand]].sum(axis=1)
                for hand in hand_classes], axis=1)
            for index, trial_id in enumerate(ids):
                prior4 = old4_lookup.get((phase, trial_id))
                if prior4 is None or prior4["label"] != parse_label(truth[index])[1]:
                    raise ValueError(f"previous four-state trial missing: {phase}/{trial_id}")
                previous_hand = np.asarray([float(prior4[f"p_{hand}"]) for hand in hand_classes])
                if int(hand_marginal[index].argmax()) != int(previous_hand.argmax()):
                    raise ValueError(f"four-state hard decision changed: {phase}/{trial_id}")
                max_hand_replay_error = max(max_hand_replay_error, *(
                    abs(hand_marginal[index, k] - float(prior4[f"p_{hand}"]))
                    for k, hand in enumerate(hand_classes)))
            arm_values[arm] = (truth, probability)
        if max_hand_replay_error > 1e-5:
            raise ValueError(f"source hand branch differs from prior 4-state model: {max_hand_replay_error}")
        truth = arm_values[protocol["arms"][0]][0]
        if not np.array_equal(truth, arm_values[protocol["arms"][1]][0]):
            raise ValueError(f"paired truth differs: {phase}")
        old_rows = [old28_lookup.get((phase, trial_id)) for trial_id in ids]
        if any(row is None or row["label"] != label for row, label in zip(old_rows, truth)):
            raise ValueError(f"previous 28-state trial coverage differs: {phase}")
        old_probability = np.asarray([[float(row[f"p_{label}"]) for label in classes]
                                      for row in old_rows])
        old_correct = classes[old_probability.argmax(axis=1)] == truth
        source_correct = classes[arm_values[protocol["arms"][0]][1].argmax(axis=1)] == truth
        guided_correct = classes[arm_values[protocol["arms"][1]][1].argmax(axis=1)] == truth
        paired[phase] = {
            "source_vs_previous_joint": {"corrected": int(np.sum(source_correct & ~old_correct)),
                                         "new_errors": int(np.sum(~source_correct & old_correct))},
            "guided_vs_same_hand_source_arm": {"corrected": int(np.sum(guided_correct & ~source_correct)),
                                                "new_errors": int(np.sum(~guided_correct & source_correct))},
        }
    verification = {"status": "verified", "prediction_rows": len(rows),
                    "heldout_native_trials": expected // len(protocol["arms"]),
                    "metric_groups_read_back": len(protocol["arms"]) * 2,
                    "max_hand_probability_replay_error": max_hand_replay_error,
                    "all_calibration_blocks_preformal": True,
                    "paired": paired}
    (ROOT / "SONG_ARM_CAL_VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    return verification


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
