"""Independent frame-table read-back for the signed-arm continuous replay."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

from benchmarks.song_real8_study import parse_label


ROOT = Path(__file__).resolve().parent


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scores(truth: list[str], predicted: list[str], labels: list[str]) -> tuple[float, float]:
    return (float(accuracy_score(truth, predicted)),
            float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)))


def run() -> dict:
    result = json.loads((ROOT / "SONG_ARM_SIGNED_CONTINUOUS_RESULTS.json").read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "SONG_JOINT28_CONTINUOUS_RESULTS.json").read_text(encoding="utf-8"))
    bound = {
        "protocol_sha256": "SONG_ARM_SIGNED_CONTINUOUS_PROTOCOL.json",
        "signed_trial_results_sha256": "SONG_ARM_SIGNED_RESULTS.json",
        "baseline_continuous_results_sha256": "SONG_JOINT28_CONTINUOUS_RESULTS.json",
        "baseline_frame_csv_sha256": "SONG_JOINT28_CONTINUOUS_FRAMES.csv",
        "candidate_arm_model_sha256": "SONG_ARM_SIGNED_ARM_MODEL.json",
        "frame_csv_sha256": "SONG_ARM_SIGNED_CONTINUOUS_FRAMES.csv",
    }
    for field, name in bound.items():
        if result[field] != _sha(ROOT / name):
            raise ValueError(f"artifact digest changed: {name}")
    model = json.loads((ROOT / "SONG_ARM_SIGNED_ARM_MODEL.json").read_text(encoding="utf-8"))
    if (model["source_sessions"] != ["S01", "S02"] or len(model["classes"]) != 7
            or np.asarray(model["coef"]).shape != (7, 19)
            or np.asarray(model["scaler_mean"]).shape != (19,)):
        raise ValueError("candidate model shape/provenance invalid")
    if model["source_hdf5_sha256"] != {sid: result["source_hdf5_sha256"][sid]
                                             for sid in ("S01", "S02")}:
        raise ValueError("candidate model source hashes differ")
    trial_study = json.loads((ROOT / "SONG_ARM_SIGNED_RESULTS.json").read_text(encoding="utf-8"))
    classes = trial_study["joint_classes"]
    arm_classes = sorted(model["classes"])
    hand_classes = sorted({parse_label(label)[1] for label in classes})
    with (ROOT / "SONG_JOINT28_CONTINUOUS_FRAMES.csv").open(newline="", encoding="utf-8") as stream:
        original = {(row["session"], row["emg_end_index"]): row for row in csv.DictReader(stream)}
    with (ROOT / "SONG_ARM_SIGNED_CONTINUOUS_FRAMES.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != result["frame_rows"] or len(rows) != len(original):
        raise ValueError("candidate/baseline frame count differs")
    seen = set()
    checks = {}
    max_hand_error = 0.0
    for sid in ("S03", "S04"):
        session_rows = [row for row in rows if row["session"] == sid]
        expected = result["sessions"][sid]
        old = baseline["sessions"][sid]
        if len(session_rows) != expected["emitted_frames"] or len(session_rows) != old["emitted_frames"]:
            raise ValueError(f"session frame count differs: {sid}")
        truth, predicted, trials = [], [], defaultdict(list)
        hand_truth, hand_pred, arm_truth, arm_pred = [], [], [], []
        rest = rest_argmax = rest_display = events = 0
        event_intervals = Counter()
        active = candidate = None
        candidate_count = 0
        for row in session_rows:
            key = (sid, row["emg_end_index"])
            if key in seen or key not in original:
                raise ValueError(f"duplicate/missing baseline frame: {key}")
            seen.add(key)
            reference = original[key]
            if any(row[field] != reference[field] for field in
                   ("interval_kind", "trial_id", "cue_label")):
                raise ValueError(f"changed native cue interval: {key}")
            label = row["peak_label"]
            if label not in classes:
                raise ValueError(f"unknown peak class: {key}")
            probability = float(row["peak_probability"])
            next_candidate = label if probability >= .15 else None
            if next_candidate == candidate:
                candidate_count += 1
            else:
                candidate, candidate_count = next_candidate, 1
            fired = False
            if candidate_count >= 3 and next_candidate != active:
                active = next_candidate
                fired = active is not None
            if row["display_label"] != (active or "") or int(row["event_fired"]) != int(fired):
                raise ValueError(f"online event state differs: {key}")
            if fired:
                events += 1
                event_intervals[row["interval_kind"]] += 1
            if row["interval_kind"] == "formal_stable":
                p = np.asarray(json.loads(row["joint_probabilities"]))
                q = np.asarray(json.loads(reference["joint_probabilities"]))
                if (p.shape != (28,) or not np.isfinite(p).all()
                        or not np.isclose(p.sum(), 1, atol=1e-12)
                        or classes[int(np.argmax(p))] != label
                        or not np.isclose(p.max(), probability, atol=1e-12)):
                    raise ValueError(f"invalid formal frame probabilities: {key}")
                for hand in hand_classes:
                    members = [i for i, name in enumerate(classes) if parse_label(name)[1] == hand]
                    max_hand_error = max(max_hand_error,
                                         abs(float(p[members].sum() - q[members].sum())))
                cue = row["cue_label"]
                truth.append(cue); predicted.append(label)
                hand_truth.append(parse_label(cue)[1]); hand_pred.append(parse_label(label)[1])
                arm_truth.append(parse_label(cue)[0]); arm_pred.append(parse_label(label)[0])
                trials[(row["trial_id"], cue)].append(p)
            elif row["interval_kind"] == "calibration_rest_stable":
                rest += 1
                rest_argmax += int(parse_label(label)[1] != "neutral")
                rest_display += int(active is not None and parse_label(active)[1] != "neutral")
            elif row["interval_kind"] != "unlabelled":
                raise ValueError(f"unexpected interval kind: {key}")
        trial_truth = [label for _, label in trials]
        trial_pred = [classes[int(np.argmax(np.mean(p, axis=0)))] for p in trials.values()]
        for prefix, y, p, labels in (("stable_windows", truth, predicted, classes),
                                     ("trial_mean_joint_probability", trial_truth, trial_pred, classes)):
            accuracy, f1 = _scores(y, p, labels)
            if not np.isclose(accuracy, expected[prefix]["accuracy"], atol=1e-12):
                raise ValueError(f"{sid}/{prefix} accuracy differs")
            if not np.isclose(f1, expected[prefix]["macro_f1"], atol=1e-12):
                raise ValueError(f"{sid}/{prefix} macro-F1 differs")
        for prefix, y, p, labels in (("hand", hand_truth, hand_pred, hand_classes),
                                     ("arm", arm_truth, arm_pred, arm_classes)):
            if not np.isclose(accuracy_score(y, p), expected[f"stable_{prefix}_accuracy"], atol=1e-12):
                raise ValueError(f"{sid}/{prefix} accuracy differs")
            recalled = recall_score(y, p, labels=labels, average=None, zero_division=0)
            for name, value in zip(labels, recalled):
                if not np.isclose(value, expected[f"stable_{prefix}_recall"][name], atol=1e-12):
                    raise ValueError(f"{sid}/{prefix}/{name} recall differs")
        if (rest != expected["rest_frames"] or rest_argmax != expected["rest_active_hand_argmax_frames"]
                or rest_display != expected["rest_active_hand_display_frames"]
                or events != expected["continuous_state_transitions_to_non_null"]
                or dict(event_intervals) != expected["state_transitions_by_scored_interval"]
                or len(trials) != expected["scored_formal_trials"]):
            raise ValueError(f"{sid} rest/event/trial result differs")
        checks[sid] = {"frames": len(session_rows), "formal_windows": len(truth),
                       "scored_trials": len(trials), "rest_frames": rest,
                       "active_hand_display_on_explicit_rest": rest_display}
    if seen != set(original) or max_hand_error > 1e-6:
        raise ValueError(f"frame grid or unchanged hand marginal differs: {max_hand_error}")
    output = {"status": "verified", "frame_rows": len(rows),
              "maximum_baseline_hand_marginal_error": max_hand_error, "sessions": checks,
              "boundary": "Verifier checks saved frames and metrics; no independent hardware or physiological onset labels."}
    (ROOT / "SONG_ARM_SIGNED_CONTINUOUS_VERIFICATION.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
