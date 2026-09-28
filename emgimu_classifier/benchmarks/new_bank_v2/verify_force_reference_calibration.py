"""Independent matched-trial and burden audit for one-reference force calibration."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from emgimu.datasets.libemg_force import FILE_RE
from benchmarks.new_bank_v2.verify_force_calibration import _compare, _score

ROOT = Path(__file__).resolve().parent
RAW = Path("D:/emg-imu-benchmarks/data/raw/libemg_force/official/ContractionIntensity-main")
CLASSES = set(range(7))
ARMS = {"F0v2", "F0v2+F2a"}
METHODS = {"source_only": 0, "one_shot": 1, "two_shot": 2}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _duration(trial: str) -> float:
    match = FILE_RE.fullmatch(trial + ".csv")
    if match is None:
        raise AssertionError("invalid native trial name")
    path = RAW / f"S{match['subject']}" / f"{trial}.csv"
    with path.open(encoding="utf-8") as stream:
        return sum(1 for _ in stream) / 1000.0


def verify() -> dict:
    protocol_path = ROOT / "FORCE_REF_CAL_PROTOCOL.json"
    result = json.loads((ROOT / "FORCE_REF_CAL_RESULTS.json").read_text(encoding="utf-8"))
    if result["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        raise AssertionError("reference protocol hash changed")
    rows = _read(ROOT / "FORCE_REF_CAL_PREDICTIONS.csv")
    parent = _read(ROOT / "FORCE_CAL_PREDICTIONS.csv")
    if len(rows) != 3360 or len(parent) != 3360 or len(result["assignments"]) != 4:
        raise AssertionError("prediction/assignment count changed")
    if result["parent_predictions_sha256"] != hashlib.sha256(
            (ROOT / "FORCE_CAL_PREDICTIONS.csv").read_bytes()).hexdigest():
        raise AssertionError("parent matched predictions changed")
    parent_lookup = {(r["phase"], r["arm"], r["method"], r["trial_id"]): r for r in parent}
    parent_result = json.loads((ROOT / "FORCE_CAL_RESULTS.json").read_text(encoding="utf-8"))
    lookup = {}
    grouped = defaultdict(list)
    for row in rows:
        key = (row["phase"], row["arm"], row["method"], row["trial_id"])
        if key in lookup or row["arm"] not in ARMS or row["method"] not in METHODS:
            raise AssertionError("duplicate/unexpected trial prediction")
        lookup[key] = row
        match = FILE_RE.fullmatch(row["trial_id"] + ".csv")
        assignment = f"{row['phase']}_{row['subject']}"
        if (match is None or assignment not in result["assignments"] or
                int(match["subject"]) != int(row["subject"]) or
                match["condition"] != row["condition"] or
                int(match["label"]) - 1 != int(row["label"]) or
                int(match["repetition"]) not in (3, 4) or
                row["trial_id"] not in result["assignments"][assignment]["evaluation"]):
            raise AssertionError("evaluation native trial identity invalid")
        p = np.asarray([float(row[f"p_{c}"]) for c in range(7)])
        if np.any(p < 0) or not np.all(np.isfinite(p)) or not np.isclose(p.sum(), 1, atol=1e-12):
            raise AssertionError("invalid class probability vector")
        if row["method"] == "source_only":
            prior = parent_lookup[key]
            if not np.allclose(p, [float(prior[f"p_{c}"]) for c in range(7)], atol=1e-12, rtol=0):
                raise AssertionError("zero-shot probability differs from parent")
        grouped[(row["phase"], row["arm"], row["method"])].append(row)
    burden = {}
    parent_burden = {}
    for key, assignment in result["assignments"].items():
        phase, subject = key.split("_")
        evaluation = set(assignment["evaluation"])
        source = set(assignment["source"])
        cal = assignment["calibration"]
        if (len(evaluation) != 140 or set(cal) != {"0", "1", "2"} or
                cal["0"] != [] or len(cal["1"]) != 7 or len(cal["2"]) != 14 or
                not set(cal["1"]) <= set(cal["2"]) or
                source & evaluation or source & set(cal["2"]) or evaluation & set(cal["2"])):
            raise AssertionError("reference assignment incomplete or leaky")
        if Counter(FILE_RE.fullmatch(t + ".csv")["condition"] for t in evaluation) != Counter(
                {c: 14 for c in result["protocol"]["target_conditions"]}):
            raise AssertionError("ten-condition matched evaluation incomplete")
        burden[key] = {}
        parent_assignments = [v for name, v in parent_result["assignments"].items()
                              if name.startswith(key + "_")]
        if len(parent_assignments) != 10:
            raise AssertionError("per-condition parent assignment count changed")
        parent_burden[key] = {}
        for shot in (1, 2):
            trials = cal[str(shot)]
            identities = [FILE_RE.fullmatch(t + ".csv") for t in trials]
            if any(m is None or int(m["subject"]) != int(subject) or m["condition"] != "Medium" or
                   int(m["repetition"]) not in ({1} if shot == 1 else {1, 2}) for m in identities):
                raise AssertionError("reference calibration trial is wrong condition or repetition")
            if Counter(int(m["label"]) - 1 for m in identities) != Counter({c: shot for c in CLASSES}):
                raise AssertionError("reference calibration class count invalid")
            burden[key][str(shot)] = {"trials": len(trials),
                                      "recorded_signal_seconds": round(sum(_duration(t) for t in trials), 3)}
            prior_trials = [trial for item in parent_assignments
                            for trial in item["calibration"][str(shot)]]
            parent_burden[key][str(shot)] = {
                "trials": len(prior_trials),
                "recorded_signal_seconds": round(sum(_duration(t) for t in prior_trials), 3)}
        for arm in ARMS:
            for method in METHODS:
                if not all((phase, arm, method, trial) in lookup for trial in evaluation):
                    raise AssertionError("missing evaluation prediction")
    paired = {}
    for phase in ("validation", "final"):
        for arm in sorted(ARMS):
            baseline = {r["trial_id"]: r for r in grouped[(phase, arm, "source_only")]}
            for method in METHODS:
                subset = grouped[(phase, arm, method)]
                if len(subset) != 280:
                    raise AssertionError("phase-arm-budget score group incomplete")
                score = result["scores"][phase][arm][method]
                _compare(score["pooled"], _score(subset), f"{phase}/{arm}/{method}/pooled")
                for axis, column in (("by_subject", "subject"), ("by_condition", "condition")):
                    for value, saved in score[axis].items():
                        _compare(saved, _score([r for r in subset if r[column] == value]),
                                 f"{phase}/{arm}/{method}/{axis}/{value}")
                if not np.isclose(score["minimum_subject_macro_f1"],
                                  min(v["macro_f1"] for v in score["by_subject"].values()), atol=1e-12):
                    raise AssertionError("minimum subject score changed")
                if not np.isclose(score["worst_condition_macro_f1"],
                                  min(v["macro_f1"] for v in score["by_condition"].values()), atol=1e-12):
                    raise AssertionError("worst-condition score changed")
                if method == "source_only":
                    continue
                corrected = created = 0
                for later in subset:
                    old = baseline[later["trial_id"]]
                    was = np.argmax([float(old[f"p_{c}"]) for c in range(7)]) == int(old["label"])
                    now = np.argmax([float(later[f"p_{c}"]) for c in range(7)]) == int(later["label"])
                    corrected += int(not was and now)
                    created += int(was and not now)
                paired[f"{phase}_{arm}_{method}"] = {"corrected": corrected, "new_errors": created}
    audit = {"status": "ok", "prediction_rows": len(rows), "assignments": len(result["assignments"]),
             "score_groups_recomputed": len(grouped),
             "parent_zero_shot_replay": "within 1e-12 on identical native trials",
             "per_user_reference_burden": burden,
             "per_user_per_condition_parent_burden": parent_burden,
             "paired_vs_source_only": paired,
             "boundary": "recorded signal only, excluding setup/transitions; public trial-level evidence"}
    (ROOT / "FORCE_REF_CAL_VERIFICATION.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "prediction_rows": len(rows), "assignments": len(result["assignments"])}))
    return audit


if __name__ == "__main__":
    verify()
