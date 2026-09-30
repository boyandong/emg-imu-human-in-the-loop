"""Native-data parity of label-free and offline session prediction interfaces."""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2 import wearing_v1_extension_run as wearing
from emgimu.datasets.electrode_shift import PATH_RE
from emgimu.feature_bank.force_nested_oof import SubjectWindows
from emgimu.feature_bank.session_pipeline import BRANCHES, FAMILIES, SessionCalibrationPipeline


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "SESSION_UNLABELED_PROTOCOL.json"
RESULT = ROOT / "SESSION_UNLABELED_RESULTS.json"
PREDICTIONS = ROOT / "SESSION_UNLABELED_PREDICTIONS.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (sha(ROOT / "WEARING_PROTOCOL.json") != protocol["parent_wearing_protocol_sha256"]
            or sha(wearing.ARCHIVE) != protocol["archive_sha256"]):
        raise AssertionError("frozen wearing native protocol/archive changed")
    source = wearing.load(wearing.ARCHIVE, 15, ("training",))
    target = wearing.load(wearing.ARCHIVE, 15, ("trial_1",))
    source_trials = set(source.trials.tolist())
    if len(source_trials) != 25 or source.batch.channels != 8 or source.batch.sample_rate_hz != 200:
        raise AssertionError("native source profile contract changed")
    repetitions = np.array([int(PATH_RE.fullmatch(str(trial))["rep"]) for trial in target.trials])
    calibration = target.take(np.flatnonzero(repetitions == 0))
    evaluation = target.take(np.flatnonzero(repetitions == 1))
    cal_trials, eval_trials = set(calibration.trials.tolist()), set(evaluation.trials.tolist())
    if (len(cal_trials) != 5 or len(eval_trials) != 5 or
            source_trials & cal_trials or source_trials & eval_trials or cal_trials & eval_trials or
            set(calibration.labels.tolist()) != set(range(5)) or
            set(evaluation.labels.tolist()) != set(range(5))):
        raise AssertionError("one-per-class disjoint native session split changed")
    print("Session parity: fit one long-term source profile", flush=True)
    pipeline = SessionCalibrationPipeline(rest_label=2, ring_topology=True).fit_long_term(source)
    state = pipeline.calibrate_session(calibration)
    frozen = pickle.dumps((pipeline, state))
    print("Session parity: compare unlabeled and offline predictions", flush=True)
    unlabeled, trial_ids = pipeline.predict_unlabeled(
        evaluation.batch, evaluation.subjects, evaluation.trials, state)
    scored, truths, scored_trials = pipeline.predict(evaluation, state)
    wrong_truth = SubjectWindows(evaluation.batch, np.zeros_like(evaluation.labels),
                                 evaluation.subjects, evaluation.trials)
    wrong, _, wrong_trials = pipeline.predict(wrong_truth, state)
    np.testing.assert_array_equal(trial_ids, scored_trials)
    np.testing.assert_array_equal(trial_ids, wrong_trials)
    if set(trial_ids.tolist()) != eval_trials or len(trial_ids) != 5:
        raise AssertionError("native evaluation trial ordering/coverage changed")
    rows = []
    maximum = 0.
    for branch in BRANCHES:
        for family in FAMILIES:
            p = unlabeled[branch][family]
            q = scored[branch][family]
            r = wrong[branch][family]
            if p.shape != (5, 5) or not np.isfinite(p).all() or np.any(p < 0):
                raise AssertionError("invalid native session probability shape")
            np.testing.assert_allclose(p.sum(1), 1., atol=1e-12)
            maximum = max(maximum, float(np.max(np.abs(p - q))),
                          float(np.max(np.abs(p - r))))
            for trial, truth, values in zip(trial_ids, truths, p):
                rows.append({"trial_id": str(trial), "branch": branch, "family": family,
                             "offline_truth_for_audit": int(truth),
                             **{f"p_{label}": float(values[label]) for label in range(5)}})
    if maximum > 1e-12 or frozen != pickle.dumps((pipeline, state)):
        raise AssertionError("labels affected predictions or source/session state changed")
    with PREDICTIONS.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {"status": "native_unlabeled_session_api_parity",
              "protocol_sha256": sha(PROTOCOL),
              "archive_sha256": protocol["archive_sha256"],
              "source_trial_ids": sorted(source_trials),
              "calibration_trial_ids": sorted(cal_trials),
              "evaluation_trial_ids": sorted(eval_trials),
              "fitted_state_sha256": hashlib.sha256(frozen).hexdigest(),
              "prediction_rows_sha256": sha(PREDICTIONS),
              "prediction_rows": len(rows),
              "branches": list(BRANCHES), "families": list(FAMILIES),
              "max_abs_probability_difference": maximum,
              "wrong_offline_truth_changed_predictions": False,
              "source_and_session_immutable": True,
              "scope": protocol["boundary"]}
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Session parity: {len(rows)} provider rows; maximum difference {maximum:.1g}", flush=True)
    return result


if __name__ == "__main__":
    evaluate()
