"""Compare frozen and document-exact session normalizers on native public trials."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2 import wearing_v1_extension_run as wearing
from emgimu.datasets.electrode_shift import PATH_RE
from emgimu.feature_bank.session_pipeline import (
    BRANCHES, FAMILIES, DocumentSessionCalibrationPipelineV2,
    SessionCalibrationPipeline,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "DOCUMENT_SESSION_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
PARENT = ROOT / "SESSION_UNLABELED_PREDICTIONS.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_repetition(data, repetition):
    indices = np.flatnonzero(np.asarray([
        int(PATH_RE.fullmatch(str(trial))["rep"]) == repetition for trial in data.trials
    ]))
    return data.take(indices)


def run() -> dict:
    if (sha(wearing.ARCHIVE) != PROTOCOL["archive_sha256"]
            or sha(ROOT / "SESSION_UNLABELED_PROTOCOL.json") != PROTOCOL["parent_unlabeled_protocol_sha256"]
            or sha(PARENT) != PROTOCOL["parent_unlabeled_predictions_sha256"]):
        raise AssertionError("frozen native wearing parent changed")
    with PARENT.open(newline="", encoding="utf-8") as stream:
        parent = {(row["branch"], row["family"], row["trial_id"]): row
                  for row in csv.DictReader(stream)}
    if len(parent) != 60:
        raise AssertionError("frozen label-free parent coverage changed")
    rows, blocks, parent_max_error = [], [], 0.0
    for phase, users in (("validation", PROTOCOL["validation_users"]),
                         ("descriptive_final", PROTOCOL["descriptive_final_users"])):
        for user in users:
            source = wearing.load(wearing.ARCHIVE, user, (PROTOCOL["source_domain"],))
            legacy = SessionCalibrationPipeline(rest_label=2, ring_topology=True).fit_long_term(source)
            exact = DocumentSessionCalibrationPipelineV2(rest_label=2, ring_topology=True).fit_long_term(source)
            source_ids = set(source.trials.tolist())
            if len(source_ids) != 25 or legacy.profile_id_ == exact.profile_id_:
                raise AssertionError("source profile inventory or formula identity changed")
            for domain in PROTOCOL["target_domains"]:
                target = wearing.load(wearing.ARCHIVE, user, (domain,))
                calibration = select_repetition(target, PROTOCOL["calibration_repetition"])
                evaluation = select_repetition(target, PROTOCOL["evaluation_repetition"])
                cal_ids = set(calibration.trials.tolist())
                eval_ids = set(evaluation.trials.tolist())
                if (len(cal_ids) != 5 or len(eval_ids) != 5 or source_ids & cal_ids
                        or source_ids & eval_ids or cal_ids & eval_ids
                        or set(calibration.labels.tolist()) != set(range(5))):
                    raise AssertionError("native source/calibration/evaluation leakage or class gap")
                legacy_state = legacy.calibrate_session(calibration)
                exact_state = exact.calibrate_session(calibration)
                if legacy_state["profile_id"] == exact_state["profile_id"]:
                    raise AssertionError("different normalization formulas share a profile identity")
                old, old_trials = legacy.predict_unlabeled(
                    evaluation.batch, evaluation.subjects, evaluation.trials, legacy_state)
                new, new_trials = exact.predict_unlabeled(
                    evaluation.batch, evaluation.subjects, evaluation.trials, exact_state)
                np.testing.assert_array_equal(old_trials, new_trials)
                blocks.append({"phase": phase, "subject": user, "domain": domain,
                               "source_trial_ids": sorted(source_ids),
                               "calibration_trial_ids": sorted(cal_ids),
                               "evaluation_trial_ids": sorted(eval_ids),
                               "legacy_profile_id": legacy.profile_id_,
                               "document_profile_id": exact.profile_id_,
                               "minimum_legacy_source_q95": float(np.min(legacy.normalizer_.scale_)),
                               "minimum_document_source_q95": float(np.min(exact.normalizer_.scale_)),
                               "minimum_legacy_session_q95": float(np.min(legacy_state["normalizer"].scale_)),
                               "minimum_document_session_q95": float(np.min(exact_state["normalizer"].scale_))})
                for branch in BRANCHES:
                    for family in FAMILIES:
                        prior = old[branch][family]
                        candidate = new[branch][family]
                        if prior.shape != candidate.shape or prior.shape != (5, 5):
                            raise AssertionError("native provider dimensions changed")
                        for trial, first, second in zip(old_trials, prior, candidate):
                            identity = str(trial)
                            if user == 15 and domain == "trial_1":
                                frozen = parent.get((branch, family, identity))
                                if frozen is None:
                                    raise AssertionError("frozen legacy prediction identity changed")
                                reference = np.asarray([float(frozen[f"p_{h}"]) for h in range(5)])
                                parent_max_error = max(parent_max_error,
                                                       float(np.max(np.abs(first - reference))))
                            rows.append({"phase": phase, "subject": user, "domain": domain,
                                         "branch": branch, "family": family, "trial_id": identity,
                                         "truth_for_audit": int(np.unique(
                                             evaluation.labels[evaluation.trials == trial]).item()),
                                         **{f"legacy_p_{h}": float(first[h]) for h in range(5)},
                                         **{f"document_p_{h}": float(second[h]) for h in range(5)}})
            print(f"Document session {phase}: subject {user} complete", flush=True)
    if len(rows) != 1440 or len(blocks) != 24 or parent_max_error > 1e-12:
        raise AssertionError("native paired prediction or frozen parent replay failed")
    output = ROOT / "DOCUMENT_SESSION_PREDICTIONS.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    phase_scores = {}
    for phase in ("validation", "descriptive_final"):
        selected = [row for row in rows if row["phase"] == phase]
        difference = np.asarray([max(abs(row[f"legacy_p_{h}"] - row[f"document_p_{h}"])
                                     for h in range(5)) for row in selected])
        changed = sum(np.argmax([row[f"legacy_p_{h}"] for h in range(5)]) !=
                      np.argmax([row[f"document_p_{h}"] for h in range(5)]) for row in selected)
        phase_scores[phase] = {"prediction_rows": len(selected),
                               "maximum_absolute_probability_difference": float(difference.max()),
                               "changed_argmax_rows": int(changed)}
    result = {"protocol_sha256": sha(PROTOCOL_PATH), "archive_sha256": sha(wearing.ARCHIVE),
              "parent_prediction_sha256": sha(PARENT), "prediction_sha256": sha(output),
              "parent_legacy_max_abs_error": parent_max_error,
              "prediction_rows": len(rows), "blocks": blocks,
              "minimum_source_q95": min(b["minimum_document_source_q95"] for b in blocks),
              "minimum_session_q95": min(b["minimum_document_session_q95"] for b in blocks),
              "phase_scores": phase_scores, "scope": PROTOCOL["scope"]}
    (ROOT / "DOCUMENT_SESSION_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "phase_scores": phase_scores}), flush=True)
    return result


if __name__ == "__main__":
    run()
