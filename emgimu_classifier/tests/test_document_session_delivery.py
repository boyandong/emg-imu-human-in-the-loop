"""Read back the exact-versus-frozen native session normalization experiment."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_document_session_native_pairing_and_frozen_parent():
    protocol = json.loads((ROOT / "DOCUMENT_SESSION_PROTOCOL.json").read_text())
    result = json.loads((ROOT / "DOCUMENT_SESSION_RESULTS.json").read_text())
    assert result["protocol_sha256"] == sha(ROOT / "DOCUMENT_SESSION_PROTOCOL.json")
    assert result["parent_prediction_sha256"] == protocol["parent_unlabeled_predictions_sha256"]
    assert result["prediction_sha256"] == sha(ROOT / "DOCUMENT_SESSION_PREDICTIONS.csv")
    assert result["parent_legacy_max_abs_error"] == 0
    with (ROOT / "DOCUMENT_SESSION_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (ROOT / "SESSION_UNLABELED_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        parent = {(row["branch"], row["family"], row["trial_id"]): row
                  for row in csv.DictReader(stream)}
    assert len(rows) == result["prediction_rows"] == 1440
    assert len(result["blocks"]) == 24
    keys = {(row["phase"], row["subject"], row["domain"], row["branch"],
             row["family"], row["trial_id"]) for row in rows}
    assert len(keys) == len(rows)
    for block in result["blocks"]:
        source = set(block["source_trial_ids"])
        calibration = set(block["calibration_trial_ids"])
        evaluation = set(block["evaluation_trial_ids"])
        assert (len(source), len(calibration), len(evaluation)) == (25, 5, 5)
        assert not source & calibration and not source & evaluation and not calibration & evaluation
        assert block["legacy_profile_id"] != block["document_profile_id"]
        assert min(block["minimum_document_source_q95"],
                   block["minimum_document_session_q95"]) > 0
        selected = [row for row in rows if row["phase"] == block["phase"]
                    and int(row["subject"]) == block["subject"]
                    and row["domain"] == block["domain"]]
        assert len(selected) == 60
        assert {row["trial_id"] for row in selected} == evaluation
        for row in selected:
            truth = int(row["trial_id"].rsplit("_C_", 1)[1].removesuffix(".csv"))
            assert int(row["truth_for_audit"]) == truth
            legacy = np.asarray([float(row[f"legacy_p_{h}"]) for h in range(5)])
            document = np.asarray([float(row[f"document_p_{h}"]) for h in range(5)])
            np.testing.assert_allclose(legacy, document, rtol=0, atol=0)
            np.testing.assert_allclose(legacy.sum(), 1, rtol=0, atol=1e-12)
            if block["subject"] == 15 and block["domain"] == "trial_1":
                frozen = parent[(row["branch"], row["family"], row["trial_id"])]
                np.testing.assert_allclose(legacy,
                    [float(frozen[f"p_{h}"]) for h in range(5)], rtol=0, atol=0)
    for phase in ("validation", "descriptive_final"):
        selected = [row for row in rows if row["phase"] == phase]
        assert len(selected) == result["phase_scores"][phase]["prediction_rows"] == 720
        assert result["phase_scores"][phase]["maximum_absolute_probability_difference"] == 0
        assert result["phase_scores"][phase]["changed_argmax_rows"] == 0
