"""Check the committed posture evidence without requiring the external archive."""
import json

from benchmarks.new_bank_v2.envelope_posture_analysis import build as verify_envelope
from benchmarks.new_bank_v2.roam_posture_paired import build as verify_pairs
from benchmarks.new_bank_v2.roam_posture_run import ROOT


def test_roam_posture_frozen_evidence_and_subject_boundary():
    result = json.loads((ROOT / "ROAM_POSTURE_RESULTS.json").read_text(encoding="utf-8"))
    protocol = result["protocol"]
    source = set(protocol["source_subjects"])
    validation = set(protocol["validation_subjects"])
    final = set(protocol["final_subjects"])
    assert not (source & validation or source & final or validation & final)
    assert result["source_rest_windows"] > 1000
    assert len(result["source_trial_ids"]) == 162
    assert len(result["validation_trial_ids"]) == len(result["final_trial_ids"]) == 180
    assert result["validation_selected_arm"] == "F0v2"
    verify_pairs(verify=True)
    verify_envelope(verify=True)
