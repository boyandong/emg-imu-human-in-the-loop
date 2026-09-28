"""Verify committed external-Rest MANUS evidence without raw archives."""
import json

from benchmarks.new_bank_v2.envelope_speed_analysis import build as verify_envelope
from benchmarks.new_bank_v2.manus_rest_transfer_paired import build as verify_pairs
from benchmarks.new_bank_v2.manus_rest_transfer_run import ROOT


def test_external_rest_speed_trial_boundary_and_saved_scores():
    result = json.loads((ROOT / "MANUS_REST_TRANSFER_RESULTS.json").read_text(encoding="utf-8"))
    reduced = json.loads((ROOT / "MANUS_SPATIAL_RESULTS.json").read_text(encoding="utf-8"))
    assert result["external_rest_windows"] == 1828
    assert result["validation_selected_arm"] == "F0v2+F2a"
    assert result["feature_dimensions"] == {"F0v2": 48, "F2a": 36, "F3c": 20}
    for phase in ("validation", "final"):
        split = result["split_trial_ids"][phase]
        assert len(split["source_trials"]) == len(split["target_trials"]) == 108
        assert not set(split["source_trials"]) & set(split["target_trials"])
        assert split["target_trials"] == reduced["split_trial_ids"][phase]["target_trials"]
    verify_pairs(verify=True)
    verify_envelope(verify=True)
