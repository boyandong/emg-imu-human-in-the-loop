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


def test_roam_discovery_manifest_matches_frozen_native_audit():
    discovery = ROOT.parent / "discovery"
    manifest = json.loads((discovery / "DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    records = [row for row in manifest["datasets"] if row["id"] == "roam_emg"]
    audit = json.loads((discovery / "ROAM_NATIVE_AUDIT.json").read_text(encoding="utf-8"))
    assert len(records) == 1
    assert records[0]["sha256"] == audit["archive_sha256"]
    assert records[0]["size"] == audit["archive_bytes"]
    assert audit["zip_crc_verified"] and audit["static_files_verified"] == 112
