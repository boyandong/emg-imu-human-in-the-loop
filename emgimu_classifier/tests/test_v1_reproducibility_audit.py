"""New-version experiment packages expose reproducible split/model evidence."""
from __future__ import annotations

from pathlib import Path

from benchmarks.new_bank_v2.roam_posture_run import sha256
from benchmarks.new_bank_v2.v1_reproducibility_audit import build

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"


def test_seven_new_version_packages_recheck_without_drift() -> None:
    path = ROOT / "V1_REPRODUCIBILITY_AUDIT.json"
    digest = sha256(path)
    audit = build()
    assert sha256(path) == digest
    assert audit["screens"] == 7
    assert audit["prediction_rows"] == 19060
    assert all(item["channels"] == 8 and item["source_trials"] > 0
               and item["validation_trials"] > 0 and item["final_trials"] > 0
               and "random_state=" in item["model"] for item in audit["experiments"])
