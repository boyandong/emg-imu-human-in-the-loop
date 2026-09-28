"""Cohort-level checks that individual session readiness cannot establish."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py

from emgforce.collection_protocol import SESSION_MANIFEST_FILENAME
from emgforce.quality.cohort_readiness import audit_cohort


def _session(root: Path, session: str, stamp: str, *, subject: str = "P001",
             notes: str = "electrodes removed and reapplied") -> Path:
    folder = root / session
    folder.mkdir()
    path = folder / "session.h5"
    with h5py.File(path, "w") as handle:
        attrs = handle.create_group("meta").attrs
        attrs["participant_id"] = subject
        attrs["session_id"] = session
        attrs["date"] = stamp[:10]
        attrs["start_datetime"] = stamp
        attrs["donning_notes"] = notes
        attrs["model_frozen_confirmed"] = session == "S04"
    (folder / SESSION_MANIFEST_FILENAME).write_text(json.dumps({
        "status": "passed", "hdf5_file": path.name,
        "hdf5_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}), encoding="utf-8")
    return path


def _four(root: Path, *, s03: str = "2026-09-19T09:00:00+08:00",
          s04_notes: str = "electrodes removed and reapplied") -> list[Path]:
    return [_session(root, "S01", "2026-09-18T09:00:00+08:00"),
            _session(root, "S02", "2026-09-18T11:00:00+08:00"),
            _session(root, "S03", s03),
            _session(root, "S04", "2026-09-19T11:00:00+08:00", notes=s04_notes)]


def test_cross_day_attested_cohort_passes(tmp_path: Path) -> None:
    result = audit_cohort(_four(tmp_path))
    assert result["status"] == "passed"
    assert result["problems"] == []
    assert set(result["sessions"]) == {"S01", "S02", "S03", "S04"}


def test_same_day_validation_fails_even_if_individual_manifests_pass(tmp_path: Path) -> None:
    result = audit_cohort(_four(tmp_path, s03="2026-09-18T13:00:00+08:00"))
    assert result["status"] == "failed"
    assert any("later local day" in problem for problem in result["problems"])


def test_stale_manifest_and_missing_redonning_attestation_fail(tmp_path: Path) -> None:
    paths = _four(tmp_path, s04_notes="")
    with h5py.File(paths[0], "a") as handle:
        handle["meta"].attrs["operator_note"] = "modified after readiness"
    result = audit_cohort(paths)
    assert result["status"] == "failed"
    assert any("hash is stale" in problem for problem in result["problems"])
    assert any("S04: no recorded electrode re-donning" in problem for problem in result["problems"])
