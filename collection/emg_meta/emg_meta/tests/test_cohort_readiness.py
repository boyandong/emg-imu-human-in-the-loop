"""Cohort-level checks that individual session readiness cannot establish."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import Mock

import h5py
import pytest

from emgforce.collection_protocol import FORMAL_PROTOCOL_NAME, SESSION_MANIFEST_FILENAME
from emgforce.experiment.models import ParticipantInfo, ProtocolConfig, SessionInfo
from emgforce.experiment.session import ExperimentSession
from emgforce.quality.cohort_readiness import audit_cohort, validate_next_session_day


def _session(root: Path, session: str, stamp: str, *, subject: str = "P001",
             notes: str = "electrodes removed and reapplied") -> Path:
    folder = root / f"{stamp[:10]}_{session}"
    folder.mkdir(parents=True)
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
    participant_root = root / "P001"
    return [_session(participant_root, "S01", "2026-09-18T09:00:00+08:00"),
            _session(participant_root, "S02", "2026-09-18T11:00:00+08:00"),
            _session(participant_root, "S03", s03),
            _session(participant_root, "S04", "2026-09-19T11:00:00+08:00", notes=s04_notes)]


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


def test_start_gate_rejects_same_day_s03_and_accepts_later_day(tmp_path: Path) -> None:
    _four(tmp_path)
    validate_next_session_day(tmp_path, "P001", "S02", date(2026, 9, 18))
    with pytest.raises(ValueError, match="S03 是次日验证"):
        validate_next_session_day(tmp_path, "P001", "S03", date(2026, 9, 18))
    validate_next_session_day(tmp_path, "P001", "S03", date(2026, 9, 19))
    validate_next_session_day(tmp_path, "P001", "S04", date(2026, 9, 19))


def test_programmatic_formal_session_rejects_missing_redonning_before_writing(tmp_path: Path) -> None:
    today = datetime.now().astimezone().date().isoformat()
    _session(tmp_path / "P001", "S01", f"{today}T09:00:00+08:00")
    session = ExperimentSession(Mock(), tmp_path)
    protocol = ProtocolConfig(FORMAL_PROTOCOL_NAME, ["Neutral"], 1,
                              formal_collection=True, block_size=1, block_break_sec=1)
    with pytest.raises(ValueError, match="必须记录电极取下"):
        session.start(ParticipantInfo("P001"),
                      SessionInfo("S02", "formal", FORMAL_PROTOCOL_NAME), protocol)
    assert list((tmp_path / "P001").glob("*_S02")) == []


def test_programmatic_formal_session_rejects_same_day_validation(tmp_path: Path) -> None:
    today = datetime.now().astimezone().date().isoformat()
    _session(tmp_path / "P001", "S02", f"{today}T09:00:00+08:00")
    session = ExperimentSession(Mock(), tmp_path)
    protocol = ProtocolConfig(FORMAL_PROTOCOL_NAME, ["Neutral"], 1,
                              formal_collection=True, block_size=1, block_break_sec=1)
    with pytest.raises(ValueError, match="S03 是次日验证"):
        session.start(ParticipantInfo("P001"),
                      SessionInfo("S03", "formal", FORMAL_PROTOCOL_NAME,
                                  donning_notes="electrodes removed and reapplied"), protocol)
    assert list((tmp_path / "P001").glob("*_S03")) == []
