"""Four-session protocol gate across files, dates and donning attestations."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import h5py

from emgforce.collection_protocol import FORMAL_SESSION_SPLITS, SESSION_MANIFEST_FILENAME


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8").rstrip("\x00")
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_cohort(hdf5_paths: list[Path]) -> dict:
    """Check four individual readiness reports and their cross-session claims.

    A passing report is a metadata/temporal gate, not physical proof of electrode
    removal, a blinded model freeze, or real-world recognition performance.
    """
    problems: list[str] = []
    warnings: list[str] = []
    sessions: dict[str, dict] = {}
    for raw_path in hdf5_paths:
        path = Path(raw_path)
        if not path.is_file():
            problems.append(f"missing session file: {path.name}")
            continue
        digest = _sha256(path)
        manifest_path = path.with_name(SESSION_MANIFEST_FILENAME)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{path.parent.name}: readiness manifest unavailable ({exc})")
            continue
        if not isinstance(manifest, dict):
            problems.append(f"{path.parent.name}: readiness manifest is not an object")
            continue
        try:
            with h5py.File(path, "r") as handle:
                attrs = handle["meta"].attrs
                subject = _text(attrs.get("participant_id", "")).strip()
                session = _text(attrs.get("session_id", "")).strip().upper()
                recorded_date = date.fromisoformat(_text(attrs.get("date", "")))
                started = datetime.fromisoformat(_text(attrs.get("start_datetime", "")))
                if started.tzinfo is None or started.utcoffset() is None:
                    raise ValueError("start_datetime has no timezone")
                notes = _text(attrs.get("donning_notes", "")).strip()
                frozen = attrs.get("model_frozen_confirmed", False)
                freeze_attested = frozen is True or str(frozen).lower() in ("true", "1")
        except (OSError, KeyError, ValueError) as exc:
            problems.append(f"{path.parent.name}: invalid HDF5 session metadata ({exc})")
            continue
        if session in sessions:
            problems.append(f"duplicate session ID: {session}")
            continue
        if not subject:
            problems.append(f"{session}: missing participant ID")
        if started.date() != recorded_date:
            problems.append(f"{session}: date differs from start_datetime local date")
        expected_split = FORMAL_SESSION_SPLITS.get(session)
        if expected_split is None:
            problems.append(f"unexpected session ID: {session}")
        if manifest.get("status") != "passed":
            problems.append(f"{session}: individual readiness failed")
        if manifest.get("hdf5_file") != path.name or manifest.get("hdf5_sha256") != digest:
            problems.append(f"{session}: individual readiness HDF5 hash is stale")
        sessions[session] = {"participant_id": subject, "date": recorded_date.isoformat(),
                             "start_datetime": started.isoformat(), "hdf5_sha256": digest,
                             "individual_readiness": manifest.get("status", "missing"),
                             "donning_attested": bool(notes), "model_freeze_attested": freeze_attested,
                             "expected_split": expected_split}
    required = set(FORMAL_SESSION_SPLITS)
    if set(sessions) != required or len(hdf5_paths) != len(required):
        problems.append(f"cohort must contain exactly S01-S04; found {sorted(sessions)}")
    if len({item["participant_id"] for item in sessions.values()}) > 1:
        problems.append("S01-S04 do not belong to one participant")
    if set(sessions) == required:
        order = [datetime.fromisoformat(sessions[key]["start_datetime"]) for key in FORMAL_SESSION_SPLITS]
        if not all(left < right for left, right in zip(order, order[1:])):
            problems.append("S01-S04 acquisition start times are not strictly ordered")
        if order[0].date() != order[1].date():
            problems.append("S01 and S02 must be recorded on the same local day")
        if order[2].date() <= order[1].date():
            problems.append("S03 validation must occur on a later local day than S01/S02")
        if order[3].date() < order[2].date():
            problems.append("S04 final test cannot precede S03's local day")
        for key in ("S02", "S03", "S04"):
            if not sessions[key]["donning_attested"]:
                problems.append(f"{key}: no recorded electrode re-donning attestation")
        if not sessions["S04"]["model_freeze_attested"]:
            problems.append("S04: model freeze was not attested")
    warnings.append("Donning notes and freeze confirmation are operator attestations, not independent physical or historical proof")
    return {"status": "passed" if not problems else "failed",
            "scope": "four-session formal collection cohort; individual readiness, one participant, chronology, day split and operator attestations",
            "sessions": sessions, "problems": problems, "warnings": warnings,
            "boundary": "A pass does not establish live recognition accuracy, real reapplication, or immutable model files; those require independent measurements and archived model hashes."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs=4, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_cohort(args.sessions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps({"status": result["status"], "problems": result["problems"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
