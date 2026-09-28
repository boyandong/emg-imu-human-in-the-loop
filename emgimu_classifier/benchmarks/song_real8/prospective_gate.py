"""Reject previously inspected or incompatible Song sessions before blind scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import h5py


REPO = Path(__file__).resolve().parents[3]
FREEZE = Path(__file__).with_name("PROSPECTIVE_FREEZE.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_session(candidate: Path, *, freeze_path: Path = FREEZE,
                  repo: Path = REPO) -> dict:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    reasons = []
    model_hashes = {}
    for name in ("model", "model_manifest"):
        path = repo / freeze[f"{name}_relative_path"]
        value = sha256(path) if path.is_file() else None
        model_hashes[name] = value
        if value != freeze[f"{name}_sha256"]:
            reasons.append(f"frozen_{name}_hash_mismatch")

    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})_([A-Za-z][A-Za-z0-9_-]*)", candidate.name)
    session_date = None
    if match is None:
        reasons.append("session_directory_name_invalid")
    else:
        try:
            session_date = date.fromisoformat(match.group(1))
            if session_date <= date.fromisoformat(freeze["frozen_date_local"]):
                reasons.append("session_not_after_freeze_date")
        except ValueError:
            reasons.append("session_date_invalid")

    config_path = candidate / "session_config.json"
    readiness_path = candidate / "SESSION_COLLECTION_READINESS.json"
    signal_path = candidate / "session.h5"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    readiness = json.loads(readiness_path.read_text(encoding="utf-8")) if readiness_path.is_file() else {}
    session = config.get("session", {})
    session_id = session.get("session_id")
    if not config:
        reasons.append("session_config_missing")
    if not readiness:
        reasons.append("readiness_missing")
    if match and (session_id != match.group(2)):
        reasons.append("session_id_directory_mismatch")
    if session_id in freeze["existing_session_hdf5_sha256"]:
        reasons.append("previously_inspected_session_id")
    if session.get("protocol_name") != freeze["protocol_name"]:
        reasons.append("protocol_mismatch")
    if session.get("dataset_split") != "test":
        reasons.append("not_test_split")
    if session.get("model_frozen_confirmed") is not True:
        reasons.append("model_not_confirmed_frozen_at_collection")
    if readiness.get("status") != "passed":
        reasons.append("collection_readiness_not_passed")
    if readiness.get("protocol_required") != freeze["protocol_name"]:
        reasons.append("readiness_protocol_mismatch")
    checks = readiness.get("checks", {})
    rates = checks.get("sampling_rates", {})
    if not (rates.get("emg_nominal_hz") == 250.0 and
            rates.get("imu_nominal_hz") == 112.0 and
            checks.get("hdf5_v3") is True and checks.get("formal_protocol") is True):
        reasons.append("sampling_or_schema_incompatible")
    required = {f"still_{state}" for state in freeze["required_hand_states"]}
    counts = checks.get("valid_trial_counts", {})
    if any(counts.get(label, 0) < 1 for label in required):
        reasons.append("hand_state_trials_missing")

    signal_hash = sha256(signal_path) if signal_path.is_file() else None
    if signal_hash is None:
        reasons.append("session_hdf5_missing")
    elif not h5py.is_hdf5(signal_path):
        reasons.append("session_hdf5_invalid")
    else:
        with h5py.File(signal_path, "r") as handle:
            meta = handle["meta"].attrs if "meta" in handle else {}
            raw = handle.get("streams/emg/raw")
            accel = handle.get("streams/imu/accel")
            if (str(meta.get("schema_version")) != "3.0" or
                    meta.get("session_id") != session_id or
                    meta.get("date") != (session_date.isoformat() if session_date else None) or
                    meta.get("protocol_name") != freeze["protocol_name"] or
                    meta.get("dataset_split") != "test" or
                    bool(meta.get("model_frozen_confirmed")) is not True or
                    int(meta.get("num_emg_channels", 0)) != 8 or
                    int(meta.get("emg_nominal_rate_hz", 0)) != 250 or
                    int(meta.get("imu_nominal_rate_hz", 0)) != 112 or
                    raw is None or len(raw.shape) != 2 or raw.shape[1] != 8 or
                    accel is None or len(accel.shape) != 2 or accel.shape[1] != 3 or
                    "trials" not in handle):
                reasons.append("hdf5_content_incompatible")
    if signal_hash in freeze["existing_session_hdf5_sha256"].values():
        reasons.append("previously_inspected_signal_hash")
    if signal_hash != readiness.get("hdf5_sha256"):
        reasons.append("readiness_signal_hash_mismatch")
    return {
        "eligible_for_prospective_scoring": not reasons,
        "reasons": reasons,
        "candidate": str(candidate.resolve()),
        "session_date": session_date.isoformat() if session_date else None,
        "session_id": session_id,
        "signal_sha256": signal_hash,
        "model_sha256": model_hashes["model"],
        "freeze_sha256": sha256(freeze_path),
        "scope": freeze["scope"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    result = check_session(args.candidate)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["eligible_for_prospective_scoring"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
