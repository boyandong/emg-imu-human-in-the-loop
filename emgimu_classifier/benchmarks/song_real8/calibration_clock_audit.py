"""Measure the recorded guided-calibration span with independent HDF5 clocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from prospective_gate import sha256


HERE = Path(__file__).resolve().parent


def analyze_session(path: Path, expected_sha256: str) -> dict:
    signal = path / "session.h5"
    digest = sha256(signal)
    if digest != expected_sha256:
        raise ValueError(f"source session hash changed: {signal}")
    with h5py.File(signal, "r") as handle:
        events = handle["events"][:]
        starts = events[events["event_type"] == b"CALIBRATION_BLOCK_START"]
        ends = events[events["event_type"] == b"CALIBRATION_BLOCK_END"]
        if len(starts) != 26 or len(ends) != 26:
            raise ValueError(f"unexpected calibration block count: {path}")
        if (starts[0]["label"] != b"calibration_rest_initial" or
                ends[-1]["label"] != b"calibration_rest_final"):
            raise ValueError(f"unexpected calibration block order: {path}")
        hand_end = ends[ends["label"] == b"calibration_rest_after_open_2"]
        if len(hand_end) != 1:
            raise ValueError(f"hand calibration boundary missing: {path}")
        emg = handle["streams/emg"]
        indices = emg["sample_index"][:]
        if len(indices) < 2 or np.any(np.diff(indices) <= 0):
            raise ValueError(f"EMG sample indices invalid: {path}")

        def received_span(first: np.void, last: np.void) -> float:
            left = int(np.searchsorted(indices, first["sample_index"]))
            right = int(np.searchsorted(indices, last["sample_index"])) - 1
            if left >= len(indices) or right < left:
                raise ValueError(f"event has no EMG reception interval: {path}")
            return round(float(emg["pc_received_ns"][right] - emg["pc_received_ns"][left]) / 1e9, 3)

        def event_span(first: np.void, last: np.void) -> float:
            return round(float(last["pc_monotonic_ns"] - first["pc_monotonic_ns"]) / 1e9, 3)

        session_start = events[events["event_type"] == b"SESSION_START"]
        if len(session_start) != 1:
            raise ValueError(f"session start missing: {path}")
        return {
            "session": str(handle["meta"].attrs["session_id"]),
            "hdf5_sha256": digest,
            "calibration_blocks": len(starts),
            "recorded_event_first_to_last_seconds": event_span(starts[0], ends[-1]),
            "recorded_emg_reception_first_to_last_seconds": received_span(starts[0], ends[-1]),
            "recorded_event_hand_sequence_seconds": event_span(starts[0], hand_end[0]),
            "recorded_emg_reception_hand_sequence_seconds": received_span(starts[0], hand_end[0]),
            "session_start_to_calibration_end_event_seconds": event_span(session_start[0], ends[-1]),
        }


def build(source: Path) -> dict:
    freeze = json.loads((HERE / "PROSPECTIVE_FREEZE.json").read_text(encoding="utf-8"))
    rows = []
    for session_id, expected in freeze["existing_session_hdf5_sha256"].items():
        candidates = sorted(source.glob(f"*_{session_id}"))
        if len(candidates) != 1:
            raise ValueError(f"expected exactly one {session_id} directory")
        rows.append(analyze_session(candidates[0], expected))
    return {
        "status": "recorded_in_protocol_elapsed_only",
        "freeze_sha256": sha256(HERE / "PROSPECTIVE_FREEZE.json"),
        "sessions": rows,
        "limitations": [
            "All four sessions are one person and one day; S01-S03 failed collection readiness.",
            "Recorded monotonic event and EMG reception clocks measure only in-app guided-protocol time.",
            "Electrode preparation, operator clicks before session start, recognition calibration outside this protocol, and user-perceived burden were not recorded.",
            "No inference about new-day or new-user calibration effectiveness is possible.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "CALIBRATION_CLOCK_AUDIT.json")
    args = parser.parse_args()
    result = build(args.source)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "sessions": len(result["sessions"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
