"""Check cue-event rows against S04 native annotations and saved aggregate."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
from pathlib import Path

import h5py

from benchmarks.song_cue_event_audit import PROTOCOL_PATH, ROOT, summarize
from benchmarks.song_real8_study import _hash, _text, parse_label


def verify(source: Path = Path("E:/qxy/emg_meta/emg_meta/data/Song"),
           protocol_path: Path = PROTOCOL_PATH, prefix: str = "CUE_EVENT") -> dict:
    audit = json.loads((ROOT / f"{prefix}_AUDIT.json").read_text(encoding="utf-8"))
    session = source / "2026-09-18_S04" / "session.h5"
    if (_hash(session) != audit["source_sha256"]
            or audit["source_sha256"] != audit["protocol"]["source_sha256"]
            or audit["model_sha256"] != audit["protocol"]["model_sha256"]
            or audit["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest()):
        raise ValueError("frozen input/model/protocol hash mismatch")
    with h5py.File(session) as handle:
        native = {_text(trial["trial_id"]): trial for trial in handle["trials"][:]
                  if _text(trial["trial_kind"]) == "formal" and bool(trial["valid"])
                  and _text(trial["completion_status"]) == "completed"}
    with (ROOT / f"{prefix}_ROWS.csv").open(encoding="utf-8", newline="") as stream:
        saved = list(csv.DictReader(stream))
    if len(saved) != 144 or len({row["trial_id"] for row in saved}) != 144 or set(native) != {
            row["trial_id"] for row in saved}:
        raise ValueError("formal event coverage differs from S04 annotations")
    rows = []
    for row in saved:
        trial = native[row["trial_id"]]
        hand = parse_label(_text(trial["label"]))[1]
        if (row["label"] != _text(trial["label"]) or row["hand"] != hand
                or int(row["stable_start_sample"]) != int(trial["stable_start_sample"])
                or int(row["stable_end_sample"]) != int(trial["stable_end_sample"])):
            raise ValueError(f"native cue identity/boundary differs: {row['trial_id']}")
        parsed = {**row}
        for field in ("stable_start_sample", "stable_end_sample", "stable_frame_count",
                      "decoded_correct_frames", "decoded_unknown_frames", "late_rest_frame_count",
                      "late_rest_active_frame_count"):
            parsed[field] = int(row[field])
        for field in ("event_success", "endpoint_correct"):
            if row[field] not in ("True", "False"):
                raise ValueError(f"invalid boolean {field}")
            parsed[field] = row[field] == "True"
        if row["pre_prompt_activation"] not in ("", "True", "False"):
            raise ValueError("invalid pre-prompt activation flag")
        parsed["pre_prompt_activation"] = (None if row["pre_prompt_activation"] == ""
                                           else row["pre_prompt_activation"] == "True")
        parsed["first_correct_latency_seconds"] = (None if row["first_correct_latency_seconds"] == ""
                                                   else float(row["first_correct_latency_seconds"]))
        if (parsed["stable_frame_count"] < 1
                or parsed["decoded_correct_frames"] + parsed["decoded_unknown_frames"] > parsed["stable_frame_count"]
                or parsed["event_success"] != (parsed["decoded_correct_frames"] > 0)
                or (parsed["endpoint_correct"] and not parsed["event_success"])
                or parsed["late_rest_active_frame_count"] > parsed["late_rest_frame_count"]
                or parsed["pre_prompt_activation"] != (
                    bool(parsed["late_rest_active_frame_count"]) if parsed["late_rest_frame_count"] else None)):
            raise ValueError(f"event row inconsistency: {row['trial_id']}")
        delay = parsed["first_correct_latency_seconds"]
        if hand == "neutral" or not parsed["event_success"]:
            if delay is not None:
                raise ValueError("latency reported for neutral or missed event")
        elif delay is None or not 0 <= delay <= (
                parsed["stable_end_sample"] - parsed["stable_start_sample"]) / 250:
            raise ValueError("latency outside stable cue interval")
        rows.append(parsed)
    summary = summarize(rows)
    if any(audit[key] != value for key, value in summary.items()):
        raise ValueError("saved event summary differs from native-row readback")
    result = {"status": "verified", "native_formal_trials_checked": len(native),
              "event_rows_read_back": len(rows), "summary_groups_read_back": len(summary),
              "protocol_sha256": audit["protocol_sha256"],
              "boundary": "Native cue markers and saved rows agree; physiological onset, USB and screen latency remain unmeasured."}
    (ROOT / f"{prefix}_VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("E:/qxy/emg_meta/emg_meta/data/Song"))
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--prefix", default="CUE_EVENT")
    args = parser.parse_args()
    print(json.dumps(verify(args.source, args.protocol, args.prefix), indent=2))
