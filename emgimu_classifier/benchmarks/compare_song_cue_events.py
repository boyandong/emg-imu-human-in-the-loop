"""Paired S04 cue-event comparison of the saved F0 and current F0+SPD models."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "song_real8"


def load(prefix: str) -> tuple[dict, dict[str, dict]]:
    audit = json.loads((ROOT / f"{prefix}_AUDIT.json").read_text(encoding="utf-8"))
    verification = json.loads((ROOT / f"{prefix}_VERIFICATION.json").read_text(encoding="utf-8"))
    if verification["status"] != "verified" or verification["event_rows_read_back"] != 144:
        raise ValueError(f"{prefix} event audit is not verified")
    with (ROOT / f"{prefix}_ROWS.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keyed = {row["trial_id"]: row for row in rows}
    if len(rows) != 144 or len(keyed) != 144:
        raise ValueError(f"{prefix} event row coverage differs")
    return audit, keyed


def compare() -> dict:
    old, a = load("CUE_EVENT")
    current, b = load("SPD_CUE_EVENT")
    if (a.keys() != b.keys() or old["source_sha256"] != current["source_sha256"]
            or old["model_sha256"] == current["model_sha256"]):
        raise ValueError("paired cue-event sources/models do not align")
    active = Counter()
    rest = Counter()
    by_class = {}
    for trial_id in a:
        left, right = a[trial_id], b[trial_id]
        for key in ("label", "hand", "stable_start_sample", "stable_end_sample",
                    "stable_frame_count", "late_rest_frame_count"):
            if left[key] != right[key]:
                raise ValueError(f"paired event annotation differs: {trial_id} {key}")
        rest[(left["pre_prompt_activation"] == "True", right["pre_prompt_activation"] == "True")] += 1
        if left["hand"] != "neutral":
            pair = (left["event_success"] == "True", right["event_success"] == "True")
            active[pair] += 1
            by_class.setdefault(left["hand"], Counter())[pair] += 1
    if sum(active.values()) != 108 or sum(rest.values()) != 144:
        raise ValueError("paired event totals differ")
    names = {(True, True): "both", (False, True): "current_only",
             (True, False): "f0_only", (False, False): "neither"}
    artifact = {
        "status": "verified_paired_cue_comparison",
        "same_s04_source_sha256": old["source_sha256"],
        "f0_model_sha256": old["model_sha256"],
        "current_spd_model_sha256": current["model_sha256"],
        "active_event_detection": {names[pair]: active[pair] for pair in names},
        "active_event_detection_by_class": {
            name: {names[pair]: counts[pair] for pair in names}
            for name, counts in sorted(by_class.items())},
        "late_rest_active_decoded_state": {
            names[pair]: rest[pair] for pair in names},
        "boundary": "Same saved S04 and fixed decoder rule; after-the-fact paired comparison on an already explored one-person session. Late rest may retain prior movement. Neither cue-relative delay nor active state establishes physical-device false triggers or physiological onset latency.",
    }
    (ROOT / "CUE_EVENT_COMPARISON.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2))
