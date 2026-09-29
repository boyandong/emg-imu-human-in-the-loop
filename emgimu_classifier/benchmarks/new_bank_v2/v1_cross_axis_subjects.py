"""Read-back per-subject effects across the seven independent-family screens."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2.force_v1_extension_analysis import write_csv
from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parent
PREFIX = "V1_CROSS_AXIS_SUBJECT"
SOURCES = {
    "V1_EXTENSION_FAMILY_SCREEN.csv": "a40b64222ae5405381317e3bbf44f979e2ed9947a2fc70a607ae3e781c9f602a",
    "FORCE_V1_EXTENSION_FAMILY.csv": "a71419ff8b26a29a18fe8add23750fab9a78a938da8bf74ecc014739cde522e4",
    "WEARING_V1_EXTENSION_FAMILY.csv": "b4bd74f7342eb1ea9c0ca8e72ae537d70d3389b0cf3337ffc040093f98910c2a",
    "MANUS_V1_SPEED_FAMILY.csv": "8084912efc1f59dafcbbc08da34810e300c1d889570048011cbe791fcecf7224",
    "ROAM_V1_QUALITY_FAMILY.csv": "b7022eb087576e38bb3ecaa3348fdc9839e2bbb51e4fa404232b673d6f2811f0",
}
STUDIES = {"roam_posture": "posture", "grab_user": "unseen_user", "grab_day": "cross_day"}
DIRECT = {"FORCE_V1_EXTENSION_FAMILY.csv": "force_intensity",
          "WEARING_V1_EXTENSION_FAMILY.csv": "wearing_shift",
          "MANUS_V1_SPEED_FAMILY.csv": "observed_speed"}
ARMS = ["F0v2", "F0v2+scale_pattern", "F0v2+ring_lag",
        "F0v2+correlation_spectrum", "F0v2+frequency_direction"]


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def build() -> dict:
    for filename, digest in SOURCES.items():
        if sha256(ROOT / filename) != digest:
            raise AssertionError(f"frozen family screen changed: {filename}")
    indexed: dict[tuple[str, str, str, str], dict] = {}
    for row in read(ROOT / "V1_EXTENSION_FAMILY_SCREEN.csv"):
        if row["scope"] == "subject":
            key = (STUDIES[row["study"]], row["phase"], row["subject"], row["feature_family"])
            indexed[key] = {"macro_f1": float(row["macro_f1"]),
                            "log_loss": float(row["log_loss"])}
    for filename, axis in DIRECT.items():
        for row in read(ROOT / filename):
            if row["scope"] == "subject" and row["condition"] == "ALL":
                key = (axis, row["phase"], row["subject"], row["feature_family"])
                indexed[key] = {"macro_f1": float(row["macro_f1"]),
                                "log_loss": float(row["log_loss"])}
    quality: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for row in read(ROOT / "ROAM_V1_QUALITY_FAMILY.csv"):
        if row["scope"] == "subject":
            key = (row["phase"], row["subject"], row["feature_family"])
            quality[key][row["condition"]] = row
    for (phase, subject, arm), faults in quality.items():
        if len(faults) != 14:
            raise AssertionError("quality subject fault grid incomplete")
        mean_dropout = float(np.mean([float(faults[f"dropout_ch{i}"]["macro_f1"])
                                      for i in range(8)]))
        other = ["gain_half_all", "clip_q95_all", "line50_half_rms_all",
                 "baseline_ramp_half_rms_all", "burst_ch0_2q95"]
        family_mean = float(np.mean([mean_dropout,
                                     *(float(faults[name]["macro_f1"]) for name in other)]))
        fault_loss = float(np.mean([float(row["log_loss"]) for name, row in faults.items()
                                    if name != "clean"]))
        indexed[("synthetic_quality", phase, subject, arm)] = {
            "macro_f1": family_mean, "log_loss": fault_loss}
    expected_subjects = {"posture": 5, "unseen_user": 2, "cross_day": 8,
                         "force_intensity": 2, "wearing_shift": 3,
                         "observed_speed": 6, "synthetic_quality": 5}
    cells, summary = [], []
    for axis, count in expected_subjects.items():
        for phase in ("validation", "final"):
            users = sorted({subject for name, state, subject, arm in indexed
                            if name == axis and state == phase and arm == "F0v2"}, key=int)
            if len(users) != count:
                raise AssertionError(f"subject count changed: {axis}/{phase}")
            for arm in ARMS:
                deltas = []
                values = []
                baselines = []
                for subject in users:
                    current = indexed[(axis, phase, subject, arm)]
                    base = indexed[(axis, phase, subject, "F0v2")]
                    delta = current["macro_f1"] - base["macro_f1"]
                    deltas.append(delta)
                    values.append(current["macro_f1"])
                    baselines.append(base["macro_f1"])
                    cells.append({"axis": axis, "phase": phase, "subject": subject,
                                  "arm": arm, "macro_f1": current["macro_f1"],
                                  "log_loss": current["log_loss"],
                                  "delta_macro_f1": delta,
                                  "delta_log_loss": current["log_loss"] - base["log_loss"]})
                summary.append({"axis": axis, "phase": phase, "arm": arm,
                                "subjects": count,
                                "improved_subjects": sum(v > 1e-10 for v in deltas),
                                "worsened_subjects": sum(v < -1e-10 for v in deltas),
                                "unchanged_subjects": sum(abs(v) <= 1e-10 for v in deltas),
                                "mean_subject_delta_macro_f1": float(np.mean(deltas)),
                                "minimum_subject_macro_f1": min(values),
                                "baseline_minimum_subject_macro_f1": min(baselines)})
    if len(cells) != 310 or len(summary) != 70:
        raise AssertionError("cross-axis subject coverage changed")
    cells_path = ROOT / f"{PREFIX}_CELLS.csv"
    summary_path = ROOT / f"{PREFIX}_SUMMARY.csv"
    write_csv(cells_path, cells)
    write_csv(summary_path, summary)
    audit = {"status": "ok", "source_sha256": SOURCES,
             "subject_cells": len(cells), "summary_cells": len(summary),
             "subject_cells_sha256": sha256(cells_path),
             "summary_cells_sha256": sha256(summary_path),
             "boundary": "Subjects are paired within each axis and phase. ROAM quality uses the fixed six-coordinate mean per subject. Axis populations and GRAB/ROAM recordings overlap; no cross-dataset inference or own-device population claim."}
    (ROOT / f"{PREFIX}_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps({"subject_cells": len(cells), "summary_cells": len(summary)}), flush=True)
    return audit


if __name__ == "__main__":
    build()
