"""Recheck author ROAM archive integrity and the exact static-study input."""
from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

from benchmarks.new_bank_v2.roam_posture_run import extract_archive, sha256

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path("D:/emg-imu-benchmarks/data/raw/roam_emg/data.zip")
EXPECTED_SHA256 = "c9de0e25c187c216e4771d17ccfa19acb0e4028f907ea726b2a87a18d2db5343"


def audit() -> dict:
    if sha256(ARCHIVE) != EXPECTED_SHA256:
        raise ValueError("author ROAM archive SHA-256 mismatch")
    with zipfile.ZipFile(ARCHIVE) as z:
        infos = z.infolist()
        if len(infos) != 187515 or z.testzip() is not None:
            raise ValueError("author ROAM ZIP inventory or CRC mismatch")
        native = [item.filename for item in infos if item.filename.startswith("data/ROAM_EMG/")
                  and item.filename.endswith(".csv")]
    if len(native) != 252:
        raise ValueError("ROAM CSV inventory differs")
    batch, _, identities, _, counts = extract_archive(ARCHIVE)
    subjects = sorted({row["subject"] for row in identities})
    postures = sorted({row["condition"] for row in identities})
    if (subjects != list(range(1, 29)) or postures != ["hanging", "reaching", "resting", "unsupported"]
            or len(counts) != 112 or len(identities) != 1008 or batch.windows != 25624
            or batch.channels != 8 or batch.emg.shape[1] != 40):
        raise ValueError("native ROAM static study inventory changed")
    labels = Counter(str(row["label"]) for row in identities)
    if labels != {"0": 448, "1": 336, "2": 224}:
        raise ValueError("native ROAM class counts changed")
    report = {"status": "ok", "source": "https://github.com/roamlab/reactemg",
              "archive_path": str(ARCHIVE), "archive_bytes": ARCHIVE.stat().st_size,
              "archive_sha256": EXPECTED_SHA256, "zip_crc_verified": True,
              "zip_members": len(infos), "roam_csv_files": len(native),
              "static_files_verified": len(counts), "subjects": subjects,
              "postures": postures, "nominal_sample_rate_hz": 200,
              "timing_and_finite_channel_contract": "all 112 static native CSVs passed source reader",
              "channels": batch.channels, "window_samples": batch.emg.shape[1],
              "disjoint_pure_label_windows": batch.windows,
              "native_label_bouts": len(identities), "bout_labels": dict(sorted(labels.items())),
              "min_file_samples": min(item["samples"] for item in counts.values()),
              "max_file_samples": max(item["samples"] for item in counts.values()),
              "min_windows_per_bout": min(row["windows"] for row in identities),
              "max_windows_per_bout": max(row["windows"] for row in identities),
              "scope": "Public Myo eight-channel raw-count CSVs; static labelled sequences and ZIP CRC verified. Archive also holds other datasets. No user's 250 Hz device, real fault or live transition claim."}
    (ROOT / "ROAM_NATIVE_AUDIT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "static_files": len(counts),
                      "subjects": len(subjects), "bouts": len(identities)}))
    return report


if __name__ == "__main__":
    audit()
