"""Replay the exported opt-in signed bundle on the frozen whole-stream grid."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

from benchmarks.new_bank_v2.export_song_arm_signed_bundle import OUTPUT, ROOT
from benchmarks.new_bank_v2.song_joint28_continuous_replay import RUNTIME_FILE, SOURCE


FRAME_FILE = ROOT / "SONG_ARM_SIGNED_CONTINUOUS_FRAMES.csv"
RESULT_FILE = ROOT / "SONG_ARM_SIGNED_CONTINUOUS_RESULTS.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _classes():
    spec = importlib.util.spec_from_file_location("song_signed_export_runtime", RUNTIME_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SongJoint28WindowRuntime, module.SongJoint28Stream


def run(source: Path = SOURCE, bundle: Path = OUTPUT) -> dict:
    result = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    manifest_path = bundle / "song_joint28_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["reference_continuous_results_sha256"] != _sha(RESULT_FILE)
            or _sha(FRAME_FILE) != result["frame_csv_sha256"]):
        raise ValueError("bundle is not bound to saved continuous candidate")
    Window, Stream = _classes()
    model = Window(bundle)
    if model.arm_feature_kind != "f6_signed_19":
        raise ValueError("exported model is not signed candidate")
    with FRAME_FILE.open(newline="", encoding="utf-8") as handle:
        saved = {(row["session"], int(row["emg_end_index"])): row for row in csv.DictReader(handle)}
    if len(saved) != result["frame_rows"]:
        raise ValueError("saved frame identities are not unique")
    maximum_joint_error = maximum_peak_error = 0.0
    checked = set()
    sessions = {}
    for sid in ("S03", "S04"):
        path = source / f"2026-09-18_{sid}" / "session.h5"
        if _sha(path) != result["source_hdf5_sha256"][sid]:
            raise ValueError(f"native recording changed: {sid}")
        with h5py.File(path) as handle:
            raw = handle["streams/emg/raw"][:]
            emg_indices = handle["streams/emg/sample_index"][:]
            imu_indices = handle["streams/imu/emg_sample_index"][:]
            accel = handle["streams/imu/accel"][:]
            gyro = handle["streams/imu/gyro"][:]
        if not np.array_equal(emg_indices, np.arange(len(raw))) or np.any(np.diff(imu_indices) < 0):
            raise ValueError(f"native stream indices changed: {sid}")
        stream = Stream(model)
        emitted = []
        imu_offset = 0
        for left in range(0, len(raw), 25):
            right = min(left + 25, len(raw))
            gap, frames = stream.ingest_emg(raw[left:right], emg_indices[left:right])
            if gap:
                raise ValueError(f"unexpected EMG gap: {sid}")
            emitted.extend(frames)
            last = int(np.searchsorted(imu_indices, right, side="right"))
            emitted.extend(stream.ingest_imu(accel[imu_offset:last], gyro[imu_offset:last],
                                             imu_indices[imu_offset:last]))
            imu_offset = last
        if imu_offset < len(imu_indices):
            emitted.extend(stream.ingest_imu(accel[imu_offset:], gyro[imu_offset:],
                                             imu_indices[imu_offset:]))
        if (len(emitted) != result["sessions"][sid]["emitted_frames"]
                or stream.dropped_frames != 0):
            raise ValueError(f"exported model frame population differs: {sid}")
        for end, probability in emitted:
            key = (sid, int(end))
            row = saved.get(key)
            if row is None or key in checked:
                raise ValueError(f"missing or duplicated exported frame: {key}")
            checked.add(key)
            peak = int(np.argmax(probability))
            if model.joint_classes[peak] != row["peak_label"]:
                raise ValueError(f"exported hard label differs: {key}")
            maximum_peak_error = max(maximum_peak_error,
                                     abs(float(probability[peak]) - float(row["peak_probability"])))
            if row["interval_kind"] == "formal_stable":
                reference = np.asarray(json.loads(row["joint_probabilities"]))
                maximum_joint_error = max(maximum_joint_error,
                                          float(np.max(np.abs(probability - reference))))
        sessions[sid] = {"frames": len(emitted), "dropped": stream.dropped_frames}
        print(f"{sid}: exported candidate replay {len(emitted)} native frames", flush=True)
    if checked != set(saved) or maximum_peak_error > 1e-6 or maximum_joint_error > 1e-6:
        raise ValueError(f"export replay differs: {len(saved) - len(checked)} missing, "
                         f"peak={maximum_peak_error}, joint={maximum_joint_error}")
    audit = {"status": "exported_candidate_continuous_replay_verified",
             "model_sha256": model.sha256,
             "manifest_sha256": _sha(manifest_path),
             "runtime_source_sha256": _sha(RUNTIME_FILE),
             "continuous_results_sha256": _sha(RESULT_FILE),
             "continuous_frame_csv_sha256": _sha(FRAME_FILE),
             "matched_frames": len(checked), "maximum_peak_probability_error": maximum_peak_error,
             "maximum_formal_joint_probability_error": maximum_joint_error,
             "sessions": sessions,
             "boundary": "Exported software stream reproduces frozen replay; physical hardware and new-day/person use remain unverified."}
    (bundle / "song_joint28_replay_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
