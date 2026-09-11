from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from emgforce.collection_protocol import FORMAL_PROTOCOL_NAME, QUALITY_THRESHOLD_VERSION
from emgforce.experiment.events import EventType
from emgforce.experiment.models import CueEvent, ExperimentEvent, TrialInfo
from emgforce.experiment.prompt_engine import PromptEngine
from emgforce.experiment.protocol_loader import ProtocolLoader
from emgforce.quality.session_readiness import generate_session_readiness
from emgforce.storage.hdf5_recorder import Hdf5Recorder
from emgforce.transfer.dataset_upload import build_upload_plan


def _formal_protocol():
    root = Path(__file__).parents[1]
    return ProtocolLoader(root / "protocols").load("jilv_music_28")


def test_formal_protocol_has_calibration_blocks_constrained_blocks_and_balanced_offsets() -> None:
    protocol = _formal_protocol()
    engine = PromptEngine(seed=42)
    sequence = engine.prepare(protocol)
    calibration_count = len(protocol.calibration_blocks)

    assert protocol.name == FORMAL_PROTOCOL_NAME
    assert protocol.formal_collection is True
    assert sum(float(block["duration_sec"]) for block in protocol.calibration_blocks) == 64.0
    assert len(sequence) == calibration_count + 144
    assert all(kind == "calibration" for kind in engine.trial_kinds[:calibration_count])
    assert all(kind == "formal" for kind in engine.trial_kinds[calibration_count:])

    counts = Counter(sequence[calibration_count:])
    assert all(counts[label] == protocol.trial_count(label) for label in protocol.labels)
    per_label_offsets: dict[str, Counter[int]] = defaultdict(Counter)
    for label, offset in zip(sequence[calibration_count:], engine.onset_offsets[calibration_count:]):
        per_label_offsets[label][offset] += 1
    assert all(set(per_label_offsets[label]) == {-200, 0, 200} for label in protocol.labels)
    assert all(max(counter.values()) - min(counter.values()) <= 1
               for counter in per_label_offsets.values())

    for block in range(1, 13):
        labels = [label for label, index in zip(sequence, engine.block_indices) if index == block]
        assert len(labels) == 12
        assert sum(label.startswith("still_") for label in labels) == 6
        assert sum(not label.startswith("still_") for label in labels) == 6


def test_complete_formal_hdf5_passes_collection_readiness(tmp_path: Path) -> None:
    protocol = _formal_protocol()
    engine = PromptEngine(seed=7)
    sequence = engine.prepare(protocol)
    path = tmp_path / "session.h5"
    metadata = {
        "participant_id": "P001", "session_id": "S01", "dominant_hand": "right",
        "recorded_arm": "right", "donning_code": "D01", "donning_id": "D01",
        "channel1_orientation": "thumb_side", "channel_1_orientation": "thumb_side",
        "anatomical_distance_mm": 60.0,
        "strap_scale": "3", "strap_tightness": 2, "skin_condition": "dry",
        "fatigue_before": 0, "fatigue_after": 2, "reference_photo_name": "ref.png",
        "reference_photo_sha256": "a" * 64, "operator_id": "OP01",
        "protocol_name": protocol.name, "protocol_version": protocol.name,
        "protocol_file_sha256": protocol.source_sha256,
        "protocol_json": json.dumps(protocol.to_dict(), ensure_ascii=False),
        "dataset_split": "train", "sampling_rate": 250,
        "emg_nominal_rate_hz": 250, "imu_nominal_rate_hz": 112,
        "measured_emg_rate_hz": 250.0, "measured_imu_rate_hz": 112.0,
        "num_emg_channels": 8, "timestamp_source": "pc_reconstructed",
        "model_frozen_confirmed": True,
        "quality_report_json": json.dumps({
            "passed": True, "threshold_version": QUALITY_THRESHOLD_VERSION,
        }),
    }
    recorder = Hdf5Recorder(batch_samples=100)
    recorder.start(path, metadata)
    recorder.enqueue_emg(np.ones((500, 8), np.int32), np.arange(500),
                         np.arange(500, dtype=np.uint8))
    recorder.enqueue_imu(np.ones((224, 3), np.float32), np.ones((224, 3), np.float32),
                         np.arange(224, dtype=np.int64), np.arange(224, dtype=np.uint8),
                         np.arange(224, dtype=np.int64))
    for trial_id, (label, kind, block, offset) in enumerate(zip(
            sequence, engine.trial_kinds, engine.block_indices, engine.onset_offsets), 1):
        event_uid = f"P001:S01:trial:{trial_id}:attempt:1"
        recorder.enqueue_trial(TrialInfo(
            trial_id, label, 1, 1, 0, 0, 10, 20, 30, True,
            relative_onset_offset_ms=offset, trial_kind=kind, block_index=block,
            event_uid=event_uid,
            stable_start_sample=12, stable_end_sample=18,
            completion_status="completed"))
        if kind == "formal":
            arm, hand = label.split("_", 1)
            base_sample = 100
            base_ns = 1_000_000_000 + trial_id * 10_000_000
            arm_sample = base_sample + (50 if offset < 0 else 0)
            hand_sample = base_sample + (50 if offset > 0 else 0)
            arm_ns = base_ns + (200_000_000 if offset < 0 else 0)
            hand_ns = base_ns + (200_000_000 if offset > 0 else 0)
            recorder.enqueue_cue_event(CueEvent(
                f"arm:{arm}", arm_sample, arm_sample / 250, trial_id, 1,
                arm_ns, arm_ns, event_uid))
            recorder.enqueue_cue_event(CueEvent(
                f"hand:{hand}", hand_sample, hand_sample / 250, trial_id, 1,
                hand_ns, hand_ns, event_uid))
            if label.endswith("_open_hand"):
                recorder.enqueue_cue_event(CueEvent(
                    "hand:release", 200, 0.8, trial_id, 1,
                    base_ns + 500_000_000, base_ns + 500_000_000, event_uid))
    event_id = 0
    for block in range(1, 12):
        for event_type in (EventType.BLOCK_BREAK_START, EventType.FATIGUE_REPORT,
                           EventType.BLOCK_BREAK_END):
            recorder.enqueue_event(ExperimentEvent(
                event_id, 0, 0.0, event_id + 1, event_type,
                note=json.dumps({"block": block, "score": 0})))
            event_id += 1
    recorder.stop()

    manifest = generate_session_readiness(path)
    assert manifest["status"] == "passed", manifest["problems"]
    assert (tmp_path / "SESSION_COLLECTION_READINESS.json").exists()
    with h5py.File(path, "r") as handle:
        assert len(handle["calibration_blocks"]) == len(protocol.calibration_blocks)
        assert handle["meta"].attrs["timestamp_source"] == "pc_reconstructed"
    plan = build_upload_plan(tmp_path)
    assert plan.sessions == 1
    assert plan.prompts == 144
    assert {item.relative_path for item in plan.items} == {
        "session.h5", "SESSION_COLLECTION_READINESS.json",
        "formal_collection_manifest.json",
    }
