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
    spans = [max(round((protocol.prompt_duration(label) + 0.5) * 250),
                 512 if kind == "formal" else 0)
             for label, kind in zip(sequence, engine.trial_kinds)]
    bases = np.cumsum([0, *spans[:-1]])
    emg_samples = sum(spans)
    emg = np.random.default_rng(7).integers(
        -1000, 1000, size=(emg_samples, 8), dtype=np.int32)
    emg_time_ns = np.rint(np.arange(emg_samples) * 1_000_000_000 / 250).astype(np.int64)
    recorder.enqueue_emg(emg, np.arange(emg_samples),
                         np.arange(emg_samples, dtype=np.uint8),
                         emg_time_ns, emg_time_ns)
    imu_samples = round(emg_samples * 112 / 250)
    imu_time_ns = np.rint(np.arange(imu_samples) * 1_000_000_000 / 112).astype(np.int64)
    imu_emg_indices = np.rint(np.arange(imu_samples) * 250 / 112).astype(np.int64)
    recorder.enqueue_imu(np.ones((imu_samples, 3), np.float32),
                         np.ones((imu_samples, 3), np.float32),
                         imu_time_ns,
                         np.arange(imu_samples, dtype=np.uint8),
                         imu_emg_indices, imu_time_ns)
    for trial_id, (label, kind, block, offset, base_sample, span) in enumerate(zip(
            sequence, engine.trial_kinds, engine.block_indices,
            engine.onset_offsets, bases, spans), 1):
        event_uid = f"P001:S01:trial:{trial_id}:attempt:1"
        prompt_start = int(base_sample) + 25
        prompt_end = prompt_start + round(protocol.prompt_duration(label) * 250)
        stable_start = prompt_start + 75 + (50 if kind == "formal" else 0)
        stable_end = prompt_end - 75
        recorder.enqueue_trial(TrialInfo(
            trial_id, label, 1, 1, int(base_sample), int(base_sample) + 10,
            prompt_start, prompt_end, int(base_sample) + span - 1, True,
            relative_onset_offset_ms=offset, trial_kind=kind, block_index=block,
            event_uid=event_uid,
            stable_start_sample=stable_start,
            stable_end_sample=stable_end,
            completion_status="completed"))
        if kind == "formal":
            arm, hand = label.split("_", 1)
            cue_sample = prompt_start + 25
            base_ns = 1_000_000_000 + round(int(base_sample) * 1_000_000_000 / 250)
            arm_sample = cue_sample + (50 if offset < 0 else 0)
            hand_sample = cue_sample + (50 if offset > 0 else 0)
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
                    "hand:release", prompt_end, prompt_end / 250, trial_id, 1,
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
    with h5py.File(path, "r+") as handle:
        trials = handle["trials"]
        formal_indices = [index for index, row in enumerate(trials[:])
                          if row["trial_kind"] == b"formal"]
        first, second = formal_indices[:2]
        original_second = trials[second]
        duplicated = trials[second]
        duplicated["stable_start_sample"] = trials[first]["stable_start_sample"]
        duplicated["stable_end_sample"] = trials[first]["stable_end_sample"]
        trials[second] = duplicated
    invalid = generate_session_readiness(path)
    assert invalid["status"] == "failed"
    assert invalid["checks"]["stable_interval_data"]["overlapping_trial_ids"]
    with h5py.File(path, "r+") as handle:
        trials = handle["trials"]
        trials[second] = original_second
        stable_start = int(trials[first]["stable_start_sample"])
        stable_end = int(trials[first]["stable_end_sample"])
        first_trial_id = int(trials[first]["trial_id"])
        handle["streams/emg/raw"][stable_start:stable_end, 0] = 0
    invalid_signal = generate_session_readiness(path)
    assert invalid_signal["status"] == "failed"
    assert str(first_trial_id) in invalid_signal["checks"][
        "stable_interval_data"]["signal_quality_failures"]
    with h5py.File(path, "r+") as handle:
        handle["streams/emg/raw"][stable_start:stable_end, 0] = emg[
            stable_start:stable_end, 0]
        trials = handle["trials"]
        original_first = trials[first]
        shortened = trials[first]
        shortened["prompt_end_sample"] = shortened["prompt_start_sample"] + 20
        trials[first] = shortened
    invalid_duration = generate_session_readiness(path)
    assert invalid_duration["status"] == "failed"
    assert first_trial_id in invalid_duration["checks"]["protocol_duration"][
        "short_prompt_trial_ids"]
    with h5py.File(path, "r+") as handle:
        trials = handle["trials"]
        trials[first] = original_first
        calibration_index = next(index for index, row in enumerate(trials[:])
                                 if row["trial_kind"] == b"calibration")
        original_calibration = trials[calibration_index]
        shortened_calibration = trials[calibration_index]
        calibration_trial_id = int(shortened_calibration["trial_id"])
        shortened_calibration["stable_end_sample"] = (
            shortened_calibration["stable_start_sample"] + 20)
        trials[calibration_index] = shortened_calibration
    invalid_calibration = generate_session_readiness(path)
    assert invalid_calibration["status"] == "failed"
    assert calibration_trial_id in invalid_calibration["checks"]["protocol_duration"][
        "short_stable_trial_ids"]
    with h5py.File(path, "r+") as handle:
        handle["trials"][calibration_index] = original_calibration
        imu_mapping = handle["streams/imu/emg_sample_index"]
        mapped = imu_mapping[:]
        mapped[(mapped >= stable_start) & (mapped < stable_end)] = stable_end
        imu_mapping[:] = mapped
    invalid_imu = generate_session_readiness(path)
    assert invalid_imu["status"] == "failed"
    assert first_trial_id in invalid_imu["checks"]["imu_coverage"]["missing_trial_ids"]
    with h5py.File(path, "r+") as handle:
        handle["streams/imu/emg_sample_index"][:] = imu_emg_indices
        first_imu = int(np.searchsorted(imu_emg_indices, stable_start))
        handle["streams/imu/gyro"][first_imu, 0] = np.nan
    invalid_imu_value = generate_session_readiness(path)
    assert invalid_imu_value["status"] == "failed"
    assert first_trial_id in invalid_imu_value["checks"]["imu_coverage"][
        "nonfinite_trial_ids"]
