from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from emgimu.baseline import HeadPrediction
from emgimu.calibration import SessionCalibration
from emgimu.cli import build_parser
from emgimu.data import DatasetError, build_feature_windows, dataset_report, discover_trials
from emgimu.hdf5_v3 import adapt_hdf5_v3, inspect_hdf5_v3
from emgimu.osc import OSC_V2_ADDRESS, decode_message, encode_message
from emgimu.runtime import HumanStateEstimator
from emgimu.state import Direction, Gesture, QualityFlag


ARMS = ("still", "up", "down", "left", "right", "forward", "backward")
HANDS = ("neutral", "index_pinch", "fist", "open_hand")


class FixedPredictor:
    def predict(self, normalized_emg, normalized_imu):
        return HeadPrediction(Direction.RIGHT, Gesture.FIST, 0.95, 0.92)


def write_hdf5_v3(path: Path, *, gate_pass: bool = True, schema: str = "3.0") -> None:
    rng = np.random.default_rng(19)
    emg_rate = 250
    imu_rate = 112
    samples_per_trial = 500
    labels = [f"{arm}_{hand}" for arm in ARMS for hand in HANDS]
    emg_count = len(labels) * samples_per_trial
    duration = emg_count / emg_rate
    imu_count = int(np.floor(duration * imu_rate))
    origin = 1_000_000_000_000
    emg_time = origin + np.rint(np.arange(emg_count) * 1e9 / emg_rate).astype(np.int64)
    imu_time = origin + np.rint(np.arange(imu_count) * 1e9 / imu_rate).astype(np.int64)
    carrier = 180 * np.sin(2 * np.pi * 35 * np.arange(emg_count) / emg_rate)
    emg = rng.normal(0, 25, size=(emg_count, 8)) + carrier[:, None]
    accel = rng.normal(0, 0.03, size=(imu_count, 3))
    accel[:, 2] += 9.81
    gyro = rng.normal(0, 0.02, size=(imu_count, 3))

    string = h5py.string_dtype("utf-8")
    trial_dtype = np.dtype([
        ("trial_id", "i8"), ("label", string), ("stage_id", "i8"),
        ("donning_id", "i8"), ("trial_start_sample", "i8"),
        ("rest_start_sample", "i8"), ("prompt_start_sample", "i8"),
        ("prompt_end_sample", "i8"), ("trial_end_sample", "i8"),
        ("valid", "?"), ("reject_reason", string), ("note", string),
        ("relative_onset_offset_ms", "i4"),
    ])
    cue_dtype = np.dtype([
        ("name", string), ("sample_index", "i8"), ("time_sec", "f8"),
        ("trial_id", "i8"), ("stage_id", "i8"),
        ("scheduled_monotonic_ns", "i8"), ("emitted_monotonic_ns", "i8"),
    ])
    audit_dtype = np.dtype([
        ("packet_type", "u1"), ("packet_seq", "u1"),
        ("pc_received_ns", "i8"), ("lost_before", "i4"),
        ("duplicate", "?"), ("out_of_order", "?"),
    ])
    trial_rows = []
    cue_rows = []
    for index, label in enumerate(labels, 1):
        start = (index - 1) * samples_per_trial
        prompt_start = start + 100
        prompt_end = start + 400
        end = start + samples_per_trial
        offset = (-200, 0, 200)[(index - 1) % 3]
        arm, hand = next((arm, label[len(arm) + 1:]) for arm in ARMS if label.startswith(arm + "_"))
        if offset < 0:
            hand_cue, arm_cue = prompt_start, prompt_start + round(abs(offset) * emg_rate / 1000)
        else:
            arm_cue, hand_cue = prompt_start, prompt_start + round(offset * emg_rate / 1000)
        trial_rows.append((
            index, label, 1, 1, start, start, prompt_start, prompt_end, end,
            True, "", "", offset,
        ))
        cue_rows.extend([
            (f"arm:{arm}", arm_cue, arm_cue / emg_rate, index, 1, -1, -1),
            (f"hand:{hand}", hand_cue, hand_cue / emg_rate, index, 1, -1, -1),
        ])
    loss_time = int(emg_time[650])
    audit = np.asarray([
        (1, 10, loss_time, 2, False, False),
        (2, 11, loss_time, 1, False, False),
    ], dtype=audit_dtype)
    quality = {
        "passed": gate_pass,
        "grade": "GOOD" if gate_pass else "BAD",
        "reasons": [] if gate_pass else ["50hz_interference"],
        "saturation_ratio": [0.0] * 8,
        "zero_ratio": [0.0] * 8,
        "mains_50hz_ratio": ([0.01] * 8 if gate_pass else [0.5] * 8),
    }
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("meta")
        meta.attrs.update({
            "schema_version": schema,
            "participant_id": "Dong",
            "session_id": "S01",
            "sampling_rate": emg_rate,
            "emg_nominal_rate_hz": emg_rate,
            "imu_nominal_rate_hz": imu_rate,
            "num_emg_channels": 8,
            "protocol_name": "jilv_music_28_v1",
            "start_datetime": "2026-09-10T09:00:00+08:00",
            "dataset_split": "train",
            "quality_report_json": json.dumps(quality),
            "tested_arm": "right",
            "channel1_orientation": "ulnar_marker",
            "anatomical_marker": "photo-01",
            "strap_setting": "3",
            "stabilization_sec": 60,
            "physical_condition": "normal",
            "measured_emg_rate_hz": 249.7,
            "measured_imu_rate_hz": 111.8,
            "lost_frames": 3,
            "duplicate_frames": 0,
            "out_of_order_frames": 0,
        })
        streams = handle.create_group("streams")
        emg_group = streams.create_group("emg")
        emg_group.create_dataset("raw", data=np.rint(emg).astype(np.int32))
        emg_group.create_dataset("sample_index", data=np.arange(emg_count, dtype=np.int64))
        emg_group.create_dataset("packet_seq", data=np.arange(emg_count, dtype=np.uint8))
        emg_group.create_dataset("pc_received_ns", data=emg_time)
        emg_group.create_dataset("sample_time_ns", data=emg_time)
        imu_group = streams.create_group("imu")
        imu_group.create_dataset("accel", data=accel.astype(np.float32))
        imu_group.create_dataset("gyro", data=gyro.astype(np.float32))
        imu_group.create_dataset("packet_seq", data=np.arange(imu_count, dtype=np.uint8))
        imu_group.create_dataset("pc_monotonic_ns", data=imu_time)
        imu_group.create_dataset("sample_time_ns", data=imu_time)
        imu_group.create_dataset(
            "emg_sample_index", data=np.rint(np.arange(imu_count) * emg_rate / imu_rate).astype(np.int64),
        )
        handle.create_dataset("trials", data=np.asarray(trial_rows, dtype=trial_dtype))
        cues = handle.create_dataset("cue_events", data=np.asarray(cue_rows, dtype=cue_dtype))
        cues.attrs["semantics"] = "ui_action_cue_not_movement_onset"
        handle.create_dataset("packet_audit", data=audit)


class Hdf5V3Tests(unittest.TestCase):
    def test_adapts_dual_timeline_all_28_labels_and_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "session.h5"
            write_hdf5_v3(source)
            check = inspect_hdf5_v3(source)
            self.assertEqual(check["status"], "ok")
            self.assertEqual(check["valid_trials"], 28)

            output = root / "dataset"
            result = adapt_hdf5_v3(source, output)
            self.assertEqual(result.trial_count, 28)
            trials = discover_trials(output)
            self.assertEqual(len(trials), 28)
            pairs = {
                (int(trial.direction[trial.stable_mask][0]), int(trial.gesture[trial.stable_mask][0]))
                for trial in trials if np.any(trial.stable_mask)
            }
            self.assertEqual(len(pairs), 28)
            open_trial = next(trial for trial in trials if trial.metadata["source_label"] == "up_open_hand")
            self.assertIn(int(Gesture.OPEN), open_trial.gesture)
            self.assertNotEqual(int(Gesture.OPEN), int(Gesture.NEUTRAL))
            self.assertTrue(np.any(open_trial.interpolated_imu))
            self.assertEqual(open_trial.channel_quality.shape, (8,))
            self.assertTrue(open_trial.quality_gate_pass)
            self.assertEqual(open_trial.metadata["wearing"]["strap_setting"], "3")
            with np.load(open_trial.path, allow_pickle=False) as raw:
                self.assertNotEqual(
                    len(raw["source_emg_timestamp_ms"]), len(raw["source_imu_timestamp_ms"]),
                )
                self.assertIn("arm_phase_label", raw.files)
                self.assertIn("hand_phase_label", raw.files)
            self.assertTrue(any(np.any(trial.missing_mask) for trial in trials))

            windows = build_feature_windows(
                trials, {"1": SessionCalibration.identity()},
            )
            self.assertGreater(len(windows.emg), 0)
            audit = dataset_report(trials)["quality_by_session"]["1"]
            self.assertTrue(audit["quality_gate_pass"])
            self.assertEqual(audit["minimum_valid_emg_channels"], 8)
            self.assertGreater(audit["interpolated_imu_ratio"], 0.0)
            report = json.loads((output / "HDF5_V3_IMPORT.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["sessions"][0]["source_sha256"]), 64)
            self.assertEqual(report["calibration_status"], "separate guided 60-second calibration still required")

    def test_end_to_end_trial_quality_to_human_state_and_osc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "session.h5"
            write_hdf5_v3(source)
            output = root / "dataset"
            adapt_hdf5_v3(source, output)
            trial = next(
                row for row in discover_trials(output)
                if row.metadata["source_label"] == "right_fist"
            )
            estimator = HumanStateEstimator(FixedPredictor(), SessionCalibration.identity())
            states = estimator.push_batch(
                trial.timestamp_ms, trial.emg, trial.accel, trial.gyro,
                window_quality=trial.window_quality,
                missing_mask=trial.missing_mask,
                channel_quality=trial.channel_quality,
                timestamp_valid=trial.timestamp_valid,
                imu_valid=trial.imu_valid,
                interpolated_imu=trial.interpolated_imu,
                quality_gate_pass=trial.quality_gate_pass,
            )
            self.assertTrue(states)
            self.assertTrue(any(state.gesture == Gesture.FIST for state in states))
            self.assertTrue(any(
                state.signal_quality.flags & QualityFlag.INTERPOLATED_IMU for state in states
            ))
            packet = encode_message(OSC_V2_ADDRESS, *states[-1].osc_v2_args())
            address, values = decode_message(packet)
            self.assertEqual(address, OSC_V2_ADDRESS)
            self.assertEqual(int(values[2]), int(states[-1].gesture))

    def test_failed_quality_gate_is_retained_but_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "session.h5"
            write_hdf5_v3(source, gate_pass=False)
            output = root / "dataset"
            adapt_hdf5_v3(source, output)
            trials = discover_trials(output)
            self.assertTrue(all(not trial.quality_gate_pass for trial in trials))
            self.assertTrue(all(not np.any(trial.stable_mask) for trial in trials))
            with self.assertRaises(DatasetError):
                build_feature_windows(trials, {"1": SessionCalibration.identity()})

            trial = trials[0]
            estimator = HumanStateEstimator(FixedPredictor(), SessionCalibration.identity())
            states = estimator.push_batch(
                trial.timestamp_ms, trial.emg, trial.accel, trial.gyro,
                quality_gate_pass=False,
            )
            self.assertTrue(states)
            self.assertTrue(all(state.direction == Direction.UNKNOWN for state in states))
            self.assertTrue(all(state.gesture == Gesture.UNKNOWN for state in states))
            self.assertTrue(all(
                state.signal_quality.flags & QualityFlag.QUALITY_GATE_FAILED for state in states
            ))

    def test_rejects_wrong_schema_and_cli_exposes_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.h5"
            write_hdf5_v3(path, schema="2.1")
            self.assertEqual(inspect_hdf5_v3(path)["status"], "error")
        parser = build_parser()
        self.assertEqual(parser.parse_args(["hdf5-v3-check", "x"]).source, "x")
        self.assertEqual(
            parser.parse_args(["hdf5-v3-adapt", "x", "--output", "y"]).output, "y",
        )


if __name__ == "__main__":
    unittest.main()
