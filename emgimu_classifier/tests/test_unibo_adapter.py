from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from emgimu.cli import build_parser
from emgimu.datasets import BenchmarkDatasetError, check_benchmark_dataset, load_benchmark_trial
from emgimu.datasets.adapters.unibo_inail import UniBoInailAdapter
from emgimu.signal import polyphase_resample
from emgimu.state import Gesture


def write_source_mat(
    path: Path,
    *,
    channels: int = 4,
    missing: str | None = None,
    nan_emg: bool = False,
    invalid_label: bool = False,
) -> None:
    rng = np.random.default_rng(42)
    emg_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    relabel_blocks: list[np.ndarray] = []
    counter_blocks: list[np.ndarray] = []

    def append_block(gesture: int, counter: int, length: int = 100) -> None:
        label = np.ones(length, dtype=np.float32)
        if gesture != 1:
            label[:60] = gesture
        relabel = label.copy()
        if gesture != 1:
            relabel[:10] = 1  # official relabel removes an onset boundary
        signal = rng.normal(0, 0.05, size=(length, channels)).astype(np.float32)
        if gesture != 1:
            signal[:60] += gesture * 0.1
        emg_blocks.append(signal)
        label_blocks.append(label)
        relabel_blocks.append(relabel)
        counter_blocks.append(np.full(length, counter, dtype=np.float32))

    append_block(1, 0, 50)
    # Two repetitions make the counter reset (2 -> 1) visible at gesture boundaries.
    for gesture in (6, 5, 4, 3, 2):
        append_block(gesture, 1)
        append_block(gesture, 2)
    values = {
        "emg": np.concatenate(emg_blocks),
        "label": np.concatenate(label_blocks)[:, None],
        "relabel": np.concatenate(relabel_blocks)[:, None],
        "gestureCounter": np.concatenate(counter_blocks)[:, None],
    }
    if nan_emg:
        values["emg"][0, 0] = np.nan
    if invalid_label:
        values["label"][60, 0] = 7
    if missing is not None:
        values.pop(missing)
    savemat(path, values)


class UniBoAdapterTests(unittest.TestCase):
    def test_adapts_native_four_channels_and_keeps_unmapped_trials_out(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            write_source_mat(source / "user1_day1_posture1.mat")
            output = root / "converted"

            result = UniBoInailAdapter().adapt(source, output)

            self.assertEqual(result.trial_count, 11)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "label_map.json").is_file())
            self.assertTrue((output / "splits.json").is_file())
            self.assertTrue((output / "reports" / "CAPABILITIES.md").is_file())
            self.assertEqual((output / ".gitignore").read_text(encoding="utf-8"), "trials/\n")
            check = check_benchmark_dataset(output)
            self.assertEqual(check["status"], "ok")

            paths = sorted((output / "trials").rglob("*.npz"))
            open_trial = load_benchmark_trial(next(path for path in paths if "g06-r01" in path.name))
            self.assertEqual(open_trial.emg.shape[1], 4)
            self.assertIsNone(open_trial.imu)
            self.assertEqual(open_trial.subject_id, "u01")
            self.assertEqual(open_trial.session_id, "d01")
            self.assertEqual(open_trial.posture_label, 1)
            self.assertTrue(open_trial.benchmark_eligible)
            self.assertIn(int(Gesture.OPEN), np.unique(open_trial.hand_label))
            changed = open_trial.source_label != open_trial.source_relabel
            self.assertTrue(np.any(changed))
            self.assertFalse(np.any(open_trial.stable_mask[changed]))

            pointing = load_benchmark_trial(next(path for path in paths if "g05-r01" in path.name))
            self.assertFalse(pointing.benchmark_eligible)
            self.assertFalse(np.any(pointing.stable_mask))
            self.assertNotIn("imu", np.load(pointing.path, allow_pickle=False).files)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["emg_channels"], 4)
            self.assertEqual(manifest["source_sample_rate_hz"], 500.0)
            self.assertEqual(manifest["target_sample_rate_hz"], 200.0)
            self.assertEqual(manifest["channel_layout"], "named_muscles_non_circular")
            self.assertFalse(manifest["has_imu"])
            statistics = json.loads(
                (output / "reports" / "statistics.json").read_text(encoding="utf-8")
            )
            self.assertGreater(statistics["unmapped_sample_fraction"], 0.0)
            self.assertIn("OPEN", statistics["stable_duration_seconds_by_hand_label"])

    def test_fixed_subject_day_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            write_source_mat(source / "user_1_day_1_posture_1.mat")
            output = root / "converted"
            UniBoInailAdapter().adapt(source, output)
            splits = json.loads((output / "splits.json").read_text(encoding="utf-8"))
            groups = splits["groups"]
            self.assertIn("u01/d01", groups["train"])
            self.assertIn("u01/d06", groups["validation"])
            self.assertIn("u01/d07", groups["test"])
            self.assertIn("u01/d08", groups["test"])
            all_groups = groups["train"] + groups["validation"] + groups["test"]
            self.assertEqual(len(all_groups), len(set(all_groups)))

    def test_invalid_source_is_atomic_and_not_silently_skipped(self):
        for problem in ("bad_channels", "missing_label", "nan_emg", "invalid_label"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source"
                source.mkdir()
                options = {
                    "bad_channels": {"channels": 8},
                    "missing_label": {"missing": "label"},
                    "nan_emg": {"nan_emg": True},
                    "invalid_label": {"invalid_label": True},
                }[problem]
                write_source_mat(source / "user_1_day_1_posture_1.mat", **options)
                output = root / "converted"
                with self.assertRaises(BenchmarkDatasetError):
                    UniBoInailAdapter().adapt(source, output)
                self.assertFalse(output.exists())

    def test_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            write_source_mat(source / "user_1_day_1_posture_1.mat")
            output = root / "converted"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("user data", encoding="utf-8")
            with self.assertRaises(BenchmarkDatasetError):
                UniBoInailAdapter().adapt(source, output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "user data")

    def test_polyphase_downsampling_suppresses_above_target_nyquist(self):
        timestamp = np.arange(2500) / 500.0
        low = np.sin(2 * np.pi * 30 * timestamp)
        high = np.sin(2 * np.pi * 150 * timestamp)
        result = polyphase_resample(np.stack([low, high], axis=1), 500, 200)
        low_rms = float(np.sqrt(np.mean(result[:, 0] ** 2)))
        high_rms = float(np.sqrt(np.mean(result[:, 1] ** 2)))
        self.assertGreater(low_rms, 0.6)
        self.assertLess(high_rms, low_rms * 0.05)

    def test_checker_rejects_duplicate_ids_and_split_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            write_source_mat(source / "user1_day1_posture1.mat")
            output = root / "converted"
            UniBoInailAdapter().adapt(source, output)

            original = next((output / "trials").rglob("*.npz"))
            duplicate = original.with_name("duplicate.npz")
            duplicate.write_bytes(original.read_bytes())
            duplicate_report = check_benchmark_dataset(output)
            self.assertEqual(duplicate_report["status"], "error")
            self.assertTrue(any("duplicate trial_id" in row for row in duplicate_report["errors"]))
            duplicate.unlink()

            splits_path = output / "splits.json"
            splits = json.loads(splits_path.read_text(encoding="utf-8"))
            splits["groups"]["validation"].append("u01/d01")
            splits_path.write_text(json.dumps(splits), encoding="utf-8")
            leakage_report = check_benchmark_dataset(output)
            self.assertEqual(leakage_report["status"], "error")
            self.assertTrue(any("multiple splits" in row for row in leakage_report["errors"]))

    def test_cli_exposes_three_benchmark_commands(self):
        parser = build_parser()
        adapt = parser.parse_args([
            "benchmark-adapt", "unibo-inail", "source", "--output", "output",
        ])
        check = parser.parse_args(["benchmark-check", "output"])
        report = parser.parse_args(["benchmark-report", "output"])
        self.assertEqual(adapt.adapter, "unibo-inail")
        self.assertEqual(check.dataset, "output")
        self.assertEqual(report.dataset, "output")


@unittest.skipUnless(os.environ.get("UNIBO_INAIL_SAMPLE"), "set UNIBO_INAIL_SAMPLE to an official MAT")
class UniBoOfficialSampleSmokeTest(unittest.TestCase):
    def test_official_sample_read_only_conversion(self):
        source_file = Path(os.environ["UNIBO_INAIL_SAMPLE"])
        self.assertTrue(source_file.is_file())
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            target = source / source_file.name
            target.write_bytes(source_file.read_bytes())
            output = Path(directory) / "converted"
            result = UniBoInailAdapter().adapt(source, output)
            self.assertGreater(result.trial_count, 0)
            self.assertEqual(check_benchmark_dataset(output)["status"], "ok")


if __name__ == "__main__":
    unittest.main()
