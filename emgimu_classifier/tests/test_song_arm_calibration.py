"""Guard native trial probability composition and pre-formal IMU provenance."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from benchmarks.new_bank_v2.song_arm_calibration import (
    PROTOCOL, calibration_imu_windows, joint_probabilities,
)
from benchmarks.song_real8_study import ARMS, HANDS, parse_label


class SongArmCalibrationTests(unittest.TestCase):
    def test_joint_probability_marginals_recover_same_hand_and_arm(self):
        hand_classes = np.asarray(sorted(HANDS))
        arm_classes = np.asarray(sorted(ARMS))
        joint_classes = np.asarray(sorted(f"{a}_{h}" for a in ARMS for h in HANDS))
        hand = np.asarray([[.1, .2, .3, .4], [.4, .3, .2, .1]])
        arm = np.asarray([[.1, .1, .1, .1, .1, .2, .3],
                          [.3, .2, .1, .1, .1, .1, .1]])
        joint = joint_probabilities(hand, arm, hand_classes, arm_classes, joint_classes)
        np.testing.assert_allclose(joint.sum(axis=1), 1)
        for index, label in enumerate(hand_classes):
            np.testing.assert_allclose(joint[:, [i for i, name in enumerate(joint_classes)
                                                if parse_label(name)[1] == label]].sum(axis=1),
                                       hand[:, index])
        for index, label in enumerate(arm_classes):
            np.testing.assert_allclose(joint[:, [i for i, name in enumerate(joint_classes)
                                                if parse_label(name)[0] == label]].sum(axis=1),
                                       arm[:, index])

    def test_calibration_loader_rejects_overlap_with_first_formal_trial(self):
        dtype = np.dtype([("trial_id", "i4"), ("label", "S40"), ("valid", "?"),
                          ("completion_status", "S20"), ("trial_start_sample", "i8"),
                          ("stable_start_sample", "i8"), ("stable_end_sample", "i8"),
                          ("trial_end_sample", "i8")])
        blocks = np.zeros(8, dtype=dtype)
        for index, name in enumerate(PROTOCOL["calibration_blocks"]):
            start = 100 + 140 * index
            blocks[index] = (index + 1, name.encode(), True, b"completed",
                             start - 10, start, start + 100, start + 110)
        formal = np.asarray([(b"formal", 1500)],
                            dtype=[("trial_kind", "S16"), ("trial_start_sample", "i8")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.h5"
            def write():
                with h5py.File(path, "w") as handle:
                    imu = handle.create_group("streams/imu")
                    imu.create_dataset("emg_sample_index", data=np.arange(0, 2000, 2))
                    imu.create_dataset("accel", data=np.zeros((1000, 3)))
                    imu.create_dataset("gyro", data=np.zeros((1000, 3)))
                    handle.create_dataset("calibration_blocks", data=blocks)
                    handle.create_dataset("trials", data=formal)
            write()
            windows, audit = calibration_imu_windows(Path(directory), "synthetic")
            self.assertEqual(set(windows), set(PROTOCOL["calibration_blocks"]))
            self.assertTrue(all(item["preformal"] for item in audit.values()))
            blocks[-1]["trial_end_sample"] = 1501
            write()
            with self.assertRaisesRegex(ValueError, "pre-formal"):
                calibration_imu_windows(Path(directory), "synthetic")


if __name__ == "__main__":
    unittest.main()
