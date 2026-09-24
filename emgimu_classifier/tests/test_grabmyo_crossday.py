import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks.grabmyo_crossday.run import CHANNELS, read_record, records, relative_file


class GrabMyoCrossdayTests(unittest.TestCase):
    def test_frozen_trial_identity_and_day_partition(self):
        rows = list(records())
        self.assertEqual(len(rows), 8 * 3 * 4 * 7)
        self.assertEqual({row["session"] for row in rows}, {1, 2, 3})
        self.assertEqual({row["gesture"] for row in rows}, {4, 15, 16, 17})
        self.assertEqual(len({row["stem"] for row in rows}), len(rows))
        for day in (1, 2, 3):
            self.assertEqual(sum(row["session"] == day for row in rows), 224)

    def test_wfdb_adc_scale_and_first_forearm_ring_only(self):
        record = next(records())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            header = root / relative_file(record, "hea")
            signal = root / relative_file(record, "dat")
            header.parent.mkdir(parents=True)
            lines = [f"{record['stem']} 32 2048 10240"]
            for channel, name in enumerate(CHANNELS):
                lines.append(f"{record['stem']}.dat 16 100(10)/mV 16 0 0 0 0 {name}")
            for channel in range(24):
                lines.append(f"{record['stem']}.dat 16 100(0)/mV 16 0 0 0 0 other{channel}")
            header.write_text("\n".join(lines) + "\n", encoding="utf-8")
            adc = np.zeros((10240, 32), dtype="<i2")
            adc[:, :8] = 10
            adc[0, 0] = 110
            adc[512, 1] = -90
            adc[:, 8:] = 32000
            adc.tofile(signal)
            windows = read_record(root, record)
            self.assertEqual(windows.shape, (20, 512, 8))
            self.assertAlmostEqual(windows[0, 0, 0], 1.0)
            self.assertAlmostEqual(windows[1, 0, 1], -1.0)
            self.assertEqual(np.count_nonzero(windows), 2)

    def test_rejects_mismatched_record_header(self):
        record = next(records())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            header = root / relative_file(record, "hea")
            header.parent.mkdir(parents=True)
            header.write_text("wrong 32 2048 10240\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected WFDB header"):
                read_record(root, record)


if __name__ == "__main__":
    unittest.main()
