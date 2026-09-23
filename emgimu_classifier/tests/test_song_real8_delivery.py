import csv
import tempfile
import unittest
from pathlib import Path

from benchmarks.export_song_real8_delivery import export


class SongReal8DeliveryTest(unittest.TestCase):
    def test_personal_anchor_curve_is_separate_from_source_spd(self):
        classifier = Path(__file__).resolve().parents[1]
        evidence = classifier / 'benchmarks' / 'song_real8'
        with tempfile.TemporaryDirectory(prefix='song_delivery_') as temporary:
            output = Path(temporary)
            manifest = export(evidence / 'PROBABILITY_CALIBRATION_AUDIT.json',
                              evidence / 'SPD_INCREMENT_RESULTS.json', output)
            with (output / 'calibration_curve.csv').open(encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 10)
            personal = [row for row in rows if row['feature_bank'] == 'F0+F7_SPD_personal_anchor']
            self.assertEqual(len(personal), 6)
            for session in ('S03', 'S04'):
                self.assertEqual([int(row['shots_per_class']) for row in personal
                                  if row['session/domain'] == session], [0, 1, 2])
            self.assertEqual(manifest['personal_anchor_max_blocks_per_class'], 2)
            final = [row for row in personal if row['session/domain'] == 'S04']
            self.assertGreater(float(final[1]['log_loss']), float(final[0]['log_loss']))
            self.assertGreater(float(final[2]['log_loss']), float(final[0]['log_loss']))
            source_spd = [row for row in rows if row['feature_bank'] == 'F0+F2c_SPD']
            self.assertEqual(len(source_spd), 2)
            self.assertTrue(all(row['shots_per_class'] == '0' for row in source_spd))


if __name__ == '__main__':
    unittest.main()
