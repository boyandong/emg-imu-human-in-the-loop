"""Read back the saved native affine-SPD screen without rerunning the archive."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / 'benchmarks/new_bank_v3/F7_AFFINE_EPN'


def _rows(name):
    with (DELIVERY / name).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def test_native_trial_calibration_is_disjoint_and_probabilities_are_valid():
    report = json.loads((DELIVERY / 'results.json').read_text(encoding='utf-8'))
    for name, expected in report['output_sha256'].items():
        assert hashlib.sha256((DELIVERY / name).read_bytes()).hexdigest() == expected
    assert report['archive_sha256'] == '4ee8db037385e7bee1e6ac6f9e9eea4f0869e25f7825f5eb7e5be0dff4f93c21'
    for name, digest in report['source_sha256'].items():
        path = (ROOT / 'src/emgimu/feature_bank' / name if name == 'affine_spd_anchor.py'
                else ROOT / 'benchmarks/new_bank_v3' / name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    selected = _rows('calibration_trial_ids.csv')
    predictions = _rows('trial_predictions.csv')
    scores = _rows('subject_scores.csv')
    assert len(scores) == 18
    assert len(predictions) == 2412
    assert len(report['pooled']) == 6
    assert all(row['phase'] in {'validation', 'descriptive_final'} for row in scores)
    for phase in ('validation', 'descriptive_final'):
        for user in report['user_phases'][phase]:
            for budget in report['budgets']:
                key = (phase, str(user), str(budget))
                cal = [r for r in selected if (r['phase'], r['subject'], r['shots_per_class']) == key]
                ev = [r for r in predictions if (r['phase'], r['subject'], r['shots_per_class']) == key]
                assert len(cal) == 6 * budget
                assert len(ev) == 150 - 6 * budget
                assert set(r['trial_id'] for r in cal).isdisjoint(r['trial_id'] for r in ev)
                assert Counter(int(r['label']) for r in cal) == {c: budget for c in range(6)}
                assert len({r['trial_id'] for r in ev}) == len(ev)
                p = np.asarray([[float(r[f'p_{name}']) for name in
                                 ('noGesture', 'fist', 'waveIn', 'waveOut', 'open', 'pinch')]
                                for r in ev])
                assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
                np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-12)
                assert all(int(r['predicted_label']) == int(np.argmax(q))
                           for r, q in zip(ev, p))
