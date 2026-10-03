"""Independent readback of matched Core/F7 trial identities and loss deltas."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from emgimu.datasets.epn612 import GESTURES
from emgimu.feature_bank.epn_study import _metrics


ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / 'benchmarks/new_bank_v3/F7_AFFINE_CORE_MATCHED'
AFFINE = ROOT / 'benchmarks/new_bank_v3/F7_AFFINE_EPN'


def _read(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_trial_match_and_uniform_control_readback():
    report = json.loads((DELIVERY / 'results.json').read_text(encoding='utf-8'))
    assert _sha(ROOT / 'benchmarks/new_bank_v3/f7_affine_core_matched.py') == report['source_sha256']
    for name, digest in report['output_sha256'].items():
        assert _sha(DELIVERY / name) == digest
    assert report['output_sha256']['core_validation.npz'] == report['input_sha256']['core_validation']
    assert report['output_sha256']['core_final.npz'] == report['input_sha256']['core_final']
    frozen = json.loads((ROOT / 'feature_bank/results/epn_shortlist_final_replay.json').read_text())
    assert report['input_sha256']['core_validation'] == frozen['source_sha256']['heldout_predictions.npz']
    assert report['input_sha256']['core_final'] == frozen['output_sha256']['heldout_predictions.npz']
    assert _sha(AFFINE / 'results.json') == report['input_sha256']['affine_results']
    scores = _read(DELIVERY / 'scores.csv')
    paired = _read(DELIVERY / 'trial_predictions.csv')
    originals = _read(AFFINE / 'trial_predictions.csv')
    assert len(scores) == 96 and len(paired) == len(originals) == 2412
    original_keys = {(r['phase'], r['subject'], r['shots_per_class'], r['trial_id'])
                     for r in originals}
    assert len(original_keys) == len(originals)
    assert {(r['phase'], r['subject'], r['shots_per_class'], r['trial_id'])
            for r in paired} == original_keys
    for phase, name in (('validation', 'core_validation.npz'),
                        ('descriptive_final', 'core_final.npz')):
        with np.load(DELIVERY / name, allow_pickle=False) as saved:
            assert set(saved.files) >= {'labels', 'users', 'trials', 'full'}
            core = {(str(user), str(trial)): (int(label), p)
                    for user, trial, label, p in zip(saved['users'], saved['trials'],
                                                      saved['labels'], saved['full'])}
        assert len(core) == 450
        for row in (r for r in paired if r['phase'] == phase):
            label, probability = core[row['subject'], row['trial_id']]
            assert int(row['true_label']) == label
            np.testing.assert_allclose(
                [float(row[f'p_core_{gesture}']) for gesture in GESTURES],
                probability, atol=1e-15)
    pooled = {(r['phase'], int(r['shots_per_class']), r['arm']): r
              for r in scores if r['subject'] == 'ALL'}
    for cell in report['pooled_increments']:
        phase, budget = cell['phase'], cell['shots_per_class']
        rows = [r for r in paired if r['phase'] == phase and int(r['shots_per_class']) == budget]
        assert len(rows) == cell['evaluation_trials']
        truth = np.asarray([int(r['true_label']) for r in rows])
        core = np.asarray([[float(r[f'p_core_{name}']) for name in GESTURES] for r in rows])
        affine = np.asarray([[float(r[f'p_affine_{name}']) for name in GESTURES] for r in rows])
        mix = np.asarray([[float(r[f'p_mix_{name}']) for name in GESTURES] for r in rows])
        np.testing.assert_allclose(mix, (core + affine) / 2, atol=1e-15)
        np.testing.assert_allclose(mix.sum(axis=1), 1., atol=1e-12)
        uniform = (core + np.ones_like(core) / len(GESTURES)) / 2
        arms = {'Core': core, 'Core_plus_Uniform': uniform,
                'Affine_F7': affine, 'Core_plus_Affine_F7': mix}
        for name, probability in arms.items():
            metric = _metrics(truth, probability, np.ones(len(truth)))
            saved = pooled[phase, budget, name]
            for field in ('macro_f1', 'accuracy', 'log_loss', 'brier', 'ece'):
                np.testing.assert_allclose(float(saved[field]), metric[field], atol=1e-12)
        assert cell['delta_logloss'] > 0
        np.testing.assert_allclose(
            cell['delta_logloss_over_uniform'],
            float(pooled[phase, budget, 'Core_plus_Uniform']['log_loss'])
            - float(pooled[phase, budget, 'Core_plus_Affine_F7']['log_loss']), atol=1e-12)
