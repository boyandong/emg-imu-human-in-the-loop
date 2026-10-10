"""Recover rejection-aware native metrics without fitting or hardware assumptions."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'benchmarks/song_real8'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_raw_quality_trial_metrics_annotations_and_soft_routing_harm():
    p = json.loads((HERE/'SONG_RAW_QUALITY_V1_PROTOCOL.json').read_text(encoding='utf8'))
    r = json.loads((HERE/'SONG_RAW_QUALITY_V1_RESULTS.json').read_text(encoding='utf8'))
    assert r['protocol_sha256'] == sha(HERE/'SONG_RAW_QUALITY_V1_PROTOCOL.json')
    for name, digest in {**p['source_sha256'], **p['artifact_sha256']}.items(): assert sha(ROOT.parent/name) == digest
    for name, digest in r['artifact_sha256'].items(): assert sha(ROOT/name) == digest
    old = json.loads((HERE/'SONG_PERSONAL_SESSION_V1_RESULTS.json').read_text(encoding='utf8'))
    assert r['evaluation_ids'] == old['evaluation_ids']
    assert len(r['source_trials']) == 285 and r['source_windows'] == 849
    assert not set(r['evaluation_ids']) & (set(r['source_trials']) | set(old['reserved_current_ids']) | set(old['personal_calibration_ids']))
    with (HERE/'song_raw_quality_v1/predictions.csv').open(encoding='utf8', newline='') as f: rows = list(csv.DictReader(f))
    assert len(rows) == 2976 and len(r['cells']) == 24
    correct = {}
    for cell in r['cells']:
        records = [v for v in rows if (v['scenario'], v['mode']) == (cell['scenario'], cell['mode'])]
        assert [v['trial_id'] for v in records] == r['evaluation_ids']
        labels = np.array([v['label'] for v in records]);pred = np.array([v['predicted_label'] for v in records])
        rejected = np.array([v['rejected'] == 'True' for v in records]);accepted = ~rejected
        assert np.array_equal(pred == 'Unknown', rejected)
        assert cell['rejected'] == int(rejected.sum())
        assert cell['coverage'] == float(accepted.mean())
        assert cell['accuracy_unknown_wrong'] == float((labels == pred).mean())
        assert cell['accepted_accuracy'] == (float((labels[accepted] == pred[accepted]).mean()) if accepted.any() else None)
        known = cell['scenario'] in p['known_injected_fault_scenarios']
        gate = cell['gate_evaluation']
        assert gate['known_trials'] == (124 if known else 0)
        assert gate['trial_balanced_metrics']['normal_false_rejection_rate'] is None
        if known: assert gate['trial_balanced_metrics']['fault_recall'] == float(rejected.mean())
        else: assert gate['trial_balanced_metrics']['fault_recall'] is None
        if cell['scenario'] == 'unmodified': correct[cell['mode']] = labels == pred
    assert np.array_equal(correct['off'], correct['structural'])
    assert correct['soft'].sum() < correct['off'].sum()
    assert all(r['primary_guards'].values()) and r['primary_pass']
    assert not r['default_promoted'] and not r['physical_validation_proven'] and not r['completion_proven']


def test_raw_quality_acceptance_binds_oracles_harm_and_full_streams():
    a = json.loads((ROOT/'feature_bank/SONG_RAW_QUALITY_ACCEPTANCE_V1.json').read_text(encoding='utf8'))
    r = json.loads((HERE/'SONG_RAW_QUALITY_V1_RESULTS.json').read_text(encoding='utf8'))
    for name, digest in a['source_sha256'].items(): assert sha(ROOT/name) == digest
    assert a['artifact_sha256'] == r['artifact_sha256']
    assert a['checked_cells'] == 24 and a['checked_trial_probabilities'] == 2976
    assert a['maximum_probability_error'] < 1e-12
    assert a['source_observation_and_quantile_oracles'] and a['baseline_probability_parity']
    with (HERE/'song_raw_quality_v1/predictions.csv').open(encoding='utf8', newline='') as f: rows = list(csv.DictReader(f))
    correctness = {mode: np.array([v['label'] == v['predicted_label'] for v in rows if v['scenario'] == 'unmodified' and v['mode'] == mode]) for mode in ('off', 'soft')}
    assert a['unmodified_soft_correct_to_wrong'] == int(np.sum(correctness['off'] & ~correctness['soft']))
    assert a['unmodified_soft_wrong_to_correct'] == int(np.sum(~correctness['off'] & correctness['soft']))
    assert a['unmodified_soft_correct_to_wrong'] > a['unmodified_soft_wrong_to_correct']
    assert {(v['arm'], v['mode']) for v in a['full_stream_records']} == {(arm, mode) for arm in ('population', 'session_5shot') for mode in ('off', 'structural')}
    for v in a['full_stream_records']:
        assert v['windows'] == 29790 and v['maximum_probability_error'] < 1e-12
        assert v['confirmation_exact'] and v['source_and_profiles_immutable']
        assert v['rejected_windows'] == 0  # Observation, not physical normal/fault truth.
    assert not a['default_promoted'] and not a['physical_validation_proven'] and not a['completion_proven']
