"""Retained native predictions, explicit target costs and immutable provenance."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'benchmarks/song_real8'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def test_extended_cells_reconstruct_all_metrics_costs_and_guards():
    p = load(HERE/'SONG_EXTENDED_WINDOW_V1_PROTOCOL.json')
    r = load(HERE/'SONG_EXTENDED_WINDOW_V1_RESULTS.json')
    assert sha(HERE/'SONG_EXTENDED_WINDOW_V1_PROTOCOL.json') == r['protocol_sha256']
    for name,digest in {**p['source_sha256'],**p['artifact_sha256'],**r['artifact_sha256']}.items():
        assert sha(ROOT/name) == digest,name
    previous = load(HERE/'SONG_PERSONAL_SESSION_V1_RESULTS.json')
    for field in ('evaluation_ids','personal_calibration_ids','reserved_current_ids'):
        assert r[field] == previous[field]
    assert not set(r['evaluation_ids']) & (set(r['personal_calibration_ids']) | set(r['reserved_current_ids']))
    last = set()
    for shots in (1,2,5):
        current = set(r['profiles'][str(shots)]['calibration_ids'])
        assert len(current) == 4*shots and last <= current
        last = current
    with (HERE/'song_extended_window_v1/predictions.csv').open(encoding='utf8',newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 9424 and len(r['cells']) == 76
    classes = ['fist','index_pinch','neutral','open_hand']
    for cell in r['cells']:
        selected = [v for v in rows if int(v['shots']) == cell['shots'] and v['arm'] == cell['arm']]
        assert [v['trial_id'] for v in selected] == r['evaluation_ids']
        assert [v['label'] for v in selected] == r['evaluation_labels']
        q = np.array([[float(v['p_'+c]) for c in classes] for v in selected])
        y = np.array([classes.index(v['label']) for v in selected]);pred = q.argmax(1)
        f1,recall = [],{}
        for i,c in enumerate(classes):
            tp = np.sum((y == i) & (pred == i));support = np.sum(y == i)
            f1.append(2*tp/(support+np.sum(pred == i)));recall[c] = float(tp/support)
        for actual,expected in ((cell['macro_f1'],np.mean(f1)),(cell['accuracy'],np.mean(y == pred)),
                (cell['log_loss'],-np.log(np.maximum(q[np.arange(len(y)),y],1e-15)).mean()),
                (cell['brier'],((q-np.eye(4)[y])**2).mean())):
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
        assert cell['recall'] == recall
        source_only = cell['arm'] in ('seven_population','seven_uniform','CSP_only')
        assert cell['long_term_calibration_trials'] == (0 if source_only else 20)
        assert cell['current_calibration_trials'] == (0 if source_only else 4*cell['shots'])
    lookup = {(c['shots'],c['arm']):c for c in r['cells']}
    for field,old,new in [('primary_guards','old_six_baseline','seven_reliability'),
                          ('full_anchor_guards','old_six_F7_F8','seven_F7_F8')]:
        a,b = lookup[5,old],lookup[5,new]
        expected = dict(lower_log_loss=b['log_loss'] < a['log_loss'],lower_brier=b['brier'] < a['brier'],
            nonworse_macro_f1=b['macro_f1'] >= a['macro_f1'],
            nonworse_all_class_recall=all(b['recall'][c] >= a['recall'][c] for c in classes))
        assert expected == r[field] and all(expected.values())
    assert r['primary_pass'] and r['F4d_context_only'] and sum(r['source_dimensions'].values()) == 281
    assert not any(r[k] for k in ('default_promoted','physical_validation_proven','completion_proven'))


def test_extended_independent_acceptance_and_full_stream_bindings():
    a = load(ROOT/'feature_bank/SONG_EXTENDED_WINDOW_ACCEPTANCE_V1.json')
    assert a['result_sha256'] == sha(HERE/'SONG_EXTENDED_WINDOW_V1_RESULTS.json')
    assert a['verifier_sha256'] == sha(HERE/'verify_extended_window_v1.py')
    assert a['checked_cells'] == 76 and a['checked_trial_probabilities'] == 9424
    assert a['maximum_probability_error'] < 1e-12 and a['maximum_CSP_eigen_residual'] < 1e-10
    for key in ('independent_source_CSP_eigen_and_variance','independent_direct_DFT_context',
                'independent_trial_balanced_prototypes_and_routing','previous_six_baseline_parity',
                'source_and_profiles_immutable','F4d_context_only'):
        assert a[key]
    p = load(HERE/'SONG_EXTENDED_GUI_V1_PROTOCOL.json')
    gui = load(ROOT/'feature_bank/SONG_EXTENDED_GUI_V1_ACCEPTANCE.json')
    assert gui['protocol_sha256'] == sha(HERE/'SONG_EXTENDED_GUI_V1_PROTOCOL.json')
    for name,digest in p['source_sha256'].items():
        assert sha(ROOT.parent/name) == digest,name
    for name,digest in p['artifact_sha256'].items():
        assert sha(ROOT/name) == digest,name
    assert sha(ROOT/gui['emissions_path']) == gui['emissions_sha256']
    assert gui['verified_windows'] == 297900 and len(gui['records']) == 10
    assert all(c['windows'] == 29790 and c['maximum_probability_error'] < 1e-12
        and c['maximum_context_error'] <= 2e-6 for c in gui['records'])
    assert next(c for c in gui['records'] if c['arm'] == 'full_dropout_structural')['rejected'] == 29790
    assert gui['independent_direct_DFT_context'] and gui['source_and_profiles_immutable']
    with np.load(ROOT/gui['emissions_path'],allow_pickle=False) as z:
        np.testing.assert_array_equal(z['output_sample_indices'],np.arange(49,297942,10))
        np.testing.assert_array_equal(z['full_off_probabilities'],z['full_structural_probabilities'])
        assert z['full_dropout_structural_rejected'].all()
        assert np.all(z['full_dropout_structural_confirmed'] == -1)


def test_extended_canonical_rows_preserve_source_only_costs_and_all_removals():
    r = load(HERE/'SONG_EXTENDED_WINDOW_V1_RESULTS.json')
    def rows(name):
        with (ROOT/'feature_bank/delivery/new_bank_v3'/name).open(encoding='utf8',newline='') as f:
            return [v for v in csv.DictReader(f) if v['run_id'] == 'song_extended_window_v1']
    family,pairs,ablations = rows('feature_family_results.csv'),rows('conditional_incremental.csv'),rows('ablation_full_bank.csv')
    assert (len(family),len(pairs),len(ablations)) == (76,72,40)
    lookup = {(int(v['calibration_budget']),v['feature_family']):v for v in family}
    for cell in r['cells']:
        row = lookup[cell['shots'],cell['arm']]
        for metric in ('macro_f1','accuracy','log_loss','brier'):
            np.testing.assert_allclose(float(row[metric]),cell[metric],rtol=0,atol=1e-12)
        cost = json.loads(row['metadata_notes_json'])['actual_total_target_calibration_trials'][cell['arm']]
        assert cost == cell['long_term_calibration_trials']+cell['current_calibration_trials']
    assert {v['removed_provider'] for v in ablations} == set(r['source_dimensions']) | {'F7_anchor','F8_router','F9_raw'}
    for row in ablations:
        assert int(row['full_target_calibration_trials_per_user']) == int(row['remaining_target_calibration_trials_per_user']) == 20+4*int(row['calibration_budget'])
