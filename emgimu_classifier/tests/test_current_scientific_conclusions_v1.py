import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'benchmarks/new_bank_v3'


def load(path): return json.loads(path.read_text(encoding='utf8'))


def test_eight_answers_bind_current_evidence_and_keep_scope_boundaries():
    r = load(ROOT / 'feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json')
    assert r['generator_sha256'] == hashlib.sha256((HERE / 'current_scientific_conclusions_v1.py').read_bytes()).hexdigest()
    for path, digest in r['source_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert r['requirement_document_sha256'] == load(ROOT / 'feature_bank/NEW_VERSION_ACCEPTANCE_AUDIT.json')['documents']['docx_goal.txt']
    assert r['requirement_lines'] == [1013, 1049]
    assert [q['question_id'] for q in r['questions']] == list('ABCDEFGH')
    assert all(q['answer'] and q['boundary'] and q['evidence'] for q in r['questions'])
    assert all(set(q['evidence']) <= set(r['source_sha256']) for q in r['questions'])
    assert all(t in r['source_sha256'] for t in r['verification_tests'])
    assert not any(r[k] for k in ('default_promoted', 'own_device_efficacy_proven', 'full_seven_axis_robustness_proven', 'completion_proven'))
    assert r['nominal_continuous_samples'] == 393776
    index = load(ROOT / 'feature_bank/delivery/INDEX.json')['current_scientific_conclusions']
    indexed = (ROOT / 'feature_bank/delivery' / index['path']).resolve()
    assert indexed == (ROOT / 'feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json').resolve()
    assert index['sha256'] == hashlib.sha256(indexed.read_bytes()).hexdigest()


def test_quantitative_answers_reconstruct_comparisons_not_old_report_numbers():
    q = {r['question_id']: r for r in load(ROOT / 'feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json')['questions']}
    native = load(HERE / 'ROAM_DEBOUNCE_CONTROL_V1_RESULTS.json')
    for cell in q['A']['measurements']:
        arms = native['summaries'][cell['phase']]; a, b = arms['raw'], arms['confirmed']
        assert abs(cell['delta_shared_macro_f1'] - (b['equal_recording_shared_macro_f1'] - a['equal_recording_shared_macro_f1'])) < 1e-12
        assert abs(cell['switch_reduction_fraction'] - ((a['maintenance_switches'] - b['maintenance_switches']) / a['maintenance_switches'])) < 1e-12
        assert cell['raw_unknown_samples'] == a['unknown_samples']
        assert cell['confirmed_unknown_samples'] == b['unknown_samples']
    fresh = load(HERE / 'F7_AFFINE_FRESH/results.json')['primary_five_shot']
    assert q['B']['measurements'] == fresh and all(fresh['criteria'].values())
    router = load(HERE / 'F8_CALIBRATED_MANUS_V2_RESULTS.json')['scores']
    before = load(HERE / 'F8_ROUTER_MANUS_V1_RESULTS.json')['scores']
    for cell in q['F']['measurements']:
        scores = router[cell['cell']]; old = before[cell['cell']]
        assert cell['calibrated_F8_vs_calibrated_uniform_delta_log_loss'] == scores['uniform']['log_loss'] - scores['F8']['log_loss']
        assert cell['calibrated_F8_vs_raw_F8_delta_macro_f1'] == scores['F8']['macro_f1'] - old['F8']['macro_f1']
    assert len(q['F']['measurements']) == 4
    epn = load(HERE / 'MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json')
    for row in q['E']['measurements']:
        values = []
        for block in epn['blocks']:
            if block['shots'] != row['shots_per_class']: continue
            y = np.asarray(block['labels']); called = np.asarray(block['probabilities'][row['method']]).argmax(1)
            f = []
            for c in range(6):
                den = np.sum(y == c) + np.sum(called == c)
                f.append(2 * np.sum((y == c) & (called == c)) / den if den else 0)
            values.append(sum(f) / 6)
        assert len(values) == 10
        assert abs(row['equal_user_mean'] - np.mean(values)) < 1e-12
        assert abs(row['minimum'] - min(values)) < 1e-12
        assert abs(row['sample_std_ddof1'] - np.std(values, ddof=1)) < 1e-12
    assert len(q['E']['measurements']) == 4
    costs = load(HERE / 'MAHALANOBIS_EPN_HOLDOUT_V2_BURDEN.json')['records']
    for row in q['H']['measurements']:
        group = [c for c in costs if c['shots_per_class'] == row['shots_per_class']]
        assert row['used_trials'] == [row['shots_per_class'] * 6]
        assert row['used_signal_seconds'] == [row['shots_per_class'] * 6 * 4 * .2]
        assert row['full_recording_seconds_min'] == min(c['used_trials_full_recording_seconds'] for c in group)
        assert row['full_recording_seconds_max'] == max(c['used_trials_full_recording_seconds'] for c in group)
        assert row['device_wall_time_seconds'] is None
    assert len(q['G']['measurements']['budget_comparisons']) == 4


def test_default_extension_claim_rechecks_native_axes_and_signed_metrics():
    audit = load(HERE / 'PUBLIC_DEFAULT_EXTENSION_AUDIT.json')
    protocol = HERE / 'PUBLIC_DEFAULT_EXTENSION_PROTOCOL.json'
    assert audit['protocol_sha256'] == hashlib.sha256(protocol.read_bytes()).hexdigest()
    for path, digest in audit['source_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    spatial = load(HERE / 'SPEC_SPATIAL_GRAB_RESULTS.json')['scores']
    temporal = load(ROOT / 'benchmarks/new_bank_v2/F5C_UNIBO_BOUT_RESULTS.json')['scores']['validation']
    anchor = load(HERE / 'F7_DOCUMENT_ANCHOR_RESULTS.json')['metrics']['validation']
    assert len(audit['candidate_checks']) == 7
    for check in audit['candidate_checks']:
        candidate = check['candidate']
        if candidate == 'F0+F2c_spec':
            axis = load(HERE / 'SPEC_F2C_CROSS_AXIS_AUDIT.json')
            assert check['validation_violations'] == axis['validation_violations']
            assert axis['validation_violations'] and not axis['candidate_eligible_for_public_default']
            continue
        if candidate in ('F0+F2a_spec', 'F0+F3c_spec'):
            base, alt = spatial['F0']['validation'], spatial[candidate]['validation']
        elif candidate == 'G5+F5c':
            base, alt = temporal['G5']['pooled'], temporal['G5+F5c']['pooled']
        else:
            budget = candidate.rsplit('_', 1)[1].replace('shot', '')
            base, alt = anchor[budget]['F0v2'], anchor[budget]['F0v2+F7_document']
        df = alt['macro_f1'] - base['macro_f1']; dl = alt['log_loss'] - base['log_loss']
        assert abs(check['validation_delta_macro_f1'] - df) < 1e-12
        assert abs(check['validation_delta_log_loss'] - dl) < 1e-12
        assert check['validation_guard_pass'] == (df >= -1e-12 and dl <= 1e-12 and (df > 1e-12 or dl < -1e-12))
    assert not any(c['validation_guard_pass'] for c in audit['candidate_checks'])
