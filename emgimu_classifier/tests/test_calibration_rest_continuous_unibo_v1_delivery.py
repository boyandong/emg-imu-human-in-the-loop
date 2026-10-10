import csv
import json
from pathlib import Path
import numpy as np
from benchmarks.new_bank_v3.verify_calibration_rest_continuous_unibo_v1 import verify, metrics

ROOT = Path(__file__).resolve().parents[1]


def test_registered_rest_native_boundaries_probabilities_metrics_and_costs_are_independent():
    actual = verify()
    saved = json.loads((ROOT/'feature_bank/CALIBRATION_REST_CONTINUOUS_UNIBO_ACCEPTANCE_V1.json').read_text())
    assert actual == saved
    assert actual['checked_detected_intervals'] == 4381 and actual['retained_probability_values'] == 192764
    assert actual['arm_cells'] == 308 and actual['supported_references'] == 596
    assert actual['old_correct'] == 367 and actual['registered_rest_correct'] == 407
    assert actual['primary_pass'] and actual['user_success_wins'] == 5
    assert actual['independent_direct_G5_max_probability_error'] < 1e-6
    assert actual['independent_vector_energy_FSM_boundaries_exact'] and not actual['default_promoted']


def test_registered_rest_canonical_rows_keep_matched_subset_and_explicit_detector_cost():
    result = json.loads((ROOT/'benchmarks/new_bank_v3/CALIBRATION_REST_CONTINUOUS_UNIBO_V1_RESULTS.json').read_text())
    blocks = {(b['user'], str(b['shots'])):b for b in result['blocks']}
    def rows(name):
        with (ROOT/'feature_bank/delivery/new_bank_v3'/name).open(encoding='utf8', newline='') as f:
            return [r for r in csv.DictReader(f) if r['run_id'] == 'calibration_rest_continuous_unibo_v1']
    assert len(rows('error_complementarity.csv')) == 364 and len(rows('ablation_full_bank.csv')) == 56
    family = rows('feature_family_results.csv'); assert len(family) == 308
    for row in family:
        b = blocks[row['subject'], row['calibration_budget']]
        with np.load(ROOT/'benchmarks/new_bank_v3/calibration_rest_continuous_unibo_v1/parts'/(b['user']+'.npz'), allow_pickle=False) as arrays:
            q = arrays[b['key']+'_'+row['feature_family']][b['matched_positions']]
        expected = metrics(np.array(b['labels']), q, np.array(b['weights']))
        for field in ('accuracy', 'macro_f1', 'log_loss', 'brier'):
            assert abs(float(row[field])-expected[field]) < 1e-12
        notes = json.loads(row['metadata_notes_json'])
        assert notes['detector_used_neutral_calibration_trials'] == b['detector_calibration_cost']
        assert notes['end_to_end_including_missed_references'] == b['end_to_end']
        assert notes['conditional_on_supported_matched_references'] and notes['no_rest_ground_truth']


def test_report_distinguishes_detection_gain_from_changed_matched_classification_subset():
    d = json.loads((ROOT/'feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json').read_text(encoding='utf8'))
    native = d['calibration_rest_continuous_unibo']
    assert native['primary_pass'] and native['primary_user_success_wins'] == 5
    assert not d['default_promoted'] and not d['own_device_efficacy_proven']
    report = (ROOT/'feature_bank/REPORT.md').read_text(encoding='utf8')
    assert '407/596' in report and '192764' in report and 'matched subsets differ' in report
