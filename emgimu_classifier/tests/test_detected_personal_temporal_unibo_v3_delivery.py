import csv
import json
from pathlib import Path
import numpy as np
from benchmarks.new_bank_v3.verify_detected_personal_temporal_unibo_v3 import verify, metrics

ROOT = Path(__file__).resolve().parents[1]


def test_native_continuous_fusion_all_cells_splits_and_misses_are_independently_verified():
    actual = verify()
    assert actual == json.loads((ROOT/'feature_bank/DETECTED_PERSONAL_TEMPORAL_UNIBO_ACCEPTANCE_V3.json').read_text())
    assert actual['retained_predictions'] == 273328 and actual['arm_cells'] == 616
    assert actual['totals']['supported_references'] == 596 and actual['totals']['supported_matched'] == 527
    assert actual['totals']['unsupported_matched'] == 481 and actual['totals']['unmatched_detections'] == 18
    assert not actual['primary_eligible'] and not actual['primary_pass']
    assert actual['user_loss_wins'] == 2 and not actual['default_promoted']


def test_continuous_canonical_rows_preserve_conditional_weights_costs_and_coverage():
    result = json.loads((ROOT/'benchmarks/new_bank_v3/DETECTED_PERSONAL_TEMPORAL_UNIBO_V3_RESULTS.json').read_text())
    blocks = {(b['user'], 'Day6_'+b['mode'], str(b['shots'])):b for b in result['blocks']}
    def rows(name):
        with (ROOT/'feature_bank/delivery/new_bank_v3'/name).open(encoding='utf8', newline='') as f:
            return [r for r in csv.DictReader(f) if r['run_id'] == 'detected_personal_temporal_unibo_v3']
    family = rows('feature_family_results.csv'); error = rows('error_complementarity.csv'); ablation = rows('ablation_full_bank.csv')
    assert len(family) == 616 and len(error) == 728 and len(ablation) == 112
    with np.load(ROOT/'benchmarks/new_bank_v3/detected_personal_temporal_unibo_v3/readouts.npz', allow_pickle=False) as arrays:
        for row in family:
            b = blocks[row['subject'], row['session/domain'], row['calibration_budget']]
            q = arrays[b['key']+'_'+row['feature_family']][b['matched_positions']]
            expected = metrics(np.array(b['labels']), q, np.array(b['weights']))
            for field in ('accuracy', 'macro_f1', 'log_loss', 'brier'):
                assert abs(float(row[field])-expected[field]) < 1e-12
            notes = json.loads(row['metadata_notes_json'])
            assert notes['no_rest_ground_truth'] and notes['all_reserved_recordings_excluded_at_all_budgets']
            assert notes['end_to_end_including_missed_references'] == b['end_to_end']
            assert notes['actual_additional_temporal_calibration_trials'] == b['predictive_calibration_cost']
            assert notes['observed_active_classes'] == sorted(set(b['labels']))
        for row in error:
            b = blocks[row['subject'], row['session/domain'], row['calibration_budget']]
            y = np.array(b['labels']); w = np.array(b['weights']); w /= w.sum()
            ea = arrays[b['key']+'_'+row['family_a']][b['matched_positions']].argmax(1) != y
            eb = arrays[b['key']+'_'+row['family_b']][b['matched_positions']].argmax(1) != y
            assert abs(float(row['a_correct_b_wrong_probability'])-w[~ea & eb].sum()) < 1e-12
            assert abs(float(row['a_wrong_b_correct_probability'])-w[ea & ~eb].sum()) < 1e-12
        for row in ablation:
            b = blocks[row['subject'], row['session/domain'], row['calibration_budget']]
            assert int(row['full_target_calibration_trials_per_user']) == 20+4*b['shots']
            assert 'not renormalization' in json.loads(row['metadata_notes_json'])['branch_ablation']


def test_current_conclusions_report_continuous_failure_and_missing_fist_separately_from_gui():
    current = json.loads((ROOT/'feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json').read_text(encoding='utf8'))
    native = current['detected_personal_temporal_unibo']
    assert not native['primary_eligible'] and native['primary_user_loss_wins'] == 2
    assert current['detected_personal_temporal_unibo_acceptance']['retained_predictions'] == 273328
    assert not current['temporal_live_gui_acceptance']['physical_validation_proven']
    report = (ROOT/'feature_bank/REPORT.md').read_text(encoding='utf8')
    assert '273328' in report and 'u07' in report and '596' in report
