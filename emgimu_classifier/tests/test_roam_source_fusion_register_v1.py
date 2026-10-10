"""Canonical source-selected fusion keeps cost, Unknown and scientific scope."""
import csv
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(name):return json.loads((ROOT/name).read_text(encoding='utf8'))


def test_source_selected_fusion_is_bound_in_current_report_and_index():
    receipt=read('feature_bank/ROAM_SOURCE_FUSION_V1_ACCEPTANCE.json')
    current=read('feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json')['roam_source_fusion']
    assert all(current[k]==v for k,v in receipt.items())
    assert len(current['two_shot_summary'])==12 and set(current['source_weights'])=={'source_window','reliability_window','F7F8_window','DTW','signature'}
    index=read('feature_bank/delivery/INDEX.json')['roam_source_fusion_acceptance']
    assert index['sha256']==hashlib.sha256((ROOT/'feature_bank/ROAM_SOURCE_FUSION_V1_ACCEPTANCE.json').read_bytes()).hexdigest()
    report=(ROOT/'feature_bank/REPORT.md').read_text(encoding='utf8')
    assert 'Source-only population probability fusion' in report
    assert 'source training loss is not an unbiased evaluation' in report
    assert 'Unknown is scored as wrong' in report


def test_source_fusion_canonical_provider_removals_keep_shared_queries_and_costs():
    with (ROOT/'feature_bank/delivery/new_bank_v3/ablation_full_bank.csv').open(encoding='utf8',newline='') as stream:
        rows=[r for r in csv.DictReader(stream) if r['run_id']=='roam_source_fusion_v1']
    assert len(rows)==195 and len({(r['subject'],r['session/domain'],r['calibration_budget'],r['removed_provider']) for r in rows})==195
    for row in rows:
        notes=json.loads(row['metadata_notes_json']);shots=int(row['calibration_budget'])
        assert notes['source_policy_frozen_before_target_apply'] and notes['target_arrays_reused_without_refitting']
        assert notes['Unknown_scored_wrong'] and notes['removal_is_top_level_component_not_whole_F0_F9_family']
        assert int(row['full_target_calibration_trials_per_user'])==int(row['remaining_target_calibration_trials_per_user'])==6+3*shots
