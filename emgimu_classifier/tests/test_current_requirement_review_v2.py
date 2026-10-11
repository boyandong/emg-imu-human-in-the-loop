import csv
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.new_bank_v3.current_requirement_review_v2 import DOCUMENTS, build, verify_partition

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'feature_bank/CURRENT_REQUIREMENT_REVIEW_V2.json'


def read():
    return json.loads(REVIEW.read_text(encoding='utf-8'))


def test_current_review_binds_current_evidence_and_never_promotes_completion():
    review = read()
    for rel, expected in review['source_sha256'].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == expected, rel
    assert review['documents'] == DOCUMENTS
    assert not any(review[k] for k in ('completion_proven','default_promoted','native_experiment_rerun'))
    assert not any(s['individual_completion_proven'] for s in review['sections'])


def test_exact_original_line_partition_and_current_semantic_routes():
    review = read()
    trace = json.loads((ROOT/'feature_bank/DOCUMENT_TRACEABILITY_V1.json').read_text(encoding='utf-8'))
    records = trace['records']
    assert len(records) == review['nonempty_source_lines'] == 2586
    assert len(review['sections']) == 78
    assert sum(s['nonblank_lines'] for s in review['sections']) == len(records)
    for r in records:
        matches = [s for s in review['sections'] if s['document']==r['document']
                   and s['section_start']<=r['line']<=s['section_end']]
        assert len(matches) == 1
        s = matches[0]
        if r['major_clause']:
            assert r['major_clause'] in s['major_clause_ids'].split('|')
        else:
            supplemental = [a for a in review['supplemental_line_reviews']
                            if (a['document'],a['line'])==(r['document'],r['line'])]
            assert len(supplemental)==1
            assert supplemental[0]['supplemental_review'] in s['supplemental_review_ids'].split('|')


@pytest.mark.parametrize('mutation', ['gap','overlap','tail','unknown'])
def test_partition_rejects_missing_overlapping_and_unknown_source(mutation):
    sections = [dict(document='d',section_start=1,section_end=2),
                dict(document='d',section_start=3,section_end=4)]
    if mutation=='gap': sections[1]['section_start']=4
    elif mutation=='overlap': sections[1]['section_start']=2
    elif mutation=='tail': sections[1]['section_end']=3
    else: sections.append(dict(document='other',section_start=1,section_end=1))
    with pytest.raises(ValueError): verify_partition(sections, {'d':['a','','b','c']})


def test_review_rejects_changed_requirement_document_before_using_old_evidence(tmp_path):
    for name in DOCUMENTS:
        (tmp_path/name).write_text('changed requirement\n',encoding='utf-8')
    with pytest.raises(ValueError,match='Requirement source hash mismatch'):
        build(ROOT,tmp_path)


def test_numerical_review_and_completed_negative_studies_stay_separate():
    review = read()
    numeric = review['formula_evidence']
    assert numeric['fixture_sections']==32
    assert numeric['junit_counts']==dict(tests=142,failures=0,errors=0,skipped=0)
    assert numeric['junit_counts_include_subtests']
    assert numeric['numerical_tests_are_not_native_acceptance']
    native = review['current_native_outcomes']
    assert native['precision_detector_primary_pass']
    assert not native['document_reliability_primary_pass']
    assert not native['precision_joint_primary_pass']
    assert not native['units_physical_validation_proven']
    assert native['negative_completed_experiments_are_not_unimplemented']


def test_current_blockers_preserve_real_conditions_and_supersede_absent_legacy():
    review = read()
    old = review['historical_supersession']
    assert not old['old_code_wait_is_current_blocker']
    assert not old['ds2_force_label_wait_is_current_blocker']
    assert old['ds2_eligible_trials']==2832
    assert not old['original_chronology_reconstructed']
    assert old['historical_audits_preserved']
    assert {r['id'] for r in review['remaining_conditions']} == {
        'own_cohort','physical_device','body_frame','electrode_topology','fault_truth',
        'eligible_trial_budgets','prospective_evaluation','secondary_provenance'}
    assert review['major_status_counts']==dict(verified_scoped=26,superseded_history=3,
                                              partial_active=2,deferred_data_device=1)


def test_csv_and_canonical_entry_expose_same_review_without_digest_cycle():
    review = read()
    with REVIEW.with_suffix('.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==len(review['sections'])
    for row, expected in zip(rows,review['sections']):
        assert row == {k:str(v) for k,v in expected.items()}
    index_path=ROOT/'feature_bank/delivery/INDEX.json'
    index=json.loads(index_path.read_text(encoding='utf-8'))
    pointer=index['current_requirement_review']
    assert (index_path.parent/pointer['path']).resolve()==REVIEW.resolve()
    assert not pointer['completion_proven']
    assert 'sha256' not in pointer  # Review -> major audit -> INDEX must stay acyclic.
