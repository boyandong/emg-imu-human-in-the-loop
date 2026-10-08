import json
from pathlib import Path
import pytest
from benchmarks.new_bank_v3.document_traceability_v1 import digest, inventory
from benchmarks.new_bank_v2.v1_acceptance_audit import CLAUSES, DOCUMENTS

ROOT = Path(__file__).resolve().parents[1]


def test_coverage_keeps_unmapped_text_and_rejects_overlapping_acceptance():
    docs = {'spec': ['before', '', 'required', 'after']}
    clauses = [('X', 'spec', 3, 3)]
    rows = inventory(docs, clauses)
    assert [(r['line'], r['text'], r['major_clause']) for r in rows] == [
        (1, 'before', None), (3, 'required', 'X'), (4, 'after', None)]
    assert all(r['individual_verification'] == 'unproven' for r in rows)
    with pytest.raises(ValueError, match='Overlapping'):
        inventory(docs, clauses + [('Y', 'spec', 1, 3)])


def test_delivery_covers_source_queue_without_promoting_acceptance():
    import csv
    artifact = json.loads((ROOT / 'feature_bank/DOCUMENT_TRACEABILITY_V1.json').read_text(encoding='utf8'))
    assert artifact['documents'] == DOCUMENTS
    for name, expected in artifact['source_sha256'].items():
        assert digest(ROOT / name) == expected
    records = artifact['records']
    identities = [(r['document'], r['line']) for r in records]
    assert len(identities) == len(set(identities)) == artifact['nonempty_lines'] == 2586
    with (ROOT / 'feature_bank/DOCUMENT_SCOPE_QUEUE.csv').open(encoding='utf8', newline='') as stream:
        queue = list(csv.DictReader(stream))
    assert [(r['document'], r['line'], r['text']) for r in records] == [
        (r['document'], int(r['line']), r['text']) for r in queue]
    reconstructed = inventory({name: [next((r['text'] for r in records
                                if r['document'] == name and r['line'] == line), '')
                                for line in range(1, max(r['line'] for r in records if r['document'] == name)+1)]
                                for name in DOCUMENTS}, CLAUSES)
    assert reconstructed == records
    assert artifact['unmapped_records'] == [r for r in records if r['major_clause'] is None]
    assert artifact['unmapped_lines'] == 77
    assert artifact['mapped_lines'] + artifact['unmapped_lines'] == len(records)
    assert artifact['completion_proven'] is False
    assert all(r['individual_verification'] == 'unproven' for r in records)


def test_supplemental_semantics_preserve_open_components_and_original_lines():
    artifact = json.loads((ROOT / 'feature_bank/DOCUMENT_TRACEABILITY_V1.json').read_text(encoding='utf8'))
    reviewed = artifact['supplemental_reviews']
    assert artifact['unmapped_without_semantic_review'] == 0
    assert len(reviewed) == len(artifact['unmapped_records']) == 77
    for original, review in zip(artifact['unmapped_records'], reviewed):
        assert all(review[key] == value for key, value in original.items())
        assert review['evidence_scope'] and review['evidence_sha256']
        for path, expected in review['evidence_sha256'].items():
            assert digest(ROOT / path) == expected
    assert {r['supplemental_review'] for r in reviewed if r['disposition'] == 'partial_active'} == {
        'PRE-LICENSE', 'GOAL-COMPONENTS'}
    assert all(r['disposition'] == 'historical_prior_not_new_evidence'
               for r in reviewed if r['supplemental_review'] == 'GOAL-PRIORS')
    assert artifact['completion_proven'] is False


def test_intro_conditional_information_rule_against_actual_concatenation_results():
    import csv
    import math
    base = ROOT / 'benchmarks/new_bank_v2'
    def read(name):
        with (base / name).open(encoding='utf8', newline='') as stream:
            return list(csv.DictReader(stream))
    keys = ('dataset', 'axis', 'phase', 'scope', 'subject', 'condition')
    families = {tuple(row[k] for k in keys) + (row['feature_family'],): row
                for row in read('FAMILY_SCREEN.csv')}
    increments = read('CONDITIONAL_VALUE.csv')
    assert len(increments) == 128
    for row in increments:
        identity = tuple(row[k] for k in keys)
        core = families[identity + (row['core_bank'],)]
        added = families[identity + (row['increment_bank'],)]
        assert row['increment_bank'] == 'F0v2+F2a+F3c'
        assert row['core_bank'] in ('F0v2+F2a', 'F0v2+F3c')
        assert row['evaluation_trials'] == core['evaluation_trials'] == added['evaluation_trials']
        for metric in ('log_loss', 'brier'):
            assert math.isclose(float(row['delta_' + metric]),
                                float(core[metric]) - float(added[metric]), abs_tol=1e-12)
        assert math.isclose(float(row['delta_macro_f1']),
                            float(added['macro_f1']) - float(core['macro_f1']), abs_tol=1e-12)
        assert json.loads(row['core_per_class_f1_json']) == json.loads(core['per_class_f1_json'])
        assert json.loads(row['increment_per_class_f1_json']) == json.loads(added['per_class_f1_json'])
