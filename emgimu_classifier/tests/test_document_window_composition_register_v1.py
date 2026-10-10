"""Current delivery preserves scientific scope and the joint-family removal."""
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]


def read(name): return json.loads((ROOT/name).read_text(encoding='utf8'))


def test_current_conclusions_and_index_bind_composition_without_inventing_gain():
    receipt=read('feature_bank/DOCUMENT_WINDOW_COMPOSITION_V1_ACCEPTANCE.json')
    conclusions=read('feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json')['document_window_composition']
    assert all(conclusions[k]==v for k,v in receipt.items())
    assert conclusions['independent_rebuild_is_not_a_new_accuracy_gain']
    index=read('feature_bank/delivery/INDEX.json')['document_window_composition_acceptance']
    assert index['sha256']==hashlib.sha256((ROOT/'feature_bank/DOCUMENT_WINDOW_COMPOSITION_V1_ACCEPTANCE.json').read_bytes()).hexdigest()
    assert not receipt['default_promoted'] and not receipt['completion_proven']


def test_canonical_whole_family_F2_removal_keeps_same_trials_and_budgets():
    with (ROOT/'feature_bank/delivery/new_bank_v3/ablation_full_bank.csv').open(encoding='utf8',newline='') as stream:
        rows=[r for r in csv.DictReader(stream) if r['run_id']=='document_window_composition_v1']
    assert len(rows)==64
    f2=[r for r in rows if r['removed_provider']=='family_F2']
    assert len(f2)==4
    for row in f2:
        assert row['remaining_bank']=='minus_family_F2' and int(row['evaluation_trials'])==124
        assert json.loads(row['metadata_notes_json'])['F2_family_removes_both_covariance_and_CSP']
        shots=int(row['calibration_budget'])
        assert int(row['full_target_calibration_trials_per_user'])==20+4*shots
        assert int(row['remaining_target_calibration_trials_per_user'])==20+4*shots


def test_current_report_discloses_equivalence_and_unavailable_geometry():
    report=(ROOT/'feature_bank/REPORT.md').read_text(encoding='utf8')
    assert 'Document window composition and family removals' in report
    assert 'not a new algorithmic gain' in report
    assert 'Ring F3a/c, anatomical F6' in report


def test_composition_formal_source_and_shared_bout_tests_actually_passed():
    root=ET.parse(ROOT/'feature_bank/regression/document_window_composition_v1/classifier.xml').getroot()
    cases=root.findall('.//testcase')
    assert len(cases)==80
    assert not root.findall('.//failure') and not root.findall('.//error') and not root.findall('.//skipped')
    assert any(c.attrib['name']=='test_exact_document_composition_is_usable_by_shared_complete_bout_workflow' for c in cases)
