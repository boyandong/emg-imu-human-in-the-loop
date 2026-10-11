"""Current navigation over exact source lines; never promotes evidence scope.

Historical section boundaries are retained, but historical conclusions are not
copied. Major reviews, supplemental reviews and numerical fixtures remain
separate: coverage is not scientific acceptance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = {
    'docx_pre.txt': '9b70fb41ebbb4f8183e19533facf80df81b1d7ce73bd3a976a2714347d3f11d8',
    'docx_goal.txt': '4da8b372c8f07936c1156935114849f85c0d83957c1ecacc6a0b6462bdf3b1f0',
}
INPUTS = (
    'feature_bank/DOCUMENT_SECTION_AUDIT.csv',
    'feature_bank/NEW_VERSION_ACCEPTANCE_AUDIT.csv',
    'feature_bank/DOCUMENT_TRACEABILITY_V1.json',
    'feature_bank/FORMULA_FAMILY_SCOPE_AUDIT.csv',
    'feature_bank/FORMULA_NUMERICAL_ACCEPTANCE.json',
    'feature_bank/FORMULA_SUBSECTION_COVERAGE_AUDIT.json',
    'feature_bank/BODY_FRAME_UNITS_V3_ACCEPTANCE.json',
    'feature_bank/ROAM_DOCUMENT_RELIABILITY_V3_ACCEPTANCE.json',
    'feature_bank/ROAM_PRECISION_TRANSITION_V2_ACCEPTANCE.json',
    'benchmarks/discovery/CURRENT_DISCOVERY_STATE.json',
    'benchmarks/discovery/DS2_V9_FORCE_LABEL_AUDIT.json',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def section_lines(section, documents):
    name = section['document']
    start, end = int(section['section_start']), int(section['section_end'])
    lines = documents[name]
    if not 1 <= start <= end <= len(lines):
        raise ValueError('Invalid section interval')
    return [(name, i) for i in range(start, end + 1) if lines[i - 1].strip()]


def verify_partition(sections, documents):
    # Check blank lines too: an overlap hidden in whitespace is still an error.
    for name, lines in documents.items():
        intervals = sorted((int(s['section_start']), int(s['section_end']))
                           for s in sections if s['document'] == name)
        cursor = 1
        for start, end in intervals:
            if start != cursor or end < start or end > len(lines):
                raise ValueError('Section gap, overlap or out-of-range interval')
            cursor = end + 1
        if cursor != len(lines) + 1:
            raise ValueError('Incomplete section partition')
    if any(s['document'] not in documents for s in sections):
        raise ValueError('Unknown section document')


def build(root, documents_root):
    root, documents_root = Path(root), Path(documents_root)
    documents = {}
    for name, expected in DOCUMENTS.items():
        path = documents_root / name
        if sha(path) != expected:
            raise ValueError('Requirement source hash mismatch: ' + name)
        documents[name] = path.read_text(encoding='utf-8-sig').splitlines()
    bound = {}

    def bind(rel, expected=None):
        actual = sha(root / rel)
        if expected is not None and actual != expected:
            raise ValueError('Evidence hash mismatch: ' + rel)
        if rel in bound and bound[rel] != actual:
            raise ValueError('Conflicting evidence: ' + rel)
        bound[rel] = actual
        return actual

    def read(rel):
        bind(rel)
        return json.loads((root / rel).read_text(encoding='utf-8'))

    def csv_rows(rel):
        bind(rel)
        with (root / rel).open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))

    sections = csv_rows(INPUTS[0])
    verify_partition(sections, documents)
    major = csv_rows(INPUTS[1])
    if len(major) != 32 or len({r['requirement_id'] for r in major}) != 32:
        raise ValueError('Major clause inventory mismatch')
    for row in major:
        name = row['document']
        if (name not in DOCUMENTS or row['document_sha256'] != DOCUMENTS[name]
                or not 1 <= int(row['line_start']) <= int(row['line_end']) <= len(documents[name])):
            raise ValueError('Major clause source contract mismatch')
    trace = read(INPUTS[2])
    families = csv_rows(INPUTS[3])
    numeric = read(INPUTS[4])
    appendix = read(INPUTS[5])
    units, reliability, precision = (read(p) for p in INPUTS[6:9])
    discovery, ds2 = (read(p) for p in INPUTS[9:])
    for row in major + families:
        for rel, expected in json.loads(row['evidence_sha256_json']).items():
            bind(rel, expected)
    for rel, expected in trace['source_sha256'].items():
        bind(rel, expected)
    for row in trace['supplemental_reviews']:
        for rel, expected in row['evidence_sha256'].items():
            bind(rel, expected)
    records = {(r['document'], r['line']): r for r in trace['records']}
    expected_lines = {(name, i) for name, lines in documents.items()
                      for i, line in enumerate(lines, 1) if line.strip()}
    if len(records) != len(trace['records']) or set(records) != expected_lines:
        raise ValueError('Lossless line inventory mismatch')
    for (name, i), row in records.items():
        if row['text'] != documents[name][i - 1]:
            raise ValueError('Source line text mismatch')
    supplemental = {(r['document'], r['line']): r for r in trace['supplemental_reviews']}
    for row in numeric['sections'] + numeric['additional_clause_fixtures']:
        for key in ('test_sha256', 'reviewed_source_sha256'):
            for rel, expected in row[key].items():
                bind(rel, expected)
    bind('benchmarks/formula_numerical_acceptance.py', numeric['generator_sha256'])
    junit = numeric['regression_junit']
    bind(junit['path'], junit['sha256'])
    suites = ET.parse(root / junit['path']).getroot().findall('testsuite')
    counts = {key: sum(int(s.get(key, 0)) for s in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    if counts['tests'] != 142 or any(counts[k] for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Unexpected numerical fixture JUnit')
    if ds2['status'] != 'publisher_v9_force_codes_exactly_joined_to_public_raw_trials':
        raise ValueError('DS2 force label resolution no longer supported')
    formula_starts = {r['section_start'] for r in numeric['sections']}
    rows = []
    for s in sections:
        name, start, end = s['document'], int(s['section_start']), int(s['section_end'])
        if s['source_sha256'] != DOCUMENTS[name]:
            raise ValueError('Historical partition belongs to different source')
        keys = section_lines(s, documents)
        clauses = [r for r in major if r['document'] == name
                   and int(r['line_start']) <= end and int(r['line_end']) >= start]
        supplement_ids = sorted({supplemental[k]['supplemental_review'] for k in keys if k in supplemental})
        if any(not records[k]['major_clause'] and k not in supplemental for k in keys):
            raise ValueError('Source line has no semantic review route')
        rows.append(dict(document=name, source_sha256=DOCUMENTS[name],
            section_start=start, section_end=end, title=documents[name][start-1],
            nonblank_lines=len(keys), major_clause_ids='|'.join(r['requirement_id'] for r in clauses),
            major_status_counts_json=json.dumps(dict(Counter(r['status'] for r in clauses)), sort_keys=True),
            supplemental_review_ids='|'.join(supplement_ids),
            numeric_fixture_available=name == 'docx_goal.txt' and start in formula_starts,
            disposition='navigation_with_scoped_reviews', individual_completion_proven=False))
    if len(rows) != 78 or sum(r['nonblank_lines'] for r in rows) != 2586:
        raise ValueError('Original specification coverage changed')
    for rel in ('benchmarks/new_bank_v3/current_requirement_review_v2.py',
                'tests/test_current_requirement_review_v2.py'):
        bind(rel)
    remaining = [
        ('own_cohort', 'Independent multi-user, multi-day, re-donning eight-channel recordings',
         'Song is one person on one day; public axes cannot establish own-device cohort efficacy.'),
        ('physical_device', 'Available hardware, packet/timestamp truth and synchronized end-to-end measurement',
         'Software replay and Qt lifecycle checks do not measure physical throughput, clocks or live accuracy.'),
        ('body_frame', 'Measured IMU units/rate/order, anatomical forward direction and neutral calibration',
         'V3 validates declarations and numerical transforms; strict native calibrated F6 remains ineligible.'),
        ('electrode_topology', 'Authoritative physical electrode-to-channel layout',
         'Unattested or partly inferred ring order does not establish physical spatial features.'),
        ('fault_truth', 'Independent natural/controlled quality-fault labels and ADC/packet metadata',
         'Synthetic perturbations and finite raw observations do not establish physical fault sensitivity or false rejection.'),
        ('eligible_trial_budgets', 'Enough independent calibration trials and complete sequences for each selected native variant',
         'Dependent windows are not extra shots; unsupported high-dimensional covariance budgets stay ineligible.'),
        ('prospective_evaluation', 'Fresh independent held-out evaluation after a new version is frozen',
         'Previously inspected cohorts remain retrospective; finished negative studies are not missing implementation.'),
        ('secondary_provenance', 'Resolve secondary-source terms, physical layout and remaining full-paper provenance',
         'Current discovery retains secondary review and DS2 license conflicts; label resolution is not a redistribution license.'),
    ]
    result = dict(schema='current_requirement_review_v2', documents=DOCUMENTS,
        sections=rows, nonempty_source_lines=2586, major_clauses=major,
        major_status_counts=dict(Counter(r['status'] for r in major)),
        supplemental_line_reviews=trace['supplemental_reviews'],
        formula_evidence=dict(fixture_sections=numeric['fixture_sections'],
            numerical_suite_summary=numeric['suite_summary'], junit_counts=counts,
            junit_counts_include_subtests=True, appendix_coverage=appendix,
            numerical_tests_are_not_native_acceptance=True),
        families=families,
        historical_supersession=dict(old_code_wait_is_current_blocker=False,
            ds2_force_label_wait_is_current_blocker=False,
            ds2_status=ds2['status'], ds2_eligible_trials=ds2['eligible_subject_gesture_force_trials'],
            original_chronology_reconstructed=False, historical_audits_preserved=True),
        current_native_outcomes=dict(document_reliability_primary_pass=reliability['primary_pass'],
            precision_detector_primary_pass=precision['detector_primary_pass'],
            precision_joint_primary_pass=precision['within_detector_joint_pass'],
            units_physical_validation_proven=units['physical_validation_proven'],
            negative_completed_experiments_are_not_unimplemented=True),
        remaining_conditions=[dict(id=i, needed=n, reason=r) for i,n,r in remaining],
        discovery_remaining=discovery['remaining'], source_sha256=bound,
        native_experiment_rerun=False, default_promoted=False, completion_proven=False,
        scope='Current evidence navigation, not exhaustive per-line scientific acceptance. '
              'Historical section statuses are superseded only as current navigation; their files remain unchanged. '
              'Major scoped reviews and 77 supplemental lines are retained without promoting their scope. '
              'Numerical fixtures do not establish native efficacy, universal positive gain or physical validation. '
              'Remaining conditions do not shrink the original two-document objective.')
    return result


def export(root, documents_root):
    root = Path(root)
    result = build(root, documents_root)
    output = root / 'feature_bank/CURRENT_REQUIREMENT_REVIEW_V2.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with output.with_suffix('.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result['sections'][0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(result['sections'])
    print(json.dumps(dict(sections=len(result['sections']), source_lines=result['nonempty_source_lines'],
                         bound_paths=len(result['source_sha256']), completion_proven=False)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('documents_root', type=Path)
    args = parser.parse_args()
    export(ROOT, args.documents_root)
