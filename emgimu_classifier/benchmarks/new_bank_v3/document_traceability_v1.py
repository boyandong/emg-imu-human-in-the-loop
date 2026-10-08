"""Lossless source-line coverage, separate from scientific acceptance claims."""
import argparse
import hashlib
import json
from pathlib import Path
from benchmarks.new_bank_v2.v1_acceptance_audit import CLAUSES, DOCUMENTS

ROOT = Path(__file__).resolve().parents[2]

# Manual semantic review of the source text omitted by the major clause matrix.
# These dispositions preserve scientific limits; they do not inherit "passed".
SUPPLEMENTAL = (
    ('PRE-INTRO', 'docx_pre.txt', 1, 15, 'superseded_phase_history',
     ['benchmarks/discovery/CURRENT_DISCOVERY_STATE.json'],
     'Discovery infrastructure exists. Original no-training phase chronology is historical; user now authorizes new independent experiments. No retrospective chronology proof.'),
    ('PRE-FAILURES', 'docx_pre.txt', 1126, 1126, 'verified_scoped',
     ['benchmarks/discovery/BENCHMARK_SELECTION_REPORT.md'],
     'The failed UniBo master download and successful replacement are recorded; this is not a complete historical network-failure log.'),
    ('PRE-LICENSE', 'docx_pre.txt', 1127, 1127, 'partial_active',
     ['benchmarks/discovery/CURRENT_DISCOVERY_STATE.json', 'benchmarks/discovery/SECONDARY_PRIMARY_REVIEW_V1.json'],
     'License and access problems are recorded, including unresolved secondary data terms and DS2 conflicting labels; no universal redistribution determination.'),
    ('PRE-DIGEST', 'docx_pre.txt', 1128, 1128, 'verified_scoped',
     ['benchmarks/discovery/CORE_ARCHIVE_DIGEST_AUDIT.json', 'benchmarks/discovery/DS2_V9_FORCE_LABEL_AUDIT.json'],
     'Core archives have full recorded digests; v9 force files and exact joins are separately bound. Current review does not reread every multi-GB raw byte.'),
    ('PRE-SANITY', 'docx_pre.txt', 1129, 1129, 'verified_scoped',
     ['benchmarks/discovery/SANITY_AUDIT.json', 'benchmarks/discovery/electrode_replacement_native_sanity.json'],
     'Native sample sanity reports include rejected missing fields. Sample checks do not prove full population or physical sensor quality.'),
    ('PRE-ORDER', 'docx_pre.txt', 1130, 1133, 'superseded_phase_history',
     ['benchmarks/discovery/CURRENT_DISCOVERY_STATE.json'],
     'Historical stage ordering is not an unmet request to repeat discovery without training. New independent experiments are explicitly authorized.'),
    ('GOAL-INTRO', 'docx_goal.txt', 1, 8, 'context',
     ['feature_bank/REPORT.md'],
     'Introduction and section separators; substantive research criteria reviewed below.'),
    ('GOAL-PRIORS', 'docx_goal.txt', 9, 20, 'historical_prior_not_new_evidence',
     ['feature_bank/REPORT.md'],
     'User-supplied X1/CES/RLCS/X2 statements are research priors, not reproduced evidence. User superseded unavailable legacy code.'),
    ('GOAL-HYPOTHESIS', 'docx_goal.txt', 21, 29, 'hypothesis_not_proven',
     ['benchmarks/new_bank_v2/PAIR_AUDIT.json', 'benchmarks/new_bank_v2/V1_CROSS_AXIS_DECISION_AUDIT.json'],
     'Pairwise interactions and heterogeneous axes tested; full person-force-wearing-posture-day-phase-quality factorial interaction is not proven.'),
    ('GOAL-COMPONENTS', 'docx_goal.txt', 30, 36, 'partial_active',
     ['feature_bank/HARDWARE_PREPARATION_ACCEPTANCE.json', 'benchmarks/new_bank_v3/F8_CALIBRATED_MANUS_V2_RESULTS.json'],
     'Shared families, personal/session calibration and quality APIs exist. Native body-frame, physical mapping and real-fault/live multi-day validation remain unavailable.'),
    ('GOAL-QUESTIONS', 'docx_goal.txt', 37, 45, 'verified_scoped',
     ['feature_bank/REPORT.md', 'benchmarks/new_bank_v2/FULL_V1_LOFO_AUDIT.json', 'benchmarks/new_bank_v2/CONDITIONAL_AUDIT.json'],
     'Failure-specific, conditional, paired-error and calibration questions have bounded public results including negative shortlist/LOFO findings; a universally improved compact bank is not claimed.'),
    ('GOAL-INFO', 'docx_goal.txt', 46, 84, 'verified_scoped',
     ['benchmarks/new_bank_v2/CONDITIONAL_PROTOCOL.json', 'benchmarks/new_bank_v2/CONDITIONAL_VALUE.csv', 'benchmarks/new_bank_v2/CONDITIONAL_AUDIT.json', 'benchmarks/new_bank_v2/PAIR_AUDIT.json'],
     '128 paired concatenation increments across 64 cells implement core-minus-added log loss/Brier and added-minus-core F1, with per-class/per-subject outputs. These are predictive proxies, not direct mutual information; stability is not inferred from one positive pooled delta.'),
    ('GOAL-HOLDOUT', 'docx_goal.txt', 85, 89, 'verified_scoped',
     ['benchmarks/new_bank_v2/V1_REPRODUCIBILITY_AUDIT.json', 'benchmarks/new_bank_v3/F8_CALIBRATED_MANUS_V2_PROTOCOL.json'],
     'Versioned source-fit and disjoint trial/OOF evidence exists. Previously inspected final cohorts are descriptive; this does not prove blind historical execution or exhaustive no-leakage for every older run.'),
    ('GOAL-TAIL', 'docx_goal.txt', 2641, 2642, 'context',
     ['benchmarks/new_bank_v2/V1_CROSS_AXIS_DECISION_AUDIT.json'],
     'Continuation of the preceding family-selection criterion: failure specialization or calibration-dependent complementarity. No new standalone command or artifact.'),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(documents, clauses):
    records = []
    for name, lines in documents.items():
        for number, text in enumerate(lines, 1):
            if not text.strip():
                continue
            covering = [clause[0] for clause in clauses
                        if clause[1] == name and clause[2] <= number <= clause[3]]
            if len(covering) > 1:
                raise ValueError(f'Overlapping clause intervals: {name}:{number}')
            records.append({'document': name, 'line': number, 'text': text,
                            'major_clause': covering[0] if covering else None,
                            'individual_verification': 'unproven'})
    return records


def build(pre, goal):
    paths = {'docx_pre.txt': Path(pre), 'docx_goal.txt': Path(goal)}
    for name, path in paths.items():
        if digest(path) != DOCUMENTS[name]:
            raise ValueError(f'Changed specification: {name}')
    records = inventory({name: path.read_text(encoding='utf-8-sig').splitlines()
                         for name, path in paths.items()}, CLAUSES)
    missing = [record for record in records if record['major_clause'] is None]
    reviewed = []
    for record in missing:
        matching = [entry for entry in SUPPLEMENTAL if entry[1] == record['document']
                    and entry[2] <= record['line'] <= entry[3]]
        if len(matching) != 1:
            raise ValueError(f'Unreviewed or multiply reviewed omitted line: {record["document"]}:{record["line"]}')
        entry = matching[0]
        reviewed.append({**record, 'supplemental_review': entry[0],
                         'disposition': entry[4], 'evidence_scope': entry[6],
                         'evidence_sha256': {path: digest(ROOT / path) for path in entry[5]}})
    acceptance = ROOT / 'feature_bank/NEW_VERSION_ACCEPTANCE_AUDIT.json'
    result = {
        'schema': 'document_traceability_v1',
        'documents': DOCUMENTS,
        'source_sha256': {
            'benchmarks/new_bank_v3/document_traceability_v1.py': digest(Path(__file__)),
            'tests/test_document_traceability_v1.py': digest(ROOT / 'tests/test_document_traceability_v1.py'),
            'benchmarks/new_bank_v2/v1_acceptance_audit.py': digest(ROOT / 'benchmarks/new_bank_v2/v1_acceptance_audit.py'),
            'feature_bank/NEW_VERSION_ACCEPTANCE_AUDIT.json': digest(acceptance)},
        'nonempty_lines': len(records),
        'mapped_lines': len(records) - len(missing),
        'unmapped_lines': len(missing),
        'records': records,
        'unmapped_records': missing,
        'supplemental_reviews': reviewed,
        'unmapped_without_semantic_review': 0,
        'completion_proven': False,
        'scope': 'Every nonempty source line is retained exactly once. Major-clause coverage is a navigation aid, not individual requirement verification. All lines omitted by that matrix now have explicit supplemental semantic review, including scoped evidence, background and open requirements. Mapped lines still remain individually unproven; supplemental review does not establish complete scientific acceptance.'}
    output = ROOT / 'feature_bank/DOCUMENT_TRACEABILITY_V1.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({key: result[key] for key in
                      ('nonempty_lines', 'mapped_lines', 'unmapped_lines', 'completion_proven')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pre')
    parser.add_argument('goal')
    args = parser.parse_args()
    build(args.pre, args.goal)
