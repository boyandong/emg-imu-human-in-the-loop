"""Lossless source-line coverage, separate from scientific acceptance claims."""
import argparse
import hashlib
import json
from pathlib import Path
from benchmarks.new_bank_v2.v1_acceptance_audit import CLAUSES, DOCUMENTS

ROOT = Path(__file__).resolve().parents[2]


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
    acceptance = ROOT / 'feature_bank/NEW_VERSION_ACCEPTANCE_AUDIT.json'
    result = {
        'schema': 'document_traceability_v1',
        'documents': DOCUMENTS,
        'source_sha256': {
            'benchmarks/new_bank_v3/document_traceability_v1.py': digest(Path(__file__)),
            'benchmarks/new_bank_v2/v1_acceptance_audit.py': digest(ROOT / 'benchmarks/new_bank_v2/v1_acceptance_audit.py'),
            'feature_bank/NEW_VERSION_ACCEPTANCE_AUDIT.json': digest(acceptance)},
        'nonempty_lines': len(records),
        'mapped_lines': len(records) - len(missing),
        'unmapped_lines': len(missing),
        'records': records,
        'unmapped_records': missing,
        'completion_proven': False,
        'scope': 'Every nonempty source line is retained exactly once. Major-clause coverage is a navigation aid, not individual requirement verification. Unmapped lines require explicit semantic review; mapped lines also remain individually unproven. No status is inherited as scientific acceptance.'}
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
