"""Index every nonblank specification line without claiming scientific completion.

The index is an audit queue, not an automatic requirement classifier. Formula
continuations, examples and prohibitions remain in the queue so none disappear
because a heading or keyword heuristic failed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def build(documents, output):
    output.mkdir(parents=True, exist_ok=True)
    rows, sources = [], []
    for document in documents:
        raw = document.read_bytes()
        lines = raw.decode('utf-8-sig').splitlines()
        digest = hashlib.sha256(raw).hexdigest()
        selected = []
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            selected.append(number)
            rows.append(dict(document=document.name, source_sha256=digest,
                line=number, text=line, verification_status='unverified',
                evidence='', evidence_boundary='Requires contextual requirement-by-requirement review; inclusion is not completion.'))
        sources.append(dict(document=document.name, source_sha256=digest,
            total_lines=len(lines), indexed_nonblank_lines=len(selected),
            omitted_nonblank_lines=0))
    target = output / 'DOCUMENT_SCOPE_QUEUE.csv'
    with target.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Read back actual output; check exact ordering, Unicode and source coverage.
    with target.open(encoding='utf-8', newline='') as handle:
        actual = list(csv.DictReader(handle))
    expected = [{key: str(value) for key, value in row.items()} for row in rows]
    if actual != expected:
        raise AssertionError('Document line coverage or exact text changed')
    audit = dict(status='source_index_verified_scientific_completion_unproven',
        documents=sources, indexed_rows=len(rows),
        queue_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        limitation='Every nonblank line is indexed, including context and separators. Rows are not independent requirements. No existing test or summary status is promoted to completion.')
    (output / 'DOCUMENT_SCOPE_AUDIT.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'indexed_rows': len(rows), 'documents': len(sources),
        'status': audit['status']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('documents', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build(args.documents, args.output)
