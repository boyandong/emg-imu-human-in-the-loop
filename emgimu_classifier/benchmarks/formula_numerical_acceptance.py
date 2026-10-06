"""Run the explicit per-section arithmetic acceptance suite, separate from efficacy."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Reviewed equation fixtures, not inferred from source names or output dimensions.
EVIDENCE = {
    1110: ('test_new_bank_v2.py', 'test_document_scale_v3.py'),
    1274: ('test_spec_spatial_v3.py',),
    1314: ('test_document_signal.py',),
    1365: ('test_spec_spatial_v3.py',),
    1405: ('test_f3a_rlcs_direct_oracle.py',),
    1444: ('test_document_ces_v3.py',),
    1480: ('test_spec_spatial_v3.py',),
    1551: ('test_document_spectral_v3.py',),
    1591: ('test_document_spectral_v3.py',),
    1626: ('test_document_spectral_v3.py',),
    1663: ('test_relative_spectrum.py',),
    1697: ('test_document_temporal_v3.py',),
    1705: ('test_document_temporal_v3.py',),
    1777: ('test_document_path_v3.py',),
    1826: ('test_document_path_v3.py',),
    1867: ('test_body_frame.py',),
    1923: ('test_posture_context_oracle.py',),
    1941: ('test_f7_anchor_known_geometry.py', 'test_document_personal_anchor_v2.py'),
    1968: ('test_f7_anchor_known_geometry.py',),
    1975: ('test_document_personal_anchor_v2.py',),
    1985: ('test_document_personal_anchor_v2.py',),
    1993: ('test_feature_bank.py',),
    2005: ('test_affine_spd_anchor.py',),
    2047: ('test_document_session_v3.py', 'test_session_shift_summary.py'),
    2140: ('test_document_quality_v3.py',),
    2146: ('test_document_quality_v3.py',),
    2169: ('test_document_quality_v3.py',),
    2185: ('test_document_quality_v3.py', 'test_f9_known_signal.py'),
    2205: ('test_document_quality_v3.py', 'test_f9_known_signal.py'),
    2224: ('test_document_quality_v3.py',),
    2244: ('test_document_quality_v3.py',),
    2271: ('test_quality_mask_v1.py', 'test_feature_bank.py'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    import csv
    table = ROOT / 'feature_bank/FORMULA_SUBSECTION_COVERAGE_AUDIT.csv'
    with table.open(encoding='utf-8', newline='') as stream:
        sections = list(csv.DictReader(stream))
    assert {int(s['section_start']) for s in sections if s['mapping'] != 'context_heading'} == set(EVIDENCE)
    additional = {'clause_start': 2399, 'clause_end': 2474,
                  'title': 'Personal feature reliability and population shrinkage (D/E)',
                  'test_paths': ['tests/test_document_reliability_v2.py',
                                 'tests/test_document_reliability_direct_oracle.py'],
                  'source_paths': ['src/emgimu/feature_bank/document_reliability_v2.py']}
    files = sorted({f'tests/{name}' for names in EVIDENCE.values() for name in names}
                   | set(additional['test_paths']))
    for path in files:
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *files], cwd=ROOT,
                            capture_output=True, text=True)
    print(result.stdout, end='')
    if result.returncode:
        print(result.stderr, end='')
        raise SystemExit(result.returncode)
    rows = []
    for section in sections:
        names = EVIDENCE.get(int(section['section_start']), ())
        paths = [f'tests/{name}' for name in names]
        sources = section['source_paths'].split('|') if section['source_paths'] else []
        rows.append({
            'section_start': int(section['section_start']), 'title': section['title'],
            'arithmetic_status': 'reviewed_fixture_suite_passed' if names else 'context_heading',
            'test_sha256': {p: sha(ROOT / p) for p in paths},
            'reviewed_source_sha256': {p: sha(ROOT / p) for p in sources},
            'scope': 'Versioned new implementations; each fixture establishes only its explicit assertions. Historical reproduction, unspecified preprocessing and universal efficacy are not inferred.',
            'scientific_completion': False,
        })
    payload = {
        'schema': 'formula_numerical_acceptance_v1', 'sections': rows,
        'fixture_sections': len(EVIDENCE), 'suite_exit_code': result.returncode,
        'additional_clause_fixtures': [{
            'clause_start': additional['clause_start'], 'clause_end': additional['clause_end'],
            'title': additional['title'],
            'test_sha256': {p: sha(ROOT/p) for p in additional['test_paths']},
            'reviewed_source_sha256': {p: sha(ROOT/p) for p in additional['source_paths']},
            'arithmetic_status': 'reviewed_fixture_suite_passed', 'scientific_completion': False,
            'boundary': 'Known three-class nonzero distances verify B/W, additive epsilon, temperature2, six independent trials and population shrinkage. Does not prove source-CV parameter selection or predictive benefit.'}],
        'suite_summary': result.stdout.strip().splitlines()[-1],
        'inventory_sha256': sha(table), 'generator_sha256': sha(Path(__file__)),
        'completion_proven': False,
        'boundary': 'Equation fixtures and source hashes are review evidence, not an efficacy acceptance. F6 native body-frame, physical electrode order and F9 physical faults remain deferred; F7 high-dimensional budget, autonomous bout boundaries and effective F8 routing are separate tasks.',
    }
    (ROOT / 'feature_bank/FORMULA_NUMERICAL_ACCEPTANCE.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return payload


if __name__ == '__main__':
    run()
