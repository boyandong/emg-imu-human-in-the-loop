"""Current evidence hashes and real JUnit records, not filename-only coverage."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]


def test_formula_register_uses_current_source_bytes_and_actual_passing_fixture_run():
    register=json.loads((ROOT/'feature_bank/FORMULA_NUMERICAL_ACCEPTANCE.json').read_text(encoding='utf8'))
    clauses={r['clause_start']:r for r in register['additional_clause_fixtures']}
    assert set(clauses)=={2310,2370,2399,2514}
    assert clauses[2514]['clause_end']==2560
    assert register['fixture_sections']==32 and register['suite_exit_code']==0
    for row in register['sections']+list(clauses.values()):
        for key in ('test_sha256','reviewed_source_sha256'):
            for name,digest in row[key].items():
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        assert not row['scientific_completion']
    junit=ROOT/register['regression_junit']['path']
    assert hashlib.sha256(junit.read_bytes()).hexdigest()==register['regression_junit']['sha256']
    suites=list(ET.parse(junit).getroot().iter('testsuite'))
    assert all(int(s.attrib[k])==0 for s in suites for k in ('failures','errors','skipped'))
    tests=[r.attrib['name'] for r in ET.parse(junit).getroot().iter('testcase')]
    assert 'test_independent_source_quantile_product_and_four_summaries_with_exact_boundaries' in tests
    assert 'test_rational_quality_mixture_unknown_fallback_tiny_positive_and_persistence' in tests
    assert 'test_document_force_envelope_known_waveforms_and_within_gesture_spread' in tests
    assert 'test_document_personal_normalizer_uses_raw_q95_plus_epsilon' in tests
    assert not register['completion_proven']


def test_quality_verification_is_discoverable_without_promoting_accuracy_or_completion():
    index=json.loads((ROOT/'feature_bank/delivery/INDEX.json').read_text(encoding='utf8'))
    for key in ('available_quality_fusion_acceptance','formula_numerical_acceptance'):
        row=index[key];path=(ROOT/'feature_bank/delivery'/row['path']).resolve()
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    conclusions=json.loads((ROOT/'feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json').read_text(encoding='utf8'))
    quality=conclusions['available_quality_fusion']
    assert quality['verification_cases']==288 and quality['rejected_invalid_quality_calls']==96
    assert not quality['default_promoted'] and not quality['physical_validation_proven']
    assert not conclusions['completion_proven']
    report=(ROOT/'feature_bank/REPORT.md').read_text(encoding='utf8')
    assert 'Available-provider quality fusion V2' in report and 'synthetic quality-one/zero' in report
