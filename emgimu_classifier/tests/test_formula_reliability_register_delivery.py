import hashlib
import json
from pathlib import Path


def test_reliability_clause_registered_with_current_source_and_fixture_hashes():
    root = Path(__file__).resolve().parents[1]
    register = json.loads((root/'feature_bank/FORMULA_NUMERICAL_ACCEPTANCE.json').read_text(encoding='utf8'))
    clause, = register['additional_clause_fixtures']
    assert (clause['clause_start'], clause['clause_end']) == (2399,2474)
    assert clause['arithmetic_status'] == 'reviewed_fixture_suite_passed'
    assert not clause['scientific_completion'] and not register['completion_proven']
    assert register['generator_sha256'] == hashlib.sha256((root/'benchmarks/formula_numerical_acceptance.py').read_bytes()).hexdigest()
    for key in ('test_sha256','reviewed_source_sha256'):
        for path, digest in clause[key].items():
            assert hashlib.sha256((root/path).read_bytes()).hexdigest() == digest
