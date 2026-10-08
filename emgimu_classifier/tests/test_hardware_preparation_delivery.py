import hashlib
import json
from pathlib import Path


def test_hardware_free_acceptance_binds_current_inputs_without_physical_claims():
    root = Path(__file__).resolve().parents[1]
    r = json.loads((root/'feature_bank/HARDWARE_PREPARATION_ACCEPTANCE.json').read_text(encoding='utf8'))
    assert r['suite_exit_code'] == 0
    for path,digest in r['source_sha256'].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest() == digest
    assert {item['item'] for item in r['items']} == {5,6,7}
    assert all(item['remaining'] and item['verified_by'] for item in r['items'])
    assert not r['physical_validation_proven'] and not r['completion_proven'] and not r['default_promoted']
