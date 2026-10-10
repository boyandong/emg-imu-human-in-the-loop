import json
from pathlib import Path

from benchmarks.new_bank_v3.verify_available_bank_quality_v2 import verify

ROOT=Path(__file__).resolve().parents[1]


def test_frozen_native_probability_algebra_and_source_state_are_independently_verified():
    saved=json.loads((ROOT/'feature_bank/AVAILABLE_BANK_QUALITY_ACCEPTANCE_V2.json').read_text(encoding='utf8'))
    assert verify()==saved
    assert saved['verification_cases']==288 and saved['rejected_invalid_quality_calls']==96
    assert saved['maximum_probability_error']<1e-12 and len(saved['records'])==24
    assert all(r['source_and_fusion_state_immutable'] for r in saved['records'])
    assert not saved['default_promoted'] and not saved['physical_validation_proven'] and not saved['completion_proven']
