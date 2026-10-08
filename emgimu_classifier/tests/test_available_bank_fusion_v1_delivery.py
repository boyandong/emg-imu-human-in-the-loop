import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_availability_acceptance_binds_source_and_covers_all_frozen_native_blocks():
    d = json.loads((ROOT/'feature_bank/AVAILABLE_BANK_FUSION_ACCEPTANCE_V1.json').read_text(encoding='utf8'))
    native = json.loads((ROOT/'benchmarks/new_bank_v3/F8_CALIBRATED_MANUS_V2_RESULTS.json').read_text(encoding='utf8'))
    for path,digest in d['source_sha256'].items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    assert len(d['records'])==len(native['blocks'])==d['native_full_cases']==24
    assert d['native_missing_provider_cases']==d['rejected_invalid_native_calls']==96
    assert not d['default_promoted'] and not d['physical_validation_proven'] and not d['completion_proven']
    for row,block in zip(d['records'],native['blocks']):
        assert (row['phase'],row['user'],row['shots_per_class'])==(block['phase'],block['user'],block['shots'])
        assert row['evaluation_trials']==len(block['evaluation_trials'])
        assert row['full_max_probability_error'] < 1e-12 and row['state_immutable']
        omissions=row['missing_provider_cases']; names=('TD24','pattern','SPD','log_bands')
        assert {c['missing_provider'] for c in omissions}==set(names)
        for case in omissions:
            expected_names=[name for name in names if name!=case['missing_provider']]
            expected=np.array([block['weights'][names.index(name)] for name in expected_names]); expected/=sum(expected)
            assert case['remaining_providers']==expected_names
            np.testing.assert_allclose(case['renormalized_weights'],expected,rtol=0,atol=1e-12)
            assert case['max_probability_error'] < 1e-12
