import hashlib
import json
from pathlib import Path
import numpy as np


def test_native_f8_descriptor_inventory_and_frozen_source_evidence():
    root=Path(__file__).resolve().parents[1]
    result=json.loads((root/'benchmarks/new_bank_v3/F8_DOCUMENT_MANUS_RESULTS.json').read_text())
    assert result['profiles']==24
    assert result['dimension']==180
    assert not result['ring_enabled']
    assert not result['evaluation_windows_used_for_descriptor']
    assert not result['classifier_fit']
    for path,digest in result['source_hashes'].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest
    records=result['records']
    assert {(r['phase'],r['user'],r['shots']) for r in records}=={
        (phase,user,shots) for phase in ('validation','descriptive_final')
        for user in range(3,9) for shots in (1,2)}
    names=records[0]['descriptor']['phi_feature_names']
    assert len(names)==len(set(names))==180
    for row in records:
        source=set(row['source_trials']);cal=set(row['calibration_trials']);evaluation=set(row['evaluation_trials'])
        assert not(source&cal or source&evaluation or cal&evaluation)
        assert len(cal)==6*row['shots']
        descriptor=row['descriptor']
        assert descriptor['user_id']==str(row['user'])
        assert descriptor['source_sessions']==['1']
        assert descriptor['current_sessions']==(['2'] if row['phase']=='validation' else ['3'])
        assert descriptor['phi_feature_names']==names
        vector=np.asarray(descriptor['phi_session'])
        assert vector.shape==(180,) and np.isfinite(vector).all()
        for label,values in descriptor['classes'].items():
            expected=values['channel_quality_score_shift']
            positions=[names.index(f'F8.channel_quality_shift.{label}.ch{c}') for c in range(1,9)]
            np.testing.assert_allclose(vector[positions],expected)
            assert vector[names.index(f'F8.spd_shift.{label}')]==values['affine_invariant_trace_covariance_distance']
