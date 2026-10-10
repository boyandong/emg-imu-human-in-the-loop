import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pytest
from emgimu.feature_bank.class_transition_policy_v1 import stream_from_frozen_class_policy

ROOT=Path(__file__).resolve().parents[1]


def inputs():
    folder=ROOT/'benchmarks/new_bank_v3'
    bank=pickle.loads((folder/'roam_native_joint_v1/source_bank.pkl').read_bytes())
    path=folder/'roam_class_transition_v1/source/policy.json'
    kwargs=dict(expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_recording_ids=('source',),
        forbidden_recording_ids=('cal',),user_id='portable',rest_label='relax',sample_rate_hz=200.,
        channel_ids=tuple(f'emg{i}' for i in range(8)),source_channel_ids=tuple(f'emg{i}' for i in range(8)),
        preprocessing_id='roam_native_myo200_no_added_filter_v1',source_preprocessing_id='roam_native_myo200_no_added_filter_v1')
    return bank,path,kwargs


def test_portable_source_policy_loads_without_archive_labels_or_target_profiles():
    bank,path,kwargs=inputs();before=pickle.dumps(bank)
    s=stream_from_frozen_class_policy(bank,path,**kwargs)
    assert s.detector.confirmations==5 and s.detector.smoothing==1 and s.detector.confidence==0.
    assert s.feed(np.zeros((100,8),np.float32),0,recording_id='query')==[]
    source=json.loads(path.read_text())['source_query_recordings'][0]
    with pytest.raises(ValueError,match='Disjoint'):s.feed(np.zeros((100,8)),0,recording_id=source)
    assert pickle.dumps(bank)==before


def test_policy_checksum_parameter_override_and_incompatible_bank_are_rejected():
    bank,path,kwargs=inputs()
    with pytest.raises(ValueError,match='checksum'):stream_from_frozen_class_policy(bank,path,**dict(kwargs,expected_sha256='0'*64))
    with pytest.raises(ValueError,match='overridden'):stream_from_frozen_class_policy(bank,path,**kwargs,confirmations=1)
    bank.bank_id_='different'
    with pytest.raises(ValueError,match='bank'):stream_from_frozen_class_policy(bank,path,**kwargs)
