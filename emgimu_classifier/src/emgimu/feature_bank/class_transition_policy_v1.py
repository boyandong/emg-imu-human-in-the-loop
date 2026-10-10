"""Load an exact frozen source policy for label-free native class streaming."""
import hashlib
import json
from pathlib import Path
from .native_class_transition_stream_v1 import NativeClassTransitionStreamV1


def stream_from_frozen_class_policy(bank,policy_path,*,expected_sha256,**kwargs):
    payload=Path(policy_path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected_sha256:
        raise ValueError('Source class-transition policy checksum differs')
    policy=json.loads(payload)
    if (policy.get('schema')!='native_class_transition_policy_v1'
            or policy.get('inference_bank_id')!=bank.bank_id_
            or policy.get('sample_rate_hz')!=bank.sensor_contract_[0]
            or tuple(policy.get('classes',()))!=tuple(bank.classes_)
            or kwargs.get('rest_label')!=policy.get('rest_label','relax')):
        raise ValueError('Frozen policy bank, native rate or class axis differs')
    config=policy.get('config',{})
    if set(config)!={'confirmations','smoothing_windows','confidence'} or set(config)&set(kwargs):
        raise ValueError('Frozen detector parameters cannot be omitted or overridden')
    source=policy.get('source_query_recordings',[])
    if not source or len(set(source))!=len(source):raise ValueError('Source policy-selection recordings required')
    caller_sources=tuple(kwargs.pop('source_recording_ids'))
    # Source policy-selection queries are training inputs for this policy too.
    return NativeClassTransitionStreamV1(bank,source_recording_ids=tuple(sorted(set(source)|set(caller_sources))),**config,**kwargs)
