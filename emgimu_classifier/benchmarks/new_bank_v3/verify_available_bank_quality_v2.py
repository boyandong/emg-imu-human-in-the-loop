"""Read-only algebra replay on already frozen MANUS provider probabilities.

No raw signals, classifier fitting, label scoring, tuning or native experiment
rerun. Quality-one/zero inputs are explicitly software fixtures. They do not
represent measured native signal quality or a physical fault experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from emgimu.feature_bank.available_bank_fusion_v1 import AvailableBankFusionPolicyV1
from emgimu.feature_bank.available_bank_quality_fusion_v2 import AvailableBankQualityFusionV2

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PROTOCOL=HERE/'AVAILABLE_BANK_QUALITY_V2_PROTOCOL.json'
OUT=ROOT/'feature_bank/AVAILABLE_BANK_QUALITY_ACCEPTANCE_V2.json'
NATIVE=HERE/'F8_CALIBRATED_MANUS_V2_RESULTS.json'
NAMES=('TD24','pattern','SPD','log_bands')
SOURCES=('src/emgimu/feature_bank/available_bank_quality_fusion_v2.py',
         'src/emgimu/feature_bank/available_bank_fusion_v1.py',
         'src/emgimu/feature_bank/document_reliability_v2.py',
         'src/emgimu/feature_bank/quality_mask_v1.py',
         'tests/test_available_bank_quality_fusion_v2.py',
         'tests/test_quality_mask_product_oracle_v2.py',
         'tests/test_available_bank_quality_v2_delivery.py',
         'benchmarks/new_bank_v3/verify_available_bank_quality_v2.py')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if PROTOCOL.exists():raise FileExistsError('Existing verification protocol is immutable')
    payload=dict(schema='available_bank_quality_v2_protocol',
        source_sha256={p:sha(ROOT/p) for p in SOURCES},
        native_probability_artifact_sha256={NATIVE.relative_to(ROOT).as_posix():sha(NATIVE)},
        purpose='Read-only algebra verification of frozen probabilities, not a new efficacy experiment.',
        cases=['unobserved quality identity','unit quality identity','four provider omissions',
               'four synthetic single-zero quality fixtures','synthetic all-zero Unknown',
               'partial unavailable quality observation'],
        forbidden='No source refit, native raw-signal replay, target labels, temperature/threshold/weight selection or device claim.',
        default_promoted=False,completion_proven=False)
    PROTOCOL.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8',newline='\n')
    return payload


def independent_mixture(weights,probabilities,indices):
    # Deliberately scalar accumulation, separate from production tensordot/einsum.
    arrays=[np.asarray(q,float) for q in probabilities]
    denominator=sum(float(weights[i]) for i in indices)
    result=np.zeros_like(arrays[0])
    for row in range(len(result)):
        for label in range(result.shape[1]):
            result[row,label]=sum(float(weights[i])*float(arrays[i][row,label])
                                  for i in indices)/denominator
    return result


def verify():
    protocol=json.loads(PROTOCOL.read_text(encoding='utf8'))
    for path,digest in {**protocol['source_sha256'],**protocol['native_probability_artifact_sha256']}.items():
        assert sha(ROOT/path)==digest,path
    native=json.loads(NATIVE.read_text(encoding='utf8'))
    records=[];case_count=0;probability_values=0;maximum=0.;invalid=0
    for block in native['blocks']:
        source=AvailableBankFusionPolicyV1(tuple(range(6)),NAMES,(.25,)*4,1.,1.,
            'imported_native_f8:'+native['protocol_sha256'],tuple(sorted(set(block['source_trials']))))
        policy=AvailableBankQualityFusionV2(source,'software-fixture-only:no-measured-quality')
        state=source.from_fitted_weights(block['weights'],calibration_trials=tuple(block['calibration_trials']))
        probabilities=dict(zip(NAMES,np.asarray(block['providers'],float)))
        trials=tuple(block['evaluation_trials']);before=pickle.dumps((policy,state,probabilities))
        def call(values,quality=None,quality_axes=None,quality_id=None):
            return policy.predict(state,values,evaluation_trials=trials,
                provider_trial_ids={n:trials for n in values},provider_classes={n:tuple(range(6)) for n in values},
                quality=quality,quality_trial_ids=({n:trials for n in quality} if quality_axes is None and quality is not None else quality_axes),
                quality_policy_id=(policy.quality_policy_id if quality_id is None and quality is not None else quality_id))
        def check(actual,expected):
            nonlocal maximum,case_count,probability_values
            error=float(np.max(abs(actual['probabilities']-expected)))
            assert error<1e-12
            assert np.isfinite(actual['effective_weights']).all()
            assert np.max(abs(actual['effective_weights'].sum(1)-1))<1e-12
            maximum=max(maximum,error);case_count+=1;probability_values+=expected.size
        base=call(probabilities)
        check(base,np.asarray(block['probabilities']['F8']))
        assert not any(base['quality_available']) and base['quality_unavailable']==NAMES
        assert not base['rejected'].any()
        unit=call(probabilities,{n:np.ones(len(trials)) for n in NAMES})
        check(unit,independent_mixture(block['weights'],block['providers'],range(4)))
        assert all(unit['quality_available']) and not unit['rejected'].any()
        for drop in NAMES:
            remaining=[i for i,n in enumerate(NAMES) if n!=drop]
            expected=independent_mixture(block['weights'],block['providers'],remaining)
            omitted=call({n:q for n,q in probabilities.items() if n!=drop})
            check(omitted,expected)
            assert omitted['missing_at_prediction']==(drop,) and not omitted['rejected'].any()
            quality={n:np.full(len(trials),float(n!=drop)) for n in NAMES}
            zero=call(probabilities,quality)
            check(zero,expected)
            assert not zero['missing_at_prediction'] and not zero['rejected'].any()
        rejected=call(probabilities,{n:np.zeros(len(trials)) for n in NAMES})
        check(rejected,base['probabilities'])
        assert rejected['rejected'].all() and set(rejected['labels'])=={'Unknown'}
        assert set(rejected['rejection_reason'])=={'all_quality_rejected'}
        partial=call(probabilities,{NAMES[0]:np.zeros(len(trials))})
        check(partial,independent_mixture(block['weights'],block['providers'],range(1,4)))
        assert partial['quality_available']==(True,False,False,False)
        assert partial['quality_unavailable']==NAMES[1:] and not partial['rejected'].any()
        # Wrong quality rows/policy/value and unknown provider must fail even
        # when the probability matrix itself is a valid frozen native matrix.
        for quality,axes,identity in (
            ({NAMES[0]:np.ones(len(trials))},{NAMES[0]:tuple(reversed(trials))},policy.quality_policy_id),
            ({NAMES[0]:np.ones(len(trials))},{NAMES[0]:trials},'different-quality-policy'),
            ({NAMES[0]:np.full(len(trials),1.01)},{NAMES[0]:trials},policy.quality_policy_id),
            ({'unknown':np.ones(len(trials))},{'unknown':trials},policy.quality_policy_id)):
            try:call(probabilities,quality,axes,identity)
            except ValueError:invalid+=1
            else:raise AssertionError('Invalid quality contract was accepted')
        assert before==pickle.dumps((policy,state,probabilities))
        records.append(dict(phase=block['phase'],user=block['user'],shots_per_class=block['shots'],
            native_evaluation_trials=len(trials),cases=12,source_and_fusion_state_immutable=True,
            weights_mode=state.mode,quality_inputs='Explicit software fixtures; no measured native quality.'))
    assert len(records)==24 and case_count==288 and invalid==96
    return dict(schema='available_bank_quality_acceptance_v2',protocol_sha256=sha(PROTOCOL),
        source_sha256=protocol['source_sha256'],native_probability_artifact_sha256=protocol['native_probability_artifact_sha256'],
        records=records,verification_cases=case_count,checked_probability_values=probability_values,
        rejected_invalid_quality_calls=invalid,maximum_probability_error=maximum,
        default_promoted=False,physical_validation_proven=False,completion_proven=False,
        scope='Frozen public native probabilities verify arithmetic and source-state parity only. Quality scores are synthetic verification inputs; no native quality-effect, F0-F9 representation-completeness, fitted-policy, device or accuracy claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    result=prepare() if args.prepare else verify()
    if not args.prepare:OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    print('Prepared frozen quality-fusion verification' if args.prepare else
          f"Verified {result['verification_cases']} arithmetic cases; max error {result['maximum_probability_error']:.3g}; no refit")
