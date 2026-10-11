"""Source-selected document D/E weights around immutable native joint profiles.

The source bank and old profiles remain unchanged. Exact long-term reliability
shrinks toward the source population; current-session reliability shrinks toward
that new long-term prior. Existing F7/F8/path readouts retain their identities.
"""
from dataclasses import dataclass,asdict,replace
import hashlib
import json
from pathlib import Path
import numpy as np
from .document_reliability_v2 import DocumentReliabilityWeightsV2
from .native_bout_window_adapter_v2 import native_windows
from .native_joint_bout_workflow_v2 import NativeJointBoutWorkflowV2


def policy_digest(body):
    return hashlib.sha256(json.dumps({k:v for k,v in body.items() if k!='policy_id'},
        sort_keys=True,separators=(',',':')).encode()).hexdigest()


def calibration_features(bank,profile,user_id):
    batch,ids,offsets,_=native_windows(profile.calibration,bank.sensor_contract_[1],round(.04*bank.sensor_contract_[0]))
    read=bank.predict_providers(batch,ids,window_offsets=offsets,user_id=user_id)
    labels=dict(profile.calibration_labels)
    axis=read['trial_ids'];y=np.array([labels[t] for t in axis])
    return {g:(read['source_standardized_features'][g],y,axis) for g in bank.providers_}


def hierarchical_weights(classes,providers,population,n0,temperature,long,current=None,*,source_policy_id,forbidden_trials=()):
    first=DocumentReliabilityWeightsV2(tuple(classes),tuple(providers),tuple(population),n0,temperature,source_policy_id)
    long_result=first.calculate(long,forbidden_trial_ids=forbidden_trials)
    current_result=None;weights=long_result['final']
    if current is not None:
        long_ids=next(iter(long.values()))[2]
        second=DocumentReliabilityWeightsV2(tuple(classes),tuple(providers),tuple(weights),n0,temperature,source_policy_id+':current')
        current_result=second.calculate(current,forbidden_trial_ids=tuple(forbidden_trials)+tuple(long_ids))
        weights=current_result['final']
    return weights,long_result,current_result


def compose_views(raw,decision,dtw,signature,weights,routing,providers):
    """Probability-only composition; no labels or parameter fitting."""
    providers=tuple(providers);w=np.asarray(weights,float);r=np.asarray(routing,float)
    if (tuple(raw)!=providers or tuple(decision)!=providers or w.shape!=(len(providers),)
            or r.shape!=w.shape or not np.isfinite(w).all() or not np.isfinite(r).all()
            or np.any(w<0) or not np.isclose(w.sum(),1.,atol=1e-12,rtol=0) or np.any(r<=0)):
        raise ValueError('Aligned normalized document weights and positive routing required')
    arrays=[np.asarray(v,float) for mapping in (raw,decision) for v in mapping.values()]+[np.asarray(dtw,float),np.asarray(signature,float)]
    shape=arrays[0].shape
    if (len(shape)!=2 or shape[1]<2 or any(v.shape!=shape or not np.isfinite(v).all() or np.any(v<0)
            or not np.allclose(v.sum(1),1.,atol=1e-12,rtol=0) for v in arrays)):
        raise ValueError('Every source/anchor/path probability needs the same normalized row/class axis')
    routed=w*r;routed/=routed.sum()
    reliability=sum(w[i]*np.asarray(raw[g]) for i,g in enumerate(providers));reliability/=reliability.sum(1,keepdims=True)
    window=sum(routed[i]*np.asarray(decision[g]) for i,g in enumerate(providers));window/=window.sum(1,keepdims=True)
    joint=.75*window+.125*np.asarray(dtw)+.125*np.asarray(signature);joint/=joint.sum(1,keepdims=True)
    return dict(document_reliability=reliability,document_window=window,document_joint=joint,
                document_weights=tuple(w),routed_weights=tuple(routed))


@dataclass(frozen=True)
class NativeReliabilityStateV3:
    policy_id:str
    bank_id:str
    user_id:str
    personal_profile_id:str
    session_profile_id:str|None
    session_id:str
    calibration_trial_ids:tuple
    calibration_recording_ids:tuple
    long_trials:int
    current_trials:int
    weights:tuple
    routing:tuple
    state_id:str=''


def state_digest(state):
    return hashlib.sha256(json.dumps({k:v for k,v in asdict(state).items() if k!='state_id'},
        sort_keys=True,separators=(',',':')).encode()).hexdigest()


class NativeDocumentReliabilityV3:
    def __init__(self,workflow,policy_path,*,expected_sha256):
        if not isinstance(workflow,NativeJointBoutWorkflowV2):raise ValueError('Frozen native joint workflow required')
        data=Path(policy_path).read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected_sha256:raise ValueError('Source reliability policy checksum differs')
        p=json.loads(data);bank=workflow.window.bank
        if (p.get('schema')!='native_document_reliability_policy_v3' or p.get('policy_id')!=policy_digest(p)
                or p.get('inference_bank_id')!=bank.bank_id_ or tuple(p.get('providers',()))!=bank.providers_
                or tuple(p.get('classes',()))!=bank.classes_ or tuple(p.get('sensor_contract',()))!=bank.sensor_contract_
                or tuple(p.get('channel_ids',()))!=workflow.window.channels
                or p.get('preprocessing_id')!=workflow.window.preprocessing_id
                or not np.array_equal(p.get('population'),bank.policy_.population)):
            raise ValueError('Exact source policy/bank/provider/class/sensor contract differs')
        records=p.get('source_query_recording_ids',[])
        if not records or len(records)!=len(set(records)):raise ValueError('Policy source recording provenance required')
        DocumentReliabilityWeightsV2(bank.classes_,bank.providers_,tuple(p['population']),p['n0'],p['temperature'],p['policy_id'])
        self.workflow=workflow;self.bank=bank;self.policy=p;self.policy_id=p['policy_id'];self.source_records=frozenset(records)

    def prepare_state(self,*,personal,user_id,session_id,session=None):
        self.workflow._profile(personal,user_id)
        if session is not None:self.workflow._profile(session,user_id,personal=personal,session_id=session_id)
        profiles=(personal,) if session is None else (personal,session)
        records=tuple(sorted({r for p in profiles for r in p.calibration.recording_ids}))
        if self.source_records.intersection(records):raise ValueError('Source policy-selection recordings cannot calibrate a target')
        long=calibration_features(self.bank,personal,user_id)
        current=None if session is None else calibration_features(self.bank,session,user_id)
        weights,lr,cr=hierarchical_weights(self.bank.classes_,self.bank.providers_,self.policy['population'],
            self.policy['n0'],self.policy['temperature'],long,current,source_policy_id=self.policy_id,
            forbidden_trials=self.bank.policy_.source_trials)
        routing=np.ones(len(weights)) if session is None else session.window.decision.routing['multipliers']
        trials=tuple(sorted({t for p in profiles for t in p.calibration.trial_ids}))
        state=NativeReliabilityStateV3(self.policy_id,self.bank.bank_id_,user_id,personal.profile_id,
            None if session is None else session.profile_id,session_id,trials,records,lr['n_cal_trials'],
            0 if cr is None else cr['n_cal_trials'],tuple(weights),tuple(routing))
        return replace(state,state_id=state_digest(state))

    def compose(self,state,raw,decision,dtw,signature,*,trial_ids,recording_ids,user_id,session_id,provider_trial_ids,provider_classes):
        ids=tuple(trial_ids);records=tuple(recording_ids)
        if (not isinstance(state,NativeReliabilityStateV3) or state.policy_id!=self.policy_id
                or state.state_id!=state_digest(state)
                or state.bank_id!=self.bank.bank_id_ or state.user_id!=user_id or state.session_id!=session_id
                or not ids or len(ids)!=len(set(ids)) or len(ids)!=len(records)
                or any(not isinstance(v,str) or not v.strip() for v in (*ids,*records))
                or set(provider_trial_ids)!=set(self.bank.providers_)
                or set(provider_classes)!=set(self.bank.providers_)
                or any(tuple(v)!=ids for v in provider_trial_ids.values())
                or any(tuple(v)!=self.bank.classes_ for v in provider_classes.values())
                or set(ids)&(set(state.calibration_trial_ids)|set(self.bank.policy_.source_trials))
                or set(records)&(set(state.calibration_recording_ids)|set(self.source_records))):
            raise ValueError('Disjoint query/profile/user/session and exact provider row identities required')
        result=compose_views(raw,decision,dtw,signature,state.weights,state.routing,self.bank.providers_)
        if result['document_joint'].shape!=(len(ids),len(self.bank.classes_)):raise ValueError('Query row/class axis differs')
        result.update(trial_ids=ids,classes=self.bank.classes_,policy_id=self.policy_id,
            calibration_trials=state.long_trials+state.current_trials,long_calibration_trials=state.long_trials,
            current_calibration_trials=state.current_trials,default_promoted=False,physical_validation_proven=False)
        return result

    def predict(self,batch,*,personal,user_id,session_id,session=None,state=None):
        if state is None:state=self.prepare_state(personal=personal,session=session,user_id=user_id,session_id=session_id)
        if (not isinstance(state,NativeReliabilityStateV3) or state.personal_profile_id!=personal.profile_id
                or state.session_profile_id!=(None if session is None else session.profile_id)):
            raise ValueError('Derived reliability state belongs to different registration profiles')
        old=self.workflow.predict(batch,personal=personal,session=session,user_id=user_id,session_id=session_id)
        windows,ids,offsets,_=native_windows(batch,self.bank.sensor_contract_[1],round(.04*self.bank.sensor_contract_[0]))
        read=self.bank.predict_providers(windows,ids,window_offsets=offsets,user_id=user_id)
        order=[read['trial_ids'].index(t) for t in batch.trial_ids]
        raw={g:q[order] for g,q in read['probabilities'].items()}
        index=[old['window']['trial_ids'].index(t) for t in batch.trial_ids]
        decision={g:q[index] for g,q in old['window']['decision_provider_probabilities'].items()}
        paths=old['temporal']['arms']
        return self.compose(state,raw,decision,paths['DTW_blended'],paths['signature_blended'],
            trial_ids=batch.trial_ids,recording_ids=batch.recording_ids,user_id=user_id,session_id=session_id,
            provider_trial_ids={g:batch.trial_ids for g in self.bank.providers_},provider_classes={g:self.bank.classes_ for g in self.bank.providers_})
