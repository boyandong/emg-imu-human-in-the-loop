"""Joint complete-action registration for frozen window and temporal branches.

This opt-in workflow does not replace the desktop entry. It reuses the existing
seven-provider source bank, F7/F8 window decision, complete-bout F5 and source
raw-quality gate. One registration is charged once even though both branches
consume it. Full captured native samples are retained in the saved profile.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import pickle
import zipfile

import numpy as np

from .available_bank_fusion_v1 import AvailableBankFusionPolicyV1
from .available_bank_quality_fusion_v2 import AvailableBankQualityFusionV2
from .calibration_rest_detector_v1 import fit_calibration_rest_detector
from .complete_bout_window_adapter_v1 import CompleteBoutWindowAdapterV1
from .core import FeatureBatch
from .extended_window_decision_v1 import ExtendedWindowDecisionV1
from .personal_temporal_bouts_v1 import PersonalTemporalBoutsV1, TemporalBoutBatchV1


@dataclass(frozen=True)
class JointBoutProfileV1:
    contract_id: str
    profile_id: str
    user_id: str
    session_id: str
    kind: str
    parent_profile_id: str | None
    window: object
    temporal: object
    detector: object
    detector_receipt: dict
    calibration: TemporalBoutBatchV1
    raw_calibration: TemporalBoutBatchV1 | None
    calibration_labels: tuple
    calibration_cost: dict


def profile_digest(profile):
    """Hash values, independently of pickle memoization and array strides."""
    def canonical(value):
        if isinstance(value,np.generic):return canonical(value.item())
        if value is None:return ['none']
        if isinstance(value,bool):return ['bool',value]
        if isinstance(value,int):return ['int',str(value)]
        if isinstance(value,float):return ['float',value.hex()]
        if isinstance(value,str):return ['str',value]
        if isinstance(value,np.ndarray):
            if value.dtype.hasobject:raise ValueError('Object arrays cannot enter joint profiles')
            return ['array',value.dtype.str,list(value.shape),
                    hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()]
        if isinstance(value,dict):
            pairs=[(canonical(k),canonical(v)) for k,v in value.items()]
            return ['dict',sorted(pairs,key=lambda p:json.dumps(p[0],sort_keys=True))]
        if isinstance(value,(set,frozenset)):
            return [type(value).__name__,sorted([canonical(v) for v in value],key=json.dumps)]
        if isinstance(value,(tuple,list,deque)):
            return [type(value).__name__,getattr(value,'maxlen',None),[canonical(v) for v in value]]
        kind=type(value).__module__+'.'+type(value).__qualname__
        if is_dataclass(value):
            return [kind,[(f.name,canonical(getattr(value,f.name))) for f in fields(value)]]
        if hasattr(value,'__dict__'):return [kind,canonical(vars(value))]
        raise ValueError('Unsupported joint profile content: '+kind)
    body=canonical(replace(profile,profile_id=''))
    return hashlib.sha256(json.dumps(body,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


class JointBoutWorkflowV1:
    def __init__(self,window_workflow):
        if not isinstance(window_workflow,ExtendedWindowDecisionV1):
            raise ValueError('Source-frozen extended window workflow required')
        self.window=window_workflow
        self.temporal=PersonalTemporalBoutsV1(source_bank_id=self.window.bank.bank_id_,
            sample_rate_hz=250.,channel_ids=self.window.channels,
            preprocessing_id=self.window.preprocessing_id,class_names=tuple(self.window.bank.classes_))
        self.adapter=CompleteBoutWindowAdapterV1(self.window,self.temporal)
        self.contract_id=hashlib.sha256(json.dumps(['joint_bout_workflow_v1',
            self.window.policy_id,self.temporal.contract_id,
            'same_native_registration_once','recording_disjoint_queries',
            'preserve_existing_temporal_outer_mixture_and_quality_Unknown']).encode()).hexdigest()

    def _input(self,batch,raw=None):
        self.temporal._input(batch)
        if raw is not None:
            if not isinstance(raw,TemporalBoutBatchV1):raise ValueError('Separate raw complete-bout batch required')
            raw.validate()
            if (raw.trial_ids!=batch.trial_ids or raw.recording_ids!=batch.recording_ids
                    or raw.starts!=batch.starts or raw.channel_ids!=batch.channel_ids
                    or raw.sample_rate_hz!=batch.sample_rate_hz or raw.boundary_kind!=batch.boundary_kind
                    or raw.preprocessing_id!='raw_pre_software_highpass'
                    or any(a.shape!=b.shape for a,b in zip(raw.sequences,batch.sequences))):
                raise ValueError('Raw/filtered native trial, recording, sample or boundary axes differ')

    @staticmethod
    def _windows(batch):
        pieces=[];ids=[];offsets=[];tails=[]
        for trial,x in zip(batch.trial_ids,batch.sequences):
            starts=list(range(0,len(x)-49,10))
            pieces.extend(np.asarray(x)[a:a+50] for a in starts)
            ids.extend([trial]*len(starts));offsets.extend(range(len(starts)))
            tails.append(len(x)-(starts[-1]+50))
        return FeatureBatch(np.stack(pieces),250.),np.asarray(ids),np.asarray(offsets,int),tails

    def _profile(self,profile,user_id,*,personal=None,session_id=None):
        if (not isinstance(profile,JointBoutProfileV1) or profile.contract_id!=self.contract_id
                or profile.user_id!=user_id or profile.profile_id!=profile_digest(profile)):
            raise ValueError('Joint profile checksum, source or user differs')
        self.temporal._profile(profile.temporal,user_id)
        if (profile.temporal.session_id!=profile.session_id or profile.temporal.kind!=profile.kind
                or profile.temporal.trial_ids!=profile.calibration.trial_ids
                or profile.temporal.recording_ids!=profile.calibration.recording_ids
                or profile.window.base.user_id!=profile.user_id
                or profile.window.base.session_id!=profile.session_id
                or set(profile.window.base.calibration_trials)!=set(profile.calibration.trial_ids)):
            raise ValueError('Joint temporal registration identity differs')
        if personal is None:
            if profile.kind!='personal' or profile.parent_profile_id is not None:
                raise ValueError('Joint long-term personal profile required')
            self.window._personal(profile.window,user_id)
        else:
            self._profile(personal,user_id)
            if (profile.kind!='session' or profile.parent_profile_id!=personal.profile_id
                    or profile.session_id!=session_id
                    or profile.temporal.personal_profile_id!=personal.temporal.profile_id):
                raise ValueError('Joint session parent or session identity differs')
            self.window._session(profile.window,personal.window,user_id,session_id)

    def enroll(self,batch,labels,*,user_id,session_id,personal=None,raw_batch=None,
               quality_mode='off',forbidden_trial_ids=(),forbidden_recording_ids=()):
        self._input(batch,raw_batch)
        # Caller-owned arrays must not invalidate an already registered profile.
        batch=copy.deepcopy(batch);raw_batch=copy.deepcopy(raw_batch)
        if batch.boundary_kind!='complete_cued':raise ValueError('Joint registration requires complete cued actions')
        if quality_mode not in ('off','structural','soft'):raise ValueError('Explicit raw quality mode required')
        if personal is not None:self._profile(personal,user_id)
        forbidden_trials=tuple(self.window.bank.policy_.source_trials)+tuple(forbidden_trial_ids)
        if set(batch.trial_ids)&set(forbidden_trials) or set(batch.recording_ids)&set(forbidden_recording_ids):
            raise ValueError('Joint registration overlaps source or forbidden evaluation provenance')
        windows,ids,offsets,tails=self._windows(batch)
        if quality_mode!='off':
            if raw_batch is None:raise ValueError('Enabled calibration quality requires separate raw input')
            raw,_,_,_=self._windows(raw_batch)
            observations=self.window.gate.observe(raw,ids,observed_channel_ids=batch.channel_ids)
            bad=(observations['trial_structural_invalid_channels'].any(axis=1) if quality_mode=='structural'
                 else observations['trial_soft_channel_quality'].max(axis=1)==0)
            if bad.any():raise ValueError('Raw quality rejected joint calibration; no profile was registered')
        # Stage both branch states locally. A failure returns neither; the
        # already fitted source bank and existing parent are never updated.
        temporal=self.temporal.enroll(batch,labels,user_id=user_id,session_id=session_id,
            personal=None if personal is None else personal.temporal,
            forbidden_trial_ids=forbidden_trials,forbidden_recording_ids=forbidden_recording_ids)
        kwargs=dict(window_offsets=offsets,user_id=user_id,session_id=session_id,
            observed_channel_ids=batch.channel_ids,preprocessing_id=batch.preprocessing_id,
            forbidden_evaluation_trials=forbidden_trial_ids)
        if personal is None:window=self.window.enroll_user(windows,ids,labels,**kwargs)
        else:window=self.window.calibrate_session(windows,ids,labels,personal=personal.window,**kwargs)
        detector,receipt=fit_calibration_rest_detector(self.temporal,batch,labels,temporal,
            user_id=user_id,rest_label=self.window.rest_label,
            forbidden_trial_ids=forbidden_trials,forbidden_recording_ids=forbidden_recording_ids)
        cost=dict(unique_native_calibration_trials=len(batch.trial_ids),
            native_signal_samples=sum(len(x) for x in batch.sequences),
            native_signal_seconds=sum(len(x) for x in batch.sequences)/250.,
            window_rows=len(ids),window_unrepresented_tail_samples=tails,
            detector_neutral_trials=receipt['neutral_trial_ids'],
            shared_registration_consumed_by=('window_F7_F8','full_bout_F5','neutral_detector'),
            counted_once=True,wall_clock_seconds=None,physical_validation_proven=False)
        profile=JointBoutProfileV1(self.contract_id,'',user_id,session_id,
            'personal' if personal is None else 'session',None if personal is None else personal.profile_id,
            window,temporal,detector,receipt,batch,raw_batch,
            tuple((t,labels[t]) for t in batch.trial_ids),cost)
        return replace(profile,profile_id=profile_digest(profile))

    def predict(self,batch,*,personal,user_id,session_id,session=None,raw_batch=None,
                quality_mode='off',use_anchor=True,use_session_routing=True,anchor_mode='blended'):
        self._input(batch,raw_batch);self._profile(personal,user_id)
        if session is not None:self._profile(session,user_id,personal=personal,session_id=session_id)
        for profile in (personal,session):
            if profile is not None and (set(batch.trial_ids)&set(profile.calibration.trial_ids)
                    or set(batch.recording_ids)&set(profile.calibration.recording_ids)):
                raise ValueError('Query overlaps a joint calibration trial or recording')
        original=self.adapter.predict(batch,temporal_personal=personal.temporal,
            temporal_session=None if session is None else session.temporal,
            personal=personal.window,session=None if session is None else session.window,
            user_id=user_id,session_id=session_id,raw_batch=raw_batch,quality_mode=quality_mode,
            use_anchor=use_anchor,use_session_routing=use_session_routing,anchor_mode=anchor_mode)
        arms=original['temporal']['arms'];names=('window','F5b_DTW','F5c_signature')
        mixture=self.temporal.mix;weights=(1-mixture,mixture/2,mixture/2)
        source=AvailableBankFusionPolicyV1(self.temporal.class_names,names,weights,1.,1.,
            self.contract_id,self.window.bank.policy_.source_trials)
        calibration=tuple(sorted(set(personal.calibration.trial_ids)
            | (set() if session is None else set(session.calibration.trial_ids))))
        state=source.from_fitted_weights(weights,calibration_trials=calibration)
        quality_id=self.window.gate.policy_id+':'+quality_mode+':preserve_window_Unknown'
        fusion=AvailableBankQualityFusionV2(source,quality_id)
        values=dict(zip(names,(arms['base'],arms['DTW_blended'],arms['signature_blended'])))
        quality=None if quality_mode=='off' else {n:(~original['rejected']).astype(float) for n in names}
        result=fusion.predict(state,values,evaluation_trials=batch.trial_ids,
            provider_trial_ids={n:batch.trial_ids for n in names},
            provider_classes={n:self.temporal.class_names for n in names},quality=quality,
            quality_trial_ids=None if quality is None else {n:batch.trial_ids for n in names},
            quality_policy_id=None if quality is None else quality_id)
        return dict(**original,composition=result,
            probabilities=result['probabilities'],labels=result['labels'],
            joint_personal_profile_id=personal.profile_id,
            joint_session_profile_id=None if session is None else session.profile_id,
            joint_calibration_trials=len(calibration),profile_contract_id=self.contract_id,
            scope='Joint seven-window-provider/F7/F8/raw-F9/full-bout-F5 composition and shared registration. Not physical ring/F6, all subfamilies, chronology, independent cohort or device efficacy.')

    def make_detector(self,profile,*,user_id,personal=None,session_id=None):
        """Return independent stream state without modifying the saved profile."""
        self._profile(profile,user_id,personal=personal,session_id=session_id)
        detector=copy.deepcopy(profile.detector)
        detector.reset()
        return detector

    def save_profile(self,profile,path):
        if (not isinstance(profile,JointBoutProfileV1) or profile.contract_id!=self.contract_id
                or profile_digest(profile)!=profile.profile_id):
            raise ValueError('Invalid joint profile content/source checksum')
        payload=pickle.dumps(profile,protocol=pickle.HIGHEST_PROTOCOL)
        manifest=dict(schema='joint_bout_profile_v1',contract_id=self.contract_id,
            profile_id=profile.profile_id,user_id=profile.user_id,session_id=profile.session_id,
            kind=profile.kind,parent_profile_id=profile.parent_profile_id,
            payload_sha256=hashlib.sha256(payload).hexdigest(),complete_native_calibration_saved=True)
        target=Path(path);created=False
        try:
            with target.open('xb') as stream:
                created=True
                with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr('manifest.json',json.dumps(manifest,sort_keys=True))
                    archive.writestr('profile.pkl',payload)
        except Exception:
            if created:target.unlink()
            raise

    def load_profile(self,path,*,user_id,session_id=None,personal=None):
        with zipfile.ZipFile(path) as archive:
            if sorted(archive.namelist())!=['manifest.json','profile.pkl']:
                raise ValueError('Unexpected joint profile archive members')
            manifest=json.loads(archive.read('manifest.json'));payload=archive.read('profile.pkl')
        if (manifest.get('schema')!='joint_bout_profile_v1' or manifest.get('contract_id')!=self.contract_id
                or manifest.get('user_id')!=user_id
                or hashlib.sha256(payload).hexdigest()!=manifest.get('payload_sha256')):
            raise ValueError('Joint bundle checksum, source or user differs')
        profile=pickle.loads(payload)
        self._profile(profile,user_id,personal=personal,session_id=session_id)
        if (profile.profile_id!=manifest['profile_id'] or profile.session_id!=manifest['session_id']
                or profile.kind!=manifest['kind'] or profile.parent_profile_id!=manifest['parent_profile_id']):
            raise ValueError('Joint manifest and payload differ')
        return profile
