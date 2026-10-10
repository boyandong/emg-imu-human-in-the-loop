"""Shared registration at the source's native200/250Hz; V1 remains frozen."""
from dataclasses import replace
import hashlib
import json
from .extended_window_decision_v1 import ExtendedWindowDecisionV1
from .joint_bout_workflow_v1 import JointBoutWorkflowV1,profile_digest
from .personal_temporal_bouts_v1 import PersonalTemporalBoutsV1
from .native_bout_window_adapter_v2 import NativeBoutWindowAdapterV2,native_windows
from .available_bank_fusion_v1 import AvailableBankFusionPolicyV1
from .available_bank_quality_fusion_v2 import AvailableBankQualityFusionV2


class NativeJointBoutWorkflowV2(JointBoutWorkflowV1):
    def __init__(self,window_workflow):
        if not isinstance(window_workflow,ExtendedWindowDecisionV1):raise ValueError('Source-frozen window workflow required')
        self.window=window_workflow
        self.rate,self.samples,channels=self.window.bank.sensor_contract_
        if (self.rate,self.samples,channels) not in ((200.,40,8),(250.,50,8)):
            raise ValueError('Native eight-channel200/250Hz source contract required')
        self.hop=round(.04*self.rate)
        self.temporal=PersonalTemporalBoutsV1(source_bank_id=self.window.bank.bank_id_,sample_rate_hz=self.rate,
            channel_ids=self.window.channels,preprocessing_id=self.window.preprocessing_id,
            class_names=tuple(self.window.bank.classes_))
        self.adapter=NativeBoutWindowAdapterV2(self.window,self.temporal)
        self.contract_id=hashlib.sha256(json.dumps(['native_joint_bout_workflow_v2',self.window.policy_id,
            self.temporal.contract_id,self.rate,self.samples,self.hop,'same_native_calibration_counted_once']).encode()).hexdigest()

    def _windows(self,batch):return native_windows(batch,self.samples,self.hop)

    def enroll(self,*args,**kwargs):
        if kwargs.get('quality_mode','off')!='off' and self.window.gate is None:
            raise ValueError('Native source has no bound quality policy')
        profile=super().enroll(*args,**kwargs)
        cost=dict(profile.calibration_cost)
        cost['native_signal_seconds']=cost['native_signal_samples']/self.rate
        cost.update(native_sample_rate_hz=self.rate,source_window_samples=self.samples,hop_samples=self.hop,
            resampled=False)
        profile=replace(profile,calibration_cost=cost,profile_id='')
        return replace(profile,profile_id=profile_digest(profile))

    def predict(self,batch,*,personal,user_id,session_id,session=None,raw_batch=None,
                quality_mode='off',use_anchor=True,use_session_routing=True,anchor_mode='blended',
                available_window_providers=None):
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
            use_anchor=use_anchor,use_session_routing=use_session_routing,anchor_mode=anchor_mode,
            available_window_providers=available_window_providers)
        arms=original['temporal']['arms'];names=('window','F5b_DTW','F5c_signature')
        mixture=self.temporal.mix;weights=(1-mixture,mixture/2,mixture/2)
        source=AvailableBankFusionPolicyV1(self.temporal.class_names,names,weights,1.,1.,
            self.contract_id,self.window.bank.policy_.source_trials)
        calibration=tuple(sorted(set(personal.calibration.trial_ids)
            | (set() if session is None else set(session.calibration.trial_ids))))
        state=source.from_fitted_weights(weights,calibration_trials=calibration)
        quality_id=(self.contract_id+':unobserved_native_quality' if self.window.gate is None else
            self.window.gate.policy_id+':'+quality_mode+':preserve_window_Unknown')
        fusion=AvailableBankQualityFusionV2(source,quality_id)
        values=dict(zip(names,(arms['base'],arms['DTW_blended'],arms['signature_blended'])))
        quality=None if quality_mode=='off' else {n:(~original['rejected']).astype(float) for n in names}
        result=fusion.predict(state,values,evaluation_trials=batch.trial_ids,
            provider_trial_ids={n:batch.trial_ids for n in names},
            provider_classes={n:self.temporal.class_names for n in names},quality=quality,
            quality_trial_ids=None if quality is None else {n:batch.trial_ids for n in names},
            quality_policy_id=None if quality is None else quality_id)
        return dict(**original,composition=result,quality_policy_available=self.window.gate is not None,
            probabilities=result['probabilities'],labels=result['labels'],
            joint_personal_profile_id=personal.profile_id,
            joint_session_profile_id=None if session is None else session.profile_id,
            joint_calibration_trials=len(calibration),profile_contract_id=self.contract_id,
            scope='Native-rate available window/F7/F8/full-cue-path composition and shared registration. Supplied boundaries are not physiological annotations or autonomous detection. Raw F9 is available only with a separately bound policy; no physical ring/F6, all-subfamily, chronology or device efficacy claim.')
