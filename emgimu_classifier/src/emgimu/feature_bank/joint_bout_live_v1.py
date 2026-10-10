"""Shared cued full-action capture and joint profiles for the collection UI."""
from dataclasses import replace
import time

from .joint_bout_workflow_v1 import JointBoutWorkflowV1, profile_digest
from .temporal_bout_live_v1 import TemporalBoutLiveV1


class JointBoutLiveV1(TemporalBoutLiveV1):
    def __init__(self,window,filters,**kwargs):
        super().__init__(window,filters,**kwargs)
        self.joint=JointBoutWorkflowV1(window)

    def info(self):
        result=super().info()
        result.update(profile_schema='joint_bout_profile_v1',shared_window_temporal_registration=True,
            calibration_trials_counted_once=True,
            calibration_cost=None if self.personal is None else self.personal.calibration_cost,
            session_calibration_cost=None if self.session is None else self.session.calibration_cost)
        return result

    def begin(self,kind,shots):
        super().begin(kind,shots)
        self.capture['began_at']=time.monotonic()
        self.capture['quality_mode']=None

    def end_trial(self,quality_mode):
        if self.capture is not None:
            existing=self.capture['quality_mode']
            if existing is not None and existing!=quality_mode:
                raise ValueError('One fixed quality policy is required throughout shared calibration')
        result=super().end_trial(quality_mode)
        if not result['calibration_rejected']:self.capture['quality_mode']=quality_mode
        return result

    def save(self,path):
        c=self.capture
        if c is None or c['trial'] is not None or len(c['rows'])!=len(c['order']):
            raise ValueError('Complete every shared cued action before saving')
        parent=self.personal if c['kind']=='session' else None
        profile=self.joint.enroll(self._batch(c['rows']),{r['id']:r['label'] for r in c['rows']},
            user_id=self.user,session_id=self.session_id,personal=parent,
            raw_batch=self._batch(c['rows'],raw=True),quality_mode=c['quality_mode'])
        cost=dict(profile.calibration_cost,quality_mode=c['quality_mode'],
            begin_to_save_elapsed_seconds=time.monotonic()-c['began_at'],
            elapsed_scope='software registration interval; not measured physical device latency')
        receipt=dict(profile.detector_receipt,registration_quality_mode=c['quality_mode'],
            boundaries='user_marked_cued_intervals_not_physiological_ground_truth',
            settle_samples_discarded=0,raw_and_filtered_samples_saved=True)
        profile=replace(profile,calibration_cost=cost,detector_receipt=receipt)
        profile=replace(profile,profile_id=profile_digest(profile))
        # Prepare detector state before creating a file; a failed save retains
        # both the captured trials and the previously active profiles.
        detector=self.joint.make_detector(profile,user_id=self.user,personal=parent,
            session_id=self.session_id if parent is not None else None)
        self.joint.save_profile(profile,path)
        if parent is None:
            self.personal,self.long_detector=profile,detector
            self.session=self.local_detector=None
        else:self.session,self.local_detector=profile,detector
        self.reset()
        return dict(saved_joint_profile_path=str(path),saved_temporal_profile_path=str(path),
            temporal_saved_kind=profile.kind,shared_calibration_cost=cost)

    def load(self,personal,session):
        if self.capture is not None:raise ValueError('Save/cancel shared calibration before loading profiles')
        p=None if not personal else self.joint.load_profile(personal,user_id=self.user)
        if session and p is None:raise ValueError('Joint session requires its personal profile')
        s=None if not session else self.joint.load_profile(session,user_id=self.user,
            session_id=self.session_id,personal=p)
        d=None if p is None else self.joint.make_detector(p,user_id=self.user)
        e=None if s is None else self.joint.make_detector(s,user_id=self.user,personal=p,session_id=self.session_id)
        self.personal,self.session,self.long_detector,self.local_detector=p,s,d,e
        self.reset()

    def _read(self,row,kind,available_at,**options):
        controls={k:options[k] for k in ('quality_mode','use_anchor','use_session_routing','anchor_mode')}
        result=self.joint.predict(self._batch([row],kind),personal=self.personal,session=self.session,
            user_id=self.user,session_id=self.session_id,raw_batch=self._batch([row],kind,raw=True),**controls)
        arms=result['temporal']['arms']
        return dict(trial_id=row['id'],recording_id=row['recording'],start=row['start'],end=row['start']+len(row['raw']),
            available_at_sample_index=available_at,boundary_kind=kind,decision_after_interval=True,
            label=result['labels'][0],rejected=bool(result['rejected'][0]),classes=list(self.temporal.class_names),
            probabilities=result['probabilities'][0],base_probabilities=arms['base'][0],
            DTW_probabilities=arms['DTW_blended'][0],signature_probabilities=arms['signature_blended'][0],
            window_unrepresented_tail_samples=result['window_unrepresented_tail_samples'][0],
            joint_personal_profile_id=self.personal.profile_id,
            joint_session_profile_id=None if self.session is None else self.session.profile_id,
            shared_registration=True,physical_validation_proven=False,default_promoted=False)
