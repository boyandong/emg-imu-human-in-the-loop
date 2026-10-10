"""Label-free native eight-channel detection followed by frozen joint inference.

No cue edges are accepted. All outputs have estimated boundaries; release
confirmation belongs to the captured interval and decisions become available
only at its exclusive end. Quiet intervals produce no gesture events.
"""
import numpy as np
from .native_joint_bout_workflow_v2 import NativeJointBoutWorkflowV2
from .personal_temporal_bouts_v1 import TemporalBoutBatchV1


class NativeJointBoutStreamV1:
    def __init__(self, workflow, *, personal, user_id, session_id,
                 source_recording_ids, sample_rate_hz, channel_ids,
                 preprocessing_id, session=None):
        if not isinstance(workflow, NativeJointBoutWorkflowV2):
            raise ValueError('Explicit native joint workflow required')
        if (sample_rate_hz != workflow.rate or tuple(channel_ids) != workflow.window.channels
                or preprocessing_id != workflow.window.preprocessing_id):
            raise ValueError('Native stream rate, channel order or preprocessing differs')
        source = tuple(source_recording_ids)
        if not source or any(not isinstance(s, str) or not s.strip() for s in source):
            raise ValueError('Explicit source recording identities required')
        workflow._profile(personal, user_id)
        if session is not None:
            workflow._profile(session, user_id, personal=personal, session_id=session_id)
        self.workflow, self.personal, self.session = workflow, personal, session
        self.user_id, self.session_id = user_id, session_id
        self.forbidden = set(source) | set(personal.calibration.recording_ids)
        if session is not None:
            self.forbidden.update(session.calibration.recording_ids)
        self.detector = workflow.make_detector(session or personal, user_id=user_id,
            personal=personal if session is not None else None,
            session_id=session_id if session is not None else None)
        self.recording = None
        self.ordinal = 0

    def reset(self):
        self.detector.reset()
        self.recording = None
        self.ordinal = 0

    def finish(self):
        censored = self.detector.finish()
        self.recording = None
        self.ordinal = 0
        return censored

    def feed(self, samples, first_sample_index, *, recording_id):
        if (not isinstance(recording_id, str) or not recording_id.strip()
                or recording_id in self.forbidden):
            self.reset()
            raise ValueError('Disjoint query recording identity required')
        if self.recording is not None and self.recording != recording_id:
            self.reset()
            raise ValueError('Finish/reset before changing recording')
        try:
            events = self.detector.feed(samples, first_sample_index, trial_id=recording_id)
        except (ValueError, RuntimeError):
            self.reset()
            raise
        self.recording = recording_id
        result = []
        w = self.workflow
        for event in events:
            identity = f'{recording_id}:estimated{self.ordinal}'
            self.ordinal += 1
            batch = TemporalBoutBatchV1((event.emg,), (identity,), (recording_id,), (event.start,),
                w.rate, w.window.channels, w.window.preprocessing_id, 'estimated')
            joint = w.predict(batch, personal=self.personal, session=self.session,
                user_id=self.user_id, session_id=self.session_id)
            b, ids, offsets, tails = w._windows(batch)
            source = w.window.bank.predict(b, ids, window_offsets=offsets, user_id=self.user_id)
            probabilities = dict(source_window=source['probabilities'][0],
                window_full=joint['temporal']['arms']['base'][0], joint_full=joint['probabilities'][0])
            classes = tuple(w.temporal.class_names)
            result.append(dict(trial_id=identity, recording_id=recording_id,
                start=event.start, end=event.end, available_at_sample_index=event.end,
                boundary_kind='estimated', class_names=classes,
                probabilities={k:np.asarray(q).tolist() for k,q in probabilities.items()},
                labels={k:classes[int(np.argmax(q))] for k,q in probabilities.items()},
                unrepresented_tail_samples=tails[0],
                personal_profile_id=self.personal.profile_id,
                session_profile_id=None if self.session is None else self.session.profile_id,
                quality_policy='off', physical_validation_proven=False))
        return result
