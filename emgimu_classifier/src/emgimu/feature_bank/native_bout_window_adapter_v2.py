"""Complete supplied cue intervals with native-rate window and path readouts."""
import numpy as np
from .core import FeatureBatch
from .extended_window_decision_v1 import ExtendedWindowDecisionV1
from .personal_temporal_bouts_v1 import PersonalTemporalBoutsV1,TemporalBoutBatchV1


def native_windows(batch,window_samples,hop_samples):
    batch.validate()
    pieces=[];ids=[];offsets=[];tails=[]
    for trial,x in zip(batch.trial_ids,batch.sequences):
        starts=list(range(0,len(x)-window_samples+1,hop_samples))
        if not starts:raise ValueError('Native cue interval shorter than source window')
        pieces.extend(np.asarray(x)[a:a+window_samples] for a in starts)
        ids.extend([trial]*len(starts));offsets.extend(range(len(starts)))
        tails.append(len(x)-(starts[-1]+window_samples))
    return FeatureBatch(np.stack(pieces),batch.sample_rate_hz),np.asarray(ids),np.asarray(offsets,int),tails


class NativeBoutWindowAdapterV2:
    def __init__(self,window_workflow,temporal_workflow):
        if not isinstance(window_workflow,ExtendedWindowDecisionV1) or not isinstance(temporal_workflow,PersonalTemporalBoutsV1):
            raise ValueError('Explicit window and temporal workflows required')
        w,t=window_workflow,temporal_workflow;rate,size,channels=w.bank.sensor_contract_
        if ((rate,size,channels) not in ((200.,40,8),(250.,50,8))
                or t.source_bank_id!=w.bank.bank_id_ or t.rate!=rate or t.channel_ids!=tuple(w.channels)
                or t.preprocessing_id!=w.preprocessing_id or t.class_names!=tuple(w.bank.classes_)):
            raise ValueError('Native window/path sensor, source or class contracts differ')
        self.window,self.temporal=w,t;self.samples=size;self.hop=round(.04*rate)

    def predict(self,batch,*,temporal_personal,user_id,session_id,temporal_session=None,
                personal=None,session=None,raw_batch=None,quality_mode='off',
                use_anchor=False,use_session_routing=False,anchor_mode='blended',available_window_providers=None):
        self.temporal._input(batch)
        if raw_batch is not None:
            if not isinstance(raw_batch,TemporalBoutBatchV1):raise ValueError('Separate raw native cue intervals required')
            raw_batch.validate()
            if (raw_batch.trial_ids!=batch.trial_ids or raw_batch.recording_ids!=batch.recording_ids
                    or raw_batch.starts!=batch.starts or raw_batch.channel_ids!=batch.channel_ids
                    or raw_batch.sample_rate_hz!=batch.sample_rate_hz or raw_batch.boundary_kind!=batch.boundary_kind
                    or raw_batch.preprocessing_id!='raw_pre_software_highpass'
                    or any(a.shape!=b.shape for a,b in zip(raw_batch.sequences,batch.sequences))):
                raise ValueError('Raw/filtered interval identity/sample axes differ')
        b,ids,offsets,tails=native_windows(batch,self.samples,self.hop)
        raw=None if raw_batch is None else native_windows(raw_batch,self.samples,self.hop)[0]
        window=self.window.predict(b,ids,window_offsets=offsets,personal=personal,session=session,
            user_id=user_id,session_id=session_id,observed_channel_ids=batch.channel_ids,
            preprocessing_id=batch.preprocessing_id,raw_batch=raw,quality_mode=quality_mode,
            use_anchor=use_anchor,use_session_routing=use_session_routing,anchor_mode=anchor_mode,
            available=available_window_providers)
        order=[window['trial_ids'].index(t) for t in batch.trial_ids]
        temporal=self.temporal.predict(batch,personal=temporal_personal,session=temporal_session,
            user_id=user_id,session_id=session_id,base_probabilities=window['probabilities'][order],
            base_trial_ids=batch.trial_ids,base_class_names=window['class_names'])
        rejected=np.zeros(len(batch.trial_ids),bool)
        if window.get('quality_decision') is not None:rejected=np.asarray(window['quality_decision']['rejected'])[order]
        labels=np.asarray(temporal['class_names'],object)[temporal['arms']['base_full'].argmax(1)]
        labels[rejected]='Unknown'
        return dict(window=window,temporal=temporal,trial_ids=batch.trial_ids,predicted_labels=labels,rejected=rejected,
            window_unrepresented_tail_samples=tails,temporal_uses_all_native_samples=True,
            output_sample_indices=[int(a)+len(x)-1 for a,x in zip(batch.starts,batch.sequences)],
            decision_available_after_interval=True,physical_latency_proven=False,default_promoted=False)
