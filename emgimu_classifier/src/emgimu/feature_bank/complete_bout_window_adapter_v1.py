"""Eight-channel full-bout temporal fusion around the frozen window workflow."""
import numpy as np
from .core import FeatureBatch
from .extended_window_decision_v1 import ExtendedWindowDecisionV1
from .personal_temporal_bouts_v1 import PersonalTemporalBoutsV1, TemporalBoutBatchV1


class CompleteBoutWindowAdapterV1:
    def __init__(self,window_workflow,temporal_workflow):
        if not isinstance(window_workflow,ExtendedWindowDecisionV1) or not isinstance(temporal_workflow,PersonalTemporalBoutsV1):
            raise ValueError('Versioned seven-provider and full-bout workflows required')
        w,t = window_workflow,temporal_workflow
        if (t.source_bank_id != w.bank.bank_id_ or t.channel_ids != tuple(w.channels)
                or t.rate != 250. or t.preprocessing_id != w.preprocessing_id
                or t.class_names != tuple(w.bank.classes_)):
            raise ValueError('Window and complete-bout source/sensor/class contracts differ')
        self.window,self.temporal = w,t

    def predict(self,batch,*,temporal_personal,user_id,session_id,temporal_session=None,
                personal=None,session=None,raw_batch=None,quality_mode='off',
                use_anchor=False,use_session_routing=False,anchor_mode='blended'):
        self.temporal._input(batch)
        if raw_batch is not None:
            if not isinstance(raw_batch,TemporalBoutBatchV1):
                raise ValueError('Separate raw native bouts required')
            raw_batch.validate()
            if (raw_batch.trial_ids != batch.trial_ids or raw_batch.recording_ids != batch.recording_ids
                    or raw_batch.starts != batch.starts or raw_batch.channel_ids != batch.channel_ids
                    or raw_batch.sample_rate_hz != batch.sample_rate_hz
                    or raw_batch.boundary_kind != batch.boundary_kind
                    or raw_batch.preprocessing_id != 'raw_pre_software_highpass'
                    or any(np.asarray(a).shape != np.asarray(b).shape for a,b in zip(raw_batch.sequences,batch.sequences))):
                raise ValueError('Raw/filtered full-bout identity/sample/processing axes differ')
        pieces,raw_pieces,ids,offsets,tails = [],[],[],[],[]
        for i,(trial,x) in enumerate(zip(batch.trial_ids,batch.sequences)):
            starts = list(range(0,len(x)-49,10))
            pieces.extend(np.asarray(x)[a:a+50] for a in starts)
            if raw_batch is not None:
                raw_pieces.extend(np.asarray(raw_batch.sequences[i])[a:a+50] for a in starts)
            ids.extend([trial]*len(starts));offsets.extend(range(len(starts)))
            tails.append(len(x)-(starts[-1]+50))
        window = self.window.predict(FeatureBatch(np.stack(pieces),250.),ids,window_offsets=offsets,
            personal=personal,session=session,user_id=user_id,session_id=session_id,
            observed_channel_ids=batch.channel_ids,preprocessing_id=batch.preprocessing_id,
            raw_batch=None if raw_batch is None else FeatureBatch(np.stack(raw_pieces),250.),
            quality_mode=quality_mode,use_anchor=use_anchor,use_session_routing=use_session_routing,anchor_mode=anchor_mode)
        lookup = {t:i for i,t in enumerate(window['trial_ids'])}
        order = [lookup[t] for t in batch.trial_ids]
        temporal = self.temporal.predict(batch,personal=temporal_personal,session=temporal_session,
            user_id=user_id,session_id=session_id,base_probabilities=window['probabilities'][order],
            base_trial_ids=batch.trial_ids,base_class_names=window['class_names'])
        rejected = np.zeros(len(batch.trial_ids),bool)
        if window.get('quality_decision') is not None:
            rejected = np.asarray(window['quality_decision']['rejected'])[order]
        # A temporal branch cannot override the window branch's raw-quality Unknown.
        labels = np.asarray(temporal['class_names'],dtype=object)[temporal['arms']['base_full'].argmax(1)]
        labels[rejected] = 'Unknown'
        return dict(window=window,temporal=temporal,trial_ids=batch.trial_ids,
            predicted_labels=labels,rejected=rejected,
            window_unrepresented_tail_samples=tails,temporal_uses_all_native_samples=True,
            output_sample_indices=[int(a)+len(x)-1 for a,x in zip(batch.starts,batch.sequences)],
            decision_available_after_interval=True,physical_latency_proven=False,default_promoted=False)
