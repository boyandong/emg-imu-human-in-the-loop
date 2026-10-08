"""Opt-in source-frozen Song stream; no cues or labels enter inference."""
from numbers import Integral
import numpy as np
from .core import FeatureBatch
from .song_f0_runtime_v1 import SongF0RuntimeV1
from .causal_label_debounce_v1 import CausalLabelDebounceV1


class SongF0StreamV1:
    def __init__(self,runtime,*,arm,recording_id,hop_samples=10,confirmations=2):
        if not isinstance(runtime,SongF0RuntimeV1) or arm not in runtime.models_:
            raise ValueError('Explicit source-frozen Song model arm required')
        if not isinstance(recording_id,str) or not recording_id.strip():raise ValueError('Explicit recording identity required')
        if isinstance(hop_samples,bool) or not isinstance(hop_samples,Integral) or not 1<=hop_samples<=50:
            raise ValueError('Integer hop in1..50 required')
        self.runtime=runtime;self.arm=arm;self.recording_id=recording_id
        self.hop=int(hop_samples);self.confirmations=confirmations
        self.classes=tuple(runtime.classes_)
        self.columns=tuple(list(runtime.models_[arm][1][-1].classes_).index(c) for c in self.classes)
        self.reset(first_sample_index=0)

    def reset(self,*,first_sample_index):
        if isinstance(first_sample_index,bool) or not isinstance(first_sample_index,Integral) or first_sample_index<0:
            raise ValueError('Explicit nonnegative first sample index required')
        self.origin=int(first_sample_index);self.received=self.origin
        self.buffer_start=self.origin;self.next_end=self.origin+49
        self.buffer=np.empty((0,8),dtype=np.float32);self.filter=self.runtime.new_filter()
        self.confirmation=CausalLabelDebounceV1(tuple(range(len(self.classes))),self.confirmations)

    def push(self,raw,*,first_sample_index,sample_rate_hz):
        x=np.asarray(raw)
        if isinstance(first_sample_index,bool) or not isinstance(first_sample_index,Integral) or first_sample_index!=self.received:
            raise ValueError('Missing, duplicated or reordered samples require explicit stream reset')
        if sample_rate_hz!=250.:raise ValueError('Song stream requires250Hz')
        if x.ndim!=2 or x.shape[1]!=8 or not np.isfinite(x).all():raise ValueError('Finite ordered raw eight-channel samples required')
        family,model=self.runtime.models_[self.arm]
        if tuple(list(model[-1].classes_)[i] for i in self.columns)!=self.classes:
            raise ValueError('Source model class axis changed')
        if not len(x):return []
        combined=np.concatenate([self.buffer,self.filter.process(x)])
        received=self.received+len(x);ends=np.arange(self.next_end,received,self.hop,dtype=int)
        records=[]
        if len(ends):
            windows=np.stack([combined[e-49-self.buffer_start:e+1-self.buffer_start] for e in ends])
            q=model.predict_proba(family.transform(FeatureBatch(windows,250.)))[:,self.columns]
            if q.shape!=(len(ends),len(self.classes)) or not np.isfinite(q).all() or np.any(q<0) or not np.allclose(q.sum(1),1.,atol=1e-12):
                raise ValueError('Invalid source classifier probabilities')
            for end,probability in zip(ends,q):
                raw_label=int(probability.argmax());confirmed=self.confirmation.update(raw_label)
                records.append({'end_sample':int(end),'window_start_sample':int(end)-49,
                    'recording_id':self.recording_id,'available_at_nominal_seconds':(int(end)-self.origin+1)/250.,
                    'probabilities':probability.copy(),'raw_class_index':raw_label,
                    'confirmed_class_index':confirmed,'raw_label':self.classes[raw_label],
                    'confirmed_label':self.classes[confirmed] if confirmed>=0 else 'UNKNOWN'})
            self.next_end=int(ends[-1])+self.hop
        start=max(self.origin,self.next_end-49)
        self.buffer=combined[start-self.buffer_start:].copy()
        self.buffer_start=start;self.received=received
        return records
