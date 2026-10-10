"""Source-frozen window readouts and class-transition detection at native rate."""
import numpy as np
from .core import FeatureBatch
from .frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1
from .class_transition_bouts_v1 import ConfirmedClassBoutDetectorV1


def window_scores(bank,windows,ends,*,recording_id,user_id):
    ids=np.array([f'{recording_id}:source_window{int(e):012d}' for e in ends])
    result=bank.predict(FeatureBatch(np.asarray(windows,np.float32),bank.sensor_contract_[0]),ids,
        window_offsets=np.zeros(len(ids),int),user_id=user_id)
    if tuple(result['trial_ids'])!=tuple(ids):raise ValueError('Native window identity/order differs')
    return result['probabilities']


class NativeClassTransitionStreamV1:
    def __init__(self,bank,*,source_recording_ids,forbidden_recording_ids,user_id,rest_label,
                 sample_rate_hz,channel_ids,source_channel_ids,preprocessing_id,source_preprocessing_id,
                 confirmations=3,smoothing_windows=1,confidence=0.):
        if not isinstance(bank,FrozenEmgProviderBankV1):raise ValueError('Source-frozen bank required')
        rate,size,channels=bank.sensor_contract_
        if ((rate,size,channels) not in ((200.,40,8),(250.,50,8)) or sample_rate_hz!=rate
                or len(tuple(channel_ids))!=8 or tuple(channel_ids)!=tuple(source_channel_ids)
                or preprocessing_id!=source_preprocessing_id):raise ValueError('Explicit native sensor/preprocessing contract differs')
        sources=tuple(source_recording_ids)
        if not sources or any(not isinstance(v,str) or not v.strip() for v in (*sources,*forbidden_recording_ids)):
            raise ValueError('Explicit source/forbidden recording identities required')
        self.bank=bank;self.user=user_id;self.forbidden=set(sources)|set(forbidden_recording_ids)
        self.detector=ConfirmedClassBoutDetectorV1(bank_id=bank.bank_id_,class_names=bank.classes_,rest_label=rest_label,
            sample_rate_hz=rate,confirmations=confirmations,smoothing_windows=smoothing_windows,confidence=confidence)
        self.size=size;self.hop=round(rate*.04);self.reset()

    def reset(self):
        self.detector.reset();self.buffer=np.empty((0,8),np.float32);self.next=None;self.recording=None;self.next_end=None

    def finish(self):
        censored=self.detector.finish();self.reset();return censored

    def feed(self,samples,first_sample_index,*,recording_id):
        x=np.asarray(samples)
        if (not isinstance(recording_id,str) or not recording_id.strip() or recording_id in self.forbidden
                or isinstance(first_sample_index,bool) or not isinstance(first_sample_index,int) or first_sample_index<0
                or x.ndim!=2 or x.shape[1]!=8 or not len(x) or not np.isfinite(x).all()
                or (self.next is not None and (first_sample_index!=self.next or recording_id!=self.recording))):
            self.reset();raise ValueError('Disjoint recording and contiguous finite native stream required')
        values=np.concatenate([self.buffer,x.astype(np.float32)]);origin=first_sample_index-len(self.buffer)
        first_end=first_sample_index+self.size if self.next_end is None else self.next_end
        stop=first_sample_index+len(x);ends=np.arange(first_end,stop+1,self.hop,dtype=np.int64)
        q=np.empty((0,len(self.bank.classes_)))
        if len(ends):
            windows=np.stack([values[e-self.size-origin:e-origin] for e in ends])
            q=window_scores(self.bank,windows,ends,recording_id=recording_id,user_id=self.user)
        events=self.detector.feed(x,first_sample_index,recording_id=recording_id,probabilities=q,
            window_end_indices=ends,bank_id=self.bank.bank_id_)
        self.next=stop;self.next_end=first_end+len(ends)*self.hop;self.recording=recording_id
        self.buffer=values[-(self.size-1):].copy()
        return events
