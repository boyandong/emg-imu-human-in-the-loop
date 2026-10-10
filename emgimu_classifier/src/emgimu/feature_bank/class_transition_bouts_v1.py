"""Causal estimated bouts from confirmed class changes, including active-active.

Probabilities must have the explicit source class/window axis. A complete
initial active interval is never invented, Unknown breaks discard candidates,
and EOF does not manufacture a release. This is not physiological annotation.
"""
from collections import deque
import math
import numpy as np


class ConfirmedClassBoutDetectorV1:
    def __init__(self, *, bank_id, class_names, rest_label, sample_rate_hz,
                 confirmations=3, smoothing_windows=1, confidence=0.,
                 min_duration_s=1., max_duration_s=30.):
        self.classes=tuple(class_names)
        if (not isinstance(bank_id,str) or not bank_id.strip() or len(self.classes)<2
                or len(set(self.classes))!=len(self.classes) or rest_label not in self.classes
                or 'Unknown' in self.classes or any(not isinstance(c,str) or not c.strip() for c in self.classes)):
            raise ValueError('Explicit source bank and unique class axis required')
        if sample_rate_hz not in (200.,250.):raise ValueError('Native200/250Hz required')
        if any(isinstance(v,bool) or not isinstance(v,int) or not 1<=v<=50 for v in (confirmations,smoothing_windows)):
            raise ValueError('Integer confirmation/smoothing counts in1..50 required')
        if not np.isfinite(confidence) or not 0<=confidence<=1:raise ValueError('Probability confidence required')
        if not 0<min_duration_s<max_duration_s<=30:raise ValueError('Finite1..30s capture contract required')
        self.bank_id=bank_id;self.rest=rest_label;self.rate=float(sample_rate_hz)
        self.size=round(.2*self.rate);self.hop=round(.04*self.rate)
        self.confirmations=confirmations;self.smoothing=smoothing_windows;self.confidence=float(confidence)
        self.minimum=math.ceil(min_duration_s*self.rate);self.maximum=math.ceil(max_duration_s*self.rate)
        self.reset()

    def reset(self):
        self.next_sample=None;self.next_window_end=None;self.recording=None
        self.buffer=np.empty((0,8),np.float32);self.buffer_start=None
        self.probability_history=deque(maxlen=self.smoothing)
        self.stable=None;self.pending=None;self.count=0;self.pending_boundary=None
        self.start=None;self.known_start=False;self.ordinal=0;self.discarded=0

    def finish(self):
        censored=self.stable not in (None,self.rest,'Unknown') or self.count>0
        self.reset()
        return censored

    def feed(self,samples,first_sample_index,*,recording_id,probabilities,window_end_indices,bank_id):
        x=np.asarray(samples);q=np.asarray(probabilities,float);ends=np.asarray(window_end_indices)
        if (bank_id!=self.bank_id or not isinstance(recording_id,str) or not recording_id.strip()
                or isinstance(first_sample_index,bool) or not isinstance(first_sample_index,int) or first_sample_index<0
                or x.ndim!=2 or x.shape[1]!=8 or not len(x) or not np.isfinite(x).all()
                or ends.ndim!=1 or ends.dtype.kind not in 'iu' or q.shape!=(len(ends),len(self.classes))
                or not np.isfinite(q).all() or np.any(q<0) or not np.allclose(q.sum(1),1.,rtol=0,atol=1e-12)):
            self.reset();raise ValueError('Finite native samples and explicit source probability/window axes required')
        if self.next_sample is not None and (self.next_sample!=first_sample_index or self.recording!=recording_id):
            self.reset();raise ValueError('Gap, overlap or changed recording invalidates detection')
        first_end=first_sample_index+self.size if self.next_window_end is None else self.next_window_end
        stop=first_sample_index+len(x)
        expected=np.arange(first_end,stop+1,self.hop,dtype=np.int64)
        if not np.array_equal(ends,expected):
            self.reset();raise ValueError('Exact causal native window grid required; no omitted or future windows')
        if self.buffer_start is None:self.buffer_start=first_sample_index
        self.buffer=np.concatenate([self.buffer,x.astype(np.float32)])
        self.recording=recording_id;events=[]
        for probability,end in zip(q,ends):
            self.probability_history.append(probability.copy())
            smooth=np.mean(self.probability_history,axis=0)
            label=self.classes[int(smooth.argmax())] if smooth.max()>=self.confidence else 'Unknown'
            if label==self.stable:
                self.pending=None;self.count=0
            else:
                if label!=self.pending:
                    self.pending=label;self.count=0;self.pending_boundary=int(end)-self.size//2
                self.count+=1
                if self.count>=self.confirmations:
                    boundary=self.pending_boundary;old=self.stable
                    if old not in (None,self.rest,'Unknown'):
                        length=boundary-self.start
                        if label!='Unknown' and self.known_start and self.minimum<=length<=self.maximum:
                            a=self.start-self.buffer_start;b=boundary-self.buffer_start
                            if not 0<=a<b<=len(self.buffer):raise RuntimeError('Native segment retention invariant failed')
                            native=self.buffer[a:b].copy();native.setflags(write=False)
                            events.append(dict(trial_id=f'{recording_id}:class_estimated{self.ordinal}',
                                recording_id=recording_id,start=self.start,end=boundary,
                                algorithmic_available_at_sample_index=int(end),
                                returned_after_sample_index=stop,estimated_class=old,
                                boundary_kind='estimated',emg=native))
                            self.ordinal+=1
                        else:self.discarded+=1
                    self.stable=label;self.start=boundary if label not in (self.rest,'Unknown') else None
                    self.known_start=old is not None;self.pending=None;self.count=0
            if self.start is not None and int(end)-self.start>self.maximum+self.size:
                self.known_start=False
        self.next_sample=stop;self.next_window_end=first_end+len(ends)*self.hop
        retain=self.maximum+self.size+self.hop*(self.confirmations+self.smoothing)
        if len(self.buffer)>retain:
            excess=len(self.buffer)-retain;self.buffer=self.buffer[excess:].copy();self.buffer_start+=excess
        return events
