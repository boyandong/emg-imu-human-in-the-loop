"""Opt-in Song250Hz F0 runtime: source-frozen models and causal preprocessing."""
import copy
import hashlib
import pickle
import numpy as np
from scipy.signal import sosfilt
from .core import FeatureBatch


class SongCausalFilterV1:
    def __init__(self, stages):
        self.stages=tuple(np.asarray(s,dtype=float).copy() for s in stages)
        if len(self.stages)!=3 or any(s.ndim!=2 or s.shape[1]!=6 or not np.isfinite(s).all() for s in self.stages):
            raise ValueError('Three finite source-frozen SOS stages required')
        self.reset()

    def reset(self):
        self.states=[np.zeros((len(s),2,8)) for s in self.stages]
        self.samples=0

    def process(self, raw):
        x=np.asarray(raw,dtype=float)
        if x.ndim!=2 or x.shape[1]!=8 or not np.isfinite(x).all():
            raise ValueError('Finite raw samples by eight channels required')
        if not len(x):return np.empty((0,8),dtype=np.float32)
        for i,stage in enumerate(self.stages):x,self.states[i]=sosfilt(stage,x,axis=0,zi=self.states[i])
        self.samples+=len(x)
        return x.astype(np.float32)


class SongF0RuntimeV1:
    preprocessing_id='song250_causal_hp40_order4_notch50_100_Q30_zero_session_initial_v1'

    def __init__(self, models, *, source_trial_ids, source_state_sha256, class_names, filter_stages):
        if set(models)!={'pooled_source_threshold','rest_only_threshold'}:
            raise ValueError('Both unchanged source threshold arms required')
        if len(class_names)!=4 or len(set(class_names))!=4:raise ValueError('Explicit four-class native order required')
        for arm,state in models.items():
            if hashlib.sha256(pickle.dumps(state)).hexdigest()!=source_state_sha256[arm]:
                raise ValueError('Source fitted family/model fingerprint differs')
            if set(state[1][-1].classes_)!=set(class_names):raise ValueError('Source class axis differs')
        self.models_=copy.deepcopy(models);self.classes_=tuple(class_names)
        self.source_trial_ids_=tuple(sorted(source_trial_ids))
        if not self.source_trial_ids_ or len(set(self.source_trial_ids_))!=len(self.source_trial_ids_):
            raise ValueError('Unique source trial identities required')
        self.filter_stages_=SongCausalFilterV1(filter_stages).stages

    def new_filter(self):return SongCausalFilterV1(self.filter_stages_)

    def predict_windows(self,batch,trial_ids,*,window_offsets,preprocessing_id):
        if preprocessing_id!=self.preprocessing_id:raise ValueError('Explicit matching causal preprocessing required')
        if (batch.sample_rate_hz,batch.emg.shape[1],batch.channels)!=(250.,50,8):
            raise ValueError('Song requires eight channels at250Hz and50 samples per window')
        ids=np.asarray(trial_ids,dtype=object);offsets=np.asarray(window_offsets)
        if ids.shape!=(batch.windows,) or not len(ids) or any(not isinstance(t,str) or not t.strip() for t in ids):
            raise ValueError('Explicit nonempty window-aligned trial identities required')
        if offsets.shape!=ids.shape or offsets.dtype.kind not in 'iu' or np.any(offsets<0):
            raise ValueError('Aligned integer chronological window offsets required')
        unique=np.unique(ids)
        if set(unique)&set(self.source_trial_ids_):raise ValueError('Source trials cannot become evaluation')
        order=[]
        for trial in unique:
            positions=np.flatnonzero(ids==trial)
            if not np.array_equal(np.sort(offsets[positions]),np.arange(len(positions))):
                raise ValueError('Trial window offsets must be distinct contiguous from zero')
            order.append(positions[np.argsort(offsets[positions])])
        outputs={}
        for arm,(family,model) in self.models_.items():
            columns=[list(model[-1].classes_).index(c) for c in self.classes_]
            q=model.predict_proba(family.transform(batch))[:,columns]
            # This native protocol averages probabilities, not features.
            output=np.stack([q[positions].mean(0) for positions in order])
            outputs[arm]={'probabilities':output,'predicted_labels':tuple(self.classes_[i] for i in output.argmax(1))}
        return {'trial_ids':tuple(unique),'class_names':self.classes_,'arms':outputs,
                'preprocessing_id':self.preprocessing_id,'default_promoted':False}

    def predict_recording(self,raw,window_starts,trial_ids,*,window_offsets,sample_rate_hz):
        if sample_rate_hz!=250.:raise ValueError('Raw recording rate must be250Hz')
        starts=np.asarray(window_starts)
        if starts.ndim!=1 or not len(starts) or starts.dtype.kind not in 'iu' or np.any(starts<0):
            raise ValueError('Explicit integer recording window starts required')
        data=np.asarray(raw)
        if data.ndim!=2 or np.any(starts+50>len(data)):raise ValueError('Windows must lie wholly inside recording')
        filtered=self.new_filter().process(data)
        windows=np.stack([filtered[start:start+50] for start in starts])
        return self.predict_windows(FeatureBatch(windows,250.),trial_ids,window_offsets=window_offsets,
                                    preprocessing_id=self.preprocessing_id)
