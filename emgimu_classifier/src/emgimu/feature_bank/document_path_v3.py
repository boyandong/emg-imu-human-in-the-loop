"""Document F5 complete-envelope paths, with additive epsilon and trial provenance."""
import numpy as np
from .core import FeatureFamily
from .temporal import require_complete_sequences

EPS=1e-10


def document_envelope_path(window):
    x=np.asarray(window,dtype=np.float64)
    if x.ndim!=2 or len(x)<3 or not x.shape[1] or not np.isfinite(x).all() or np.any(x<0):
        raise ValueError('Finite nonnegative complete envelope path required')
    return x/(np.linalg.norm(x,axis=1,keepdims=True)+EPS)


def document_dtw_distance(first,second,band):
    a,b=np.asarray(first,dtype=float),np.asarray(second,dtype=float)
    if a.ndim!=2 or b.ndim!=2 or not len(a) or not len(b) or a.shape[1]!=b.shape[1] or not a.shape[1]:
        raise ValueError('Nonempty matching DTW coordinates required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Finite DTW coordinates required')
    if isinstance(band,bool) or not isinstance(band,(int,np.integer)) or band<0:
        raise ValueError('Nonnegative integer warp band required')
    if abs(len(a)-len(b))>band:raise ValueError('Warp band cannot reach the sequence endpoint')
    costs=np.full((len(a)+1,len(b)+1),np.inf);lengths=np.zeros_like(costs,dtype=np.int64)
    costs[0,0]=0.
    for i in range(1,len(a)+1):
        for j in range(max(1,i-band),min(len(b),i+band)+1):
            # Minimize cumulative Euclidean cost; among exactly equal costs use
            # shortest path. Divide the selected path cost by its own length.
            previous=min(((costs[i-1,j-1],lengths[i-1,j-1]),
                          (costs[i-1,j],lengths[i-1,j]),
                          (costs[i,j-1],lengths[i,j-1])),key=lambda v:(v[0],v[1]))
            costs[i,j]=previous[0]+np.linalg.norm(a[i-1]-b[j-1]);lengths[i,j]=previous[1]+1
    return float(costs[-1,-1]/lengths[-1,-1])


class DocumentTemporalTemplatesV3(FeatureFamily):
    family_id='F5b_document_dtw_v3'

    def __init__(self,band_fraction=.1):
        self.band_fraction=float(band_fraction)

    @staticmethod
    def _ids(batch,trial_ids):
        ids=np.asarray(trial_ids,dtype=object)
        if ids.ndim!=1 or len(ids)!=batch.windows or not len(ids):
            raise ValueError('One explicit trial ID per complete sequence required')
        if any(not isinstance(i,str) or not i.strip() for i in ids):
            raise ValueError('Nonempty string trial IDs required')
        if len(set(ids))!=len(ids):raise ValueError('Duplicate complete calibration/evaluation trials')
        return ids

    def fit(self,batch,labels=None,*,trial_ids=None):
        require_complete_sequences(batch);ids=self._ids(batch,trial_ids)
        y=np.asarray(labels)
        if y.ndim!=1 or len(y)!=batch.windows:raise ValueError('Aligned one-dimensional calibration labels required')
        if np.issubdtype(y.dtype,np.number) and not np.isfinite(y).all():raise ValueError('Finite labels required')
        if not np.isfinite(self.band_fraction) or not 0<self.band_fraction<=1:raise ValueError('Warp fraction must be in (0,1]')
        self.channels_=batch.channels;self.samples_=batch.emg.shape[1]
        self.rate_=batch.sample_rate_hz;self.band_=max(1,round(self.samples_*self.band_fraction))
        self.classes_=np.unique(y);self.calibration_trial_ids_=frozenset(ids)
        paths=[document_envelope_path(x) for x in batch.emg]
        self.templates_=[];self.medoid_trial_ids_=[]
        for label in self.classes_:
            positions=np.flatnonzero(y==label);distances=np.zeros((len(positions),len(positions)))
            for i in range(len(positions)):
                for j in range(i+1,len(positions)):
                    distance=document_dtw_distance(paths[positions[i]],paths[positions[j]],self.band_)
                    distances[i,j]=distances[j,i]=distance
            selected=positions[np.argmin(distances.sum(axis=1))]
            self.templates_.append(paths[selected].copy());self.medoid_trial_ids_.append(ids[selected])
        self.calibration_distances_=self._distances(paths)
        self.fitted_=True;return self

    def _distances(self,paths):
        return np.asarray([[document_dtw_distance(path,template,self.band_) for template in self.templates_]
                           for path in paths],dtype=np.float64)

    def transform(self,batch,*,trial_ids=None):
        self._check();require_complete_sequences(batch);ids=self._ids(batch,trial_ids)
        if self.calibration_trial_ids_.intersection(ids):raise ValueError('Evaluation overlaps calibration trials')
        if batch.channels!=self.channels_ or batch.emg.shape[1]!=self.samples_ or batch.sample_rate_hz!=self.rate_:
            raise ValueError('Complete path channel/sample/rate contract differs')
        return self._distances([document_envelope_path(x) for x in batch.emg])

    @property
    def feature_names(self):
        self._check();return tuple(f'F5bv3.dtw.{h}' for h in self.classes_)


class DocumentPathSignatureV3(FeatureFamily):
    family_id='F5c_document_signature_v3'

    def fit(self,batch,labels=None):
        require_complete_sequences(batch)
        for x in batch.emg:document_envelope_path(x)
        self.channels_=batch.channels;self.fitted_=True;return self

    def transform(self,batch):
        self._check();require_complete_sequences(batch)
        if batch.channels!=self.channels_:raise ValueError('Signature channel contract differs')
        paths=np.stack([document_envelope_path(x) for x in batch.emg])
        delta=np.diff(paths-paths[:,:1],axis=1)
        level1=delta.sum(axis=1);prefix=np.cumsum(delta,axis=1)-delta
        level2=np.einsum('ntc,ntd->ncd',prefix,delta)+.5*np.einsum('ntc,ntd->ncd',delta,delta)
        return np.concatenate((level1,level2.reshape(len(paths),-1)),axis=1)

    @property
    def feature_names(self):
        self._check()
        return tuple(f'F5cv3.level1.ch{c+1}' for c in range(self.channels_))+tuple(
            f'F5cv3.level2.ch{a+1}.ch{b+1}' for a in range(self.channels_) for b in range(self.channels_))
