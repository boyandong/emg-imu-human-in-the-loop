"""Independent document candidates; do not rewrite historical reference families."""
import numpy as np
from .core import FeatureBatch
from .families import LocalDetailFamily, CspSpatialFamily, EPS, _sym_power


class RestNoiseLocalDetailFamily(LocalDetailFamily):
    family_id = 'F0_rest_noise_detail'

    def __init__(self, *, rest_label):
        super().__init__()
        self.rest_label = rest_label

    def fit(self, batch: FeatureBatch, labels=None):
        if labels is None or np.asarray(labels).shape != (batch.windows,):
            raise ValueError('Explicit source labels and native Rest are required')
        rest = np.asarray(labels) == self.rest_label
        if not rest.any():
            raise ValueError('Noise thresholds require source/calibration Rest windows')
        # Rest windows are isolated; never concatenate across trial/window boundaries.
        super().fit(batch.take(np.flatnonzero(rest)))
        self.threshold_source_ = 'median absolute Rest adjacent difference + 3 * 1.4826 MAD'
        self.rest_windows_ = int(rest.sum())
        return self


class DocumentCspFamily(CspSpatialFamily):
    family_id = 'F2b_document_csp'

    def __init__(self, components_per_tail=2, regularization=1e-5):
        super().__init__(components_per_tail,regularization)

    @staticmethod
    def normalized_second_moments(x):
        x = np.asarray(x,dtype=float)
        matrices = np.einsum('ntc,ntd->ncd',x,x)
        return matrices / (np.trace(matrices,axis1=1,axis2=2)[:,None,None]+EPS)

    def fit(self,batch: FeatureBatch,labels=None):
        y = np.asarray(labels)
        if y.shape != (batch.windows,) or len(np.unique(y))<2:
            raise ValueError('CSP requires aligned source labels and >=2 native classes')
        if not isinstance(self.components_per_tail,int) or self.components_per_tail<1:
            raise ValueError('Positive integer tail component count required')
        if not np.isfinite(self.regularization) or self.regularization<=0:
            raise ValueError('Positive source-fixed regularization required')
        matrices = self.normalized_second_moments(batch.emg)
        self.classes_ = np.unique(y); filters=[]; self.eigenvalues_=[]
        keep = min(self.components_per_tail,batch.channels//2)
        if keep<1: raise ValueError('CSP requires >=2 channels')
        for label in self.classes_:
            positive = matrices[y==label].mean(0)
            negative = matrices[y!=label].mean(0)
            total = positive+negative+self.regularization*np.eye(batch.channels)
            inverse = _sym_power(total,-.5)
            whitened = inverse @ positive @ inverse
            values,vectors = np.linalg.eigh((whitened+whitened.T)*.5)
            indices = np.r_[np.arange(len(values)-keep,len(values)),np.arange(keep)]
            filters.append(inverse @ vectors[:,indices]); self.eigenvalues_.append(values[indices])
        self.filters_ = np.concatenate(filters,axis=1)
        self.fitted_ = True
        return self

    def transform(self,batch):
        self._check()
        if batch.channels!=self.filters_.shape[0]: raise ValueError('CSP channel mismatch')
        projected = np.einsum('ntc,ck->ntk',np.asarray(batch.emg,dtype=float),self.filters_)
        variance = np.var(projected,axis=1)
        return np.log(variance/(variance.sum(1,keepdims=True)+EPS)+EPS).astype(np.float32)
