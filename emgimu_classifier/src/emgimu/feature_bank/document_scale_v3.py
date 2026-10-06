"""Document F1 additive-epsilon pattern, with global scale exposed separately."""
import numpy as np
from .core import FeatureFamily

EPS=1e-10


def document_activation_coordinates(windows):
    x=np.asarray(windows,dtype=np.float64)
    if x.ndim!=3 or not x.shape[1] or not x.shape[2] or not np.isfinite(x).all():
        raise ValueError('Finite [windows,samples,channels] input required')
    rms=np.sqrt(np.mean(x*x,axis=1))
    scale=np.sqrt(np.mean(rms*rms,axis=1,keepdims=True))
    return rms/(scale+EPS),np.log(scale+EPS)


class DocumentScalePatternV3(FeatureFamily):
    family_id='F1_document_scale_pattern_v3'

    def fit(self,batch,labels=None):
        self.channels_=batch.channels;self.rate_=batch.sample_rate_hz
        self.fitted_=True;return self

    def _coordinates(self,batch):
        self._check()
        if batch.channels!=self.channels_ or batch.sample_rate_hz!=self.rate_:
            raise ValueError('F1 source channel/rate contract differs')
        return document_activation_coordinates(batch.emg)

    def transform(self,batch):
        return self._coordinates(batch)[0]

    def activation_context(self,batch):
        """One context coordinate; deliberately absent from the H feature vector."""
        return self._coordinates(batch)[1]

    @property
    def feature_names(self):
        self._check();return tuple(f'F1v3.rms_pattern.ch{c+1}' for c in range(self.channels_))
