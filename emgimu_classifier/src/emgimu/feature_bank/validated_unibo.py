"""Explicit adapters for existing native UniBo formulas, without rewriting them."""
import numpy as np
from .core import FeatureFamily,FeatureBatch


class ValidatedUniBoFamily(FeatureFamily):
    def __init__(self,group='G5'):
        if group not in ('G0','G5'):raise ValueError('validated adapter supports G0 or G5')
        self.group=group;self.family_id=f'{"F0" if group=="G0" else "F5"}_validated_unibo_{group}'
        self.transformer_=None

    def fit(self,batch:FeatureBatch,labels=None,sample_weight=None):
        from emgimu.datasets.unibo_physiology import PhysiologyFeatureTransformer
        self._validate(batch)
        if labels is None:raise ValueError('supply explicit training labels')
        w=np.ones(batch.windows) if sample_weight is None else np.asarray(sample_weight)
        self.transformer_=PhysiologyFeatureTransformer((self.group,)).fit(batch.emg,np.asarray(labels),w)
        self.fitted_=True;return self

    @staticmethod
    def _validate(batch):
        if batch.channels!=4:raise ValueError('validated UniBo formulas require the native four named muscles')
        if batch.sample_rate_hz!=200:raise ValueError('validated benchmark adapter requires the established 200 Hz processed protocol')

    def transform(self,batch):
        self._check();self._validate(batch)
        return self.transformer_.transform(batch.emg,np.ones(batch.windows,dtype=np.int16))

    @property
    def feature_names(self):
        self._check();return self.transformer_.feature_names_
