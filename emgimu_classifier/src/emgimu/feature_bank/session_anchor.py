"""Immutable long-term/session prototype fusion in a fixed source coordinate system."""
import copy
import numpy as np
from .calibration import PersonalAnchor


class SessionPrototypeAnchor:
    def fit_long_term(self, features, labels):
        self.anchor_=PersonalAnchor().fit(features,labels)
        y=np.asarray(labels)
        self.counts_=np.array([np.sum(y==h) for h in self.anchor_.classes_],dtype=float)
        return self

    def from_calibration(self, features=None, labels=None, *, mode='blended'):
        if mode not in ('long_term','local','blended'):raise ValueError('Unknown session prototype mode')
        if not hasattr(self,'anchor_'):raise RuntimeError('Long-term profile required')
        result=copy.deepcopy(self.anchor_)
        if mode=='long_term' or features is None:
            return result,np.ones(len(self.counts_))
        x=np.asarray(features,dtype=float);y=np.asarray(labels)
        if x.ndim!=2 or len(x)!=len(y) or x.shape[1]!=result.prototypes_.shape[1] or not np.all(np.isfinite(x)):
            raise ValueError('Invalid session calibration matrix')
        if set(y)!=set(result.classes_):raise ValueError('Session calibration must cover exactly the long-term classes')
        local=np.stack([x[y==h].mean(0) for h in result.classes_])
        count=np.array([np.sum(y==h) for h in result.classes_],dtype=float)
        beta=self.counts_/(self.counts_+count) if mode=='blended' else np.zeros(len(count))
        result.prototypes_=beta[:,None]*result.prototypes_+(1-beta[:,None])*local
        # Source scale remains fixed so prototype modes use the same distance metric.
        return result,beta
