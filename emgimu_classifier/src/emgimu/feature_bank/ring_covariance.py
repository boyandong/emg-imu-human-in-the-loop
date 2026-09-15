"""Document F3c candidate; distinct from the historical envelope-ring proxy."""
import numpy as np

from .core import FeatureBatch, FeatureFamily
from .families import _covariances


class RawRingCovarianceFamily(FeatureFamily):
    family_id = 'F3c_raw_ring_covariance'

    def __init__(self, *, ring_topology: bool = False, shrinkage: float = .05):
        self.ring_topology = ring_topology
        self.shrinkage = shrinkage
        self.channels_ = None

    def fit(self, batch: FeatureBatch, labels=None):
        if not self.ring_topology or batch.channels < 4:
            raise ValueError('verified circular channel topology is required')
        if not 0 <= self.shrinkage < 1:
            raise ValueError('shrinkage must be in [0,1)')
        self.channels_ = batch.channels
        self.fitted_ = True
        return self

    def transform(self, batch: FeatureBatch):
        self._check()
        if batch.channels != self.channels_:
            raise ValueError('channel count differs from source state')
        x = np.asarray(batch.emg, dtype=float)
        midpoint = x.shape[1] // 2
        covariance = _covariances(x, self.shrinkage)
        early = _covariances(x[:, :midpoint], self.shrinkage)
        late = _covariances(x[:, midpoint:], self.shrinkage)
        index = np.arange(self.channels_)
        pieces = []
        for lag in range(1, self.channels_ // 2 + 1):
            other = (index + lag) % self.channels_
            values = covariance[:, index, other]
            pieces.extend((values.mean(1), np.median(values, axis=1), values.std(1),
                           np.quantile(values, .25, axis=1), np.quantile(values, .75, axis=1),
                           np.abs(late[:, index, other].mean(1) - early[:, index, other].mean(1))))
        return np.stack(pieces, axis=1).astype(np.float32)

    @property
    def feature_names(self):
        self._check()
        return tuple(f'F3c.raw_ringcov.lag{lag}.{metric}'
                     for lag in range(1, self.channels_ // 2 + 1)
                     for metric in ('mean', 'median', 'std', 'q25', 'q75', 'temporal_delta'))
