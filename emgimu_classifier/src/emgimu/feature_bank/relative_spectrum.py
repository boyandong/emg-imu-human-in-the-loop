"""Log-band observations and explicit calibration-relative spectral coordinates."""
import numpy as np
from .core import FeatureBatch, FeatureFamily
from .families import SpectralStateFamily


class LogBandEnergyFamily(FeatureFamily):
    family_id='F4d_log_band_energy'

    def fit(self,batch:FeatureBatch,labels=None):
        definition=SpectralStateFamily().fit(batch)
        self.bands_=definition.bands_;self.channels_=batch.channels;self.rate_=batch.sample_rate_hz
        self.fitted_=True;return self

    def transform(self,batch:FeatureBatch):
        self._check()
        if batch.channels!=self.channels_ or batch.sample_rate_hz!=self.rate_:
            raise ValueError('Log-band input must match fitted channel count and sample rate')
        x=np.asarray(batch.emg,dtype=float)
        tapered=(x-x.mean(1,keepdims=True))*np.hanning(x.shape[1])[None,:,None]
        psd=np.abs(np.fft.rfft(tapered,axis=1))**2/max(x.shape[1],1)
        frequency=np.fft.rfftfreq(x.shape[1],1/self.rate_);pieces=[]
        for i,(low,high) in enumerate(self.bands_):
            mask=(frequency>=low)&(frequency<=high if i==len(self.bands_)-1 else frequency<high)
            pieces.append(np.log(psd[:,mask].sum(1)+1e-10))
        return np.concatenate(pieces,axis=1).astype(np.float32)

    @property
    def feature_names(self):
        self._check()
        return tuple(f'F4d.log_band{b+1}.ch{c+1}' for b in range(len(self.bands_)) for c in range(self.channels_))


class RelativeSpectrumCoordinates:
    def fit_calibration(self,log_band_energy,sample_weight=None):
        x=np.asarray(log_band_energy,dtype=float)
        if x.ndim!=2 or not len(x) or not np.all(np.isfinite(x)):raise ValueError('Finite calibration log-band matrix required')
        self.reference_=np.average(x,axis=0,weights=sample_weight);return self

    def transform(self,log_band_energy):
        if not hasattr(self,'reference_'):raise RuntimeError('Explicit calibration reference required')
        x=np.asarray(log_band_energy,dtype=float)
        if x.ndim!=2 or x.shape[1]!=len(self.reference_) or not np.all(np.isfinite(x)):raise ValueError('Invalid log-band observation')
        return (x-self.reference_).astype(np.float32)
