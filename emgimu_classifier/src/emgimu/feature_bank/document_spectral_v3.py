"""F4a/b/c with additive epsilon and raw spectral entropy from the goal.

PSD convention is source-fixed: demeaned Hann, rFFT magnitude squared / T.
Cepstral summary uses orthonormal DCT-II, excluding DC; it is not CCA.
"""
import numpy as np
from .core import FeatureBatch,FeatureFamily
from .families import SpectralStateFamily


EPS=1e-10


class DocumentSpectralStateV3(FeatureFamily):
    family_id='F4_document_spectral_v3'

    def __init__(self,band_count=4,cepstral_coefficients=4):
        for value in (band_count,cepstral_coefficients):
            if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1:
                raise ValueError('Positive integer band and cepstral counts required')
        self.band_count=band_count;self.cepstral_coefficients=cepstral_coefficients

    def fit(self,batch:FeatureBatch,labels=None):
        bins=batch.emg.shape[1]//2+1
        if self.cepstral_coefficients>=bins:
            raise ValueError('Cepstral count must exclude DC and fit the source frequency grid')
        definition=SpectralStateFamily(self.band_count,self.cepstral_coefficients).fit(batch)
        self.bands_=definition.bands_;self.channels_=batch.channels
        self.samples_=batch.emg.shape[1];self.rate_=batch.sample_rate_hz
        self.orders_=tuple(range(1,self.cepstral_coefficients+1))
        frequency=np.fft.rfftfreq(self.samples_,1/self.rate_)
        self.masks_=tuple((frequency>=a)&(frequency<=b if i==len(self.bands_)-1 else frequency<b)
                          for i,(a,b) in enumerate(self.bands_))
        if any(not mask.any() for mask in self.masks_):
            raise ValueError('Source frequency grid has an empty EMG band')
        self._names=tuple(f'F4v3.band{b+1}.orientation.ch{c+1}'
                         for b in range(len(self.bands_)) for c in range(self.channels_))
        self._names+=tuple(f'F4v3.{metric}.ch{c+1}' for metric in
                           ('total_power','centroid','median_frequency','entropy') for c in range(self.channels_))
        self._names+=tuple(f'F4v3.cepstral{k}.{metric}' for k in self.orders_ for metric in ('mean','std'))
        self.fitted_=True;return self

    def transform(self,batch:FeatureBatch):
        self._check()
        if batch.channels!=self.channels_ or batch.emg.shape[1]!=self.samples_ or batch.sample_rate_hz!=self.rate_:
            raise ValueError('F4 source channel/sample/rate contract differs')
        x=np.asarray(batch.emg,dtype=np.float64)
        tapered=(x-x.mean(axis=1,keepdims=True))*np.hanning(self.samples_)[None,:,None]
        power=np.abs(np.fft.rfft(tapered,axis=1))**2/self.samples_
        frequency=np.fft.rfftfreq(self.samples_,1/self.rate_)
        bands=[]
        for mask in self.masks_:
            energy=power[:,mask].sum(axis=1)
            bands.append(energy/(np.linalg.norm(energy,axis=1,keepdims=True)+EPS))
        total=power.sum(axis=1)
        centroid=(power*frequency[None,:,None]).sum(axis=1)/(total+EPS)
        median=frequency[np.argmax(np.cumsum(power,axis=1)>=.5*total[:,None,:],axis=1)]
        q=power/(total[:,None,:]+EPS)
        entropy=-(q*np.log(q+EPS)).sum(axis=1)
        log_spectrum=np.log(power+EPS);bins=len(frequency);cepstral=[]
        for order in self.orders_:
            basis=np.sqrt(2./bins)*np.cos(np.pi*(np.arange(bins)+.5)*order/bins)
            coefficient=np.sum(log_spectrum*basis[None,:,None],axis=1)
            cepstral.extend((coefficient.mean(axis=1,keepdims=True),coefficient.std(axis=1,keepdims=True)))
        output=np.concatenate((*bands,total,centroid,median,entropy,*cepstral),axis=1)
        if not np.isfinite(output).all():raise ValueError('Nonfinite document spectral features')
        return output.astype(np.float32)

    @property
    def feature_names(self):
        self._check();return self._names
