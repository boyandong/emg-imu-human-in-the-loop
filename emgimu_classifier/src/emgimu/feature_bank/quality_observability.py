"""Explicit longest-flatline and availability-aware quality observation candidate."""
import numpy as np
from .families import QualityFamily


def longest_flatline_ratio(x,tolerance):
    edges=np.abs(np.diff(x,axis=1))<=np.asarray(tolerance)[None,None,:]
    current=np.zeros((len(x),x.shape[2]),dtype=int);longest=current.copy()
    for step in range(edges.shape[1]):
        current=np.where(edges[:,step],current+1,0);longest=np.maximum(longest,current)
    return longest/max(x.shape[1],1)


def mean_channel_correlation(x,ring):
    centered=x-x.mean(1,keepdims=True);cov=np.einsum('ntc,ntd->ncd',centered,centered)
    scale=np.sqrt(np.maximum(np.diagonal(cov,axis1=1,axis2=2),0))
    correlation=cov/np.maximum(scale[:,:,None]*scale[:,None,:],1e-10)
    channels=x.shape[2]
    if ring:
        return np.stack([(correlation[:,c,(c-1)%channels]+correlation[:,c,(c+1)%channels])/2 for c in range(channels)],axis=1)
    return (correlation.sum(2)-np.diagonal(correlation,axis1=1,axis2=2))/max(channels-1,1)


class QualityObservabilityFamily(QualityFamily):
    family_id='F9v2_quality_observability'

    def __init__(self,*,pre_highpass_available=False,ring_topology=False,low_frequency_hz=10.,**kwargs):
        super().__init__(**kwargs)
        self.pre_highpass_available=bool(pre_highpass_available);self.ring_topology=bool(ring_topology)
        self.low_frequency_hz=float(low_frequency_hz)

    def fit(self,batch,labels=None):
        if self.ring_topology and batch.channels!=8:raise ValueError('Ring-neighbor quality requires verified native eight-channel topology')
        if not 0<self.low_frequency_hz<batch.sample_rate_hz/2:raise ValueError('Low-frequency band must lie below Nyquist')
        super().fit(batch,labels)
        x=np.asarray(batch.emg,dtype=float);self.channels_v2_=batch.channels;self.rate_=batch.sample_rate_hz;self.samples_=x.shape[1]
        self.flat_tolerance_=np.maximum(np.median(np.abs(np.diff(x,axis=1)),axis=(0,1))*1e-4,1e-10)
        correlation=mean_channel_correlation(x,self.ring_topology)
        self.correlation_median_=np.median(correlation,axis=0)
        self.correlation_mad_=np.maximum(np.median(np.abs(correlation-self.correlation_median_),axis=0)*1.4826,1e-4)
        frequency=np.fft.rfftfreq(self.samples_,1/self.rate_)
        line=np.abs(frequency-self.line_frequency_hz)<=1
        neighbor=((frequency>=self.line_frequency_hz-6)&(frequency<=self.line_frequency_hz-3))|((frequency>=self.line_frequency_hz+3)&(frequency<=self.line_frequency_hz+6))
        self.availability_={'adc_clipping':self.adc_min is not None and self.adc_max is not None,
            'line_noise':self.line_frequency_hz<self.rate_/2 and bool(line.any() and neighbor.any()),
            'low_frequency_pre_highpass':self.pre_highpass_available and bool(((frequency>=20)&(frequency<=min(450,self.rate_*.475))).any())}
        appended=tuple(f'F9v2.{metric}.ch{c+1}' for metric in ('longest_flatline_ratio','correlation_anomaly','low_frequency_power_ratio') for c in range(batch.channels))
        self.names_v2_=super().feature_names+appended+tuple(f'F9v2.available.{k}' for k in self.availability_)
        return self

    def transform(self,batch):
        self._check()
        if batch.channels!=self.channels_v2_ or batch.sample_rate_hz!=self.rate_ or batch.emg.shape[1]!=self.samples_:
            raise ValueError('Quality observability sensor/window contract mismatch')
        legacy=super().transform(batch);x=np.asarray(batch.emg,dtype=float)
        flat=longest_flatline_ratio(x,self.flat_tolerance_)
        correlation=np.abs(mean_channel_correlation(x,self.ring_topology)-self.correlation_median_)/self.correlation_mad_
        low=np.zeros_like(flat)
        if self.availability_['low_frequency_pre_highpass']:
            centered=(x-x.mean(1,keepdims=True))*np.hanning(self.samples_)[None,:,None]
            power=np.abs(np.fft.rfft(centered,axis=1))**2;frequency=np.fft.rfftfreq(self.samples_,1/self.rate_)
            low=power[:,frequency<=self.low_frequency_hz].sum(1)/np.maximum(power[:,(frequency>=20)&(frequency<=min(450,self.rate_*.475))].sum(1),1e-10)
        available=np.broadcast_to(np.array(list(self.availability_.values()),dtype=float)[None,:],(len(x),3))
        return np.concatenate((legacy,flat,correlation,low,available),axis=1).astype(np.float32)

    @property
    def feature_names(self):
        self._check();return self.names_v2_
