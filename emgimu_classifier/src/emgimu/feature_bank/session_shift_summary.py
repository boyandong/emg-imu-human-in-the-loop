"""Class-matched family-specific session signatures from source and calibration only."""
import numpy as np
from .families import QualityFamily,RingGeometryFamily,_covariances,_sym_power
from .relative_spectrum import LogBandEnergyFamily
from .activation_profile import trial_weights


def affine_covariance_distance(first,second):
    first=np.asarray(first,dtype=float);second=np.asarray(second,dtype=float)
    if first.ndim!=2 or first.shape[0]!=first.shape[1] or first.shape!=second.shape:
        raise ValueError('Matching square SPD covariance matrices required')
    if np.min(np.linalg.eigvalsh(first))<=0 or np.min(np.linalg.eigvalsh(second))<=0:
        raise ValueError('Positive definite covariance matrices required')
    inverse=_sym_power(first,-.5);relative=inverse@second@inverse
    values=np.linalg.eigvalsh((relative+relative.T)/2)
    return float(np.linalg.norm(np.log(np.maximum(values,1e-10))))


class FamilySessionShiftSummary:
    def fit_long_term(self,batch,labels,trials,*,ring_topology=False,rest_label=None):
        if ring_topology and batch.channels!=8:raise ValueError('This ring summary requires verified native eight-channel topology')
        self.channels_=batch.channels;self.rate_=batch.sample_rate_hz;self.rest_label_=rest_label
        self.spectral_=LogBandEnergyFamily().fit(batch)
        self.quality_=QualityFamily().fit(batch)
        self.ring_=RingGeometryFamily().fit(batch) if ring_topology else None
        self.reference_=self._profiles(batch,labels,trials)
        return self

    def _profiles(self,batch,labels,trials):
        if batch.channels!=self.channels_ or batch.sample_rate_hz!=self.rate_:raise ValueError('Session sensor contract mismatch')
        y=np.asarray(labels);trials=np.asarray(trials)
        if len(y)!=batch.windows or len(trials)!=batch.windows:raise ValueError('Window labels/trial IDs must align')
        if any(len(set(y[trials==t]))!=1 for t in set(trials)):raise ValueError('Mixed-label calibration trials unsupported')
        rms=np.sqrt(np.mean(np.asarray(batch.emg,dtype=float)**2,axis=1));scale=np.sqrt(np.mean(rms**2,axis=1))
        observations={'log_scale':np.log(scale+1e-10),'pattern':rms/np.maximum(scale[:,None],1e-10),
            'log_bands':self.spectral_.transform(batch),'covariance':_covariances(batch.emg),'quality':self.quality_.transform(batch)}
        if self.rest_label_ is not None:
            observations['rest_adjacent_diff_median']=np.median(
                np.abs(np.diff(np.asarray(batch.emg,dtype=float),axis=1)),axis=1)
        if self.ring_ is not None:observations['ring']=self.ring_.transform(batch)
        profiles={}
        for label in sorted(set(y.tolist())):
            mask=y==label;w=trial_weights(trials[mask]);w/=w.sum()
            profiles[label]={n:np.tensordot(w,x[mask],axes=(0,0)) for n,x in observations.items()}
        return profiles

    def from_calibration(self,batch,labels,trials):
        if not hasattr(self,'reference_'):raise RuntimeError('Source long-term profile required')
        local=self._profiles(batch,labels,trials)
        if set(local)!=set(self.reference_):raise ValueError('Calibration must cover the same source classes')
        names=self.quality_.feature_names;mean_quality=names.index('F9.mean_quality')
        output={}
        for h,current in local.items():
            source=self.reference_[h]
            output[str(h)]={'log_global_activation_shift':float(current['log_scale']-source['log_scale']),
                'scale_pattern_residual_norm':float(np.linalg.norm(current['pattern']-source['pattern'])),
                'mean_absolute_log_band_residual':float(np.abs(current['log_bands']-source['log_bands']).mean()),
                'affine_invariant_trace_covariance_distance':affine_covariance_distance(source['covariance'],current['covariance']),
                'ring_vector_residual_norm':float(np.linalg.norm(current['ring']-source['ring'])) if self.ring_ is not None else None,
                'mean_quality_score_shift':float(current['quality'][mean_quality]-source['quality'][mean_quality]),
                'channel_variance_residual_json':(current['quality'][self.channels_:2*self.channels_]-source['quality'][self.channels_:2*self.channels_]).tolist()}
        rest_available=self.rest_label_ is not None and self.rest_label_ in local
        rest_shift=None
        if rest_available:
            source_noise=self.reference_[self.rest_label_]['rest_adjacent_diff_median']
            current_noise=local[self.rest_label_]['rest_adjacent_diff_median']
            rest_available=bool(np.all(source_noise>1e-10) and np.all(current_noise>1e-10))
            if rest_available:
                rest_shift=np.log(current_noise/source_noise)
        result={'classes':output,'rest_class_available':self.rest_label_ is not None and self.rest_label_ in local,
            'rest_noise_shift_available':rest_available,
            'sensor_channels':self.channels_,'sample_rate_hz':self.rate_,
            'quality_scope':'relative rule scores; ADC clipping unavailable unless hardware metadata supplied',
            'pattern_scope':'new RMS/global_RMS reference; no claim of historical X1-H/RLCS equivalence'}
        if self.rest_label_ is not None:
            result['rest_noise_log_ratio_per_channel']=None if rest_shift is None else rest_shift.tolist()
            result['rest_noise_shift_norm']=None if rest_shift is None else float(np.linalg.norm(rest_shift))
        return result
