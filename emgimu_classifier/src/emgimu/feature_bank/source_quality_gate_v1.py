"""Source-frozen F9v3 raw-input masks and explicit probability/Unknown decisions."""
import hashlib
import json
import numpy as np
from .core import FeatureBatch
from .document_quality_v3 import DocumentQualityObservationsV3
from .calibration import late_fusion_decision


class SourceQualityGateV1:
    # Quantization tolerances are half a signed integer count. Transport extrema
    # are explicit acquisition metadata, not observed target extrema or a claim
    # about saturation elsewhere in the analogue front end.
    def __init__(self,*,channel_ids,adc_range,adc_range_provenance,line_frequency_hz=50,source_quantile=.995):
        channels=tuple(channel_ids)
        if len(channels)!=8 or len(set(channels))!=8 or any(not isinstance(c,str) or not c.strip() for c in channels):
            raise ValueError('Eight explicit source channel identities required')
        if not isinstance(adc_range_provenance,str) or not adc_range_provenance.strip():
            raise ValueError('Explicit acquisition range provenance required')
        if not .9<=source_quantile<1.:raise ValueError('Source quantile must be in[.9,1)')
        self.channels=channels;self.range_provenance=adc_range_provenance;self.quantile=float(source_quantile)
        self.observer=DocumentQualityObservationsV3(adc_range=adc_range,line_frequency_hz=line_frequency_hz,
            pre_highpass_available=True,zero_threshold=.5,flat_threshold=.5)

    def fit(self,raw_batch,trial_ids):
        ids=np.asarray(trial_ids,dtype=object)
        if (raw_batch.channels!=8 or raw_batch.sample_rate_hz!=250. or raw_batch.emg.shape[1]!=50
                or ids.shape!=(raw_batch.windows,) or any(not isinstance(t,str) or not t.strip() for t in ids)):
            raise ValueError('Raw pre-software-highpass Song250 windows and native source IDs required')
        self.source_trials=tuple(np.unique(ids));self.observer.fit(raw_batch)
        names=self.observer.feature_names;x=self.observer.transform(raw_batch).astype(float)
        components=[('zero_fraction',.10,.50),('longest_flatline_ratio',.10,.50),('amplitude_z',6.,None)]
        if self.observer.adc_range is not None:components.append(('clip_fraction',.01,.10))
        if self.observer.line_mask_ is not None:components.append(('line_noise_ratio',10.,None))
        if self.observer.low_available_:components.append(('low_frequency_ratio',10.,None))
        self.indices={};self.thresholds={}
        for metric,floor,ceiling in components:
            index=np.array([names.index(f'F9v3.{metric}.ch{c+1}') for c in range(8)])
            values=x[:,index]
            if metric=='amplitude_z':values=np.abs(values)
            quantile=np.quantile(values,self.quantile,axis=0)
            if ceiling is not None and np.any(quantile>=ceiling):raise ValueError('Source too degraded for '+metric)
            self.indices[metric]=index;self.thresholds[metric]=np.maximum(quantile,floor)
        fields=[self.channels,self.range_provenance,self.quantile,self.source_trials,self.observer.adc_range,
            {k:v.tolist() for k,v in self.thresholds.items()},self.observer.reference_median_.tolist(),
            self.observer.reference_mad_.tolist(),self.observer.reference_covariance_.tolist()]
        self.policy_id=hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()
        return self

    def observe(self,raw_batch,trial_ids,*,observed_channel_ids):
        if not hasattr(self,'policy_id'):raise RuntimeError('Source quality fitting required')
        channels=tuple(observed_channel_ids);ids=np.asarray(trial_ids,dtype=object)
        if (len(channels)!=8 or len(set(channels))!=8 or set(channels)!=set(self.channels)
                or ids.shape!=(raw_batch.windows,) or any(not isinstance(t,str) or not t.strip() for t in ids)
                or set(ids)&set(self.source_trials)):
            raise ValueError('Raw quality channel/trial/source contract differs')
        order=[channels.index(c) for c in self.channels]
        mapped=FeatureBatch(raw_batch.emg[:,:,order],raw_batch.sample_rate_hz)
        x=self.observer.transform(mapped).astype(float)
        soft=np.ones((len(x),8))
        for metric,threshold in self.thresholds.items():
            value=x[:,self.indices[metric]]
            if metric=='amplitude_z':value=np.abs(value)
            soft*=1.-np.clip(value/threshold-1.,0.,1.)
        structural=(x[:,self.indices['zero_fraction']]>=.5)|(x[:,self.indices['longest_flatline_ratio']]>=.5)
        if 'clip_fraction' in self.indices:structural|=x[:,self.indices['clip_fraction']]>=.1
        trials=tuple(np.unique(ids))
        return dict(trial_ids=trials,window_trial_ids=tuple(ids),observations=x,
            soft_channel_quality=soft,structural_invalid_channels=structural,
            trial_soft_channel_quality=np.stack([soft[ids==t].min(0) for t in trials]),
            trial_structural_invalid_channels=np.stack([structural[ids==t].any(0) for t in trials]),
            policy_id=self.policy_id,feature_names=self.observer.feature_names,
            raw_pre_software_highpass=True,physical_validation_proven=False)

    def decide(self,providers,weights,raw_batch,trial_ids,*,observed_channel_ids,class_names,mode,provider_trial_ids):
        if mode not in ('off','structural','soft'):raise ValueError('Explicit off/structural/soft mode required')
        observations=self.observe(raw_batch,trial_ids,observed_channel_ids=observed_channel_ids)
        names=tuple(providers);n=len(observations['trial_ids'])
        if (set(provider_trial_ids)!=set(names)
                or any(tuple(provider_trial_ids[g])!=observations['trial_ids'] for g in names)):
            raise ValueError('Every provider needs the exact raw-quality trial axis')
        if mode=='structural':
            channel_quality=(~observations['trial_structural_invalid_channels']).astype(float)
            quality={g:channel_quality.min(1) for g in names}
        elif mode=='soft':
            channel_quality=observations['trial_soft_channel_quality']
            # Predeclared heuristic: both F0 and SPD consume every measured
            # channel directly; remaining branches use the mean mask candidate.
            quality={g:channel_quality.min(1) if g in ('F0','F2ac') else channel_quality.mean(1) for g in names}
        else:
            channel_quality=np.ones((n,8));quality={g:np.ones(n) for g in names}
        decision=late_fusion_decision(providers,names,np.asarray(weights),tuple(class_names),quality,minimum_confidence=0.)
        effective=np.asarray(weights)[None,:]*np.column_stack([quality[g] for g in names])
        total=effective.sum(1);effective[total<=1e-10]=np.asarray(weights)
        effective/=effective.sum(1,keepdims=True)
        return dict(probabilities=decision.probabilities,labels=decision.labels,rejected=decision.rejected,
            rejection_reason=decision.rejection_reason,channel_quality=channel_quality,
            structural_invalid_channels=observations['trial_structural_invalid_channels'],
            bad_channel_count=(channel_quality<.5).sum(1),provider_weights=effective,
            trial_ids=observations['trial_ids'],policy_id=self.policy_id,mode=mode,
            physical_validation_proven=False,observations=observations)
