"""Calibration-only signal activation range; never a mechanical force estimate."""
import numpy as np
from .core import FeatureBatch


def trial_weights(trials):
    _,index,count=np.unique(trials,return_inverse=True,return_counts=True)
    return 1./count[index]


def weighted_quantiles(values,weights):
    unique,index=np.unique(values,return_inverse=True)
    mass=np.bincount(index,weights=weights);cdf=np.cumsum(mass)/mass.sum()
    return unique[np.minimum(np.searchsorted(cdf,[.1,.5,.9],side='left'),len(unique)-1)]


class PersonalActivationProfile:
    def fit_calibration(self,batch:FeatureBatch,labels,trials,*,rest_label):
        y=np.asarray(labels);trials=np.asarray(trials)
        if len(y)!=batch.windows or len(trials)!=batch.windows:raise ValueError('Calibration windows, labels and trial IDs must align')
        for trial in np.unique(trials):
            if len(np.unique(y[trials==trial]))!=1:raise ValueError('Each calibration trial must have one native label')
        rms=np.sqrt(np.mean(np.asarray(batch.emg,dtype=float)**2,axis=1))
        activation=np.sqrt(np.mean(rms**2,axis=1));active=(y!=rest_label)
        if not active.any():raise ValueError('Active calibration required; rest is excluded')
        self.profile_={'channels':batch.channels,'calibration_trials':sorted(set(trials.tolist())),
            'rest_label':rest_label,'rest_windows_excluded':int(np.sum(~active)),
            'quantile_rule':'inverse weighted empirical CDF; equal trial mass; no interpolation',
            'amplitude_definition':'sqrt(mean_channel(RMS_channel**2)); archive signal units',
            'pattern_definition':'RMS_channel/global_RMS; new reference, historical X1-H equivalence unverified',
            'interpretation':'observed calibration activation range, not measured mechanical force','groups':{}}
        for label in ('ALL_ACTIVE',*sorted(set(y[active].tolist()))):
            mask=active if label=='ALL_ACTIVE' else active&(y==label)
            weights=trial_weights(trials[mask]);q=weighted_quantiles(activation[mask],weights)
            valid=activation[mask]>1e-10;pattern=rms[mask][valid]/activation[mask][valid,None]
            w=weights[valid];mean=np.average(pattern,axis=0,weights=w) if valid.any() else np.zeros(batch.channels)
            spread=float(np.average(np.linalg.norm(pattern-mean,axis=1),weights=w)) if valid.any() else None
            self.profile_['groups'][str(label)]={'q10':float(q[0]),'q50':float(q[1]),'q90':float(q[2]),
                'pattern_mean':mean.tolist(),'pattern_spread':spread,'windows':int(mask.sum()),
                'trials':len(set(trials[mask].tolist())),'zero_activation_pattern_windows_excluded':int(np.sum(~valid))}
        active_groups=[v['pattern_spread'] for k,v in self.profile_['groups'].items() if k!='ALL_ACTIVE' and v['pattern_spread'] is not None]
        pooled=self.profile_['groups']['ALL_ACTIVE']
        pooled['pooled_across_gestures_pattern_spread']=pooled['pattern_spread']
        pooled['pattern_spread']=float(np.mean(active_groups)) if active_groups else None
        pooled['pattern_spread_definition']='mean of within-gesture weighted distances; excludes between-gesture separation'
        return self
