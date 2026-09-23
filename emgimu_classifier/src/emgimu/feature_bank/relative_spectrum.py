"""Log-band observations and explicit calibration-relative spectral coordinates."""
import numpy as np
from .core import FeatureBatch, FeatureFamily
from .families import SpectralStateFamily
from .activation_profile import trial_weights


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


class PersonalSessionSpectralShift:
    """F4d long-term and current-session log-band references, kept separate.

    Each calibration trial has equal mass regardless of how many windows it
    contributes. Explicit trial identities prevent reference/evaluation reuse.
    """

    @staticmethod
    def _observations(values, trial_ids, width=None):
        x = np.asarray(values, dtype=np.float64)
        original_ids = np.asarray(trial_ids, dtype=object)
        if (x.ndim != 2 or not len(x) or x.shape[1] == 0 or not np.all(np.isfinite(x)) or
                (width is not None and x.shape[1] != width) or
                original_ids.ndim != 1 or len(original_ids) != len(x) or
                any(item is None or not str(item).strip() or str(item).lower() == "nan"
                    for item in original_ids)):
            raise ValueError("Finite log-band rows and one nonempty trial ID per row required")
        trials = original_ids.astype(str)
        return x, trials

    @staticmethod
    def _equal_trial_mean(x, trials):
        weights = trial_weights(trials)
        return np.average(x, axis=0, weights=weights)

    def fit_long_term(self, log_band_energy, trial_ids):
        x, trials = self._observations(log_band_energy, trial_ids)
        self.long_reference_ = self._equal_trial_mean(x, trials)
        self.long_trial_ids_ = frozenset(trials.tolist())
        self.session_reference_ = None
        self.session_trial_ids_ = frozenset()
        return self

    def fit_session_calibration(self, log_band_energy, trial_ids):
        if not hasattr(self, "long_reference_"):
            raise RuntimeError("Long-term spectral baseline required")
        x, trials = self._observations(log_band_energy, trial_ids, len(self.long_reference_))
        if self.long_trial_ids_.intersection(trials.tolist()):
            raise ValueError("Session calibration overlaps long-term trials")
        self.session_reference_ = self._equal_trial_mean(x, trials)
        self.session_trial_ids_ = frozenset(trials.tolist())
        return self

    def transform_evaluation(self, log_band_energy, trial_ids):
        if not hasattr(self, "long_reference_") or self.session_reference_ is None:
            raise RuntimeError("Long-term and session calibration references required")
        x, trials = self._observations(log_band_energy, trial_ids, len(self.long_reference_))
        if (self.long_trial_ids_ | self.session_trial_ids_).intersection(trials.tolist()):
            raise ValueError("Evaluation overlaps spectral calibration trials")
        return {
            "window_minus_long": (x - self.long_reference_).astype(np.float32),
            "session_minus_long": (self.session_reference_ - self.long_reference_).astype(np.float32),
        }
