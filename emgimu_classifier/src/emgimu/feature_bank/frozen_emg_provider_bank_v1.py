"""Portable opt-in source-fitted provider bank; inference never accepts labels."""
from dataclasses import dataclass
import copy
import hashlib
import pickle
from collections.abc import Mapping
import numpy as np
from .available_bank_fusion_v1 import AvailableBankFusionPolicyV1


@dataclass(frozen=True, slots=True)
class FrozenEmgUserStateV1:
    user_id: str
    bank_id: str
    fusion_state: object


class FrozenEmgProviderBankV1:
    def __init__(self, fitted_families, fitted_models, temperatures, *, classes, class_names,
                 source_trial_ids, source_policy_id, sample_rate_hz, window_samples, channels,
                 population, n0, reliability_temperature):
        names=tuple(fitted_families)
        if not names or set(fitted_models)!=set(names) or set(temperatures)!=set(names):
            raise ValueError('Aligned named source-fitted families/models/temperatures required')
        if len(class_names)!=len(classes) or len(set(class_names))!=len(class_names):
            raise ValueError('Unique ordered class names required')
        if not np.isfinite(sample_rate_hz) or sample_rate_hz<=0 or not isinstance(window_samples,(int,np.integer)) or isinstance(window_samples,bool) or window_samples<3 or not isinstance(channels,(int,np.integer)) or isinstance(channels,bool) or channels<1:
            raise ValueError('Explicit source sensor/window contract required')
        for name in names:
            if not fitted_families[name] or any(not f.fitted_ for f in fitted_families[name]):
                raise ValueError('All representations must already be source-fitted')
            scaler,model=fitted_models[name]
            if tuple(model.classes_)!=tuple(classes) or len(scaler.mean_)!=model.coef_.shape[1]:
                raise ValueError('Source model/scaler/class axis mismatch')
            if not np.isfinite(temperatures[name]) or temperatures[name]<=0:
                raise ValueError('Positive source-fitted probability temperature required')
        self.providers_=names;self.classes_=tuple(classes);self.class_names_=tuple(class_names)
        self.families_=copy.deepcopy(fitted_families);self.models_=copy.deepcopy(fitted_models)
        self.temperatures_=dict(temperatures)
        self.sensor_contract_=(float(sample_rate_hz),int(window_samples),int(channels))
        self.policy_=AvailableBankFusionPolicyV1(self.classes_,names,tuple(population),n0,
            reliability_temperature,source_policy_id,tuple(source_trial_ids))
        self.bank_id_=hashlib.sha256(pickle.dumps((self.families_,self.models_,self.temperatures_,
            self.sensor_contract_,self.class_names_,self.policy_))).hexdigest()

    @staticmethod
    def _user(user_id):
        if not isinstance(user_id,str) or not user_id.strip(): raise ValueError('Explicit user identity required')

    def _active(self,available,compatible=None):
        names=self.providers_ if compatible is None else tuple(compatible)
        active=tuple(names if available is None else available)
        if not active or len(set(active))!=len(active) or not set(active)<=set(names):
            raise ValueError('Unique profile-compatible available providers required')
        return tuple(name for name in names if name in active)

    def _features(self,batch,window_trial_ids,window_offsets,providers):
        contract=(batch.sample_rate_hz,batch.emg.shape[1],batch.channels)
        if contract!=self.sensor_contract_: raise ValueError('Source channel/sample/rate contract differs')
        ids=np.asarray(window_trial_ids,dtype=object)
        if ids.shape!=(batch.windows,) or any(not isinstance(t,str) or not t.strip() for t in ids):
            raise ValueError('Explicit window-aligned native trial IDs required')
        unique=np.unique(ids)
        if set(unique)&set(self.policy_.source_trials): raise ValueError('Source trials cannot become target/calibration')
        offsets=np.asarray(window_offsets)
        if offsets.shape!=ids.shape or offsets.dtype.kind not in 'iu' or np.any(offsets<0):
            raise ValueError('Aligned nonnegative integer window offsets required')
        ordered=[]
        for trial in unique:
            positions=np.flatnonzero(ids==trial)
            if not np.array_equal(np.sort(offsets[positions]),np.arange(len(positions))):
                raise ValueError('Each trial needs distinct contiguous window offsets from zero')
            ordered.append(positions[np.argsort(offsets[positions])])
        features={}
        for name in providers:
            families=self.families_[name]
            columns=[]
            for family in families:
                values=np.asarray(family.transform(batch))
                if values.ndim!=2 or values.shape[0]!=len(ids) or not np.isfinite(values).all():
                    raise ValueError('Finite aligned representation rows required')
                # Preserve the source experiment's chronological float32 mean,
                # even when caller input rows are permuted; do not silently
                # replace it with a numerically different float64 mean.
                columns.append(np.stack([values[positions].mean(0) for positions in ordered]))
            features[name]=np.concatenate(columns,axis=1)
        return features,tuple(unique)

    def predict_providers(self,batch,window_trial_ids,*,window_offsets,user_id,available=None):
        self._user(user_id);features,ids=self._features(batch,window_trial_ids,window_offsets,self._active(available))
        probabilities={};standardized={}
        for name,x in features.items():
            scaler,model=self.models_[name];z=scaler.transform(x);standardized[name]=z
            raw=model.predict_proba(z)
            logits=np.log(np.maximum(raw,1e-15))/self.temperatures_[name]
            logits-=logits.max(1,keepdims=True);q=np.exp(logits);q/=q.sum(1,keepdims=True)
            probabilities[name]=q
        return {'trial_ids':ids,'user_id':user_id,'probabilities':probabilities,
                'source_standardized_features':standardized,'classes':self.classes_}

    def calibrate_user(self,batch,window_trial_ids,trial_labels,*,window_offsets,user_id,forbidden_evaluation_trials=(),available=None):
        self._user(user_id);active=self._active(available)
        features,ids=self._features(batch,window_trial_ids,window_offsets,active)
        if not isinstance(trial_labels,Mapping) or set(trial_labels)!=set(ids):
            raise ValueError('One explicit label per calibration native trial required')
        labels=np.array([trial_labels[t] for t in ids])
        if not set(labels)<=set(self.classes_): raise ValueError('Unknown calibration class')
        calibration={name:(self.models_[name][0].transform(features[name]),labels,ids) for name in active}
        state=self.policy_.calibrate(calibration,available=active,forbidden_evaluation_trials=forbidden_evaluation_trials)
        return FrozenEmgUserStateV1(user_id,self.bank_id_,state)

    def predict(self,batch,window_trial_ids,*,window_offsets,user_id,user_state=None,available=None):
        self._user(user_id)
        if user_state is None:state=self.policy_.calibrate({})
        else:
            if not isinstance(user_state,FrozenEmgUserStateV1) or user_state.user_id!=user_id or user_state.bank_id!=self.bank_id_:
                raise ValueError('Calibration profile user/bank differs')
            state=user_state.fusion_state
        active=self._active(available,state.providers)
        readout=self.predict_providers(batch,window_trial_ids,window_offsets=window_offsets,user_id=user_id,available=active)
        probabilities={name:readout['probabilities'][name] for name in active}
        ids=readout['trial_ids']
        fused=self.policy_.predict(state,probabilities,evaluation_trials=ids,
            provider_trial_ids={name:ids for name in active},provider_classes={name:self.classes_ for name in active})
        fused.update(trial_ids=ids,user_id=user_id,classes=self.classes_,class_names=self.class_names_,
                     predicted_labels=tuple(self.class_names_[i] for i in fused['probabilities'].argmax(1)))
        return fused
