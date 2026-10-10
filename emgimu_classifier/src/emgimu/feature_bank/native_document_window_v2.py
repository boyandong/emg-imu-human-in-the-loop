"""Document EMG views on native eight-channel 200 or 250 Hz recordings.

Sensor rate, class axis and source trials are explicit. No resampling, invented
electrode geometry, anatomical context or implicit ADC-quality policy is used.
"""
import hashlib
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .document_window_composition_v1 import document_families, GROUPS, TYPES, source_trial_features
from .document_signal import DocumentCspFamily
from .extended_window_decision_v1 import ExtendedWindowDecisionV1
from .personal_session_decision_v1 import PersonalSessionDecisionV1
from .personal_session_workflow_v1 import PersonalSessionWorkflowV1


class NativeSourceCspV2(DocumentCspFamily):
    def fit(self, batch, labels=None, *, source_trial_ids):
        ids=np.asarray(source_trial_ids)
        contract=(batch.sample_rate_hz,batch.emg.shape[1],batch.channels)
        if (contract not in ((200.,40,8),(250.,50,8)) or ids.shape!=(batch.windows,)
                or any(not isinstance(t,str) or not t.strip() for t in ids)):
            raise ValueError('Native source-fold200/250Hz eight-channel200ms windows required')
        super().fit(batch,labels)
        self.source_trial_ids_=tuple(np.unique(ids));self.sensor_contract_=contract
        return self

    def transform(self,batch):
        self._check()
        if (batch.sample_rate_hz,batch.emg.shape[1],batch.channels)!=self.sensor_contract_:
            raise ValueError('Native CSP source sensor contract differs')
        return super().transform(batch)


def fit_native_document_source(batch,labels,trial_ids,*,rest_label,forbidden_trials=(),seed=20261011):
    ids=np.asarray(trial_ids);y=np.asarray(labels)
    if ((batch.sample_rate_hz,batch.emg.shape[1],batch.channels) not in ((200.,40,8),(250.,50,8))
            or not batch.windows or ids.shape!=(batch.windows,) or y.shape!=ids.shape
            or any(not isinstance(t,str) or not t.strip() for t in ids)
            or any(not isinstance(c,str) or not c.strip() for c in y)
            or len(set(y))<2 or rest_label not in set(y) or set(ids)&set(forbidden_trials)):
        raise ValueError('Explicit disjoint native source trials, Rest and string class identities required')
    unique=tuple(np.unique(ids));classes=tuple(np.unique(y))
    if any(len(set(y[ids==t]))!=1 for t in unique):raise ValueError('Mixed labels within source trial')
    targets=np.array([y[ids==t][0] for t in unique])
    families=document_families();families['F0'][0].rest_label=rest_label
    families['F2b']=[NativeSourceCspV2()];models={};metadata={}
    for group,views in families.items():
        for family in views:
            if isinstance(family,NativeSourceCspV2):family.fit(batch,y,source_trial_ids=ids)
            else:family.fit(batch,y)
            family.native_source_trials_=unique
        x,axis=source_trial_features(views,batch,ids)
        assert axis==unique
        scaler=StandardScaler().fit(x)
        model=LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=seed).fit(scaler.transform(x),targets)
        if tuple(model.classes_)!=classes or model.n_iter_.max()>=2000:raise ValueError('Source fit class axis/convergence failed')
        models[group]=(scaler,model)
        names=tuple(n for f in views for n in f.feature_names)
        assert len(names)==x.shape[1] and len(names)==len(set(names))
        metadata[group]=dict(dimension=x.shape[1],feature_names=names,iterations=model.n_iter_.tolist())
    return families,models,metadata


class NativeDocumentWindowDecisionV2(ExtendedWindowDecisionV1):
    def __init__(self,base,*,anchor_mix=.5):
        if not isinstance(base,PersonalSessionWorkflowV1):raise ValueError('Source-frozen personal workflow required')
        bank=base.bank
        expected=TYPES[:-1]+((NativeSourceCspV2,),)
        if (bank.providers_!=GROUPS or bank.sensor_contract_ not in ((200.,40,8),(250.,50,8))
                or not isinstance(bank.families_['F2b'][0],NativeSourceCspV2)
                or any(tuple(type(f) for f in bank.families_[g])!=types for g,types in zip(GROUPS,expected))
                or any(tuple(getattr(f,'native_source_trials_',()))!=tuple(bank.policy_.source_trials)
                       for views in bank.families_.values() for f in views)):
            raise ValueError('Native source provider/sensor/fold provenance differs')
        self.decision=PersonalSessionDecisionV1(base,anchor_mix=anchor_mix,gate=None)
        self.bank,self.channels=bank,base.channels
        self.preprocessing_id,self.rest_label,self.gate=base.preprocessing_id,base.rest_label,None
        self.bands=tuple(tuple(v) for v in bank.families_['F4abc'][0].bands_)
        self.policy_id=hashlib.sha256(json.dumps(['native_document_window_v2',self.decision.policy_id,
            bank.sensor_contract_,bank.classes_,self.bands,'quality_unavailable_without_bound_metadata']).encode()).hexdigest()
        self.contract_id=self.policy_id

    def predict(self,*args,**kwargs):
        if kwargs.get('quality_mode','off')!='off':
            raise ValueError('Native window package has no bound physical/ADC quality policy')
        result=super().predict(*args,**kwargs)
        result.update(native_sample_rate_hz=self.bank.sensor_contract_[0],quality_policy_available=False,
            scope='Native-rate eight-channel document window views, source-only CSP, calibration F7/F8 and F4d context. No resampling, physical ring/F6, bound raw-quality gate, biological boundary or deployment claim.')
        return result
