"""Source-fitted personal profile with calibration-only session updates."""
import pickle
import numpy as np
from .calibration import PersonalNormalizer,SessionSignature
from .session_anchor import SessionPrototypeAnchor
from .quality_observability import QualityObservabilityFamily
from .screening import FAMILY_FACTORIES
from .force_full_fusion import aggregate
from .force_nested_oof import SubjectWindows
from .unibo_full_fusion import classifier
from .wearing_session_study import anchor_probability

FAMILIES=('F0','F3_Ring')
BRANCHES=('source_model','session_model','source_long_anchor','session_long_anchor',
    'session_local_anchor','session_blended_anchor')


class SessionCalibrationPipeline:
    """Verified native eight-channel ring; rest label must be explicit.

    Prediction rejects both source-fit and session-calibration trial identities.
    Current rest/scale/quality/signatures/prototypes never replace the long profile.
    """
    def __init__(self,*,rest_label,ring_topology=False):
        self.rest_label=rest_label
        self.ring_topology=bool(ring_topology)

    def fit_long_term(self,source):
        if not self.ring_topology or source.batch.channels!=8 or len(set(source.subjects))!=1:
            raise ValueError('Require one personal source and verified native eight-channel ring')
        self.user_=np.unique(source.subjects).item()
        self.rate_=source.batch.sample_rate_hz;self.samples_=source.batch.emg.shape[1]
        self.source_trials_=set(source.trials.tolist())
        self.classes_=set(source.labels.tolist())
        self.normalizer_=PersonalNormalizer(rest_label=self.rest_label).fit(source.batch,source.labels)
        self.quality_=QualityObservabilityFamily(ring_topology=True).fit(source.batch)
        self.quality_reference_=self.quality_.transform(source.batch).mean(0)
        normalized=SubjectWindows(self.normalizer_.transform(source.batch),source.labels,source.subjects,source.trials)
        self.families_={};self.models_={};self.profiles_={};self.signatures_={}
        for name in FAMILIES:
            family=FAMILY_FACTORIES[name]().fit(normalized.batch,normalized.labels)
            x,y,_,_=aggregate(family.transform(normalized.batch),normalized)
            scaler,model=classifier(x,y,np.ones(len(y)))
            scaled=scaler.transform(x)
            self.families_[name]=family;self.models_[name]=(scaler,model)
            self.profiles_[name]=SessionPrototypeAnchor().fit_long_term(scaled,y)
            self.signatures_[name]=SessionSignature().fit_long_term(scaled,y)
        return self

    def _features(self,data,normalizer):
        if data.batch.sample_rate_hz!=self.rate_ or data.batch.emg.shape[1:]!=(self.samples_,8):
            raise ValueError('Session sensor/window contract differs from the source model')
        if set(data.subjects.tolist())!={self.user_}:
            raise ValueError('A personal session cannot pool or replace users')
        normalized=SubjectWindows(normalizer.transform(data.batch),data.labels,data.subjects,data.trials)
        values={}
        for name in FAMILIES:
            x,y,_,trials=aggregate(self.families_[name].transform(normalized.batch),normalized)
            values[name]=self.models_[name][0].transform(x)
        return values,y,trials

    def calibrate_session(self,calibration):
        if set(calibration.trials.tolist())&self.source_trials_:
            raise ValueError('Session calibration must be disjoint from long-term source trials')
        if set(calibration.labels.tolist())!=self.classes_:
            raise ValueError('Complete native gesture coverage including explicit Rest is required')
        before=pickle.dumps(self)
        normalizer=PersonalNormalizer(rest_label=self.rest_label).fit(calibration.batch,calibration.labels)
        features,y,trials=self._features(calibration,normalizer)
        signature={};anchors={mode:{} for mode in ('long_term','local','blended')};beta={};weights=np.ones(2)
        for i,name in enumerate(FAMILIES):
            vector=self.signatures_[name].from_session_calibration(features[name],y)
            signature[name]=dict(vector=vector.tolist(),names=self.signatures_[name].feature_names)
            n=len(self.classes_);weights[i]*=np.clip((vector[n:2*n].mean()+1)/2,.05,1.)
            for mode in anchors:
                anchor,b=self.profiles_[name].from_calibration(features[name],y,mode=mode)
                anchors[mode][name]=anchor;beta[f'{name}_{mode}']=b.tolist()
        quality=self.quality_.transform(calibration.batch).mean(0)
        state=dict(normalizer=normalizer,calibration_trials=trials.tolist(),anchors=anchors,
            descriptor=dict(rest_center=normalizer.center_.tolist(),channel_scale_q95=normalizer.scale_.tolist(),
                source_rest_center=self.normalizer_.center_.tolist(),source_channel_scale_q95=self.normalizer_.scale_.tolist(),
                quality_availability=self.quality_.availability_,quality_shift=(quality-self.quality_reference_).tolist(),
                session_signatures=signature,prototype_beta=beta),context_weights=weights/weights.sum())
        if before!=pickle.dumps(self):raise AssertionError('Calibration changed long-term source profile')
        return state

    def predict(self,target,session=None):
        if set(target.trials.tolist())&self.source_trials_:
            raise ValueError('Source-fit trials cannot become held-out evaluation')
        if session is not None and set(target.trials.tolist())&set(session['calibration_trials']):
            raise ValueError('Calibration trials cannot become held-out evaluation')
        before=pickle.dumps((self,session))
        source,y,trials=self._features(target,self.normalizer_)
        current=source if session is None else self._features(target,session['normalizer'])[0]
        probabilities={branch:{} for branch in BRANCHES}
        for name in FAMILIES:
            _,model=self.models_[name]
            probabilities['source_model'][name]=model.predict_proba(source[name])
            probabilities['session_model'][name]=model.predict_proba(current[name])
            long=self.profiles_[name].from_calibration(mode='long_term')[0]
            probabilities['source_long_anchor'][name]=anchor_probability(long,source[name])
            for mode in ('long_term','local','blended'):
                anchor=long if session is None else session['anchors'][mode][name]
                probabilities[f'session_{"long" if mode=="long_term" else mode}_anchor'][name]=anchor_probability(anchor,current[name])
        if before!=pickle.dumps((self,session)):raise AssertionError('Evaluation changed source or session state')
        return probabilities,y,trials
