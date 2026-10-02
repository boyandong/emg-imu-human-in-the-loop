"""Source-fitted personal profile with calibration-only session updates."""
import hashlib
import pickle
import numpy as np
from .calibration import DocumentPersonalNormalizerV2,PersonalNormalizer,SessionSignature
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
    normalizer_type=PersonalNormalizer

    def __init__(self,*,rest_label,ring_topology=False):
        self.rest_label=rest_label
        self.ring_topology=bool(ring_topology)

    def fit_long_term(self,source):
        if not self.ring_topology or source.batch.channels!=8 or len(set(source.subjects))!=1:
            raise ValueError('Require one personal source and verified native eight-channel ring')
        self.user_=np.unique(source.subjects).item()
        self.rate_=source.batch.sample_rate_hz;self.samples_=source.batch.emg.shape[1]
        self.source_trials_=tuple(sorted(set(source.trials.tolist())))
        self.classes_=tuple(sorted(set(source.labels.tolist())))
        self.normalizer_=self.normalizer_type(rest_label=self.rest_label).fit(source.batch,source.labels)
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
        self.profile_id_=hashlib.sha256(pickle.dumps((
            self.user_,self.rate_,self.samples_,self.source_trials_,
            self.normalizer_,self.families_,self.models_,self.profiles_,self.signatures_
        ))).hexdigest()
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
        if set(calibration.trials.tolist())&set(self.source_trials_):
            raise ValueError('Session calibration must be disjoint from long-term source trials')
        if set(calibration.labels.tolist())!=set(self.classes_):
            raise ValueError('Complete native gesture coverage including explicit Rest is required')
        before=pickle.dumps(self)
        normalizer=self.normalizer_type(rest_label=self.rest_label).fit(calibration.batch,calibration.labels)
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
        state=dict(profile_id=self.profile_id_,user=self.user_,
            normalizer=normalizer,calibration_trials=trials.tolist(),anchors=anchors,
            descriptor=dict(rest_center=normalizer.center_.tolist(),channel_scale_q95=normalizer.scale_.tolist(),
                source_rest_center=self.normalizer_.center_.tolist(),source_channel_scale_q95=self.normalizer_.scale_.tolist(),
                quality_availability=self.quality_.availability_,quality_shift=(quality-self.quality_reference_).tolist(),
                session_signatures=signature,prototype_beta=beta),context_weights=weights/weights.sum())
        if before!=pickle.dumps(self):raise AssertionError('Calibration changed long-term source profile')
        return state

    def predict(self,target,session=None):
        probabilities,trials=self.predict_unlabeled(target.batch,target.subjects,target.trials,session)
        labels=np.asarray(target.labels)
        identities=np.asarray(target.trials)
        if labels.shape!=(target.batch.windows,):
            raise ValueError('Held-out labels must align with native windows for offline scoring')
        y=[]
        for trial in trials:
            values=np.unique(labels[identities==trial])
            if len(values)!=1:
                raise ValueError('One native evaluation trial must have one truth label')
            y.append(values.item())
        return probabilities,np.asarray(y),trials

    def predict_unlabeled(self,batch,subjects,trial_ids,session=None):
        """Predict native trials without evaluation labels or target fitting."""
        if not hasattr(self,'models_'):
            raise RuntimeError('Long-term source model must be fit first')
        if batch.sample_rate_hz!=self.rate_ or batch.emg.shape[1:]!=(self.samples_,8):
            raise ValueError('Session sensor/window contract differs from the source model')
        users=np.asarray(subjects)
        trials_by_window=np.asarray(trial_ids)
        if (users.shape!=(batch.windows,) or trials_by_window.shape!=(batch.windows,)
                or set(users.tolist())!={self.user_}):
            raise ValueError('Aligned identities for one fitted personal user are required')
        if any(not isinstance(value,str) or not value.strip() for value in trials_by_window):
            raise ValueError('Explicit nonempty native trial identities are required')
        if session is not None and (not isinstance(session,dict)
                                    or session.get('profile_id')!=self.profile_id_
                                    or session.get('user')!=self.user_):
            raise ValueError('Session calibration belongs to a different long-term profile')
        trials=np.unique(trials_by_window)
        if set(trials.tolist())&set(self.source_trials_):
            raise ValueError('Source-fit trials cannot become held-out evaluation')
        if session is not None and set(trials.tolist())&set(session['calibration_trials']):
            raise ValueError('Calibration trials cannot become held-out evaluation')
        before=pickle.dumps((self,session))
        def features(normalizer):
            normalized=normalizer.transform(batch)
            output={}
            for name in FAMILIES:
                windows=self.families_[name].transform(normalized)
                values=np.stack([windows[trials_by_window==trial].mean(0) for trial in trials])
                output[name]=self.models_[name][0].transform(values)
            return output
        source=features(self.normalizer_)
        current=source if session is None else features(session['normalizer'])
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
        return probabilities,trials


class DocumentSessionCalibrationPipelineV2(SessionCalibrationPipeline):
    """Opt-in session flow using the appendix's Q95+epsilon denominator.

    The original pipeline retains its frozen max(Q95,epsilon) semantics, so
    historical experiment results are not changed by the exact-formula path.
    """
    normalizer_type=DocumentPersonalNormalizerV2
