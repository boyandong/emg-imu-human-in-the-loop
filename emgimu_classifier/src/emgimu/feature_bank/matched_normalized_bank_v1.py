"""Explicitly matched training/inference domain for personal normalization.

This version requires source models fitted on document-normalized source data.
It never attaches normalization to a raw-trained model. Raw ADC quality and
session diagnostics are not inferred from normalized classifier inputs.
"""
from dataclasses import replace
import hashlib
import json
import numpy as np
from .calibration import DocumentPersonalNormalizerV2
from .frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1
from .personal_session_workflow_v1 import PersonalSessionWorkflowV1


DOMAIN = 'document_personal_normalized_v1'


class MatchedNormalizedProviderBankV1(FrozenEmgProviderBankV1):
    def __init__(self, *args, source_model_fit_trials, source_normalization_trials, **kwargs):
        super().__init__(*args, **kwargs)
        fit = tuple(source_model_fit_trials)
        calibration = {name:tuple(ids) for name,ids in source_normalization_trials.items()}
        cal = tuple(t for ids in calibration.values() for t in ids)
        if (not fit or not calibration or not cal or len(set(fit))!=len(fit) or len(set(cal))!=len(cal)
                or set(fit)&set(cal) or set(fit)|set(cal)!=set(self.policy_.source_trials)
                or any(not isinstance(t,str) or not t.strip() for t in (*fit,*cal))
                or any(not isinstance(s,str) or not s.strip() or not ids for s,ids in calibration.items())):
            raise ValueError('Disjoint explicit source model-fit and normalization calibration trials required')
        self.training_input_domain_ = DOMAIN
        self.source_model_fit_trials_ = fit
        self.source_normalization_trials_ = calibration
        fields = [self.bank_id_, DOMAIN, fit, calibration]
        self.bank_id_ = hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()


class MatchedNormalizedWorkflowV1(PersonalSessionWorkflowV1):
    def __init__(self,bank,**kwargs):
        if (not isinstance(bank,MatchedNormalizedProviderBankV1)
                or bank.training_input_domain_!=DOMAIN):
            raise ValueError('A source model explicitly trained in the normalized domain is required')
        super().__init__(bank,**kwargs)

    def _normalization_input(self,batch,kwargs):
        return self._input(batch,observed_channel_ids=kwargs['observed_channel_ids'],
                           preprocessing_id=kwargs['preprocessing_id'])

    @staticmethod
    def _normalizer(batch,trial_ids,trial_labels,rest_label):
        # Validate labels before fitting any target statistic.
        if (not isinstance(trial_labels,dict) or set(trial_labels)!=set(trial_ids)):
            raise ValueError('Exactly one separate calibration label per native trial required')
        y=np.array([trial_labels[t] for t in trial_ids])
        return DocumentPersonalNormalizerV2(rest_label=rest_label).fit(batch,y)

    def enroll_user(self,batch,trial_ids,trial_labels,**kwargs):
        raw=self._normalization_input(batch,kwargs)
        normalizer=self._normalizer(raw,trial_ids,trial_labels,self.rest_label)
        canonical=dict(kwargs,observed_channel_ids=self.channels)
        profile=super().enroll_user(normalizer.transform(raw),trial_ids,trial_labels,**canonical)
        # Identity binds raw calibration bytes and labels, not only scale-invariant inputs.
        identity=self._profile_id(kwargs['user_id'],kwargs['session_id'],trial_ids,
                                  kwargs['window_offsets'],trial_labels,raw)
        return replace(profile,normalizer=normalizer,profile_id=identity)

    def calibrate_session(self,batch,trial_ids,trial_labels,*,personal,**kwargs):
        self._personal(personal,kwargs['user_id'])
        raw=self._normalization_input(batch,kwargs)
        normalizer=self._normalizer(raw,trial_ids,trial_labels,self.rest_label)
        canonical=dict(kwargs,observed_channel_ids=self.channels)
        profile=super().calibrate_session(normalizer.transform(raw),trial_ids,trial_labels,
                                           personal=personal,**canonical)
        identity=self._profile_id(kwargs['user_id'],kwargs['session_id'],trial_ids,
                                  kwargs['window_offsets'],trial_labels,raw)
        return replace(profile,normalizer=normalizer,profile_id=identity)

    def classifier_input(self,batch,*,personal,session=None,**kwargs):
        self._session(session,personal,kwargs['user_id'],kwargs['session_id'])
        raw=self._normalization_input(batch,kwargs)
        return (personal.normalizer if session is None else session.normalizer).transform(raw)

    def predict(self,batch,trial_ids,*,personal,session=None,**kwargs):
        # There is deliberately no zero-personal fallback for a normalized model.
        transformed=self.classifier_input(batch,personal=personal,session=session,**kwargs)
        canonical=dict(kwargs,observed_channel_ids=self.channels)
        result=super().predict(transformed,trial_ids,personal=personal,session=session,**canonical)
        result.update(training_input_domain=DOMAIN,normalization_used_by_classifier=True,
            quality_observations=None,quality_feature_names=(),session_descriptor=None,
            scope='Source-trained normalized classifiers with calibration-only rest median/Q95+epsilon; source models/temperatures stay fixed. Anchor coordinates use normalized feature geometry. No raw-ADC quality, F8 routing, rejection, live efficacy or full F0-F9 claim.')
        return result
