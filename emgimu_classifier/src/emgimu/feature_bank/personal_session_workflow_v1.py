"""Explicit personal/session lifecycle around a source-frozen provider bank.

Raw source classifiers retain their original input distribution. Personal
normalization and anchor/context readouts are separate named outputs; they are
not silently substituted into a classifier trained on raw features. Quality
observations do not reject gestures. This is an opt-in offline trial workflow.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import zipfile
from collections.abc import Mapping
import numpy as np
from .core import FeatureBatch
from .frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1
from .available_bank_fusion_v1 import AvailableBankFusionPolicyV1
from .calibration import DocumentPersonalNormalizerV2, DocumentPersonalAnchorV2
from .document_quality_v3 import DocumentQualityObservationsV3
from .document_session_v3 import DocumentSessionDescriptorV3
from .activation_profile import PersonalActivationProfile


@dataclass(frozen=True)
class PersonalProfileV1:
    contract_id: str
    profile_id: str
    user_id: str
    session_id: str
    calibration_trials: tuple
    calibration_counts: tuple
    fusion_state: object
    normalizer: object
    anchors: dict
    quality: object
    session_reference: object
    activation: dict


@dataclass(frozen=True)
class SessionProfileV1:
    contract_id: str
    profile_id: str
    personal_profile_id: str
    user_id: str
    session_id: str
    calibration_trials: tuple
    fusion_policy: object
    fusion_state: object
    normalizer: object
    anchors: dict
    prototype_beta: dict
    descriptor: dict
    activation: dict


def _identity(value, name):
    if not isinstance(value,str) or not value.strip():raise ValueError('Explicit '+name+' required')


class PersonalSessionWorkflowV1:
    def __init__(self,bank,*,channel_ids,preprocessing_id,rest_label,quality_options=None):
        if not isinstance(bank,FrozenEmgProviderBankV1):raise ValueError('Source-frozen provider bank required')
        channels=tuple(channel_ids)
        if (bank.sensor_contract_[2]!=8 or len(channels)!=8 or len(set(channels))!=8
                or any(not isinstance(c,str) or not c.strip() for c in channels)):
            raise ValueError('Eight explicit source channel identities required')
        _identity(preprocessing_id,'preprocessing identity')
        if rest_label not in bank.classes_:raise ValueError('Explicit source Rest class required')
        self.bank=bank;self.channels=channels;self.preprocessing_id=preprocessing_id;self.rest_label=rest_label
        self.quality_options=dict(quality_options or {})
        if self.quality_options.get('ring_order') is not None:
            raise ValueError('This workflow does not attest physical ring geometry')
        DocumentQualityObservationsV3(**self.quality_options)
        fields=[bank.bank_id_,channels,preprocessing_id,str(rest_label),self.quality_options]
        self.contract_id=hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()

    def _input(self,batch,*,observed_channel_ids,preprocessing_id):
        names=tuple(observed_channel_ids)
        if (len(names)!=8 or len(set(names))!=8 or set(names)!=set(self.channels)
                or batch.channels!=8 or preprocessing_id!=self.preprocessing_id):
            raise ValueError('Input channel/preprocessing contract differs')
        if (batch.sample_rate_hz,batch.emg.shape[1],batch.channels)!=self.bank.sensor_contract_:
            raise ValueError('Input sample rate/window contract differs')
        if names==self.channels:return batch
        order=[names.index(c) for c in self.channels]
        return FeatureBatch(np.ascontiguousarray(np.asarray(batch.emg)[:,:,order]),batch.sample_rate_hz,batch.imu,batch.posture)

    def _personal(self,profile,user_id):
        _identity(user_id,'user identity')
        if (not isinstance(profile,PersonalProfileV1) or profile.contract_id!=self.contract_id
                or profile.user_id!=user_id):raise ValueError('Personal profile user/source contract differs')
        if (profile.fusion_state.user_id!=user_id or profile.fusion_state.bank_id!=self.bank.bank_id_
                or profile.calibration_trials!=profile.fusion_state.fusion_state.calibration_trials
                or tuple(profile.anchors)!=self.bank.providers_):
            raise ValueError('Personal profile internal calibration/source identity differs')

    def _session(self,profile,personal,user_id,session_id):
        self._personal(personal,user_id);_identity(session_id,'session identity')
        if profile is not None and (not isinstance(profile,SessionProfileV1)
                or profile.contract_id!=self.contract_id or profile.user_id!=user_id
                or profile.session_id!=session_id or profile.personal_profile_id!=personal.profile_id):
            raise ValueError('Session profile user/session/personal identity differs')
        if profile is not None and (profile.fusion_policy.source_policy_id!=personal.profile_id
                or profile.calibration_trials!=profile.fusion_state.calibration_trials):
            raise ValueError('Session profile internal calibration/personal identity differs')

    def _calibration(self,batch,trial_ids,labels,offsets,user_id,forbidden):
        if not isinstance(labels,Mapping):raise ValueError('Separate trial-to-label mapping required')
        readout=self.bank.predict_providers(batch,trial_ids,window_offsets=offsets,user_id=user_id)
        ids=readout['trial_ids']
        if set(labels)!=set(ids) or set(labels.values())!=set(self.bank.classes_):
            raise ValueError('Calibration must cover exactly the native trials and every source class')
        if set(ids)&set(forbidden):raise ValueError('Calibration overlaps source/personal/evaluation trials')
        y=np.array([labels[t] for t in ids]);wy=np.array([labels[t] for t in trial_ids])
        return readout,ids,y,wy

    def _profile_id(self,user,session,ids,offsets,labels,batch):
        digest=hashlib.sha256(np.ascontiguousarray(batch.emg).tobytes()).hexdigest()
        ordered_labels=[[t,labels[t].item() if isinstance(labels[t],np.generic) else labels[t]] for t in sorted(labels)]
        fields=[self.contract_id,user,session,list(ids),np.asarray(offsets).tolist(),ordered_labels,
                str(np.asarray(batch.emg).dtype),list(batch.emg.shape),digest]
        return hashlib.sha256(json.dumps(fields).encode()).hexdigest()

    def _activation(self,batch,labels,trials):
        return PersonalActivationProfile().fit_calibration(batch,labels,trials,rest_label=self.rest_label).profile_

    def enroll_user(self,batch,trial_ids,trial_labels,*,window_offsets,user_id,session_id,
                    observed_channel_ids,preprocessing_id,forbidden_evaluation_trials=()):
        _identity(user_id,'user identity');_identity(session_id,'session identity')
        batch=self._input(batch,observed_channel_ids=observed_channel_ids,preprocessing_id=preprocessing_id)
        readout,ids,y,wy=self._calibration(batch,trial_ids,trial_labels,window_offsets,user_id,forbidden_evaluation_trials)
        state=self.bank.calibrate_user(batch,trial_ids,trial_labels,window_offsets=window_offsets,
            user_id=user_id,forbidden_evaluation_trials=forbidden_evaluation_trials)
        anchors={k:DocumentPersonalAnchorV2().fit(v,y) for k,v in readout['source_standardized_features'].items()}
        reference=DocumentSessionDescriptorV3().fit_long_term(batch,wy,np.asarray(trial_ids),user_id=user_id,
            session_ids=np.repeat(session_id,batch.windows),ring_topology=False,rest_label=self.rest_label)
        return PersonalProfileV1(self.contract_id,self._profile_id(user_id,session_id,trial_ids,window_offsets,trial_labels,batch),user_id,session_id,
            tuple(ids),tuple(int(np.sum(y==c)) for c in self.bank.classes_),state,
            DocumentPersonalNormalizerV2(rest_label=self.rest_label).fit(batch,wy),anchors,
            DocumentQualityObservationsV3(**self.quality_options).fit(batch),reference,
            self._activation(batch,wy,np.asarray(trial_ids)))

    def calibrate_session(self,batch,trial_ids,trial_labels,*,window_offsets,user_id,session_id,personal,
                          observed_channel_ids,preprocessing_id,forbidden_evaluation_trials=()):
        self._personal(personal,user_id);_identity(session_id,'session identity')
        if session_id==personal.session_id:raise ValueError('Session calibration requires a new session identity')
        batch=self._input(batch,observed_channel_ids=observed_channel_ids,preprocessing_id=preprocessing_id)
        forbidden=tuple(personal.calibration_trials)+tuple(forbidden_evaluation_trials)
        readout,ids,y,wy=self._calibration(batch,trial_ids,trial_labels,window_offsets,user_id,forbidden)
        policy=AvailableBankFusionPolicyV1(self.bank.classes_,self.bank.providers_,
            personal.fusion_state.fusion_state.weights,self.bank.policy_.n0,self.bank.policy_.temperature,
            personal.profile_id,tuple(self.bank.policy_.source_trials)+tuple(personal.calibration_trials))
        calibration={k:(v,y,ids) for k,v in readout['source_standardized_features'].items()}
        state=policy.calibrate(calibration,forbidden_evaluation_trials=forbidden_evaluation_trials)
        anchors={};betas={}
        count=np.array([np.sum(y==c) for c in self.bank.classes_],dtype=float)
        source_count=np.array(personal.calibration_counts,dtype=float)
        beta=source_count/(source_count+count)
        for name,x in readout['source_standardized_features'].items():
            local=DocumentPersonalAnchorV2().fit(x,y)
            long=personal.anchors[name];blended=copy.deepcopy(long)
            if not np.array_equal(local.classes_,long.classes_):raise ValueError('Anchor class axis differs')
            indices=[self.bank.classes_.index(c) for c in long.classes_]
            b=beta[indices];blended.prototypes_=b[:,None]*long.prototypes_+(1-b[:,None])*local.prototypes_
            anchors[name]={'local':local,'blended':blended};betas[name]=tuple(b)
        descriptor=personal.session_reference.from_calibration(batch,wy,np.asarray(trial_ids),user_id=user_id,
            session_ids=np.repeat(session_id,batch.windows))
        return SessionProfileV1(self.contract_id,self._profile_id(user_id,session_id,trial_ids,window_offsets,trial_labels,batch),personal.profile_id,
            user_id,session_id,tuple(ids),policy,state,
            DocumentPersonalNormalizerV2(rest_label=self.rest_label).fit(batch,wy),anchors,betas,descriptor,
            self._activation(batch,wy,np.asarray(trial_ids)))

    def normalized_view(self,batch,*,personal,user_id,session_id,session=None,observed_channel_ids,preprocessing_id):
        self._session(session,personal,user_id,session_id)
        batch=self._input(batch,observed_channel_ids=observed_channel_ids,preprocessing_id=preprocessing_id)
        return (personal.normalizer if session is None else session.normalizer).transform(batch)

    def predict(self,batch,trial_ids,*,window_offsets,user_id,session_id,personal,session=None,
                observed_channel_ids,preprocessing_id,available=None):
        self._session(session,personal,user_id,session_id)
        batch=self._input(batch,observed_channel_ids=observed_channel_ids,preprocessing_id=preprocessing_id)
        forbidden=set(personal.calibration_trials)|(set() if session is None else set(session.calibration_trials))
        if set(trial_ids)&forbidden:raise ValueError('Personal/session calibration trials cannot become evaluation')
        readout=self.bank.predict_providers(batch,trial_ids,window_offsets=window_offsets,user_id=user_id,available=available)
        ids=readout['trial_ids'];probability=readout['probabilities']
        if session is None:
            policy=self.bank.policy_;state=personal.fusion_state.fusion_state
        else:policy=session.fusion_policy;state=session.fusion_state
        result=policy.predict(state,probability,evaluation_trials=ids,
            provider_trial_ids={k:ids for k in probability},provider_classes={k:self.bank.classes_ for k in probability})
        coordinates={}
        for name,x in readout['source_standardized_features'].items():
            coordinates[name]={'long_term':personal.anchors[name].transform(x)}
            if session is not None:
                coordinates[name].update({mode:anchor.transform(x) for mode,anchor in session.anchors[name].items()})
        result.update(trial_ids=ids,user_id=user_id,session_id=session_id,classes=self.bank.classes_,
            class_names=self.bank.class_names_,personal_profile_id=personal.profile_id,
            session_profile_id=None if session is None else session.profile_id,anchor_coordinates=coordinates,
            anchor_classes={k:tuple(personal.anchors[k].classes_) for k in coordinates},
            quality_observations=personal.quality.transform(batch),quality_feature_names=personal.quality.feature_names,
            session_descriptor=None if session is None else session.descriptor,
            scope='Raw source-classifier probabilities with calibration-only weights; anchors/quality/F8 are separate context outputs, not an unvalidated classifier or rejection gate.',
            physical_validation_proven=False,completion_proven=False)
        return result

    def save_profile(self,profile,path):
        if not isinstance(profile,(PersonalProfileV1,SessionProfileV1)) or profile.contract_id!=self.contract_id:
            raise ValueError('Profile type/source contract differs')
        payload=pickle.dumps(profile,protocol=pickle.HIGHEST_PROTOCOL)
        manifest={'schema':'personal_session_profile_v1','kind':'personal' if isinstance(profile,PersonalProfileV1) else 'session',
            'contract_id':self.contract_id,'bank_id':self.bank.bank_id_,'user_id':profile.user_id,
            'session_id':profile.session_id,'profile_id':profile.profile_id,
            'payload_sha256':hashlib.sha256(payload).hexdigest()}
        with Path(path).open('xb') as handle:
            with zipfile.ZipFile(handle,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('manifest.json',json.dumps(manifest,sort_keys=True))
                archive.writestr('profile.pkl',payload)

    def load_profile(self,path,*,user_id,session_id=None,personal=None):
        _identity(user_id,'user identity')
        with zipfile.ZipFile(path) as archive:
            if sorted(archive.namelist())!=['manifest.json','profile.pkl']:raise ValueError('Unexpected profile archive members')
            manifest=json.loads(archive.read('manifest.json'));payload=archive.read('profile.pkl')
        expected_kind='personal' if session_id is None else 'session'
        if (manifest.get('schema')!='personal_session_profile_v1' or manifest.get('kind')!=expected_kind
                or manifest.get('contract_id')!=self.contract_id or manifest.get('bank_id')!=self.bank.bank_id_
                or manifest.get('user_id')!=user_id or (session_id is not None and manifest.get('session_id')!=session_id)
                or manifest.get('payload_sha256')!=hashlib.sha256(payload).hexdigest()):
            raise ValueError('Saved profile checksum/user/session/source identity differs')
        profile=pickle.loads(payload)
        if isinstance(profile,PersonalProfileV1) and expected_kind=='personal':self._personal(profile,user_id)
        elif isinstance(profile,SessionProfileV1) and expected_kind=='session':self._session(profile,personal,user_id,session_id)
        else:raise ValueError('Saved profile payload kind differs')
        if profile.profile_id!=manifest['profile_id'] or profile.session_id!=manifest['session_id']:
            raise ValueError('Saved profile manifest/payload identity differs')
        return profile
