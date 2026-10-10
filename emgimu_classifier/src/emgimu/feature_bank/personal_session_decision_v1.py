"""Opt-in F7 probabilities, F8 routing and raw F9 around frozen source models.

Prototype temperatures use calibration class geometry only. The fixed anchor
mixture and routing rule are versioned controls, not target-selected defaults.
"""
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path
import pickle
import zipfile
import numpy as np
from .core import FeatureBatch
from .affine_spd_anchor import document_spd_matrices, affine_spd_distance
from .personal_session_workflow_v1 import PersonalSessionWorkflowV1
from .source_quality_gate_v1 import SourceQualityGateV1


@dataclass(frozen=True)
class DecisionPersonalProfileV1:
    policy_id: str
    profile_id: str
    base: object
    spd_prototypes: np.ndarray


@dataclass(frozen=True)
class DecisionSessionProfileV1:
    policy_id: str
    profile_id: str
    personal_profile_id: str
    base: object
    spd_local: np.ndarray
    spd_blended: np.ndarray
    routing: dict


def trial_covariances(batch, ids):
    ids=np.asarray(ids)
    return np.stack([document_spd_matrices(batch.emg[ids==t]).mean(0) for t in np.unique(ids)])


def spd_distances(matrices, prototypes):
    """Batched exact affine-invariant distances; no tangent approximation."""
    matrices=np.asarray(matrices,float);prototypes=np.asarray(prototypes,float)
    if (matrices.ndim!=3 or matrices.shape[1:]!=(8,8) or prototypes.ndim!=3
            or prototypes.shape[1:]!=(8,8) or not np.isfinite(matrices).all() or not np.isfinite(prototypes).all()
            or not np.allclose(matrices,matrices.transpose(0,2,1),atol=1e-10)
            or not np.allclose(prototypes,prototypes.transpose(0,2,1),atol=1e-10)
            or np.linalg.eigvalsh(matrices).min()<=0 or np.linalg.eigvalsh(prototypes).min()<=0):
        raise ValueError('Finite symmetric eight-channel SPD matrices required')
    columns=[]
    for prototype in prototypes:
        values,vectors=np.linalg.eigh(prototype);inverse=(vectors*values**-.5)@vectors.T
        relative=inverse[None]@matrices@inverse[None]
        eigenvalues=np.linalg.eigvalsh((relative+relative.transpose(0,2,1))/2)
        if eigenvalues.min()<=0:raise ValueError('Numerically nonpositive relative covariance')
        columns.append(np.linalg.norm(np.log(eigenvalues),axis=1))
    return np.column_stack(columns)


def anchor_probability(features, prototypes, *, spd=False):
    x=np.asarray(features,float);p=np.asarray(prototypes,float)
    if spd:
        distance=spd_distances(x,p);geometry=spd_distances(p,p)
    else:
        if x.ndim!=2 or p.ndim!=2 or x.shape[1]!=p.shape[1] or len(p)<2 or not np.isfinite(x).all() or not np.isfinite(p).all():
            raise ValueError('Finite aligned features and at least two class prototypes required')
        distance=np.linalg.norm(x[:,None]-p[None],axis=2)
        geometry=np.linalg.norm(p[:,None]-p[None],axis=2)
    temperature=float(geometry[np.triu_indices(len(p),1)].mean())
    # A collapsed calibration geometry contributes no invented discrimination.
    if temperature<=1e-10:return np.full(distance.shape,1/len(p)),distance,temperature
    logits=-distance/temperature;logits-=logits.max(1,keepdims=True)
    q=np.exp(logits);q/=q.sum(1,keepdims=True)
    return q,distance,temperature


def session_routing(personal, session, spd_long, spd_local, providers):
    """Calibration-only descriptor/risk, normalized by long-term class geometry."""
    risks=[];phi=[];names=[];specific={}
    reference=personal.session_reference.reference_;classes=tuple(personal.anchors[providers[0]].classes_)
    entries=[session.descriptor['classes'][str(c)] for c in classes]
    pairs=list(combinations(range(len(classes)),2))
    def ratio(value,scale):return float(value/scale) if scale>1e-10 else 0.
    log_scale=np.array([float(reference[c]['log_scale']) for c in classes])
    scale=float(np.mean([abs(log_scale[i]-log_scale[j]) for i,j in pairs]))
    specific['F0']=ratio(np.mean([abs(e['log_global_activation_shift']) for e in entries]),scale)+float(np.mean([np.mean(abs(np.asarray(e['channel_quality_score_shift']))) for e in entries]))
    for group,key,field in [('F1','pattern','scale_pattern_residual_norm'),('F4abc','log_bands','mean_absolute_log_band_residual')]:
        norm=(lambda a,b:np.linalg.norm(a-b)) if group=='F1' else (lambda a,b:np.abs(a-b).mean())
        scale=float(np.mean([norm(reference[classes[i]][key],reference[classes[j]][key]) for i,j in pairs]))
        specific[group]=ratio(np.mean([e[field] for e in entries]),scale)
    for group in providers:
        long=personal.anchors[group].prototypes_;local=session.anchors[group]['local'].prototypes_
        distances=np.linalg.norm(local-long,axis=1)
        cosine=np.sum(local*long,axis=1)/(np.linalg.norm(local,axis=1)*np.linalg.norm(long,axis=1)+1e-10)
        geometry=np.array([np.linalg.norm(local[i]-local[j])-np.linalg.norm(long[i]-long[j]) for i,j in pairs])
        for metric,values in [('residual_norm',distances),('cosine_agreement',cosine),('geometry_delta',geometry)]:
            phi.extend(values.tolist());names.extend(f'F8.{group}.{metric}.{i}' for i in range(len(values)))
        scale=float(np.mean([np.linalg.norm(long[i]-long[j]) for i,j in pairs]))
        drift=ratio(distances.mean(),scale);geometry_risk=ratio(abs(geometry).mean(),scale)
        if group=='F2ac':
            source_geometry=np.array([affine_spd_distance(spd_long[i],spd_long[j]) for i,j in pairs])
            current_geometry=np.array([affine_spd_distance(spd_local[i],spd_local[j]) for i,j in pairs])
            scale=float(source_geometry.mean())
            drift=ratio(np.mean([affine_spd_distance(a,b) for a,b in zip(spd_long,spd_local)]),scale)
            geometry_risk=ratio(abs(current_geometry-source_geometry).mean(),scale)
        components=[drift,geometry_risk]
        if group in specific:components.append(specific[group])
        risks.append(float(np.mean(components)))
    risks=np.asarray(risks)
    if not np.isfinite(risks).all() or np.any(risks<0):raise ValueError('Finite nonnegative calibration routing risks required')
    return dict(providers=tuple(providers),risks=risks,multipliers=np.maximum(np.exp(-np.clip(risks,0,5)),.05),
                feature_names=tuple(names),descriptor=np.asarray(phi),
                scope='Calibration-only class-matched generic shifts plus named family-specific terms; not a gesture predictor or physical-fault detector')


class PersonalSessionDecisionV1:
    def __init__(self,base,*,anchor_mix=.5,gate=None):
        if not isinstance(base,PersonalSessionWorkflowV1) or base.bank.sensor_contract_[2]!=8:
            raise ValueError('Eight-channel source workflow required')
        if not np.isfinite(anchor_mix) or not 0<=anchor_mix<=1:raise ValueError('Explicit anchor mixture in[0,1] required')
        if gate is not None and (not isinstance(gate,SourceQualityGateV1) or gate.channels!=base.channels
                                or set(gate.source_trials)!=set(base.bank.policy_.source_trials)
                                or base.bank.sensor_contract_!=(250.,50,8)):
            raise ValueError('Quality/source/channel identities differ')
        self.base=base;self.bank=base.bank;self.channels=base.channels;self.rest_label=base.rest_label
        self.preprocessing_id=base.preprocessing_id;self.anchor_mix=float(anchor_mix);self.gate=gate
        self.policy_id=hashlib.sha256(json.dumps(['personal_session_decision_v1',base.contract_id,self.anchor_mix,
                                  None if gate is None else gate.policy_id,'calibration_mean_pair_temperature','mean_drift_geometry_specific_exp_clip5_floor005']).encode()).hexdigest()
        self.contract_id=self.policy_id

    def _input(self,*args,**kwargs):return self.base._input(*args,**kwargs)

    def _personal(self,personal,user):
        if not isinstance(personal,DecisionPersonalProfileV1) or personal.policy_id!=self.policy_id:raise ValueError('Decision personal policy differs')
        self.base._personal(personal.base,user)
        spd_distances(personal.spd_prototypes,personal.spd_prototypes)
        if len(personal.spd_prototypes)!=len(self.bank.classes_):raise ValueError('Personal covariance class axis differs')

    def _session(self,session,personal,user,session_id):
        if session is not None and not isinstance(session,DecisionSessionProfileV1):raise ValueError('Decision session type differs')
        self._personal(personal,user);self.base._session(None if session is None else session.base,personal.base,user,session_id)
        if session is not None:
            if (not isinstance(session,DecisionSessionProfileV1) or session.policy_id!=self.policy_id
                    or session.personal_profile_id!=personal.profile_id):raise ValueError('Decision session/personal policy differs')
            if tuple(session.routing['providers'])!=self.bank.providers_:raise ValueError('Session routing provider axis differs')

    def _id(self,base,kind):return hashlib.sha256(json.dumps([self.policy_id,base.profile_id,kind]).encode()).hexdigest()

    def _prototypes(self,batch,ids,labels,channels,preprocessing):
        batch=self._input(batch,observed_channel_ids=channels,preprocessing_id=preprocessing)
        matrices=trial_covariances(batch,ids);y=np.asarray([labels[t] for t in np.unique(ids)])
        return np.stack([matrices[y==c].mean(0) for c in self.bank.classes_])

    def enroll_user(self,batch,trial_ids,trial_labels,**kwargs):
        old=self.base.enroll_user(batch,trial_ids,trial_labels,**kwargs)
        prototypes=self._prototypes(batch,trial_ids,trial_labels,kwargs['observed_channel_ids'],kwargs['preprocessing_id'])
        return DecisionPersonalProfileV1(self.policy_id,self._id(old,'personal'),old,prototypes)

    def calibrate_session(self,batch,trial_ids,trial_labels,*,personal,**kwargs):
        self._personal(personal,kwargs['user_id'])
        old=self.base.calibrate_session(batch,trial_ids,trial_labels,personal=personal.base,**kwargs)
        local=self._prototypes(batch,trial_ids,trial_labels,kwargs['observed_channel_ids'],kwargs['preprocessing_id'])
        beta=np.array(old.prototype_beta[self.bank.providers_[0]])
        blended=beta[:,None,None]*personal.spd_prototypes+(1-beta[:,None,None])*local
        routing=session_routing(personal.base,old,personal.spd_prototypes,local,self.bank.providers_)
        return DecisionSessionProfileV1(self.policy_id,self._id(old,'session'),personal.profile_id,old,local,blended,routing)

    def predict(self,batch,trial_ids,*,window_offsets,user_id,session_id,personal=None,session=None,
                observed_channel_ids,preprocessing_id,available=None,use_anchor=True,use_session_routing=True,
                anchor_mode='blended',quality_mode='off',raw_batch=None):
        if anchor_mode not in ('long_term','local','blended') or quality_mode not in ('off','structural','soft'):
            raise ValueError('Explicit anchor and quality modes required')
        if type(use_anchor) is not bool or type(use_session_routing) is not bool:raise ValueError('Explicit Boolean branch switches required')
        mapped=self._input(batch,observed_channel_ids=observed_channel_ids,preprocessing_id=preprocessing_id)
        if personal is None:
            if session is not None:raise ValueError('Session requires personal profile')
            result=self.bank.predict(mapped,trial_ids,window_offsets=window_offsets,user_id=user_id,available=available)
        else:
            self._session(session,personal,user_id,session_id)
            result=self.base.predict(batch,trial_ids,window_offsets=window_offsets,user_id=user_id,session_id=session_id,
                personal=personal.base,session=None if session is None else session.base,
                observed_channel_ids=observed_channel_ids,preprocessing_id=preprocessing_id,available=available)
        readout=self.bank.predict_providers(mapped,trial_ids,window_offsets=window_offsets,user_id=user_id,available=available)
        providers=readout['probabilities'];weights=np.array(result['weights'],float)
        active=tuple(providers);anchors={};distances={};temperatures={}
        effective_mode='long_term' if session is None else anchor_mode
        if personal is not None and use_anchor:
            matrices=trial_covariances(mapped,trial_ids) if 'F2ac' in active else None
            for name in active:
                if name=='F2ac':
                    p=personal.spd_prototypes if effective_mode=='long_term' else session.spd_local if effective_mode=='local' else session.spd_blended
                    q,d,t=anchor_probability(matrices,p,spd=True)
                else:
                    anchor=personal.base.anchors[name] if effective_mode=='long_term' else session.base.anchors[name][effective_mode]
                    indices=[list(anchor.classes_).index(c) for c in self.bank.classes_]
                    q,d,t=anchor_probability(readout['source_standardized_features'][name],anchor.prototypes_[indices])
                anchors[name]=q;distances[name]=d;temperatures[name]=t
            providers={name:(1-self.anchor_mix)*providers[name]+self.anchor_mix*anchors[name] for name in active}
        routing_applied=personal is not None and session is not None and use_session_routing
        if routing_applied:
            index=[self.bank.providers_.index(name) for name in active]
            weights*=session.routing['multipliers'][index];weights/=weights.sum()
        quality=None
        if quality_mode!='off':
            if self.gate is None or raw_batch is None:raise ValueError('Enabled quality needs bound source gate and separate raw pre-highpass windows')
            if raw_batch.emg.shape!=batch.emg.shape or raw_batch.sample_rate_hz!=batch.sample_rate_hz:raise ValueError('Raw/filtered sensor shape differs')
            quality=self.gate.decide(providers,weights,raw_batch,trial_ids,observed_channel_ids=observed_channel_ids,
                class_names=self.bank.class_names_,mode=quality_mode,provider_trial_ids={name:readout['trial_ids'] for name in active})
            q=quality['probabilities'];labels=quality['labels']
        else:
            q=sum(weights[i]*providers[name] for i,name in enumerate(active));q/=q.sum(1,keepdims=True)
            labels=tuple(self.bank.class_names_[i] for i in q.argmax(1))
        result.update(probabilities=q,predicted_labels=labels,weights=tuple(weights),decision_policy_id=self.policy_id,
            personal_profile_id=None if personal is None else personal.profile_id,session_profile_id=None if session is None else session.profile_id,
            anchor_enabled=bool(anchors),anchor_mode=effective_mode,anchor_probabilities=anchors,anchor_distances=distances,
            anchor_temperatures=temperatures,decision_provider_probabilities=providers,session_routing_enabled=routing_applied,
            session_routing=None if not routing_applied else session.routing,quality_decision=quality,
            quality_rejection_enabled=quality_mode!='off',physical_validation_proven=False,completion_proven=False,
            scope='Versioned six-provider source/F7 fusion with calibration-only temperatures and F8 routing; optional separately bound raw F9. Not all document subfamilies or hardware efficacy.')
        return result

    def save_profile(self,profile,path):
        if not isinstance(profile,(DecisionPersonalProfileV1,DecisionSessionProfileV1)) or profile.policy_id!=self.policy_id:
            raise ValueError('Decision profile source policy differs')
        payload=pickle.dumps(profile,protocol=pickle.HIGHEST_PROTOCOL)
        manifest=dict(schema='personal_session_decision_profile_v1',kind='personal' if isinstance(profile,DecisionPersonalProfileV1) else 'session',
            policy_id=self.policy_id,profile_id=profile.profile_id,user_id=profile.base.user_id,session_id=profile.base.session_id,
            payload_sha256=hashlib.sha256(payload).hexdigest())
        with Path(path).open('xb') as f:
            with zipfile.ZipFile(f,'w',compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr('manifest.json',json.dumps(manifest,sort_keys=True));z.writestr('profile.pkl',payload)

    def load_profile(self,path,*,user_id,session_id=None,personal=None):
        with zipfile.ZipFile(path) as z:
            if sorted(z.namelist())!=['manifest.json','profile.pkl']:raise ValueError('Unexpected decision profile members')
            m=json.loads(z.read('manifest.json'));payload=z.read('profile.pkl')
        kind='personal' if session_id is None else 'session'
        if (m.get('schema')!='personal_session_decision_profile_v1' or m.get('policy_id')!=self.policy_id
                or m.get('kind')!=kind or m.get('user_id')!=user_id
                or (session_id is not None and m.get('session_id')!=session_id)
                or m.get('payload_sha256')!=hashlib.sha256(payload).hexdigest()):raise ValueError('Saved decision profile identity/checksum differs')
        profile=pickle.loads(payload)
        if kind=='personal':self._personal(profile,user_id)
        else:self._session(profile,personal,user_id,session_id)
        if profile.profile_id!=m['profile_id'] or profile.base.session_id!=m['session_id']:raise ValueError('Decision manifest/payload differs')
        return profile
