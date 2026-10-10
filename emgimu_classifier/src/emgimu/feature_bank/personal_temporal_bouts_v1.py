"""Personal/session full-envelope medoids and low-order path probabilities.

Complete cued sequences and estimated intervals are distinct. Short/sparse
window banks cannot be passed as complete action sequences. Base classifier
probabilities are supplied on an explicit matching trial/class axis.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import pickle
import zipfile
import numpy as np
from .document_path_v3 import DocumentTemporalTemplatesV3, DocumentPathSignatureV3, document_envelope_path, document_dtw_distance
from .temporal import CompleteSequenceBatch


@dataclass(frozen=True)
class TemporalBoutBatchV1:
    sequences: tuple
    trial_ids: tuple
    recording_ids: tuple
    starts: tuple
    sample_rate_hz: float
    channel_ids: tuple
    preprocessing_id: str
    boundary_kind: str

    def validate(self):
        n = len(self.sequences)
        if (not n or any(len(v) != n for v in (self.trial_ids,self.recording_ids,self.starts))
                or len(set(self.trial_ids)) != n or not self.channel_ids
                or len(set(self.channel_ids)) != len(self.channel_ids)
                or any(not isinstance(v,str) or not v.strip() for v in (*self.trial_ids,*self.recording_ids,*self.channel_ids))
                or not isinstance(self.preprocessing_id,str) or not self.preprocessing_id.strip()
                or not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0
                or self.boundary_kind not in ('complete_cued','estimated')):
            raise ValueError('Explicit unique full-bout trial/sensor/preprocessing/boundary provenance required')
        for x,start in zip(self.sequences,self.starts):
            x = np.asarray(x)
            if (x.ndim != 2 or len(x) < 32 or x.shape[1] != len(self.channel_ids) or not np.isfinite(x).all()
                    or not 1. <= len(x)/self.sample_rate_hz <= 30.
                    or isinstance(start,bool) or not isinstance(start,(int,np.integer)) or start < 0):
                raise ValueError('Contiguous finite native bouts of1..30 seconds with nonnegative starts required')
        return self

    def paths(self):
        self.validate()
        # Every native sample contributes; no sparse window interpolation.
        return np.stack([np.stack([np.sqrt(np.mean(part.astype(float)**2,axis=0))
            for part in np.array_split(np.asarray(x),32)]) for x in self.sequences])


@dataclass(frozen=True)
class TemporalProfileV1:
    contract_id: str
    profile_id: str
    user_id: str
    session_id: str
    kind: str
    personal_profile_id: str | None
    trial_ids: tuple
    recording_ids: tuple
    medoid_trial_ids: tuple
    templates: np.ndarray
    signature_mean: np.ndarray
    signature_scale: np.ndarray
    signature_prototypes: np.ndarray


def profile_identity(profile):
    body = asdict(profile)
    body.pop('profile_id')
    return hashlib.sha256(pickle.dumps(body,protocol=4)).hexdigest()


def distance_probability(distance,geometry):
    values = geometry[np.triu_indices(len(geometry),1)]
    temperature = float(values.mean())
    if temperature <= 1e-10:
        return np.full(distance.shape,1/distance.shape[1]),temperature
    logits = -distance/temperature
    logits -= logits.max(1,keepdims=True)
    q = np.exp(logits);q /= q.sum(1,keepdims=True)
    return q,temperature


class PersonalTemporalBoutsV1:
    def __init__(self, *, source_bank_id, sample_rate_hz, channel_ids, preprocessing_id, class_names,
                 temporal_mix=.25, band_fraction=.1):
        self.channel_ids = tuple(channel_ids)
        self.class_names = tuple(class_names)
        self.rate = float(sample_rate_hz)
        self.preprocessing_id = preprocessing_id
        self.source_bank_id = source_bank_id
        self.mix = float(temporal_mix)
        self.band_fraction = float(band_fraction)
        if (not isinstance(source_bank_id,str) or not source_bank_id.strip()
                or not isinstance(preprocessing_id,str) or not preprocessing_id.strip()
                or not np.isfinite(self.rate) or self.rate <= 0 or len(self.channel_ids) < 1
                or len(set(self.channel_ids)) != len(self.channel_ids)
                or len(self.class_names) < 2 or len(set(self.class_names)) != len(self.class_names)
                or any(not isinstance(v,str) or not v.strip() for v in (*self.channel_ids,*self.class_names))
                or not np.isfinite(self.mix) or not 0 <= self.mix <= 1
                or not np.isfinite(self.band_fraction) or not 0 < self.band_fraction <= 1):
            raise ValueError('Explicit source/sensor/classes and fixed valid temporal policy required')
        self.band = max(1,round(32*self.band_fraction))
        self.contract_id = hashlib.sha256(json.dumps(['personal_temporal_bouts_v1',source_bank_id,
            self.rate,self.channel_ids,preprocessing_id,self.class_names,self.mix,self.band],sort_keys=True).encode()).hexdigest()

    def _input(self,batch):
        if not isinstance(batch,TemporalBoutBatchV1):
            raise ValueError('Native complete/estimated bout input required; window banks are ineligible')
        batch.validate()
        if (batch.sample_rate_hz != self.rate or batch.channel_ids != self.channel_ids
                or batch.preprocessing_id != self.preprocessing_id):
            raise ValueError('Temporal native sensor/preprocessing contract differs')

    def _profile(self,p,user):
        if (not isinstance(p,TemporalProfileV1) or p.contract_id != self.contract_id
                or p.user_id != user or profile_identity(p) != p.profile_id):
            raise ValueError('Temporal profile checksum/source/user identity differs')

    def enroll(self,batch,labels,*,user_id,session_id,personal=None,forbidden_trial_ids=(),forbidden_recording_ids=()):
        self._input(batch)
        if (batch.boundary_kind != 'complete_cued' or not isinstance(user_id,str) or not user_id.strip()
                or not isinstance(session_id,str) or not session_id.strip()
                or set(batch.trial_ids) & set(forbidden_trial_ids)
                or set(batch.recording_ids) & set(forbidden_recording_ids)):
            raise ValueError('Complete calibration coverage, identity and evaluation disjointness required')
        if not isinstance(labels,dict) or set(labels) != set(batch.trial_ids) or set(labels.values()) != set(self.class_names):
            raise ValueError('Separate complete calibration labels for all declared classes required')
        if personal is not None:
            self._profile(personal,user_id)
            if (personal.kind != 'personal' or personal.session_id == session_id
                    or set(batch.trial_ids) & set(personal.trial_ids)
                    or set(batch.recording_ids) & set(personal.recording_ids)):
                raise ValueError('New-session temporal calibration must be disjoint from long-term recording')
        paths = batch.paths()
        complete = CompleteSequenceBatch(paths,1.,durations_seconds=np.array([len(x)/self.rate for x in batch.sequences]),full_coverage=True)
        y = np.array([labels[t] for t in batch.trial_ids])
        templates = DocumentTemporalTemplatesV3(self.band_fraction).fit(complete,y,trial_ids=batch.trial_ids)
        order = [list(templates.classes_).index(c) for c in self.class_names]
        signature = DocumentPathSignatureV3().fit(complete).transform(complete)
        mean = signature.mean(0) if personal is None else personal.signature_mean.copy()
        scale = signature.std(0) if personal is None else personal.signature_scale.copy()
        scale = np.where(scale > 1e-10,scale,1.)
        standardized = (signature-mean)/scale
        prototypes = np.stack([standardized[y == c].mean(0) for c in self.class_names])
        p = TemporalProfileV1(self.contract_id,'',user_id,session_id,'personal' if personal is None else 'session',
            None if personal is None else personal.profile_id,tuple(batch.trial_ids),tuple(batch.recording_ids),
            tuple(templates.medoid_trial_ids_[i] for i in order),np.stack([templates.templates_[i] for i in order]),
            mean,scale,prototypes)
        return TemporalProfileV1(**{**asdict(p),'profile_id':profile_identity(p)})

    def _read(self,paths,signature,p):
        distance = np.array([[document_dtw_distance(x,t,self.band) for t in p.templates] for x in paths])
        geometry = np.array([[document_dtw_distance(a,b,self.band) for b in p.templates] for a in p.templates])
        dtw,dtw_temperature = distance_probability(distance,geometry)
        z = (signature-p.signature_mean)/p.signature_scale
        d = np.linalg.norm(z[:,None]-p.signature_prototypes[None],axis=2)
        geometry = np.linalg.norm(p.signature_prototypes[:,None]-p.signature_prototypes[None],axis=2)
        sig,sig_temperature = distance_probability(d,geometry)
        return dict(dtw=dtw,signature=sig,dtw_distances=distance,signature_distances=d,
            dtw_temperature=dtw_temperature,signature_temperature=sig_temperature)

    def predict(self,batch,*,personal,user_id,session_id,session=None,base_probabilities=None,
                base_trial_ids=None,base_class_names=None):
        self._input(batch);self._profile(personal,user_id)
        if not isinstance(session_id,str) or not session_id.strip():
            raise ValueError('Explicit evaluation session identity required')
        if personal.kind != 'personal':
            raise ValueError('A long-term personal temporal profile is required')
        if session is not None:
            self._profile(session,user_id)
            if (session.kind != 'session' or session.personal_profile_id != personal.profile_id or session.session_id != session_id):
                raise ValueError('Temporal session parent/recording identity differs')
        for profile in (personal,session):
            if profile is not None and (set(batch.trial_ids) & set(profile.trial_ids) or set(batch.recording_ids) & set(profile.recording_ids)):
                raise ValueError('Evaluation overlaps temporal calibration trial/recording')
        envelopes = batch.paths();paths = np.stack([document_envelope_path(x) for x in envelopes])
        # This direct order2 expression applies to both certified and estimated
        # intervals without converting an estimated interval into certified data.
        delta = np.diff(paths-paths[:,:1],axis=1);prefix = np.cumsum(delta,axis=1)-delta
        signature = np.concatenate((delta.sum(1),(np.einsum('ntc,ntd->ncd',prefix,delta)+
            .5*np.einsum('ntc,ntd->ncd',delta,delta)).reshape(len(paths),-1)),axis=1)
        long = self._read(paths,signature,personal)
        local = None if session is None else self._read(paths,signature,session)
        arms = {'DTW_long':long['dtw'],'signature_long':long['signature']}
        arms['DTW_local'] = long['dtw'] if local is None else local['dtw']
        arms['DTW_blended'] = long['dtw'] if local is None else .5*(long['dtw']+local['dtw'])
        arms['signature_blended'] = long['signature'] if local is None else .5*(long['signature']+local['signature'])
        if base_probabilities is not None:
            q = np.asarray(base_probabilities,dtype=float)
            if (tuple(() if base_trial_ids is None else base_trial_ids) != batch.trial_ids or tuple(() if base_class_names is None else base_class_names) != self.class_names
                    or q.shape != (len(paths),len(self.class_names)) or not np.isfinite(q).all()
                    or np.any(q < 0) or not np.allclose(q.sum(1),1.,rtol=0,atol=1e-12)):
                raise ValueError('Explicit matching finite base probability trial/class axes required')
            arms['base'] = q
            for name in ('DTW_long','DTW_blended','signature_blended'):
                arms['base_'+name] = (1-self.mix)*q+self.mix*arms[name]
            arms['base_full'] = (1-self.mix)*q+.5*self.mix*(arms['DTW_blended']+arms['signature_blended'])
            arms['base_uniform'] = (1-self.mix)*q+self.mix/len(self.class_names)
        return dict(trial_ids=batch.trial_ids,class_names=self.class_names,arms=arms,long=long,local=local,
            signature_coordinates=signature,boundary_kind=batch.boundary_kind,
            certified_full_coverage=batch.boundary_kind == 'complete_cued',session_template_enabled=session is not None,
            personal_profile_id=personal.profile_id,session_profile_id=None if session is None else session.profile_id,
            output_after_complete_interval=True,default_promoted=False,physical_validation_proven=False)

    def save_profile(self,p,path):
        self._profile(p,p.user_id)
        payload = pickle.dumps(p,protocol=pickle.HIGHEST_PROTOCOL)
        manifest = dict(schema='personal_temporal_profile_v1',contract_id=self.contract_id,
            profile_id=p.profile_id,payload_sha256=hashlib.sha256(payload).hexdigest())
        with Path(path).open('xb') as stream:
            with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('manifest.json',json.dumps(manifest));archive.writestr('profile.pkl',payload)

    def load_profile(self,path,*,user_id):
        with zipfile.ZipFile(path) as archive:
            if sorted(archive.namelist()) != ['manifest.json','profile.pkl']:
                raise ValueError('Unexpected temporal profile archive members')
            manifest = json.loads(archive.read('manifest.json'));payload = archive.read('profile.pkl')
        if (manifest.get('schema') != 'personal_temporal_profile_v1' or manifest.get('contract_id') != self.contract_id
                or manifest.get('payload_sha256') != hashlib.sha256(payload).hexdigest()):
            raise ValueError('Temporal package checksum/contract differs')
        p = pickle.loads(payload);self._profile(p,user_id)
        if p.profile_id != manifest.get('profile_id'):
            raise ValueError('Temporal manifest/profile identity differs')
        return p
